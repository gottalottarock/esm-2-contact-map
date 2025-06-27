import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional

import lightning as L
import numpy as np
from utils.metrics import ContactPredictionMetrics
from registry import BaseModelConfig, register_model
from transformers import EsmModel
from peft import LoraConfig, get_peft_model, TaskType
from torchvision.ops.focal_loss import sigmoid_focal_loss
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContactHead(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super(ContactHead, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)

        self.final_proj = nn.Linear(num_heads, 1)

    def symmetrize(self, attention_maps):
        """Symmetrize the attention maps."""
        return (attention_maps + attention_maps.transpose(-1, -2)) / 2

    def apply_apc_correction(self, attention_maps):
        """
        Apply Average Product Correction (APC) tensorized version.
        F^APC_ij = F_ij - (F_i * F_j) / F
        """
        # attention_maps: (batch_size, num_maps, seq_len, seq_len)

        # Compute row and column sums
        F_i = attention_maps.sum(dim=-1, keepdim=True)  # (batch, num_maps, seq_len, 1)
        F_j = attention_maps.sum(dim=-2, keepdim=True)  # (batch, num_maps, 1, seq_len)
        F_total = attention_maps.sum(
            dim=(-2, -1), keepdim=True
        )  # (batch, num_maps, 1, 1)

        # Apply APC correction
        F_apc = attention_maps - (F_i * F_j) / (F_total + 1e-8)

        return F_apc

    def forward(self, x, mask):
        B, L, D = x.size()  # batch, seq_len, dim
        x = x[:, 1:-1, :]  # remove first and last token (sep tokens and padding)

        q = self.q_proj(x).reshape(B, L - 2, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L - 2, self.num_heads, self.head_dim)

        # attention-like similarity
        logits = (
            torch.einsum("blhd,bshd->bhls", q, k)
            / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        ) * mask.unsqueeze(1)

        # Symmetrize
        logits = self.symmetrize(logits)

        # APC normalization
        logits_apc = self.apply_apc_correction(logits)

        # Collapse heads dimension
        logits_apc = logits_apc.permute(0, 2, 3, 1)  # (B, L-2, L-2, num_heads)
        contact_map = self.final_proj(logits_apc).squeeze(-1) * mask  # (B, L-2, L-2)

        return contact_map


@dataclass
class LossConfig:
    loss_type: str = "bce"  # bce or focal
    focal_gamma: float = 2.0
    focal_alpha: float = -1


@dataclass
class ESM2LoRAContactConfig(BaseModelConfig):
    """Configuration for ESM2 with LoRA adapters for contact prediction."""

    # Model backbone selection
    backbone: str
    loss: LossConfig = field(default_factory=LossConfig)

    # if true, will not use LoRA
    wo_lora: bool = False
    # LoRA parameters
    lora_rank: int = 8
    lora_alpha: int = 16  # 2*rnak
    lora_dropout: float = 0.1
    target_modules: Optional[list] = (
        None  # If None, will use default ["query", "key", "value"]
    )

    # Contact head parameters
    contact_head_dim: int = 4

    # Training parameters
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    backbone_init_from: Optional[str] = None


class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.9, reduction: str = "mean"):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        loss = sigmoid_focal_loss(
            input, target, gamma=self.gamma, alpha=self.alpha, reduction=self.reduction
        )
        return loss


@register_model("esm2_lora_contact", ESM2LoRAContactConfig)
class ESM2LoRAContact(L.LightningModule):
    def __init__(self, config: ESM2LoRAContactConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # Load base ESM2 model
        self.esm_model = EsmModel.from_pretrained(self.config.backbone)

        # Configure LoRA
        if self.config.target_modules is None:
            target_modules = ["query", "key", "value"]
        else:
            target_modules = self.config.target_modules

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
        )



        if not self.config.wo_lora:
            # Apply LoRA to the model
            self.esm_model = get_peft_model(self.esm_model, lora_config)

        if self.config.backbone_init_from:
            logger.info(f"Loading backbone from {self.config.backbone_init_from}")
            ckpt = torch.load(self.config.backbone_init_from, weights_only=False)
            state_dict = ckpt["state_dict"]
            esm_sd = {
                k.replace("esm_model.", ""): v
                for k, v in state_dict.items()
                if k.startswith("esm_model.")
            }
            self.esm_model.load_state_dict(esm_sd, strict=True)
        self.freeze_backbone()

        # Contact prediction head
        self.init_contact_head()

        # Loss function
        if self.config.loss.loss_type == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif self.config.loss.loss_type == "focal":
            self.loss_fn = SigmoidFocalLoss(
                gamma=self.config.loss.focal_gamma, alpha=self.config.loss.focal_alpha
            )
        else:
            raise ValueError(f"Invalid loss type: {self.config.loss.loss_type}")

        # Initialize metrics calculator
        self.contact_metrics = ContactPredictionMetrics()

        self.validation_step_outputs = []

    def init_contact_head(self):
        embed_dim = self.esm_model.config.hidden_size
        self.contact_head = ContactHead(
            embed_dim=embed_dim, num_heads=self.config.contact_head_dim
        )

    def freeze_backbone(self):
        """Freeze the backbone model (but keep LoRA adapters trainable)."""
        for name, param in self.esm_model.named_parameters():
            if "lora" not in name:
                param.requires_grad = False

    def forward(self, input_ids, attention_mask, mask_2d):
        """Forward pass through ESM2 + LoRA + ContactHead."""
        # Get ESM2 embeddings
        outputs = self.esm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Use last hidden state
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

        # Apply contact head
        contact_logits = self.contact_head(hidden_states, mask_2d)

        return contact_logits

    def create_mask(self, seq_lengths, max_length):
        """Create mask for valid positions (excluding padding and special tokens)"""
        # Create basic sequence length mask
        mask_2d = torch.zeros(
            seq_lengths.shape[0],
            max_length,
            max_length,
            device=seq_lengths.device,
            dtype=torch.bool,
        )
        for i, length in enumerate(seq_lengths):
            mask_2d[i, :length, :length] = True

        return mask_2d

    def compute_loss(self, contact_logits, contact_map, mask_2d):
        """Compute loss for valid positions"""
        # only consider lower triangular part to avoid double counting symmetric contacts
        batch_size, seq_len, _ = mask_2d.shape

        #  lower triangular mask
        tril_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=mask_2d.device, dtype=torch.bool),
            diagonal=-1,
        )
        tril_mask = tril_mask.unsqueeze(0).expand(batch_size, -1, -1).bool()

        # combine with existing mask
        final_mask = mask_2d & tril_mask

        loss = self.loss_fn(contact_logits[final_mask], contact_map[final_mask])
        return loss

    def _forward_pass(self, batch):
        """Common forward pass logic for train/val/predict steps."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        seq_lengths = batch["seq_lengths"]

        max_length = input_ids.shape[-1] - 2  # -2 for CLS and EOS tokens
        mask_2d = self.create_mask(seq_lengths, max_length)

        # Forward pass
        contact_logits = self(input_ids, attention_mask, mask_2d)

        return contact_logits, mask_2d

    def training_step(self, batch, batch_idx):
        """Training step."""
        contact_logits, mask_2d = self._forward_pass(batch)
        contact_map = batch["contact_maps"]

        # Compute loss only on valid positions
        loss = self.compute_loss(contact_logits, contact_map, mask_2d)
        self.log(
            "train_loss",
            loss,
            batch_size=batch["input_ids"].shape[0],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        contact_logits, mask_2d = self._forward_pass(batch)
        contact_map = batch["contact_maps"]

        # Compute loss only on valid positions
        loss = self.compute_loss(contact_logits, contact_map, mask_2d).item()

        # Compute comprehensive metrics
        metrics = self.contact_metrics.compute_all_metrics(
            contact_logits, contact_map, batch["seq_lengths"], mask_2d, average=False, add_other_metrics=True
        )

        self.validation_step_outputs.append({"loss": loss, "metrics": metrics})

    def on_validation_epoch_end(self):
        """Compute and log validation metrics."""
        loss = np.mean([output["loss"] for output in self.validation_step_outputs])
        metrics = {
            metric_name: np.mean(
                [
                    value
                    for output in self.validation_step_outputs
                    for value in output["metrics"][metric_name]
                ]
            )
            for metric_name in self.validation_step_outputs[0]["metrics"].keys()
        }
        self.log("val_loss", float(loss))
        for metric_name, metric_value in metrics.items():
            self.log(f"val_{metric_name}", float(metric_value))

        # Clear validation outputs for next epoch
        self.validation_step_outputs.clear()

    def predict_step(self, batch, batch_idx):
        """Prediction step for inference."""
        contact_logits, mask_2d = self._forward_pass(batch)

        return {
            "contact_logits": contact_logits,
            "mask_2d": mask_2d,
            "metadata": batch["metadata"],
        }

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        # Get all trainable parameters (LoRA adapters + contact head)
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        return optimizer
