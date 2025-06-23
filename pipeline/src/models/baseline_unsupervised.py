"""
Baseline unsupervised contact prediction using ESM2 attention maps.
Implementation based on the paper approach:
1. Extract attention maps from pretrained ESM2
2. Symmetrize and apply APC correction
3. Train logistic regression on small dataset
"""

from dataclasses import dataclass
from typing import Optional

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from utils.metrics import ContactPredictionMetrics
from registry import BaseModelConfig, register_model
from transformers import EsmModel


@dataclass
class ESM2UnsupervisedBaselineConfig(BaseModelConfig):
    """Configuration for ESM2 attention-based baseline model (paper implementation)."""

    # Model backbone selection
    backbone: str
    freeze_backbone: bool = True

    # Training parameters
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


@register_model("esm2_unsupervised_baseline", ESM2UnsupervisedBaselineConfig)
class ESM2UnsupervisedBaseline(L.LightningModule):
    """
    Baseline model from paper: ESM2 attention maps + APC + logistic regression.
    """

    def __init__(self, config: ESM2UnsupervisedBaselineConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # Load ESM2 backbone (frozen)
        self.esm_model = EsmModel.from_pretrained(self.config.backbone)
        if self.config.freeze_backbone:
            self.freeze_backbone()

        # Get actual model dimensions from loaded model config
        self.num_layers = self.esm_model.config.num_hidden_layers
        self.num_heads = self.esm_model.config.num_attention_heads

        # Freeze ESM2 - we only extract attention maps
        for param in self.esm_model.parameters():
            param.requires_grad = False

        # Simple logistic regression head (as in paper)
        # Input: processed attention map values, Output: contact probability
        self.contact_head = nn.Linear(self.num_layers * self.num_heads, 1)

        # Loss function
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Initialize metrics calculator
        self.contact_metrics = ContactPredictionMetrics()

        self.validation_step_outputs = []

    def freeze_backbone(self):
        """Freeze the backbone model."""
        for param in self.esm_model.parameters():
            param.requires_grad = False

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

    def extract_and_process_attention_maps(self, input_ids, attention_mask, mask_2d):
        """Extract attention maps from ESM2 and apply processing tensorized."""

        outputs = self.esm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        # combine all attention maps at once: (batch_size, num_layers * num_heads, seq_len, seq_len)
        combined_attentions = torch.cat(outputs.attentions, dim=1)

        # Remove CLS token (first position), keep up to max protein sequence length
        # For each sequence: [CLS, AA1, AA2, ..., AAn, EOS, PAD, ...]
        # We want: [AA1, AA2, ..., AAn]
        protein_attentions = combined_attentions[:, :, 1:-1, 1:-1]

        # Apply mask
        protein_attentions = protein_attentions * mask_2d.unsqueeze(1)

        # Symmetrize: (A + A^T) / 2
        protein_attentions = (
            protein_attentions + protein_attentions.transpose(-1, -2)
        ) / 2

        # Apply APC correction
        protein_attentions = self.apply_apc_correction(protein_attentions)

        return protein_attentions

    def forward(self, input_ids, attention_mask, mask_2d):
        """Forward pass: extract attention maps + logistic regression."""

        # Extract and process attention maps
        processed_attention = self.extract_and_process_attention_maps(
            input_ids, attention_mask, mask_2d
        )

        # For each pair (i,j), we have num_features attention values
        # Reshape for logistic regression: (batch_size, seq_len, seq_len, num_features)
        attention_features = processed_attention.permute(0, 2, 3, 1)

        # Apply logistic regression to each pair
        contact_logits = self.contact_head(attention_features).squeeze(
            -1
        )  # (batch_size, seq_len, seq_len)

        return contact_logits

    def create_mask(self, seq_lengths, max_length):
        """Create mask for valid positions (excluding padding)"""
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
        # Only consider lower triangular part to avoid double counting symmetric contacts
        batch_size, seq_len, _ = mask_2d.shape

        # Create lower triangular mask
        tril_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=mask_2d.device, dtype=torch.bool),
            diagonal=-1,
        )
        tril_mask = tril_mask.unsqueeze(0).expand(batch_size, -1, -1).bool()

        # Combine with existing mask
        final_mask = mask_2d & tril_mask

        loss = self.loss_fn(contact_logits[final_mask], contact_map[final_mask])
        return loss

    def training_step(self, batch, batch_idx):
        """Training step."""
        if self.config.freeze_backbone:
            self.esm_model.eval()

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        contact_map = batch["contact_maps"]
        seq_lengths = batch["seq_lengths"]

        max_length = input_ids.shape[-1] - 2  # -2 for CLS and EOS tokens
        mask_2d = self.create_mask(seq_lengths, max_length)

        # Forward pass
        contact_logits = self(input_ids, attention_mask, mask_2d)

        # Compute loss only on valid positions
        loss = self.compute_loss(contact_logits, contact_map, mask_2d)
        self.log(
            "train_loss",
            loss,
            batch_size=input_ids.shape[0],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        # # Compute metrics
        # with torch.no_grad():
        #     contact_probs = torch.sigmoid(contact_logits)
        #     metrics = self.contact_metrics.compute_all_metrics(
        #         contact_probs, contact_map, seq_lengths, mask_2d, average=False
        #     )

        #     # Log metrics with train prefix
        #     for metric_name, metric_value in metrics.items():
        #         self.log(
        #             f"train_{metric_name}",
        #             metric_value,
        #             prog_bar=True if "precision_at_L" in metric_name else False,
        #         )

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        contact_map = batch["contact_maps"]
        seq_lengths = batch["seq_lengths"]

        max_length = input_ids.shape[-1] - 2  # -2 for CLS and EOS tokens
        mask_2d = self.create_mask(seq_lengths, max_length)

        # Forward pass
        contact_logits = self(input_ids, attention_mask, mask_2d)

        # Compute loss only on valid positions
        loss = self.compute_loss(contact_logits, contact_map, mask_2d).item()

        # Compute comprehensive metrics
        metrics = self.contact_metrics.compute_all_metrics(
            contact_logits, contact_map, seq_lengths, mask_2d, average=False
        )

        self.validation_step_outputs.append({"loss": loss, "metrics": metrics})

    def on_validation_epoch_end(self):
        loss = np.mean([output["loss"] for output in self.validation_step_outputs])
        metrics = {
            metric_name: np.mean(
                [
                    output["metrics"][metric_name]
                    for output in self.validation_step_outputs
                ]
            )
            for metric_name in self.validation_step_outputs[0]["metrics"].keys()
        }
        self.log("val_loss", loss)
        for metric_name, metric_value in metrics.items():
            self.log(f"val_{metric_name}", metric_value)

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        # Only train the logistic regression head
        optimizer = torch.optim.AdamW(
            self.contact_head.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        return optimizer
