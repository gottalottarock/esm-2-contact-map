import torch
import torch.nn as nn
from models.esm2_lora_contactconvhead import ContactConvHead, ESM2LoRAContactConvConfig
from dataclasses import dataclass
from typing import Optional
from models.esm2_lora_contacthead import ESM2LoRAContact, ESM2LoRAContactConfig
from registry import register_model


@dataclass
class ESM2LoRAContactConvTemplateConfig(ESM2LoRAContactConfig):
    """Configuration for ESM2 with LoRA adapters for contact prediction."""

    cnn_kernel_size: int = 5
    contact_self_attention_heads: int = 4
    contact_cross_attention_heads: int = 4
    unfreeze_lora: bool = False
    use_weighted_attn_mask: bool = False
    weighted_attn_mask_alpha_init: float = -4.0
    contacthead_init_from: Optional[str] = None
    use_gated_cross_attn: bool = False
    gated_cross_attn_init: float = -4.0
    gate_lr_multiplier: float = 100
    alpha_lr_multiplier: float = 100


class BiasCNN(nn.Module):
    def __init__(self, k=7, alpha_init=-4.0, clip=8.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 4, k, padding=k // 2),
            nn.GELU(),
            nn.Conv2d(4, 1, k, padding=k // 2),
        )
        self.log_alpha = nn.Parameter(torch.tensor(alpha_init))  # лог-форме удобно
        self.clip = clip

    def forward(self, cmap):  # (B,1,L,L)
        raw = self.net(cmap).squeeze()  # (B,L,L)
        alpha = torch.exp(self.log_alpha)
        bias = torch.clamp(raw * alpha, -self.clip, self.clip)
        bias.fill_diagonal_(0.0)
        return bias


class ContactSelfAttentionLayer(nn.Module):
    """Self‑attention guided by a contact‑map mask (batch=1).
    If *contact_mask* is ``None`` → identity.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        use_weighted_attn_mask: bool = False,
        weighted_attn_mask_alpha_init: float = -4.0,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.use_weighted_attn_mask = use_weighted_attn_mask
        if use_weighted_attn_mask:
            self.bias_cnn = BiasCNN(alpha_init=weighted_attn_mask_alpha_init)

    def forward(self, x_tpl: torch.Tensor, contact_mask: torch.Tensor):
        # contact_mask: (1, L, L) → bool mask (True = block)
        if not self.use_weighted_attn_mask:
            attn_mask = contact_mask[0] < 0.5
            attn_mask = nn.functional.pad(
                attn_mask, (1, 1, 1, 1), mode="constant", value=True
            )  # pad CLS and EOS tokens not to attend to them
            attn_mask.fill_diagonal_(
                False
            )  # to make that at least one non null value is present in the diagonal
        else:
            contact_mask = nn.functional.pad(
                contact_mask, (1, 1, 1, 1), mode="constant", value=0.0
            )
            attn_mask = self.bias_cnn(contact_mask.unsqueeze(1).float())
        attn_out, _ = self.attn(x_tpl, x_tpl, x_tpl, attn_mask=attn_mask)
        return self.norm(x_tpl + attn_out)


class CrossAttentionLayer(nn.Module):
    """Classic query→key/value cross‑attention with residual+norm."""

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        use_gated_cross_attn: bool = False,
        gated_cross_attn_init: float = -4,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.use_gated_cross_attn = use_gated_cross_attn
        if self.use_gated_cross_attn:
            self.log_gate = nn.Parameter(torch.tensor(gated_cross_attn_init))

    def forward(self, x_q: torch.Tensor, x_tpl_enriched: torch.Tensor):
        attn_out, _ = self.attn(x_q, x_tpl_enriched, x_tpl_enriched)
        if self.use_gated_cross_attn:
            gate = torch.sigmoid(self.log_gate)
            attn_out = gate * attn_out
        return self.norm(x_q + attn_out)


@register_model("esm2_lora_template_contactconv", ESM2LoRAContactConvTemplateConfig)
class ESM2LoRAContactConvTemplate(ESM2LoRAContact):
    def __init__(self, config: ESM2LoRAContactConvTemplateConfig):
        super().__init__(config)

        if not self.config.unfreeze_lora:
            self.freeze_lora()

    def init_contact_head(self):
        embed_dim = self.esm_model.config.hidden_size
        self.contact_head = ContactConvHead(
            embed_dim=embed_dim,
            num_heads=self.config.contact_head_dim,
            kernel_size=self.config.cnn_kernel_size,
        )

        self.contact_self_attention_layer = ContactSelfAttentionLayer(
            embed_dim,
            self.config.contact_self_attention_heads,
            self.config.use_weighted_attn_mask,
            self.config.weighted_attn_mask_alpha_init,
        )
        self.contact_cross_attention_layer = CrossAttentionLayer(
            embed_dim,
            self.config.contact_cross_attention_heads,
            self.config.use_gated_cross_attn,
            self.config.gated_cross_attn_init,
        )
        if self.config.contacthead_init_from:
            state_dict = torch.load(
                self.config.contacthead_init_from, weights_only=False
            )["state_dict"]
            contact_head_sd = {
                k.replace("contact_head.", ""): v
                for k, v in state_dict.items()
                if k.startswith("contact_head.")
            }
            self.contact_head.load_state_dict(contact_head_sd, strict=True)

    def _forward_esm(self, input_ids, attention_mask):
        outputs = self.esm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state

    def freeze_lora(self):
        for name, param in self.esm_model.named_parameters():
            if "lora" in name:
                param.requires_grad = False

    def _forward_pass(self, batch):
        primary_sequence = batch["primary_sequence"]
        template_sequence = batch["similar_sequence"]
        batch["contact_maps"] = batch["primary_sequence"][
            "contact_maps"
        ]  # ugly fix not to change the original module
        batch["seq_lengths"] = batch["primary_sequence"]["seq_lengths"]
        batch["input_ids"] = primary_sequence["input_ids"]
        primary_mask_2d = self.create_mask(
            batch["seq_lengths"],
            batch["primary_sequence"]["input_ids"].shape[-1]
            - 2,  # -2 for CLS and EOS tokens
        )
        if template_sequence is not None:
            return self(
                input_ids=primary_sequence["input_ids"],
                attention_mask=primary_sequence["attention_mask"],
                mask_2d=primary_mask_2d,
                template_ids=template_sequence["input_ids"],
                template_attention_mask=template_sequence["attention_mask"],
                template_contact_maps=template_sequence["contact_maps"],
                mode="template",
            )
        else:
            return self(
                input_ids=primary_sequence["input_ids"],
                attention_mask=primary_sequence["attention_mask"],
                mask_2d=primary_mask_2d,
                mode="self",
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_2d: torch.Tensor,
        template_ids: Optional[torch.Tensor] = None,
        template_attention_mask: Optional[torch.Tensor] = None,
        template_contact_maps: Optional[torch.Tensor] = None,
        mode: str = "template",
    ):
        assert mode in ["template", "self"]
        hidden_states = self._forward_esm(input_ids, attention_mask)  # (B, L, D)
        if mode == "template":
            template_hidden_states = self._forward_esm(
                template_ids, template_attention_mask
            )  # (B, Lt, D)
            template_hidden_states = self.contact_self_attention_layer(
                template_hidden_states, template_contact_maps
            )  # (B, Lt, D)
            hidden_states = self.contact_cross_attention_layer(
                hidden_states, template_hidden_states
            )  # (B, L, D)
        contact_logits = self.contact_head(hidden_states, mask_2d)
        return contact_logits, mask_2d

    def configure_optimizers(self):
        params_exc = []
        groups = []
        if self.config.use_gated_cross_attn:
            gate_param = [self.contact_cross_attention_layer.log_gate]
            groups.append({
                "params": gate_param,
                "lr": self.config.gate_lr_multiplier * self.config.learning_rate,
                "weight_deay":0
            })
            params_exc.append(gate_param[0])
        if self.config.use_weighted_attn_mask:
            alpha_param = [self.contact_self_attention_layer.bias_cnn.log_alpha]
            groups.append({
                "params": alpha_param,
                "lr": self.config.alpha_lr_multiplier * self.config.learning_rate,
                "weight_deay": 0,
            })
            params_exc.append(alpha_param[0])
        main_params = [p for p in self.parameters() if not any(p is param for param in params_exc)]
        groups.append({"params": main_params, "lr": self.config.learning_rate})

        optimizer = torch.optim.AdamW(groups, weight_decay=self.config.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = super().training_step(batch, batch_idx)
        if self.config.use_weighted_attn_mask:
            log_alpha = self.contact_self_attention_layer.bias_cnn.log_alpha
            self.log("Weighted attn mask log alpha", log_alpha)
        if self.config.use_gated_cross_attn:
            log_gate = self.contact_cross_attention_layer.log_gate
            self.log("Gated cross attn log gate", torch.sigmoid(log_gate))
        return loss
