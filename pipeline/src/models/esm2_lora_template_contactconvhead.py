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


class ContactSelfAttentionLayer(nn.Module):
    """Self‑attention guided by a contact‑map mask (batch=1).
    If *contact_mask* is ``None`` → identity.
    """

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x_tpl: torch.Tensor, contact_mask: torch.Tensor):
        # todo: pad contact mask to the same length as x_tpl
        # contact_mask: (1, L, L) → bool mask (True = block)
        attn_mask = contact_mask[0] < 0.5
        attn_mask = nn.functional.pad(
            attn_mask, (1, 1, 1, 1), mode="constant", value=True
        )  # pad CLS and EOS tokens not to attend to them
        attn_out, _ = self.attn(x_tpl, x_tpl, x_tpl, attn_mask=attn_mask)
        return self.norm(x_tpl + attn_out)


class CrossAttentionLayer(nn.Module):
    """Classic query→key/value cross‑attention with residual+norm."""

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x_q: torch.Tensor, x_tpl_enriched: torch.Tensor):
        attn_out, _ = self.attn(x_q, x_tpl_enriched, x_tpl_enriched)
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
            embed_dim, self.config.contact_self_attention_heads
        )
        self.contact_cross_attention_layer = CrossAttentionLayer(
            embed_dim, self.config.contact_cross_attention_heads
        )

    def _forward_esm(self, input_ids, attention_mask):
        outputs = self.esm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return outputs.last_hidden_state

    def freeze_lora(self):
        for param in self.esm_model.parameters():
            if "lora" in param.name:
                param.requires_grad = False

    def _forward_pass(self, batch):
        primary_sequence = batch["primary_sequence"]
        template_sequence = batch["similar_sequence"]
        batch["contact_maps"] = batch["primary_sequence"][
            "contact_maps"
        ]  # ugly fix not to change the original module
        batch["seq_lengths"] = batch["primary_sequence"]["seq_lengths"]
        primary_mask_2d = self.create_mask(
            batch["seq_lengths"], batch["input_ids"].shape[-1]
        )
        if template_sequence is not None:
            return self(
                input_ids=primary_sequence["input_ids"],
                attention_mask=primary_sequence["attention_mask"],
                mask_2d=primary_mask_2d,
                template_ids=template_sequence["input_ids"],
                template_attention_mask=template_sequence["attention_mask"],
                template_contact_maps=batch["contact_maps"],
            )
        else:
            return self(
                input_ids=primary_sequence["input_ids"],
                attention_mask=primary_sequence["attention_mask"],
                mask_2d=primary_mask_2d,
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
        hiddent_states = self._forward_esm(input_ids, attention_mask)
        if mode == "template":
            template_hiddent_states = self._forward_esm(
                template_ids, template_attention_mask
            )
            template_hiddent_states = self.contact_self_attention_layer(
                template_hiddent_states, template_contact_maps
            )
            template_hiddent_states = self.contact_cross_attention_layer(
                hiddent_states, template_hiddent_states
            )
            hiddent_states = template_hiddent_states
        elif mode == "self":
            hiddent_states = self.contact_self_attention_layer(hiddent_states, mask_2d)

        contact_logits = self.contact_head(hiddent_states, mask_2d)
        return contact_logits
