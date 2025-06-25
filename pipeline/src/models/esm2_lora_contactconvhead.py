from dataclasses import dataclass

import torch
import torch.nn as nn
from models.esm2_lora_contacthead import ESM2LoRAContact, ESM2LoRAContactConfig
from registry import register_model


class ContactConvHead(nn.Module):
    def __init__(self, embed_dim, num_heads=4, kernel_size=5):
        super(ContactConvHead, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)

        self.cnn_refiner = nn.Sequential(
            nn.Conv2d(
                num_heads,
                num_heads * 2,
                kernel_size=kernel_size,
                padding=int((kernel_size - 1) / 2),
            ),
            nn.ReLU(),
            nn.Conv2d(
                num_heads * 2,
                num_heads,
                kernel_size=kernel_size,
                padding=int((kernel_size - 1) / 2),
            ),
            nn.ReLU(),
        )

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
        logits_apc = self.apply_apc_correction(logits)  # (B, num_heads, L-2, L-2)

        # Collapse heads dimension
        refined_logits = self.cnn_refiner(logits_apc)
        refined_logits = refined_logits.permute(0, 2, 3, 1)  # (B, L-2, L-2, num_heads)
        contact_map = (
            self.final_proj(refined_logits).squeeze(-1) * mask
        )  # (B, L-2, L-2)
        return contact_map


@dataclass
class ESM2LoRAContactConvConfig(ESM2LoRAContactConfig):
    """Configuration for ESM2 with LoRA adapters for contact prediction."""

    cnn_kernel_size: int = 5


@register_model("esm2_lora_contactconv", ESM2LoRAContactConvConfig)
class ESM2LoRAContactConv(ESM2LoRAContact):
    def init_contact_head(self):
        embed_dim = self.esm_model.config.hidden_size
        self.contact_head = ContactConvHead(
            embed_dim=embed_dim,
            num_heads=self.config.contact_head_dim,
            kernel_size=self.config.cnn_kernel_size,
        )
