"""VFM Noise Adapter v1b + SigmaHead — ported from ltx2-castlehill.

NoiseAdapterV1b: Temporally-aware conditional noise generation with
self-attention + cross-attention to full text + sinusoidal positions.

SigmaHead: Per-token sigma predictor for per-token timestep scheduling.

These are standalone modules that work with any diffusion backbone
(both LTX-2 and Wan2.1).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────
# Position Encoding
# ─────────────────────────────────────────────────────────────────────

class SinusoidalPositionEncoding(nn.Module):
    """Encode (t, h, w) coordinates into a fixed-dim vector via sinusoids.

    Takes spatiotemporal bounds [B, 3, seq, 2] and produces [B, seq, pos_dim].
    """

    def __init__(self, pos_dim: int = 256, num_axes: int = 3):
        super().__init__()
        self.num_axes = num_axes
        self.dim_per_axis = (pos_dim // num_axes) // 2 * 2  # must be even
        self.pos_dim = self.dim_per_axis * num_axes

        freqs = torch.exp(
            torch.arange(0, self.dim_per_axis, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.dim_per_axis)
        )
        self.register_buffer("freqs", freqs)

    def forward(self, positions: Tensor) -> Tensor:
        """[B, 3, seq_len, 2] -> [B, seq_len, pos_dim]"""
        coords = positions.mean(dim=-1)  # midpoint per axis

        encodings = []
        for axis in range(self.num_axes):
            c = coords[:, axis, :].unsqueeze(-1)
            freqs = self.freqs.to(c.device, c.dtype)
            angles = c * freqs
            enc = torch.cat([angles.sin(), angles.cos()], dim=-1)
            encodings.append(enc)

        return torch.cat(encodings, dim=-1)


# ─────────────────────────────────────────────────────────────────────
# Adapter Block
# ─────────────────────────────────────────────────────────────────────

class AdapterBlock(nn.Module):
    """Single transformer block: self-attn -> cross-attn -> FFN."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        ffn_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.self_attn_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )

        self.cross_attn_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.text_norm = nn.LayerNorm(hidden_dim)

        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_mult),
            nn.GELU(),
            nn.Linear(hidden_dim * ffn_mult, hidden_dim),
        )

    def forward(
        self,
        x: Tensor,
        text_kv: Tensor,
        text_mask: Tensor | None = None,
    ) -> Tensor:
        # Self-attention
        residual = x
        x_norm = self.self_attn_norm(x)
        x = residual + self.self_attn(x_norm, x_norm, x_norm, need_weights=False)[0]

        # Cross-attention to text
        key_padding_mask = ~text_mask if text_mask is not None else None
        residual = x
        x_norm = self.cross_attn_norm(x)
        text_kv_norm = self.text_norm(text_kv)
        x = residual + self.cross_attn(
            x_norm, text_kv_norm, text_kv_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]

        # FFN
        residual = x
        x = residual + self.ffn(self.ffn_norm(x))

        return x


# ─────────────────────────────────────────────────────────────────────
# Noise Adapter V1b
# ─────────────────────────────────────────────────────────────────────

TASK_CLASSES = {
    "i2v": 0,
    "inpaint": 1,
    "sr": 2,
    "denoise": 3,
    "t2v": 4,
    "v2v": 5,  # Added for InSpatio novel-view synthesis
}


class NoiseAdapterV1b(nn.Module):
    """Enhanced noise adapter with spatiotemporal awareness and text cross-attention.

    Produces per-token (mu, log_sigma) noise distribution parameters.
    Works with any latent dimension (128 for LTX, 16 for Wan).
    """

    def __init__(
        self,
        text_dim: int,
        latent_dim: int = 128,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        num_task_classes: int = 6,
        task_embed_dim: int = 128,
        pos_dim: int = 256,
        init_sigma: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.init_sigma = init_sigma
        self.hidden_dim = hidden_dim

        self.pos_encoder = SinusoidalPositionEncoding(pos_dim=pos_dim)
        actual_pos_dim = self.pos_encoder.pos_dim

        self.task_embedding = nn.Embedding(num_task_classes, task_embed_dim)

        self.input_proj = nn.Sequential(
            nn.LayerNorm(actual_pos_dim + task_embed_dim),
            nn.Linear(actual_pos_dim + task_embed_dim, hidden_dim),
        )

        self.text_proj = nn.Linear(text_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            AdapterBlock(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.log_sigma_head = nn.Linear(hidden_dim, latent_dim)

        # Initialize near identity (start at N(0, I) prior)
        nn.init.zeros_(self.mu_head.weight)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.zeros_(self.log_sigma_head.weight)
        nn.init.constant_(self.log_sigma_head.bias, init_sigma)

    def forward(
        self,
        text_embeddings: Tensor,
        text_mask: Tensor,
        positions: Tensor,
        task_class: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Compute per-token noise distribution parameters.

        Args:
            text_embeddings: [B, text_seq, text_dim]
            text_mask: [B, text_seq] bool
            positions: [B, 3, video_seq, 2] spatiotemporal coordinates
            task_class: [B] integer task class indices

        Returns:
            mu: [B, video_seq, latent_dim]
            log_sigma: [B, video_seq, latent_dim]
        """
        B, video_seq = positions.shape[0], positions.shape[2]

        pos_enc = self.pos_encoder(positions.float())
        task_emb = self.task_embedding(task_class).unsqueeze(1).expand(-1, video_seq, -1)

        x = torch.cat([pos_enc, task_emb], dim=-1)
        x = self.input_proj(x)

        text_kv = self.text_proj(text_embeddings.float())

        for block in self.blocks:
            x = block(x, text_kv, text_mask)

        x = self.output_norm(x)
        mu = self.mu_head(x)
        log_sigma = self.log_sigma_head(x)
        log_sigma = log_sigma.clamp(min=-1.0, max=2.0)

        return mu, log_sigma

    def sample(
        self,
        text_embeddings: Tensor,
        text_mask: Tensor,
        positions: Tensor,
        task_class: Tensor,
        temperature: float = 1.0,
    ) -> Tensor:
        """Sample structured noise z ~ q_phi(z|y)."""
        mu, log_sigma = self.forward(text_embeddings, text_mask, positions, task_class)
        sigma = torch.exp(log_sigma)
        eps = torch.randn_like(mu)
        return mu + sigma * eps * temperature


# ─────────────────────────────────────────────────────────────────────
# Sigma Head (Per-Token Timestep Scheduling)
# ─────────────────────────────────────────────────────────────────────

class SigmaHead(nn.Module):
    """Per-token sigma predictor.

    Takes clean latent x0 + adapter mu to predict per-token noise level
    sigma_i in [sigma_min, sigma_max].
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        sigma_min: float = 0.05,
        sigma_max: float = 0.95,
    ):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        input_dim = latent_dim * 2

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, mu: Tensor, x0: Tensor | None = None) -> Tensor:
        """Predict per-token sigma from clean latent x0 and adapter mu.

        Args:
            mu: [B, seq, latent_dim]
            x0: [B, seq, latent_dim] or None

        Returns:
            [B, seq] in [sigma_min, sigma_max]
        """
        if x0 is not None:
            inp = torch.cat([x0.detach(), mu], dim=-1)
        else:
            inp = torch.cat([torch.zeros_like(mu), mu], dim=-1)

        raw = self.net(inp).squeeze(-1)
        return self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(raw)
