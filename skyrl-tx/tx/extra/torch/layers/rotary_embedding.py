from __future__ import annotations

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Original Rotary Position Embedding (RoPE).

    RotaryEmbedding manages the cos and sin cache and applies rotary position embedding to query and key tensors.
    To implement scaled or extended variants, subclass and override _compute_inv_freq or _compute_cos_sin_cache.

    Args:
        head_size: The dimension of the head.
        rotary_dim: The dimension of the rotary embedding.
        max_position_embeddings: The maximum number of positions to cache.
        base: The base of the exponential function.
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary position embedding.
        dtype: The dtype of the tensor.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int | float,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        cos_cache, sin_cache = self._compute_cos_sin_cache()
        self.cos_cache: torch.Tensor
        self.sin_cache: torch.Tensor
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

    def _compute_cos_sin_cache(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)

        freqs = torch.einsum("i,j->ij", t, inv_freq)  
        freqs = torch.cat((freqs, freqs), dim=-1)  # [max_position_embeddings, rotary_dim]
        cos = freqs.cos()
        sin = freqs.sin()

        return cos, sin

    def _compute_inv_freq(self, base: int | float) -> torch.Tensor:
        """Compute the inverse frequencies."""
        # [rotary_dim // 2]
        inv_freq = 1.0 /  (
            base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float)
                / self.rotary_dim
            )
        )

        return inv_freq

    def get_cos_sin(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Slice cached cos and sin for the first seqlen positions."""
        cos = self.cos_cache[:seq_len]
        sin = self.sin_cache[:seq_len]

        return cos, sin

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embedding to query and key.

        Args:
            q: The query tensor.
            k: The key tensor.
            position_ids: The position IDs.
        
        Returns:
            A tuple of torch.Tensor comprising of the embedded query and key.
        """
        # [B, n_heads, T, head_dim]
        q_type, k_type = q.dtype, k.dtype

        position_flat = positions_ids.flatten()
        cos = self.cos_cache.index_select(0, position_flat)
        sin = self.sin_cache.index_select(0, position_flat)
        cos = cos.reshape(*positions_ids.shape, -1)
        sin = sin.reshape(*positions_ids.shape, -1)

        # Unsqueeze along the head dimension to [B, 1, T, head_dim]
        cos = cos.unsqueeze(dim=1)
        sin = sin.unsqueeze(dim=1)

        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        return q_embed.to(q_type), k_embed.to(k_type)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half of the hidden dimension of the input."""
        hidden_size = x.shape[-1]
        x1 = x[..., :hidden_size // 2]
        x2 = x[..., hidden_size // 2:]

        return torch.cat((-x2, x1), dim=-1)
