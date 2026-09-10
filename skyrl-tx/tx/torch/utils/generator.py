"""Generator utilities for autoregressive text generation with KV caching."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List
import torch


@dataclass
class KVCache:
    """Key-value cache for all layers, each entry in the list corresponds to one layer.

    Attributes:
        keys: List of key tensors, one per layer [B, n_heads, T_cache, head_dim]
        values: List of value tensors, one per layer [B, n_heads, T_cache, head_dim]
        cache_position: Current position in the cache (next position to write to)
    """

    keys: List[torch.Tensor]
    values: List[torch.Tensor]
    cache_position: int

    def pad_to_length(self, max_length: int) -> KVCache:
        """Pad KV cache to a specified maximum length.

        Args:
            max_length: Target length to pad the cache to.

        Returns:
            New KVCache with padded keys and values.
        """
        # k and v have shape [B, n_heads, T, head_dim]
        cache_pad_length = max_length - self.keys[0].shape[2]
        if cache_pad_length <= 0:
            return self

        padded_keys = []
        padded_values = []
        for k, v in zip(self.keys, self.values):
            # Pad along the sequence dimension (dim=2)
            k_padded = torch.nn.functional.pad(k, (0, 0, 0, cache_pad_length))
            v_padded = torch.nn.functional.pad(v, (0, 0, 0, cache_pad_length))
            padded_keys.append(k_padded)
            padded_values.append(v_padded)

        return KVCache(keys=padded_keys, values=padded_values, cache_position=self.cache_position)


def compute_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute positions from attention mask.

    Positions start at 0 from the first non-zero value in the attention mask
    and increment sequentially. Supports left-padding with negative positions.

    Args:
        attention_mask: [B, T] tensor with 1=valid token, 0=padding

    Returns:
        positions: [B, T] tensor with positions (can be negative for left-padding)
    """
    first_token_idx = attention_mask.argmax(dim=1, keepdim=True)  # [B, 1]
    seq_len = attention_mask.shape[1]
    positions = torch.arange(seq_len, device=attention_mask.device)[None, :] - first_token_idx
    return positions
