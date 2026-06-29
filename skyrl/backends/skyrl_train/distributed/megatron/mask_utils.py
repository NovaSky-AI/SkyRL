"""Attention-mask helpers for the Megatron forward path."""

from typing import Optional

import torch


def to_te_attention_mask(attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Convert a 2-D keep-mask to a 4-D padding mask for Transformer Engine attention.

    ``remove_left_padding`` returns a 2-D ``[batch, seq]`` keep-mask where 1 marks a real token.
    Transformer Engine's sliding-window ``get_full_mask`` expects a mask broadcastable to
    ``[batch, 1, q_seq, kv_seq]``; a 2-D mask collides the batch dimension with the sequence
    dimension and fails for ``micro_batch_size > 1``. Return a ``[batch, 1, 1, seq]`` padding mask
    where True marks padding. A ``None`` mask (packed sequences) or an already higher-rank mask is
    returned unchanged.
    """
    if attention_mask is None or attention_mask.dim() != 2:
        return attention_mask
    return (~attention_mask.bool())[:, None, None, :]
