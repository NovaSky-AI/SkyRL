"""Flex attention implementation for attention sink 

Modified from Unsloth to support attention masks: https://github.com/unslothai/unsloth-zoo/blob/main/unsloth_zoo/flex_attention/attention_sink.py
"""

# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = [
    "old_flex_attention_with_sink",
]

import torch
import functools
from torch.nn.attention.flex_attention import (
    create_block_mask as _create_block_mask,
)
from .flex_attn_utils import (
    _flex_attention as uncompiled_flex_attention,
    flex_attention,
)


def causal_mask_with_sink(batch, head, q_idx, kv_idx):
    """
      0 1 2 3     0 1 2 3
    0 X X       1   X
    1 X X X     2   X X
    2 X X X X   3   X X X
    """
    # We add (q_idx + 1) since first column is sink token
    causal_mask = (q_idx + 1) >= kv_idx
    sink_first_column = kv_idx == 0
    return causal_mask | sink_first_column


@functools.lru_cache
def generate_sliding_window_with_sink(window_size: int):
    def sliding_window(batch, head, q_idx, kv_idx):
        causal_mask = (q_idx + 1) >= kv_idx
        # Official PyTorch attends to 1 extra token
        # windowed_mask = q_idx - kv_idx <= window_size
        # HuggingFace and official GPT OSS attends to only 128 tokens not (128+1)
        windowed_mask = (q_idx + 1) - kv_idx < window_size
        sink_first_column = kv_idx == 0
        return (causal_mask & windowed_mask) | sink_first_column

    sliding_window.__name__ = sliding_window.__doc__ = f"sliding_window_{window_size}_sink"
    return sliding_window


@functools.lru_cache
def generate_sink_score_mod(
    sink_weights: torch.Tensor,
):
    def sink_score_mod(score, batch, head, q_idx, kv_idx):
        # Sink token is at the first location
        return torch.where(
            kv_idx == 0,
            sink_weights[head].to(score.dtype) + 0.0,  # Add +0 to allow gradients
            score,
        )

    return sink_score_mod


@functools.lru_cache
def generate_padding_mask(attention_mask):
    if attention_mask.ndim == 4:
        assert attention_mask.shape[1] == 1, f"invalid shape {attention_mask.shape}"
        B, _, Q, K = attention_mask.shape
        attention_mask = attention_mask.squeeze(1).flatten()

        def padding_mask(batch, head, q_idx, kv_idx):
            return ~attention_mask[batch * (Q * K) + K * q_idx + kv_idx].bool()

    else:
        assert attention_mask.ndim == 2, f"Unexpected ndim: {attention_mask.ndim}"
        B, K = attention_mask.shape
        attention_mask = attention_mask.flatten()

        def padding_mask(batch, head, q_idx, kv_idx):
            return attention_mask[batch * K + kv_idx].bool()

    return padding_mask


def old_flex_attention_with_sink(
    query,
    key,
    value,
    attention_mask=None,
    scale=None,
    sliding_window=None,
    sinks=None,
    num_key_value_groups=1,
    compile=True,
    **kwargs,
):
    """
    Allows one sink token to be attended to for full/sliding window attention
    Similar to Efficient Streaming Language Models with Attention Sinks
    Primarily for GPT-OSS 2025

    [WARNING] This only works for training. Inference fails since KV cache's
    absolute positioning will fail.
    """

    enable_gqa = num_key_value_groups != 1
    bsz, heads_Q, qlen_Q, dim = query.shape
    _, heads_KV, qlen_KV, _ = key.shape

    # Add K and V with a row of 0s to allow sinks to be placed there
    key_padded = torch.cat([key.new_zeros(bsz, heads_KV, 1, dim), key], dim=2)
    value_padded = torch.cat([value.new_zeros(bsz, heads_KV, 1, dim), value], dim=2)

    # Check for sliding window
    mask_mod = (
        generate_sliding_window_with_sink(sliding_window)
        if type(sliding_window) is int and sliding_window != 0
        else causal_mask_with_sink
    )

    if attention_mask is not None:
        attn_size = attention_mask.size()
        # In some cases, huggingface will preprocess this to be a 4d attention mask
        # 0 -> token it can attend to
        if attention_mask.ndim == 4:
            attention_mask = torch.cat([attention_mask.new_zeros((*attn_size[:-1], 1)), attention_mask], dim=-1)
        else:
            assert attention_mask.ndim == 2, f"Unexpected ndim {attention_mask.ndim}"
            attention_mask = torch.cat([attention_mask.new_ones((*attn_size[:-1], 1)), attention_mask], dim=-1)
        _padding_mask = generate_padding_mask(attention_mask)

        def combine_masks(mask1, mask2):
            def final_mask(b, h, q, k):
                return mask1(b, h, q, k) & mask2(b, h, q, k)

            return final_mask

        mask_mod = combine_masks(_padding_mask, mask_mod)

    score_mod = generate_sink_score_mod(
        sinks,
    )
    # NOTE: if the block mask is not compiled, it leads to O(N^2) memory usage
    block_mask = _create_block_mask(mask_mod, bsz, heads_Q, qlen_Q, qlen_KV + 1, device=key.device, _compile=True)
    attn_output = (flex_attention if compile else uncompiled_flex_attention)(
        query,
        key_padded,
        value_padded,
        block_mask=block_mask,
        score_mod=score_mod,
        enable_gqa=enable_gqa,
        scale=scale,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output
