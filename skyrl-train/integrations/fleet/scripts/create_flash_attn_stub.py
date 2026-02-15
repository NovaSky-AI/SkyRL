#!/usr/bin/env python3
"""
Create a minimal flash_attn stub package.

SkyRL training code needs flash_attn.bert_padding.pad_input/unpad_input.
vLLM needs flash_attn.ops.triton.rotary.apply_rotary for rotary embeddings.

These are pure Python/PyTorch implementations - no CUDA compilation needed.
This avoids all CUDA import issues (no flash_attn_2_cuda or nvcc needed).

Sources:
- bert_padding: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py
- rotary: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/triton/rotary.py
"""

import os
import site


def create_stub():
    # Create flash_attn package in site-packages
    pkg_dir = os.path.join(site.getsitepackages()[0], "flash_attn")
    os.makedirs(pkg_dir, exist_ok=True)

    # Create ops/triton directory structure for rotary embedding
    ops_dir = os.path.join(pkg_dir, "ops")
    triton_dir = os.path.join(ops_dir, "triton")
    os.makedirs(triton_dir, exist_ok=True)

    # Create __init__.py files for package structure
    with open(os.path.join(pkg_dir, "__init__.py"), "w") as f:
        f.write("# Minimal flash_attn stub - provides bert_padding and rotary ops\n")
        f.write("__version__ = '2.7.0.stub'\n")

    with open(os.path.join(ops_dir, "__init__.py"), "w") as f:
        f.write("# flash_attn.ops stub\n")

    with open(os.path.join(triton_dir, "__init__.py"), "w") as f:
        f.write("# flash_attn.ops.triton stub\n")

    # Create rotary.py with pure PyTorch implementation (no Triton needed)
    rotary_code = '''"""
Pure PyTorch implementation of rotary embeddings.
Replaces the Triton kernel version for environments without flash_attn CUDA build.

This is a fallback implementation that matches the flash_attn API but uses
standard PyTorch operations instead of Triton kernels.
"""
import torch
from typing import Optional, Union


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_emb_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
) -> torch.Tensor:
    """
    Apply rotary embedding to input tensor using PyTorch operations.

    Args:
        x: Input tensor of shape (batch, seqlen, nheads, headdim) or (seqlen, nheads, headdim)
        cos: Cosine values of shape (seqlen, rotary_dim // 2) or (seqlen, rotary_dim)
        sin: Sine values matching cos shape
        interleaved: If True, use interleaved layout

    Returns:
        Tensor with rotary embedding applied
    """
    ro_dim = cos.shape[-1] * 2 if cos.shape[-1] * 2 <= x.shape[-1] else cos.shape[-1]

    # Handle different cos/sin shapes
    if cos.dim() == 2:
        # cos is (seqlen, dim) - need to broadcast to x's shape
        seqlen = x.shape[-3] if x.dim() == 4 else x.shape[-3]
        cos = cos[:seqlen]
        sin = sin[:seqlen]

        # Reshape for broadcasting: (seqlen, 1, dim)
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)

    # Split x into rotary and pass-through parts
    x_ro = x[..., :ro_dim]
    x_pass = x[..., ro_dim:] if ro_dim < x.shape[-1] else None

    if interleaved:
        # Interleaved layout: [x0, x1, x2, x3, ...] -> rotate pairs
        x1 = x_ro[..., ::2]
        x2 = x_ro[..., 1::2]

        cos_half = cos[..., : cos.shape[-1] // 2] if cos.shape[-1] == ro_dim else cos
        sin_half = sin[..., : sin.shape[-1] // 2] if sin.shape[-1] == ro_dim else sin

        o1 = x1 * cos_half - x2 * sin_half
        o2 = x1 * sin_half + x2 * cos_half

        x_ro_out = torch.stack([o1, o2], dim=-1).flatten(-2)
    else:
        # Non-interleaved: first half and second half are separate
        cos_expanded = cos
        sin_expanded = sin

        # Handle rotary_dim being half of headdim
        if cos.shape[-1] * 2 == ro_dim:
            cos_expanded = torch.cat([cos, cos], dim=-1)
            sin_expanded = torch.cat([sin, sin], dim=-1)

        x_ro_out = x_ro * cos_expanded + _rotate_half(x_ro) * sin_expanded

    if x_pass is not None:
        return torch.cat([x_ro_out, x_pass], dim=-1)
    return x_ro_out


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved: bool = False,
    inplace: bool = False,
    conjugate: bool = False,
) -> torch.Tensor:
    """
    Apply rotary positional embedding to input tensor.

    Pure PyTorch implementation matching flash_attn.ops.triton.rotary API.

    Args:
        x: Input tensor (batch, seqlen, nheads, headdim) or (total_seqlen, nheads, headdim)
        cos: Precomputed cosine values (seqlen_ro, rotary_dim / 2)
        sin: Precomputed sine values matching cos shape
        seqlen_offsets: Integer or tensor of size (batch,) for offset adjustments
        cu_seqlens: Optional cumulative sequence lengths for variable-length sequences
        max_seqlen: Maximum sequence length (required when cu_seqlens is provided)
        interleaved: If True, use interleaved data layout
        inplace: If True, modify x in-place (ignored in PyTorch fallback)
        conjugate: If True, conjugate sine values (negate sin)

    Returns:
        Tensor with rotary embedding applied, same shape as input
    """
    if conjugate:
        sin = -sin

    # Handle sequence offsets by slicing cos/sin appropriately
    if isinstance(seqlen_offsets, int) and seqlen_offsets != 0:
        seqlen = x.shape[-3] if x.dim() == 4 else x.shape[0]
        cos = cos[seqlen_offsets : seqlen_offsets + seqlen]
        sin = sin[seqlen_offsets : seqlen_offsets + seqlen]
    elif isinstance(seqlen_offsets, torch.Tensor):
        # Per-batch offsets - more complex handling needed
        # For now, use a simple loop (can be optimized if needed)
        if x.dim() == 4:
            batch_size = x.shape[0]
            seqlen = x.shape[1]
            outputs = []
            for b in range(batch_size):
                offset = int(seqlen_offsets[b].item())
                cos_b = cos[offset : offset + seqlen]
                sin_b = sin[offset : offset + seqlen]
                outputs.append(_apply_rotary_emb_torch(x[b:b+1], cos_b, sin_b, interleaved))
            return torch.cat(outputs, dim=0)

    # Handle cu_seqlens (variable length sequences packed together)
    if cu_seqlens is not None:
        # x is (total_seqlen, nheads, headdim)
        # Need to apply rotary per-sequence
        outputs = []
        for i in range(len(cu_seqlens) - 1):
            start = int(cu_seqlens[i].item())
            end = int(cu_seqlens[i + 1].item())
            seq_len = end - start
            x_seq = x[start:end]
            cos_seq = cos[:seq_len]
            sin_seq = sin[:seq_len]
            outputs.append(_apply_rotary_emb_torch(x_seq.unsqueeze(0), cos_seq, sin_seq, interleaved).squeeze(0))
        return torch.cat(outputs, dim=0)

    return _apply_rotary_emb_torch(x, cos, sin, interleaved)
'''

    with open(os.path.join(triton_dir, "rotary.py"), "w") as f:
        f.write(rotary_code)

    print(f"Created flash_attn.ops.triton.rotary stub at {triton_dir}")

    # Create bert_padding.py with the functions SkyRL needs
    bert_padding_code = '''"""
Minimal bert_padding functions from flash-attention.
Only includes pad_input and unpad_input used by SkyRL training.
"""
import torch
from torch import Tensor
from einops import rearrange


class IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        return torch.gather(
            rearrange(input, "b ... -> b (...)"), 0,
            indices.unsqueeze(-1).expand(-1, second_dim)
        ).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        grad_input.scatter_(
            0, indices.unsqueeze(-1).expand(-1, grad_output.shape[1]), grad_output
        )
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(
            first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype
        )
        output[indices] = values
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        grad_values = grad_output[indices]
        return grad_values, None, None


index_put_first_axis = IndexPutFirstAxis.apply


def unpad_input(hidden_states, attention_mask, unused_mask=None):
    """
    Remove padding from hidden states based on attention mask.

    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), 1 for valid tokens, 0 for padding
        unused_mask: unused, kept for API compatibility

    Returns:
        hidden_states_unpad: (total_valid_tokens, ...)
        indices: (total_valid_tokens,), indices of valid tokens in flattened batch
        cu_seqlens: (batch + 1,), cumulative sequence lengths
        max_seqlen_in_batch: int, maximum sequence length
        seqlens: (batch,), sequence lengths
    """
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = int(seqlens.max().item())
    cu_seqlens = torch.nn.functional.pad(
        torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0)
    )
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
        seqlens,
    )


def pad_input(hidden_states_unpad, indices, batch, seqlen):
    """
    Pad hidden states back to original shape.

    Arguments:
        hidden_states_unpad: (total_valid_tokens, ...)
        indices: (total_valid_tokens,)
        batch: int, batch size
        seqlen: int, sequence length

    Returns:
        hidden_states: (batch, seqlen, ...)
    """
    output = index_put_first_axis(hidden_states_unpad, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)
'''

    with open(os.path.join(pkg_dir, "bert_padding.py"), "w") as f:
        f.write(bert_padding_code)

    print(f"Created flash_attn stub package at {pkg_dir}")
    print("  - bert_padding: pad_input, unpad_input")
    print("  - ops.triton.rotary: apply_rotary (pure PyTorch)")


if __name__ == "__main__":
    create_stub()
