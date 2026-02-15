#!/usr/bin/env python3
"""
Create a minimal flash_attn stub package.

SkyRL training code only needs flash_attn.bert_padding.pad_input/unpad_input.
These are pure Python functions - we copy them from flash-attn source.
This avoids all CUDA import issues (no flash_attn_2_cuda needed).

Source: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py
"""

import os
import site


def create_stub():
    # Create flash_attn package in site-packages
    pkg_dir = os.path.join(site.getsitepackages()[0], "flash_attn")
    os.makedirs(pkg_dir, exist_ok=True)

    # Create __init__.py (empty - just marks as package)
    with open(os.path.join(pkg_dir, "__init__.py"), "w") as f:
        f.write("# Minimal flash_attn stub - only provides bert_padding\n")
        f.write("__version__ = '2.7.0.stub'\n")

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


if __name__ == "__main__":
    create_stub()
