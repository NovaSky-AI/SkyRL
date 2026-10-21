"""Project the vocabulary candidates recorded for sample-support replay.

The scorer supplies flat ``(row_ids[i], token_ids[i])`` pairs: a row in the
flattened hidden-state matrix and a row in this TP rank's LM-head weight shard.
Fixed-width support supplies slots whose padding is masked by the scorer;
ragged support can supply only valid members. Neither form needs vocabulary-wide
logits. For example, pairs ``(0, 3), (0, 7), (2, 5)`` produce three scores.

One Triton program per pair computes ``dot(hidden[row], weight[token]) / temperature``
in hidden-dimension tiles, accumulating in fp32 without gathered pair matrices.
The scorer groups the returned scores by token for normalization and entropy.
Its custom autograd function retains the bounded Torch backward for both inputs.
"""

import torch
import triton
import triton.language as tl


def _autotune_pair_regime(num_pairs: int) -> int:
    """Separate under-occupied launches without keying on exact support size."""
    if num_pairs < 256:
        return 0
    if num_pairs < 4096:
        return 1
    return 2


@triton.autotune(
    # Triton's reduction kernels likewise sweep power-of-two tiles and 4/8 warps:
    # https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/reduce.py
    configs=[
        triton.Config({"BLOCK_H": 128}, num_warps=4),
        triton.Config({"BLOCK_H": 256}, num_warps=4),
        triton.Config({"BLOCK_H": 512}, num_warps=8),
        triton.Config({"BLOCK_H": 1024}, num_warps=8),
    ],
    key=["PAIR_REGIME", "HIDDEN_SIZE"],
    cache_results=True,
)
@triton.jit
def _candidate_projection_kernel(
    hidden_ptr,
    weight_ptr,
    row_ids_ptr,
    token_ids_ptr,
    output_ptr,
    PAIR_REGIME: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    stride_hidden_row,
    stride_hidden_h,
    stride_weight_row,
    stride_weight_h,
    rcp_temperature,
    BLOCK_H: tl.constexpr,
):
    pair_idx = tl.program_id(0)

    row_idx = tl.load(row_ids_ptr + pair_idx)
    token_idx = tl.load(token_ids_ptr + pair_idx)
    offsets_h = tl.arange(0, BLOCK_H)
    score = 0.0
    for start_h in range(0, HIDDEN_SIZE, BLOCK_H):
        mask_h = start_h + offsets_h < HIDDEN_SIZE
        hidden = tl.load(
            hidden_ptr + row_idx * stride_hidden_row + (start_h + offsets_h) * stride_hidden_h,
            mask=mask_h,
            other=0.0,
        )
        weight = tl.load(
            weight_ptr + token_idx * stride_weight_row + (start_h + offsets_h) * stride_weight_h,
            mask=mask_h,
            other=0.0,
        )
        # Match the output layer and custom backward: hidden is first cast to
        # the weight dtype, while the reduction itself remains in fp32.
        score += tl.sum(hidden.to(weight.dtype).to(tl.float32) * weight.to(tl.float32))
    tl.store(output_ptr + pair_idx, score * rcp_temperature)


def candidate_projection(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    row_ids: torch.Tensor,
    token_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Project ``(hidden row, weight row)`` pairs without gathered intermediates."""
    if not hidden.is_cuda:
        raise RuntimeError("Triton sample-support projection requires CUDA tensors")
    if hidden.dim() != 2 or weight.dim() != 2 or hidden.shape[1] != weight.shape[1]:
        raise ValueError("hidden and weight must be 2-D with matching hidden dimensions")
    if row_ids.shape != token_ids.shape or row_ids.dim() != 1:
        raise ValueError("row_ids and token_ids must be matching 1-D tensors")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if not hidden.is_contiguous() or not weight.is_contiguous():
        raise ValueError("hidden and weight must be contiguous")
    if not row_ids.is_contiguous() or not token_ids.is_contiguous():
        raise ValueError("row_ids and token_ids must be contiguous")
    if row_ids.device != hidden.device or token_ids.device != hidden.device or weight.device != hidden.device:
        raise ValueError("candidate projection inputs must be on the same device")

    num_pairs = token_ids.numel()
    output = torch.empty(num_pairs, dtype=torch.float32, device=hidden.device)
    if num_pairs == 0:
        return output
    _candidate_projection_kernel[(num_pairs,)](
        hidden,
        weight,
        row_ids,
        token_ids,
        output,
        _autotune_pair_regime(num_pairs),
        hidden.shape[1],
        hidden.stride(0),
        hidden.stride(1),
        weight.stride(0),
        weight.stride(1),
        1.0 / temperature,
    )
    return output
