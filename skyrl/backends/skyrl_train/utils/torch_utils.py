# This code is adapted from VERL
# https://github.com/volcengine/verl/blob/main/verl/utils/torch_functional.py
# The original copyright is reproduced below:
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from jaxtyping import Float, Integer

try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

    FLASH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = True
except ImportError:
    FLASH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = False


def chunked_cross_entropy_from_log_probs(
    logprobs: Float[torch.Tensor, "batch_size seqlen vocab_size"],
    requires_grad: bool = False,
    chunk_size: int = 1024,
) -> Float[torch.Tensor, "batch_size seqlen"]:
    cm = nullcontext() if requires_grad else torch.no_grad()
    with cm:
        # Calculate entropy in chunks to avoid OOM
        num_chunks = (logprobs.size(1) + chunk_size - 1) // chunk_size
        entropy_tensor = torch.zeros(
            (logprobs.shape[0], logprobs.shape[1]), dtype=logprobs.dtype, device=logprobs.device
        )

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, logprobs.size(1))
            # (bsz, seq, vocab_size)
            chunk = logprobs[:, start_idx:end_idx]

            # Calculate entropy for this chunk
            chunk_probs = chunk.exp()
            chunk_entropy = -(chunk_probs * chunk).sum(-1)
            entropy_tensor[:, start_idx:end_idx] = chunk_entropy
    return entropy_tensor


# NOTE: we don't actually use jaxtype for runtime type checking since it doesn't play well with torch compile
def chunked_entropy_from_logits(
    logits: Float[torch.Tensor, "batch_size seqlen vocab"],
    requires_grad: bool = False,
    attention_mask: Float[torch.Tensor, "batch_size seqlen"] = None,
    chunk_size: int = 1024,
) -> Float[torch.Tensor, "batch_size seqlen"]:
    """Chunked entropy calculation from logits.

    Avoids allocating a full log probabilities tensor to save memory. For models like Qwen with large vocab sizes, this can reduce gpu memory significantly (~O(10GB))

    Args:
        logits: Input logits of shape (batch_size, seqlen, vocab_size)
        requires_grad: Whether to enable gradient computation
        attention_mask: Optional attention mask of shape (batch_size, seqlen). When provided,
                       entropy values for padded positions (mask=0) will be zeroed out.
        chunk_size: Sequence dimension chunk size (must be a positive integer).

    Returns:
        Entropy tensor of shape (batch_size, seqlen). If attention_mask is provided,
        positions with mask=0 will have entropy=0.
    """
    # Validate attention mask shape if provided
    if attention_mask is not None:
        if attention_mask.shape != (logits.shape[0], logits.shape[1]):
            raise ValueError(
                f"attention_mask shape {attention_mask.shape} does not match logits shape "
                f"(batch_size={logits.shape[0]}, seqlen={logits.shape[1]}). "
                f"Expected attention_mask shape: ({logits.shape[0]}, {logits.shape[1]})"
            )

    cm = nullcontext() if requires_grad else torch.no_grad()
    with cm:
        # Calculate entropy in chunks to avoid OOM
        num_chunks = (logits.size(1) + chunk_size - 1) // chunk_size
        entropy_tensor = torch.zeros((logits.shape[0], logits.shape[1]), dtype=logits.dtype, device=logits.device)

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, logits.size(1))
            # (bsz, seq, vocab_size)
            chunk = logits[:, start_idx:end_idx]
            chunk_logprob = F.log_softmax(chunk, dim=-1)

            # Calculate entropy for this chunk
            chunk_probs = chunk_logprob.exp()
            chunk_entropy = -(chunk_probs * chunk_logprob).sum(-1)

            # Apply attention mask if provided
            if attention_mask is not None:
                chunk_mask = attention_mask[:, start_idx:end_idx]
                chunk_entropy = chunk_entropy * chunk_mask

            entropy_tensor[:, start_idx:end_idx] = chunk_entropy
    return entropy_tensor


# Adapt from VERL
def logprobs_from_logits(
    logits: Float[torch.Tensor, "batch_size seqlen vocab_size"],
    labels: Integer[torch.Tensor, "batch_size seqlen"],
    inplace_backward=True,
) -> Float[torch.Tensor, "batch_size seqlen"]:
    """
    Compute per-token log-probabilities for the given labels.

    Uses a Flash-Attention-based cross-entropy (if available) for efficient backward,
    otherwise falls back to a standard log-softmax+gather approach.

    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591

    Args:
        logits (Tensor): Model outputs of shape (..., vocab_size).
        labels (LongTensor): True class indices of shape matching logits[..., :-1].
        inplace_backward (bool): If True and Flash-Attn is available, perform backward in-place.

    Returns:
        Tensor: Log-probabilities of the target labels, shape logits.shape[:-1].
    """
    # The flash-attn kernel may be importable on CPU but requires CUDA.
    if FLASH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE and logits.is_cuda:
        batch_dim = logits.shape[:-1]
        last_dim = logits.shape[-1]
        logits = logits.reshape(-1, last_dim)
        labels = labels.reshape(-1)
        output = logprobs_from_logits_flash_attn(logits, labels, inplace_backward=inplace_backward)
        output = output.view(*batch_dim)
    else:
        output = logprobs_from_logits_v2(logits, labels)
    return output


def logprobs_from_logits_flash_attn(logits, labels, inplace_backward=True):
    output = cross_entropy_loss(logits, labels, inplace_backward=inplace_backward)
    assert isinstance(
        output, tuple
    ), "please make sure flash-attn>=2.4.3 where cross_entropy_loss returns Tuple[losses, z_losses]."
    return -output[0]


# Credits: https://www.tylerromero.com/posts/2025-02-selective-log-softmax/
# https://github.com/volcengine/verl/pull/220
def logprobs_from_logits_v2(
    logits: Float[torch.Tensor, "batch_size seqlen vocab_size"], labels: Integer[torch.Tensor, "batch_size seqlen"]
) -> Float[torch.Tensor, "batch_size seqlen"]:
    """
    A memory efficient implementation of logprobs_from_logits
    """
    if logits.dtype in [torch.float32, torch.float64]:
        logits_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(logit, dim=-1) for logit in logits])
        logprobs_labels = logits_labels - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        logprobs_labels = []
        for row_logits, row_labels in zip(logits, labels):  # loop to reduce peak mem consumption
            row_logprobs = F.log_softmax(row_logits, dim=-1)
            row_logprobs_labels = row_logprobs.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            logprobs_labels.append(row_logprobs_labels)
        logprobs_labels = torch.stack(logprobs_labels)
    return logprobs_labels


class _LogprobsAndTopKLogprobsFromLogits(torch.autograd.Function):
    """Label logprobs and top-k logprobs from one pass over the logits, chunked along tokens.

    Forward computes ``log_softmax(logits)`` gathered at ``labels`` and at ``topk_ids`` without
    materializing a full-vocabulary log-softmax. Backward recomputes the softmax per chunk and,
    with ``inplace_backward``, writes the logits gradient into the logits buffer itself (as the
    flash-attn cross entropy used by ``logprobs_from_logits`` does), so score centering adds no
    second vocabulary-sized gradient tensor.
    """

    @staticmethod
    def forward(ctx, logits, labels, topk_ids, chunk_size, inplace_backward):
        num_tokens = logits.shape[0]
        label_logprobs = torch.empty(num_tokens, dtype=torch.float32, device=logits.device)
        topk_logprobs = torch.empty((num_tokens, topk_ids.shape[1]), dtype=torch.float32, device=logits.device)
        for start in range(0, num_tokens, chunk_size):
            end = min(start + chunk_size, num_tokens)
            chunk = logits[start:end].float()
            lse = torch.logsumexp(chunk, dim=-1)
            label_logprobs[start:end] = chunk.gather(-1, labels[start:end, None]).squeeze(-1) - lse
            topk_logprobs[start:end] = chunk.gather(-1, topk_ids[start:end]) - lse[:, None]
        ctx.save_for_backward(logits, labels, topk_ids)
        ctx.chunk_size = chunk_size
        ctx.inplace_backward = inplace_backward
        return label_logprobs, topk_logprobs

    @staticmethod
    def backward(ctx, grad_label_logprobs, grad_topk_logprobs):
        logits, labels, topk_ids = ctx.saved_tensors
        num_tokens = logits.shape[0]
        if grad_label_logprobs is None:
            grad_label_logprobs = torch.zeros(num_tokens, dtype=torch.float32, device=logits.device)
        if grad_topk_logprobs is None:
            grad_topk_logprobs = torch.zeros(topk_ids.shape, dtype=torch.float32, device=logits.device)
        grad_logits = logits if ctx.inplace_backward else torch.empty_like(logits)
        for start in range(0, num_tokens, ctx.chunk_size):
            end = min(start + ctx.chunk_size, num_tokens)
            grad_label = grad_label_logprobs[start:end].float()
            grad_topk = grad_topk_logprobs[start:end].float()
            # d log p_v / d logit_u = delta_{uv} - p_u
            coef = grad_label + grad_topk.sum(dim=-1)
            grad_chunk = torch.softmax(logits[start:end].float(), dim=-1) * (-coef[:, None])
            grad_chunk.scatter_add_(-1, labels[start:end, None], grad_label[:, None])
            grad_chunk.scatter_add_(-1, topk_ids[start:end], grad_topk)
            grad_logits[start:end].copy_(grad_chunk)
        return grad_logits, None, None, None, None


def logprobs_and_topk_logprobs_from_logits(
    logits: Float[torch.Tensor, "... vocab_size"],
    labels: Integer[torch.Tensor, "..."],
    topk_ids: Integer[torch.Tensor, "... k"],
    chunk_size: int = 1024,
    inplace_backward: bool = True,
) -> tuple[Float[torch.Tensor, "..."], Float[torch.Tensor, "... k"]]:
    """Per-token label logprobs plus logprobs of ``k`` extra token ids per position.

    Used by score centering, which needs the trainer's logprob of every token in the sampler's
    top-k head in addition to the sampled token. Both outputs are float32 and differentiable
    with respect to ``logits``.

    Args:
        logits: ``(..., vocab_size)`` model outputs.
        labels: ``(...)`` sampled token ids.
        topk_ids: ``(..., k)`` token ids whose logprobs are also returned.
        chunk_size: Number of token positions processed per chunk to bound peak memory.
        inplace_backward: Write the logits gradient into the logits buffer.
    """
    batch_dims = logits.shape[:-1]
    vocab_size = logits.shape[-1]
    k = topk_ids.shape[-1]
    label_logprobs, topk_logprobs = _LogprobsAndTopKLogprobsFromLogits.apply(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1).long(),
        topk_ids.reshape(-1, k).long(),
        chunk_size,
        inplace_backward,
    )
    return label_logprobs.view(*batch_dims), topk_logprobs.view(*batch_dims, k)


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor | None, dim: int | None = None) -> torch.Tensor:
    """Compute the mean of tensor elements, optionally masked and reduced along a dimension."""
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim).clamp(min=1.0)


def safe_exp_delta(delta: torch.Tensor, clip: float = 20.0, out_dtype=None) -> torch.Tensor:
    """
    Clamp the delta before exponentiating to avoid potential overflow.
    """
    y = torch.clamp(delta.to(torch.float32), -clip, clip).exp()
    return y.to(out_dtype or delta.dtype)
