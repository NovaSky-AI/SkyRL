"""Scale SkyRL's policy loss for Megatron's two loss-normalization modes.

SkyRL normalizes the policy loss before the worker sees it (advantages are
pre-scaled per mini-batch, see ``apply_loss_reduction_to_advantages_minibatch``),
so the intended gradient is the plain sum of every microbatch's policy loss on
every DP x CP rank, plus regularizers (KL, entropy, MTP) averaged over DP ranks
and real microbatches. Megatron then applies fixed factors that depend on
``TransformerConfig.calculate_per_token_loss``:

* off (default): the pipeline schedule multiplies a ``(loss, metrics)`` return by
  ``cp / num_microbatches`` and DDP averages gradients over ``dp * cp``.
* on: a ``(loss, num_tokens, metrics)`` return is left unscaled, DDP sums
  gradients over ``dp * cp`` and ``finalize_model_grads`` divides them by the
  all-reduced sum of ``num_tokens``. Megatron-Bridge's Qwen-VL providers force
  this mode when CP > 1.

``megatron_loss_output`` returns the value that yields the same gradient in
both modes.
"""

from typing import Any, Optional, Tuple, Union

import torch


def megatron_loss_output(
    normalized_loss: torch.Tensor,
    regularizer: Union[torch.Tensor, float],
    metrics: Any,
    *,
    num_microbatches: int,
    num_real_microbatches: int,
    dp_size: int,
    num_tokens_local: Optional[int] = None,
    num_tokens_global: Optional[int] = None,
) -> Tuple:
    """Return Megatron's loss_func output for SkyRL's pre-normalized loss.

    Args:
        normalized_loss: pre-scaled policy loss of this microbatch.
        regularizer: per-microbatch mean term (KL - entropy, MTP draft loss).
        metrics: passed through.
        num_microbatches: microbatches in this forward_backward on this rank.
        num_real_microbatches: microbatches carrying real samples.
        dp_size: data-parallel size without context parallelism.
        num_tokens_local / num_tokens_global: set only in per-token mode. The local
            count is what this microbatch reports to Megatron; the global count is
            the all-reduced sum of the local counts over all microbatches and
            DP x CP ranks, i.e. exactly what ``finalize_model_grads`` divides by.
    """
    if num_tokens_global is None:
        # Default mode: undo the schedule's 1/num_microbatches and DDP's 1/dp (CP ranks hold
        # different tokens and are meant to be summed, which the schedule's x cp restores).
        kl_entropy_microbatch_scale = num_microbatches / max(1, num_real_microbatches)
        return normalized_loss * num_microbatches * dp_size + regularizer * kl_entropy_microbatch_scale, metrics
    # Per-token mode: multiply by the global token count so finalize's division cancels.
    loss = (normalized_loss + regularizer / (dp_size * max(1, num_real_microbatches))) * num_tokens_global
    num_tokens = torch.tensor(num_tokens_local, dtype=torch.int, device=normalized_loss.device)
    return loss, num_tokens, metrics
