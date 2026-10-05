"""Scale SkyRL's policy loss for Megatron's two loss-normalization modes.

SkyRL normalizes the policy loss before the worker sees it: advantages are
pre-scaled per mini-batch (``apply_loss_reduction_to_advantages_minibatch``,
NovaSky-AI/SkyRL#1296), so the intended gradient is the plain sum of every
microbatch's policy loss on every DP x CP rank, plus regularizers (KL, entropy,
MTP) averaged over DP ranks and real microbatches. Megatron then applies fixed
factors that depend on ``TransformerConfig.calculate_per_token_loss``:

* off (default): the pipeline schedule multiplies a ``(loss, metrics)`` return by
  ``cp / num_microbatches`` and DDP averages gradients over ``dp * cp``.
* on: a ``(loss, num_tokens, metrics)`` return is left unscaled and DDP sums
  gradients over ``dp * cp``; ``finalize_model_grads`` would then divide by the
  summed ``num_tokens``, but SkyRL calls it without a token count (see
  ``MegatronModelWrapper.run_pending_grad_sync``) because the normalization
  already lives in the advantages. Megatron-Bridge's Qwen-VL providers force
  this mode when CP > 1.

``megatron_loss_output`` returns the value that yields the same gradient in
both modes.
"""

from typing import Any, Tuple, Union

import torch


def megatron_loss_output(
    normalized_loss: torch.Tensor,
    regularizer: Union[torch.Tensor, float],
    metrics: Any,
    *,
    num_microbatches: int,
    num_real_microbatches: int,
    dp_size: int,
    per_token_loss: bool,
    num_tokens: int = 0,
) -> Tuple:
    """Return Megatron's loss_func output for SkyRL's pre-normalized loss.

    Args:
        normalized_loss: pre-scaled policy loss of this microbatch.
        regularizer: per-microbatch mean term (KL - entropy, MTP draft loss).
        metrics: passed through.
        num_microbatches: microbatches in this forward_backward on this rank.
        num_real_microbatches: microbatches carrying real samples.
        dp_size: data-parallel size without context parallelism.
        per_token_loss: ``TransformerConfig.calculate_per_token_loss``.
        num_tokens: reported to Megatron in per-token mode; informational only,
            since the final gradient step is run without a token-count division.
    """
    if not per_token_loss:
        # Default mode: undo the schedule's 1/num_microbatches and DDP's 1/dp (CP ranks hold
        # different tokens and are meant to be summed, which the schedule's x cp restores).
        kl_entropy_microbatch_scale = num_microbatches / max(1, num_real_microbatches)
        return normalized_loss * num_microbatches * dp_size + regularizer * kl_entropy_microbatch_scale, metrics
    # Per-token mode: Megatron sums losses over microbatches and gradients over DP x CP, which is
    # already the intended reduction for the pre-scaled policy loss; regularizers keep their average.
    loss = normalized_loss + regularizer / (dp_size * max(1, num_real_microbatches))
    return loss, torch.tensor(num_tokens, dtype=torch.int, device=normalized_loss.device), metrics
