# Combine the policy loss with the decoupled draft (MTP) loss.
#
# Mirrors NeMo-RL's ``DraftLossWrapper``
# (https://github.com/NVIDIA-NeMo/RL/blob/main/nemo_rl/algorithms/loss/wrapper.py):
# the policy loss and the draft loss are computed independently and combined as
# ``policy_loss + w * draft_loss``. Because the draft head is fed *detached*
# trunk hidden states and a *detached* teacher distribution, the draft term only
# produces gradients for the draft-head parameters — the addition is just a
# convenient single backward, not an entanglement of the two objectives.

from dataclasses import dataclass
from typing import Optional

import torch

from skyrl.backends.skyrl_train.mtp.soft_ce import (
    build_teacher_logits,
    draft_hard_ce,
    draft_soft_ce,
    shift_mask_for_mtp,
)


@dataclass
class DraftLossConfig:
    """Resolved knobs for the draft loss (populated from MegatronConfig)."""

    loss_weight: float = 0.1
    loss_type: str = "soft_ce"  # "soft_ce" | "hard_ce"


def combine_policy_and_draft_loss(
    policy_loss: torch.Tensor,
    student_logits_per_layer: list[torch.Tensor],
    main_logits: torch.Tensor,
    mask: torch.Tensor,
    cfg: DraftLossConfig,
    hard_labels: Optional[torch.Tensor] = None,
    global_normalization_factor: Optional[torch.Tensor] = None,
    vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    vocab_start_index: Optional[int] = None,
) -> tuple[torch.Tensor, dict]:
    """Return ``policy_loss + w * draft_loss`` and a metrics dict.

    Args:
        policy_loss: The already-computed scalar policy loss.
        student_logits_per_layer: One ``[batch, seq, vocab(/tp)]`` draft-logits
            tensor per MTP depth, aligned (same de-padding) with ``main_logits``.
        main_logits: ``[batch, seq, vocab(/tp)]`` policy logits — the soft-CE
            teacher source.
        mask: ``[batch, seq]`` token mask over the region the draft is trained on.
        cfg: Draft loss configuration.
        hard_labels: ``[batch, seq]`` base next-token labels (``seq[t+1]`` at ``t``).
            Rolled internally per MTP depth. Required when ``cfg.loss_type == "hard_ce"``.
        global_normalization_factor: Optional global valid-token count for the
            masked-mean denominator (cross-microbatch / DP correct reduction).
        vocab_parallel_group: TP group when logits are vocab-sharded.
        vocab_start_index: This rank's vocab offset (required for vocab-parallel
            hard CE).

    Returns:
        ``(combined_loss, metrics)`` where ``metrics`` carries the (detached)
        draft loss for logging.
    """
    draft_losses = []
    for layer_idx, student_logits in enumerate(student_logits_per_layer):
        layer_mask = shift_mask_for_mtp(mask, layer_idx)
        if cfg.loss_type == "hard_ce":
            assert hard_labels is not None, "hard_ce requires hard_labels"
            layer_labels = torch.roll(hard_labels, shifts=-(layer_idx + 1), dims=1)
            draft_losses.append(
                draft_hard_ce(
                    student_logits,
                    layer_labels,
                    layer_mask,
                    global_normalization_factor=global_normalization_factor,
                    vocab_parallel_group=vocab_parallel_group,
                    vocab_start_index=vocab_start_index,
                )
            )
        else:
            teacher_logits = build_teacher_logits(main_logits, layer_idx, detach=True)
            draft_losses.append(
                draft_soft_ce(
                    student_logits,
                    teacher_logits,
                    layer_mask,
                    global_normalization_factor=global_normalization_factor,
                    vocab_parallel_group=vocab_parallel_group,
                )
            )

    draft_loss = torch.stack(draft_losses).mean()
    combined = policy_loss + cfg.loss_weight * draft_loss
    metrics = {
        "draft_loss": draft_loss.detach().item(),
        "mtp_loss": draft_loss.detach().item(),  # alias kept for existing dashboards
    }
    return combined, metrics


class DraftLossWrapper:
    """Object form of :func:`combine_policy_and_draft_loss` for backends that
    prefer a stateful seam (matches NeMo-RL's ``DraftLossWrapper`` shape). The
    FSDP backend will reuse this unchanged once its adapter lands."""

    def __init__(self, cfg: DraftLossConfig):
        self.cfg = cfg

    def __call__(self, policy_loss, student_logits_per_layer, main_logits, mask, **kwargs):
        return combine_policy_and_draft_loss(
            policy_loss,
            student_logits_per_layer,
            main_logits,
            mask,
            self.cfg,
            **kwargs,
        )
