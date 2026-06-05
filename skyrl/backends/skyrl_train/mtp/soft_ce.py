# Explicit draft-head losses for decoupled Multi-Token Prediction (MTP).
#
# Ported from NeMo-RL's ``DraftCrossEntropyLossFn`` and the ``DRAFT`` branch of
# ``prepare_loss_input``:
#   https://github.com/NVIDIA-NeMo/RL/blob/main/nemo_rl/algorithms/loss/loss_functions.py
#   https://github.com/NVIDIA-NeMo/RL/blob/main/nemo_rl/algorithms/loss/utils.py
#
# The soft cross-entropy matches the forward-KL student gradient
# (``softmax(student) - softmax(teacher)``) and is the default supervision for
# the draft head: the teacher is the *policy's own* next-token distribution
# (detached), so training the draft head never pulls on the policy trunk.

from typing import Optional

import torch
import torch.distributed
import torch.nn.functional as F

from skyrl.backends.skyrl_train.utils.torch_utils import masked_mean


def build_teacher_logits(
    main_logits: torch.Tensor,
    mtp_layer_number: int = 0,
    detach: bool = True,
) -> torch.Tensor:
    """Build the soft-distillation teacher for a given MTP depth.

    The policy's main logits at position ``t`` are the model's distribution over
    ``seq[t + 1]``. MTP layer ``k`` (0-indexed) predicts ``seq[t + k + 2]`` at
    position ``t``, whose teacher distribution (conditioned on the verified
    prefix) is the main model's distribution at position ``t + k + 1`` — i.e.
    ``main_logits`` rolled left by ``k + 1``. This generalizes NeMo-RL's
    single-step roll (its Eagle draft is one layer, rolled by ``-1``).

    Args:
        main_logits: ``[batch, seq, vocab]`` policy logits.
        mtp_layer_number: 0-indexed MTP depth ``k``.
        detach: Detach the teacher so no gradient flows back into the policy
            trunk (the decoupling that makes this "draft" training).

    Returns:
        ``[batch, seq, vocab]`` teacher logits, rolled and (optionally) detached.
        The wrapped-around tail positions are invalid; the caller's loss mask
        must zero them out (see ``shift_mask_for_mtp``).
    """
    teacher = main_logits.detach() if detach else main_logits
    return torch.roll(teacher, shifts=-(mtp_layer_number + 1), dims=1)


def shift_mask_for_mtp(mask: torch.Tensor, mtp_layer_number: int = 0) -> torch.Tensor:
    """Roll a ``[batch, seq]`` loss mask to align with an MTP teacher/label.

    Positions that wrap around the sequence end have no valid target, so they
    are zeroed. This mirrors how Megatron's ``roll_tensor`` zeros the wrapped
    boundary inside ``process_mtp_loss``.
    """
    shift = mtp_layer_number + 1
    rolled = torch.roll(mask, shifts=-shift, dims=1)
    rolled[:, -shift:] = 0
    return rolled


class _VocabParallelSoftCrossEntropy(torch.autograd.Function):
    """Soft cross-entropy ``-sum_v softmax(teacher)_v * log_softmax(student)_v``
    when ``student``/``teacher`` logits are sharded across the vocab (TP) dim.

    Forward reduces across the tensor-parallel group so the softmax denominators
    and the teacher/student dot product are computed over the *global* vocab.
    Backward returns the classic cross-entropy gradient
    ``softmax(student) - softmax(teacher)`` (the teacher is treated as a
    constant; no gradient flows to it). This is the TP analogue of NeMo-RL's
    ``DistributedCrossEntropy``.
    """

    @staticmethod
    def forward(ctx, student_vp_logits, teacher_vp_logits, tp_group):
        student = student_vp_logits.float()
        teacher = teacher_vp_logits.float()

        def global_softmax(logits):
            local_max = logits.max(dim=-1, keepdim=True).values
            torch.distributed.all_reduce(local_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)
            shifted = logits - local_max
            exp = shifted.exp()
            sum_exp = exp.sum(dim=-1, keepdim=True)
            torch.distributed.all_reduce(sum_exp, group=tp_group)
            log_sum_exp = local_max + sum_exp.log()
            return exp / sum_exp, log_sum_exp

        student_softmax, student_lse = global_softmax(student)
        teacher_softmax, _ = global_softmax(teacher)

        # dot = sum_v teacher_softmax_v * student_logit_v  (global over vocab)
        dot = (teacher_softmax * student).sum(dim=-1, keepdim=True)
        torch.distributed.all_reduce(dot, group=tp_group)

        # soft CE = sum_v teacher_p_v * (log_sum_exp - student_logit_v)
        #         = log_sum_exp - dot                (since sum_v teacher_p_v = 1)
        per_token_loss = (student_lse - dot).squeeze(-1)

        ctx.save_for_backward(student_softmax, teacher_softmax)
        return per_token_loss

    @staticmethod
    def backward(ctx, grad_output):
        student_softmax, teacher_softmax = ctx.saved_tensors
        grad_student = (student_softmax - teacher_softmax) * grad_output.unsqueeze(-1)
        return grad_student.to(student_softmax.dtype), None, None


def draft_soft_ce(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    global_normalization_factor: Optional[torch.Tensor] = None,
    vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """Masked-mean soft cross-entropy between draft (student) and policy (teacher).

    Args:
        student_logits: ``[batch, seq, vocab(/tp)]`` draft-head logits (require grad).
        teacher_logits: ``[batch, seq, vocab(/tp)]`` policy logits (detached).
        mask: ``[batch, seq]`` token mask (1 for supervised tokens).
        global_normalization_factor: Optional global valid-token count used as the
            masked-mean denominator (for correct cross-microbatch / DP reduction).
            When ``None``, uses the local masked mean.
        vocab_parallel_group: TP group when logits are vocab-sharded; ``None`` for
            full-vocab logits (single device / FSDP).

    Returns:
        Scalar loss.
    """
    if vocab_parallel_group is not None and torch.distributed.get_world_size(vocab_parallel_group) > 1:
        per_token_loss = _VocabParallelSoftCrossEntropy.apply(
            student_logits, teacher_logits, vocab_parallel_group
        )
    else:
        teacher_probs = F.softmax(teacher_logits.float(), dim=-1)
        student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
        per_token_loss = -(teacher_probs * student_log_probs).sum(dim=-1)

    if global_normalization_factor is not None:
        return (per_token_loss * mask).sum() / global_normalization_factor.clamp(min=1.0)
    return masked_mean(per_token_loss, mask)


def _onehot_vp_logits(
    labels: torch.Tensor,
    like: torch.Tensor,
    vocab_start_index: int,
) -> torch.Tensor:
    """Build vocab-parallel logits whose global softmax is a one-hot over ``labels``.

    Each rank holds ``vocab_size`` columns starting at ``vocab_start_index``. The
    column matching the label (on whichever rank owns it) is set high and all
    others low, so a *global* softmax across the TP group recovers the one-hot
    distribution. Reused to express hard cross-entropy as soft cross-entropy with
    a one-hot teacher.
    """
    vocab_size = like.shape[-1]
    local_idx = labels.long() - vocab_start_index  # [batch, seq]
    holds = (local_idx >= 0) & (local_idx < vocab_size)
    onehot = torch.full_like(like, -30.0)
    safe_idx = local_idx.clamp(0, vocab_size - 1).unsqueeze(-1)
    hot = torch.where(holds.unsqueeze(-1), 30.0, -30.0).to(like.dtype)
    onehot.scatter_(-1, safe_idx, hot)
    return onehot


def draft_hard_ce(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    global_normalization_factor: Optional[torch.Tensor] = None,
    vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    vocab_start_index: Optional[int] = None,
) -> torch.Tensor:
    """Masked-mean hard cross-entropy of the draft head against next-token labels.

    This is the standard MTP-pretraining objective (supervise against the actual
    future token rather than the policy distribution). Use when
    ``mtp_loss_type="hard_ce"``.

    Args:
        student_logits: ``[batch, seq, vocab(/tp)]`` draft-head logits.
        labels: ``[batch, seq]`` target token ids (already rolled for this MTP depth).
        mask: ``[batch, seq]`` token mask.
        global_normalization_factor: see :func:`draft_soft_ce`.
        vocab_parallel_group: TP group when logits are vocab-sharded.
        vocab_start_index: This rank's vocab offset (required when TP-sharded).
    """
    if vocab_parallel_group is not None and torch.distributed.get_world_size(vocab_parallel_group) > 1:
        assert vocab_start_index is not None, "vocab_start_index is required for vocab-parallel hard CE"
        # Hard CE == soft CE with a one-hot teacher; reuse the distributed soft-CE
        # path so the global (TP) softmax / gradient logic lives in one place.
        teacher_onehot = _onehot_vp_logits(labels, student_logits, vocab_start_index)
        per_token_loss = _VocabParallelSoftCrossEntropy.apply(
            student_logits, teacher_onehot, vocab_parallel_group
        )
    else:
        log_probs = F.log_softmax(student_logits.float(), dim=-1)
        per_token_loss = -log_probs.gather(dim=-1, index=labels.unsqueeze(-1).long()).squeeze(-1)

    if global_normalization_factor is not None:
        return (per_token_loss * mask).sum() / global_normalization_factor.clamp(min=1.0)
    return masked_mean(per_token_loss, mask)
