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

    MTP depth ``k`` supervises position ``t`` against a target read at ``t + k + 1``
    (the policy distribution / label for ``seq[t + k + 2]``). For that supervision to
    be valid, BOTH the *source* position ``t`` (where the draft head produced its
    logits) and the *target* position ``t + k + 1`` (where the teacher/label is read)
    must be real tokens. We therefore require ``mask[t] AND mask[t + shift]``:

    * ``rolled = roll(mask, -shift)`` is the target-side validity ``mask[t + shift]``,
      with the wrapped tail zeroed (no valid target past the sequence end).
    * multiplying by the original ``mask`` re-applies the source-side validity.

    The source-side ``AND`` is what makes this correct under **left padding** (and any
    padding layout): rolling a left-padded mask left turns the last pad slot into a
    ``1`` (its target is the first real token), which would otherwise pull a de-padded
    *zero-logit* (→ uniform) pad position into the loss and inflate it by up to
    ``log(V)`` per leaked position. Re-ANDing the source mask drops those positions, so
    the loss no longer depends on the de-pad zero-fill happening to be masked out.

    Note: each de-padded row holds a single sequence, so the one wrapped boundary is
    handled by zeroing ``rolled``'s tail. If multiple sequences are ever packed into a
    single row, per-sequence boundaries would need ``roll_tensor``-style cu_seqlens
    handling here.
    """
    shift = mtp_layer_number + 1
    rolled = torch.roll(mask, shifts=-shift, dims=1)
    rolled[:, -shift:] = 0
    return mask * rolled


def unpadded_vocab_shard_width(
    true_vocab_size: Optional[int],
    vocab_shard_width: int,
    tp_rank: int,
) -> int:
    """Width of this rank's vocab shard excluding Megatron's padded tail.

    When the global vocab is padded to divide across TP (``should_pad_vocab``), the padding occupies
    the tail of the last rank(s)' shards (the vocab is partitioned contiguously). Padded weight rows
    are never trained (zero-filled by the HF conversion), so their logits must not enter the draft
    loss: in the teacher softmax they leak ~uniform probability mass, and in the teacher top-k they
    can steal slots from real tokens. Callers slice ``logits[..., :width]`` — a view, so autograd
    zero-fills the padded tail's gradient and per-rank widths may differ (the loss collectives only
    reduce ``[..., 1]``/``[...]``-shaped tensors, never the vocab dim).

    Returns ``vocab_shard_width`` unchanged when ``true_vocab_size`` is ``None`` (unknown) or the
    shard holds no padding.
    """
    if true_vocab_size is None:
        return vocab_shard_width
    start = tp_rank * vocab_shard_width
    return max(0, min(vocab_shard_width, true_vocab_size - start))


def _vocab_parallel_softmax(vocab_parallel_logits, group):
    """Global softmax over a TP-sharded vocab dim. Allocates a single full-vocab output tensor (the
    ``- logits_max`` subtraction) and does the rest in place on it; the input is NOT mutated (so it is
    safe even when the caller passes float32 logits, where ``.float()`` is a no-op). Mirrors NeMo-RL's
    ``_compute_distributed_softmax``."""
    logits_max = torch.amax(vocab_parallel_logits, dim=-1, keepdim=True)
    torch.distributed.all_reduce(logits_max, op=torch.distributed.ReduceOp.MAX, group=group)
    exp_logits = (vocab_parallel_logits - logits_max).exp_()  # new buffer, then in place
    sum_exp = exp_logits.sum(dim=-1, keepdim=True)
    torch.distributed.all_reduce(sum_exp, op=torch.distributed.ReduceOp.SUM, group=group)
    return exp_logits.div_(sum_exp)


def _vocab_parallel_log_softmax(vocab_parallel_logits, group):
    """Global log-softmax over a TP-sharded vocab dim. Uses ``torch.logsumexp`` (a fused
    ``[.,.,V]->[.,.,1]`` reduction) so it never materializes a full-vocab ``exp`` temporary -- the key
    memory fix at a 248K vocab, where that temp is multiple GiB (NeMo-RL's
    ``_compute_distributed_log_softmax`` allocates it via an explicit ``.exp().sum()``). Allocates one
    full-vocab output buffer; the input is NOT mutated."""
    logits_max = torch.amax(vocab_parallel_logits, dim=-1, keepdim=True)
    torch.distributed.all_reduce(logits_max, op=torch.distributed.ReduceOp.MAX, group=group)
    shifted = vocab_parallel_logits - logits_max  # new buffer (input untouched); values now <= 0
    # exp() of the *local* logsumexp recovers this shard's sum-of-exp, which reduces across TP.
    local_sum_exp = torch.logsumexp(shifted, dim=-1, keepdim=True).exp_()
    torch.distributed.all_reduce(local_sum_exp, op=torch.distributed.ReduceOp.SUM, group=group)
    return shifted.sub_(local_sum_exp.log_())  # in place on the owned buffer -> log-softmax


class _VocabParallelSoftCrossEntropy(torch.autograd.Function):
    """Soft cross-entropy ``-sum_v softmax(teacher)_v * log_softmax(student)_v`` when
    ``student``/``teacher`` logits are sharded across the vocab (TP) dim.

    Memory-lean port of NeMo-RL's ``DistributedCrossEntropy`` (``nemo_rl/distributed/model_utils.py``):
    the per-token cross-entropy is a single ``einsum`` (no full-vocab product tensor) and the student
    log-prob buffer is reused **in place** for the backward softmax, so only two full-vocab tensors
    are kept (vs ~6-8 in the naive form -- the difference is several GiB at a 248K vocab). Forward
    all-reduces across the TP group so the softmax normalizers and the teacher/student dot are over
    the *global* vocab. Backward returns the classic cross-entropy gradient
    ``softmax(student) - softmax(teacher)`` (teacher is a detached target; no gradient flows to it).
    """

    @staticmethod
    def forward(ctx, student_vp_logits, teacher_vp_logits, tp_group):
        ctx.input_dtype = student_vp_logits.dtype
        # Detached teacher target distribution (global softmax over the sharded vocab).
        target_probs = _vocab_parallel_softmax(teacher_vp_logits.float(), tp_group)
        # Student global log-probs; the same buffer is turned into softmax(student) in place below.
        student_log_probs = _vocab_parallel_log_softmax(student_vp_logits.float(), tp_group)
        # soft CE = -sum_v p_teacher_v * log q_student_v: dot over the local shard, then reduce.
        per_token_loss = torch.einsum("...v,...v->...", target_probs, student_log_probs).neg_()
        torch.distributed.all_reduce(per_token_loss, op=torch.distributed.ReduceOp.SUM, group=tp_group)
        # exp_ in place: log_softmax(student) -> softmax(student) for the gradient (no new alloc).
        student_probs = student_log_probs.exp_()
        ctx.save_for_backward(student_probs, target_probs)
        return per_token_loss.contiguous()

    @staticmethod
    def backward(ctx, grad_output):
        student_probs, target_probs = ctx.saved_tensors
        # d(H(p, q))/d(student_logit_v) = softmax(student)_v - softmax(teacher)_v. Done in place on
        # student_probs (a private buffer from forward, used nowhere else) to avoid allocating extra
        # full-vocab temporaries during backward -- at a 248K vocab each would be multiple GiB.
        grad_student = student_probs.sub_(target_probs).mul_(grad_output.unsqueeze(-1))
        return grad_student.to(ctx.input_dtype), None, None


def _masked_mean_or_global(per_token_loss, mask, global_normalization_factor):
    """Masked mean of a ``[batch, seq]`` per-token loss, using either the local token count or a
    provided global valid-token count as the denominator."""
    if global_normalization_factor is not None:
        return (per_token_loss * mask).sum() / global_normalization_factor.clamp(min=1.0)
    return masked_mean(per_token_loss, mask)


def _chunked_masked_loss(per_token_fn, sliceable_args, mask, chunk_size, global_normalization_factor):
    """Masked-mean loss computed in ``chunk_size`` slices along the sequence dim.

    ``per_token_fn(*sliced_args)`` returns the ``[batch, chunk]`` per-token loss for a slice. Each
    chunk is gradient-checkpointed, so the (large, full-vocab) softmax tensors are recomputed in
    backward instead of all being held at once -- this bounds peak memory to a *single* chunk's vocab
    tensors. Required for large-vocab models (Qwen3.5's 248K vocab) where the full-sequence softmax
    OOMs even after the lean kernel. The result is numerically identical to the un-chunked masked mean
    (same global denominator); only the activation memory differs.
    """
    import torch.utils.checkpoint as checkpoint

    seq_len = sliceable_args[0].shape[1]
    masked_sum = None
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        sliced = [a[:, start:end] for a in sliceable_args]
        chunk_mask = mask[:, start:end]

        def _chunk_masked_sum(*args, _m=chunk_mask):
            return (per_token_fn(*args) * _m).sum()

        chunk_sum = checkpoint.checkpoint(_chunk_masked_sum, *sliced, use_reentrant=False)
        masked_sum = chunk_sum if masked_sum is None else masked_sum + chunk_sum

    if masked_sum is None:  # empty sequence
        masked_sum = mask.sum() * 0.0
    denom = (
        global_normalization_factor.clamp(min=1.0)
        if global_normalization_factor is not None
        else mask.sum().clamp(min=1.0)
    )
    return masked_sum / denom


def draft_soft_ce(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    global_normalization_factor: Optional[torch.Tensor] = None,
    vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    chunk_size: Optional[int] = None,
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
        chunk_size: If set, compute the loss in sequence-chunks of this size with gradient
            checkpointing to bound peak memory (large-vocab models). ``None`` computes the whole
            sequence at once. The result is identical either way.

    Returns:
        Scalar loss.
    """
    use_vp = vocab_parallel_group is not None and torch.distributed.get_world_size(vocab_parallel_group) > 1

    def per_token(student, teacher):
        if use_vp:
            return _VocabParallelSoftCrossEntropy.apply(student, teacher, vocab_parallel_group)
        teacher_probs = F.softmax(teacher.float(), dim=-1)
        student_log_probs = F.log_softmax(student.float(), dim=-1)
        return -(teacher_probs * student_log_probs).sum(dim=-1)

    if chunk_size is not None and student_logits.shape[1] > chunk_size:
        return _chunked_masked_loss(
            per_token, (student_logits, teacher_logits), mask, chunk_size, global_normalization_factor
        )
    return _masked_mean_or_global(per_token(student_logits, teacher_logits), mask, global_normalization_factor)


class _VocabParallelTopkSoftCE(torch.autograd.Function):
    """Top-k approximation of the soft cross-entropy ``-sum_v softmax(teacher)_v * log_softmax(student)_v``
    that never materializes a full-vocab softmax.

    Only the teacher's top-k tokens are distilled, with teacher and student renormalized over that set.
    Memory is ``O(seq * k)`` instead of ``O(seq * vocab)`` -- it fits at a 248K vocab *and* avoids the
    allocator fragmentation that the full-vocab / chunked paths cause.

    Parallelism: the vocab is sharded over the tensor-parallel ``group``. We take **each rank's local
    top-k** of its shard and reconcile the softmax normalizers and the per-token cross-entropy across
    the group with ``all_reduce`` (MAX for the stable-softmax shift, SUM for the denominators and the
    loss). No ``all_gather`` / global-index gather is needed because student and teacher share the same
    vocab partition, so a rank's local top-k indices address both. This works for ANY tensor-parallel
    size and is correct across nodes (the collectives run over the configured ``group``); ``group=None``
    / world-size 1 is the single-device path. The distilled set is the union of the per-rank top-k
    (<= ``k * tp_size`` tokens) -- a benign, richer signal at larger TP, not a correctness issue.

    Backward is manual (scatter the ``softmax(student) - softmax(teacher)`` gradient back to this rank's
    own top-k columns), so no gradient flows through the collectives.
    """

    @staticmethod
    def forward(ctx, student_logits, teacher_logits, k, group, roll_shift):
        ws = torch.distributed.get_world_size(group) if group is not None else 1

        def ar(t, op):
            if ws > 1:
                torch.distributed.all_reduce(t, op=op, group=group)
            return t

        k_eff = min(int(k), teacher_logits.shape[-1])
        # Teacher's local top-k (teacher is detached). When roll_shift != 0, teacher_logits is the
        # UN-rolled policy logits: position t's draft target is the policy distribution at t+roll_shift,
        # so we top-k the policy logits once (no full-vocab copy) and roll only the small [B, S, k]
        # result -- avoiding a ~[S, vocab] rolled-teacher copy. The wrapped boundary positions are
        # zeroed by the caller's shifted loss mask.
        t_vals, t_idx = teacher_logits.topk(k_eff, dim=-1)
        if roll_shift:
            t_vals = torch.roll(t_vals, shifts=-int(roll_shift), dims=1)
            t_idx = torch.roll(t_idx, shifts=-int(roll_shift), dims=1)
        t_vals = t_vals.float()
        # student at position t, teacher's top-k indices for position t (already rolled).
        s_vals = student_logits.gather(-1, t_idx).float()

        # Stable-softmax shift = global max over the union (across the TP group).
        t_max = ar(t_vals.max(dim=-1, keepdim=True).values.clone(), torch.distributed.ReduceOp.MAX)
        s_max = ar(s_vals.max(dim=-1, keepdim=True).values.clone(), torch.distributed.ReduceOp.MAX)

        # Teacher probs over the union (denominator summed across the group).
        t_exp = (t_vals - t_max).exp()
        t_denom = ar(t_exp.sum(dim=-1, keepdim=True), torch.distributed.ReduceOp.SUM)
        t_p = t_exp / t_denom

        # Student probs / log-probs over the union (denominator summed across the group).
        s_exp = (s_vals - s_max).exp()
        s_denom = ar(s_exp.sum(dim=-1, keepdim=True), torch.distributed.ReduceOp.SUM)
        q_s = s_exp / s_denom
        s_logp = (s_vals - s_max) - s_denom.log()

        # Per-rank partial CE summed across the group -> soft CE over the global union.
        per_token_loss = ar(-(t_p * s_logp).sum(dim=-1), torch.distributed.ReduceOp.SUM)

        ctx.save_for_backward(q_s, t_p, t_idx)
        ctx.vocab_size = student_logits.shape[-1]
        ctx.input_dtype = student_logits.dtype
        return per_token_loss

    @staticmethod
    def backward(ctx, grad_output):
        q_s, t_p, t_idx = ctx.saved_tensors
        # d(H(p,q))/d(student_logit_v) = softmax(student)_v - softmax(teacher)_v, over the union; zero
        # elsewhere. Scatter the k per-token grads back to this rank's own vocab columns. The full-vocab
        # buffer is allocated directly in the input dtype (the only fp32->input cast is the tiny [.., k]
        # grad); an fp32 buffer here would double the largest transient of the whole loss.
        grad_topk = (q_s - t_p) * grad_output.unsqueeze(-1)
        grad_student = torch.zeros(*t_idx.shape[:-1], ctx.vocab_size, dtype=ctx.input_dtype, device=grad_topk.device)
        grad_student.scatter_(-1, t_idx, grad_topk.to(ctx.input_dtype))
        return grad_student, None, None, None, None


def draft_soft_ce_topk(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    k: int,
    global_normalization_factor: Optional[torch.Tensor] = None,
    vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    roll_shift: int = 0,
) -> torch.Tensor:
    """Masked-mean **top-k** soft cross-entropy between draft (student) and policy (teacher).

    Distills only the teacher's top-``k`` tokens (renormalized over that set) -- ``O(seq*k)`` memory,
    no full-vocab softmax, so it fits + avoids fragmentation at large vocab. See
    :class:`_VocabParallelTopkSoftCE` for the (parallelism-scalable) implementation. Approximate: drops
    the teacher tail, which is benign for a draft head (acceptance is governed by the top tokens).

    Args mirror :func:`draft_soft_ce`, plus:
        k: number of top teacher tokens to distill.
        roll_shift: if non-zero, ``teacher_logits`` is the *un-rolled* policy logits and the draft
            target for position ``t`` is the policy distribution at ``t + roll_shift`` (= the MTP-depth
            roll). Top-k is taken on the un-rolled logits and only the small ``[B, S, k]`` result is
            rolled, avoiding a full ``[S, vocab]`` rolled-teacher copy. ``0`` means ``teacher_logits``
            is already aligned/rolled (the caller's ``build_teacher_logits`` path).
    """
    per_token_loss = _VocabParallelTopkSoftCE.apply(
        student_logits, teacher_logits.detach(), k, vocab_parallel_group, roll_shift
    )
    return _masked_mean_or_global(per_token_loss, mask, global_normalization_factor)


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
    chunk_size: Optional[int] = None,
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
        chunk_size: see :func:`draft_soft_ce`. Sequence-chunked + checkpointed when set.
    """
    use_vp = vocab_parallel_group is not None and torch.distributed.get_world_size(vocab_parallel_group) > 1
    if use_vp:
        assert vocab_start_index is not None, "vocab_start_index is required for vocab-parallel hard CE"

    def per_token(student, lbl):
        if use_vp:
            # Hard CE == soft CE with a one-hot teacher; reuse the distributed soft-CE path so the
            # global (TP) softmax / gradient logic lives in one place.
            teacher_onehot = _onehot_vp_logits(lbl, student, vocab_start_index)
            return _VocabParallelSoftCrossEntropy.apply(student, teacher_onehot, vocab_parallel_group)
        log_probs = F.log_softmax(student.float(), dim=-1)
        return -log_probs.gather(dim=-1, index=lbl.unsqueeze(-1).long()).squeeze(-1)

    if chunk_size is not None and student_logits.shape[1] > chunk_size:
        return _chunked_masked_loss(per_token, (student_logits, labels), mask, chunk_size, global_normalization_factor)
    return _masked_mean_or_global(per_token(student_logits, labels), mask, global_normalization_factor)
