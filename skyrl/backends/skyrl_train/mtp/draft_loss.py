# MTP draft-loss helpers: the hard-label loss and its alignment (see ``mtp_action_aligned``), the distribution
# loss's packed mask, and the per-depth token counts both share. Megatron is imported inside the functions that use
# it, so CPU CI can import this module without the megatron extra.

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from skyrl.backends.skyrl_train.mtp.soft_ce import shift_mask_for_mtp
from skyrl.train.config.config import MTPLossType
from skyrl.train.fused_lm_head import FusedLmHeadBackend


def _action_columns(cols: int, num_actions: int, target_shift: int) -> Tuple[int, int]:
    """``[lo, end)``: the log-prob columns that score the action tokens (see ``mtp_action_aligned``)."""
    # With L = cols + 1 tokens, A = num_actions and k = target_shift: action j is token L - A + j, which
    # column L - A + j - k scores, so the actions occupy columns [L - A - k, L - k) = [end - A, end).
    end = cols - (target_shift - 1)  # exclusive; column end - 1 scores seq[cols] (= the last token)
    start = end - num_actions
    # start < 0 when the prompt region (L - A columns, padding included) is shorter than k: the first
    # min(-start, A) actions have no head position in the sequence, so only columns [lo, end) exist.
    return max(start, 0), end


def mtp_action_valid(
    cols: int,
    num_actions: int,
    target_shift: int,
    loss_mask: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """``[batch, num_actions]`` bool: the action targets that count for ``mtp_action_aligned`` with ``cols``
    log-prob columns. It depends only on the masks and shapes, so a batch's count can be taken before any forward.
    """
    lo, end = _action_columns(cols, num_actions, target_shift)
    kept = end - lo
    pos_ok = torch.zeros(loss_mask.shape, dtype=torch.bool, device=loss_mask.device)
    if kept > 0:
        pos_ok[:, num_actions - kept :] = attention_mask[:, lo:end].to(torch.bool)
    return (loss_mask > 0) & pos_ok


def mtp_action_aligned(
    token_lp: torch.Tensor,
    num_actions: int,
    target_shift: int,
    loss_mask: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MTP log-probs aligned to the action tokens, and the mask of positions that count.

    Args:
        token_lp: ``[batch, seq - 1]``; column ``s`` scores ``seq[s + target_shift]``.
        num_actions: number of action (generated-region) columns, as for the policy.
        target_shift: ``d + 2`` for MTP depth ``d``.
        loss_mask: ``[batch, num_actions]``; trainable generated tokens.
        attention_mask: ``[batch, seq]``; real tokens.

    Returns:
        ``(lp, valid)``, both ``[batch, num_actions]``: ``lp[:, i]`` scores action token ``i`` from the
        position ``target_shift`` earlier, zeroed where invalid; ``valid`` (in ``lp``'s dtype) is 1 where
        that target is a trainable token and the head's position is a real token of the same row.
        Actions whose column would sit before the sequence start (a prompt region shorter than
        ``target_shift``) are invalid. Targets the log-prob helpers wrap or take from padding lie past the
        row's last real token, so their column is ``>= seq - target_shift`` (never selected) or their action
        is padding (loss mask 0).
    """
    batch, cols = token_lp.shape
    lo, end = _action_columns(cols, num_actions, target_shift)
    kept = end - lo
    lp = token_lp.new_zeros(batch, num_actions)
    if kept > 0:
        # The missing columns are the earliest actions, so the kept ones fill the last ``kept`` action slots.
        lp[:, num_actions - kept :] = token_lp[:, lo:end]
    valid = mtp_action_valid(cols, num_actions, target_shift, loss_mask, attention_mask)
    return lp.masked_fill(~valid, 0.0), valid.to(lp.dtype)


def segment_real_lengths(attention_mask: torch.Tensor, sub_seq_lengths=None) -> torch.Tensor:
    """Real tokens per packed segment, in packing order: one segment per row, or each row's ``sub_seq_lengths``."""
    if sub_seq_lengths is None:
        return attention_mask.sum(-1, dtype=torch.long)
    return torch.cat([torch.as_tensor(row, dtype=torch.long).flatten() for row in sub_seq_lengths])


def distribution_valid_lengths(segment_lengths: torch.Tensor, depth: int) -> torch.Tensor:
    """Per packed segment, how many leading real tokens the distribution loss trains at MTP ``depth``: those whose
    target, ``depth + 1`` tokens on, is a real token of the same segment."""
    return (segment_lengths - (depth + 1)).clamp(min=0)


def distribution_valid_mask(segment_lengths: torch.Tensor, depth: int, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Boolean ``[1, T]`` mask over a packed stream whose segments start at ``cu_seqlens``: the first
    ``distribution_valid_lengths`` tokens of each segment."""
    cu_seqlens = cu_seqlens.to(torch.long)
    keep = distribution_valid_lengths(segment_lengths.to(cu_seqlens.device), depth)
    pos = torch.arange(int(cu_seqlens[-1]), device=cu_seqlens.device)
    seg = torch.searchsorted(cu_seqlens[1:], pos, right=True)
    return ((pos - cu_seqlens[seg]) < keep[seg]).unsqueeze(0)


def mtp_draft_token_counts(
    micro_batches: List[dict], num_depths: int, *, loss_type: MTPLossType, packed: bool
) -> torch.Tensor:
    """Per MTP depth, the valid draft-loss tokens of the whole mini-batch: every microbatch, summed over DP.

    The draft loss is a token mean over the mini-batch, like the policy loss's reduction, so each microbatch divides
    its CE sum by these counts. They are taken up front: under PP a microbatch's backward runs before later
    microbatches' forwards. Every DP rank runs ``forward_backward_mini_batch`` in lockstep, and its TP / CP / PP peers
    hold the same full sequences, so each rank counts its own microbatches once.
    """
    import megatron.core.parallel_state as mpu

    counts = torch.zeros(num_depths, dtype=torch.long)
    for batch in micro_batches:
        attention_mask = batch["attention_mask"].to(torch.bool)
        for depth in range(num_depths):
            if loss_type == MTPLossType.HARD_LABEL:
                cols = attention_mask.shape[1] - 1
                valid = mtp_action_valid(cols, batch["num_actions"], depth + 2, batch["loss_mask"], attention_mask)
            elif not packed:  # MTPLossType.DISTRIBUTION's mask: real positions whose target is a real token
                valid = shift_mask_for_mtp(attention_mask.float(), depth)
            else:  # the same per packed segment (see distribution_valid_mask)
                valid = distribution_valid_lengths(
                    segment_real_lengths(attention_mask, batch.get("sub_seq_lengths")), depth
                )
            counts[depth] += int(valid.sum())
    counts = counts.to(torch.cuda.current_device())
    torch.distributed.all_reduce(counts, group=mpu.get_data_parallel_group(with_context_parallel=False))
    return counts.float()


def mtp_hard_label_draft_loss(
    student_outputs: List[torch.Tensor],
    *,
    fused: bool,
    lm_head_weight: Optional[torch.Tensor],
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    sub_seq_lengths: Optional[List[List[int]]],
    num_actions: int,
    loss_mask: torch.Tensor,
    packed_seq_params,
    packed_targets: Optional[torch.Tensor],
    tp_grp,
    tp_rank: int,
    chunk_size: Optional[int],
    fused_backend: FusedLmHeadBackend,
    token_counts: torch.Tensor,
    softcap: Optional[float] = None,
) -> torch.Tensor:
    """For each MTP depth ``d``, the sum over generated tokens of ``-log q_d(seq[t + 2 + d])`` divided by
    ``token_counts[d]`` (the mini-batch's valid tokens; see ``mtp_draft_token_counts``): this microbatch's
    share of the mini-batch token mean. Averaged over depths. ``student_outputs`` holds each depth's hidden states
    when ``fused`` (scored with ``lm_head_weight`` through the fused LM-head op), else its logits.

    Scored through the policy's own packed / CP / fused-LM-head log-prob path, with the target shifted
    ``d + 2`` positions instead of one. Only generated (loss-mask) targets count: they are samples from
    the sampler that verifies the drafts. The output weight is detached, so only the heads get gradient, as
    with Megatron's ``mtp_detach_heads=True`` (see ``MTPLossType``). ``chunk_size`` and
    ``fused_backend`` are the policy's ``mtp_loss_chunk_size`` and fused LM-head backend. ``softcap`` is the
    model's final logit softcap, which the fused op applies itself (the logits path gets it from the output layer).
    """
    import megatron.core.parallel_state as mpu

    from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
        from_parallel_hidden_to_logprobs,
        from_parallel_hidden_to_logprobs_packed_sequences,
        from_parallel_logits_to_logprobs,
        from_parallel_logits_to_logprobs_packed_sequences,
    )

    extra = {}
    if fused:
        lm_head_weight = lm_head_weight.detach()
        # The block-sparse backend needs the policy's active spans; score every position here.
        extra["fused_backend"] = (
            FusedLmHeadBackend.TRITON if fused_backend == FusedLmHeadBackend.TRITON_BLOCK_SPARSE else fused_backend
        )
        extra["softcap"] = softcap
        packed_fn, unpacked_fn = from_parallel_hidden_to_logprobs_packed_sequences, from_parallel_hidden_to_logprobs
    else:
        packed_fn, unpacked_fn = from_parallel_logits_to_logprobs_packed_sequences, from_parallel_logits_to_logprobs
    per_depth = []
    for depth, head in enumerate(student_outputs):
        shift = depth + 2
        # The fused ops take the hidden states and the output weight in place of the logits.
        inputs = (head, lm_head_weight) if fused else (head,)
        v_local = int(lm_head_weight.shape[0]) if fused else int(head.shape[-1])
        kwargs = dict(
            vocab_start_index=tp_rank * v_local,
            vocab_end_index=(tp_rank + 1) * v_local,
            inference_only=False,
            chunk_size=chunk_size,
            **extra,
        )
        if packed_seq_params is not None:
            # Labels come from the full packed sequence every rank holds, so they're right across CP chunk boundaries.
            token_lp = packed_fn(
                *inputs,
                packed_targets,
                packed_seq_params.cu_seqlens_q_padded,
                sequences.shape[1],
                group=tp_grp,
                cp_group=mpu.get_context_parallel_group(),
                attention_mask=attention_mask,
                sub_seq_lengths=sub_seq_lengths,
                target_shift=shift,
                **kwargs,
            )
        else:
            # The helper rolls by one more, so column s scores seq[(s + shift) % seq_len]; its shift - 1
            # wrapped columns sit at or past seq_len - shift, mtp_action_aligned's (exclusive) end.
            rolled = torch.roll(sequences, shifts=-(shift - 1), dims=1)
            token_lp = unpacked_fn(*inputs, rolled, tp_group=tp_grp, cp_group=None, **kwargs)
        lp, valid = mtp_action_aligned(token_lp, num_actions, shift, loss_mask, attention_mask)
        per_depth.append(-(lp * valid).sum() / token_counts[depth].clamp(min=1.0))
    return torch.stack(per_depth).mean()
