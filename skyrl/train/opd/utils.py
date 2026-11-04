"""Pure helpers for on-policy distillation, kept free of trainer state so they can be tested."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch

from skyrl.backends.skyrl_train.utils.packed_ragged_tensor import PackedRaggedTensor
from skyrl.backends.skyrl_train.utils.sample_support import (
    CompactSampleSupport,
    SampleSupportColumn,
)
from skyrl.backends.skyrl_train.utils.torch_utils import masked_mean
from skyrl.train.generators.base import GeneratorInput

if TYPE_CHECKING:
    from skyrl.train.opd.config import OPDConfig

TEACHER_LOGPROBS_KEY = "teacher_logprobs"
ROLLOUT_CORRECT_KEY = "rollout_correct"
# Floor on the student's logprob in the OPD term. A -inf would make the term inf and the pointwise KL NaN.
STUDENT_LOGPROB_FLOOR = -99999.0


def split_generator_input(input_batch: GeneratorInput, group_size: int) -> List[GeneratorInput]:
    """Split a batch into consecutive groups of ``group_size`` rows.

    ``prepare_generator_input`` lays a batch out prompt-major, ``n_samples_per_prompt`` rows per
    prompt, so with ``group_size = n_samples_per_prompt`` each group is one prompt's samples. Per-row
    fields are sliced; ``sampling_params`` and ``batch_metadata`` are shared (the generator never
    mutates them). The inverse of ``concatenate_generator_outputs``.
    """
    num_rows = len(input_batch["prompts"])
    if group_size <= 0 or num_rows % group_size != 0:
        raise ValueError(f"cannot split {num_rows} rows into groups of {group_size}")

    def rows(key: str, start: int):
        values = input_batch.get(key)
        return None if values is None else values[start : start + group_size]

    return [
        {
            "prompts": rows("prompts", start),
            "env_classes": rows("env_classes", start),
            "env_extras": rows("env_extras", start),
            "sampling_params": input_batch.get("sampling_params"),
            "trajectory_ids": rows("trajectory_ids", start),
            "batch_metadata": input_batch.get("batch_metadata"),
        }
        for start in range(0, num_rows, group_size)
    ]


def candidate_matrix(response_ids: List[int], support: Optional[CompactSampleSupport] = None) -> np.ndarray:
    """``[tokens, 1 + k]`` int32 ids for the teacher to score: each response token, then its row's candidates.

    The sampled token is always column 0, so with no ``support`` this is ``[tokens, 1]``, sampled-token OPD.
    ``-1`` pads a row with fewer than ``support.top_k`` candidates, including a token with none.
    """
    sampled = np.asarray(response_ids, dtype=np.int32).reshape(-1, 1)
    if support is None:
        return sampled
    if len(support) != len(response_ids):
        raise ValueError(f"{len(support)} candidate rows for {len(response_ids)} response tokens")
    return np.concatenate([sampled, support.candidate_grid()], axis=1)


def pad_teacher_logprobs(
    teacher_logprobs: List[List[float]],
    response_mask: torch.Tensor,
    pad_size: int,
) -> torch.Tensor:
    """Right-align per-row teacher logprobs into a ``(batch, max_response)`` tensor.

    Mirrors how ``convert_prompts_responses_to_batch_tensors`` lays out ``rollout_logprobs``:
    each row's values occupy its last ``len(row)`` positions, matching ``response_mask``. The
    ``pad_size`` rows ``pad_training_input_batch`` appends are copies of row 0 (they carry
    ``loss_mask == 0``, so their values never matter).

    Args:
        teacher_logprobs: one list per unpadded row, one float per response token.
        response_mask: the padded batch's ``response_mask``, ``(batch + pad_size, max_response)``.
        pad_size: number of trailing padding rows in the batch.
    """
    num_rows = len(teacher_logprobs)
    if response_mask.shape[0] != num_rows + pad_size:
        raise ValueError(
            f"response_mask has {response_mask.shape[0]} rows but got {num_rows} teacher rows and pad_size={pad_size}"
        )
    max_response = response_mask.shape[1]
    out = torch.zeros((num_rows, max_response), dtype=torch.float32)
    for i, row in enumerate(teacher_logprobs):
        expected = int(response_mask[i].sum().item())
        if len(row) != expected:
            raise ValueError(f"row {i}: {len(row)} teacher logprobs for {expected} response tokens")
        if row:
            out[i, max_response - len(row) :] = torch.tensor(row, dtype=torch.float32)
    if pad_size:
        if num_rows == 0:
            raise ValueError("cannot pad an empty batch")
        out = torch.cat([out, out[:1].expand(pad_size, -1)], dim=0)
    return out


@dataclass(frozen=True)
class TeacherSignal:
    signal: torch.Tensor
    """The regulated ``log pi_teacher - log pi_student``, before ``kl_coef``."""
    divergence_clipped: torch.Tensor
    """bool: the divergence clip zeroed the token's signal."""
    advantage_clipped: torch.Tensor
    """bool: the advantage bounds changed the token's signal."""


def teacher_signal(student_logprobs: torch.Tensor, teacher_logprobs: torch.Tensor, opd: OPDConfig) -> TeacherSignal:
    """``opd``'s regulation of ``a = log pi_teacher - log pi_student``, elementwise. Callers apply their own mask.

    The student's logprob is floored at ``STUDENT_LOGPROB_FLOOR``. Then the divergence clip, then either the
    advantage bounds or log compression (alternatives, see ``validate_opd_cfg``). A zero signal stays zero, since
    the bounds straddle it, so masking before or after regulating gives the same values.
    """
    student_logprobs = student_logprobs.clamp(min=STUDENT_LOGPROB_FLOOR)
    signal = teacher_logprobs - student_logprobs
    divergence_clipped = torch.zeros_like(signal, dtype=torch.bool)
    if opd.divergence_clip is not None:
        divergence_clipped = student_logprobs.exp() * -signal > opd.divergence_clip
        signal = torch.where(divergence_clipped, torch.zeros_like(signal), signal)
    advantage_clipped = torch.zeros_like(signal, dtype=torch.bool)
    if opd.advantage_clip_min is not None or opd.advantage_clip_max is not None:
        clamped = signal.clamp(min=opd.advantage_clip_min, max=opd.advantage_clip_max)
        advantage_clipped = clamped != signal
        signal = clamped
    if opd.advantage_log_compress:
        signal = signal.sign() * signal.abs().log1p()
    return TeacherSignal(signal, divergence_clipped, advantage_clipped)


def apply_opd_to_advantages(
    advantages: torch.Tensor,
    action_log_probs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    opd: OPDConfig,
    sampled_in_topk: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """``advantages + kl_coef * a`` on trainable tokens, with ``a = log pi_teacher - log pi_student``.

    The per-token term is the k1 (sampled-token) estimate of the reverse KL to the teacher; as a
    detached per-token coefficient it is the exact policy-gradient signal for minimizing that KL
    (only k1 has this property, so ``kl_estimator_type`` is deliberately not consulted). Applied
    *after* the advantage estimator so it composes with any of them: with zero task rewards the
    estimator emits zeros and this is pure on-policy distillation; with task rewards it is the
    additive recipe.

    ``opd``'s stabilization settings act on ``a`` alone, before ``kl_coef`` (``teacher_signal``), so a task-reward
    advantage passes through untouched. With none set this is
    ``advantages - kl_coef * (log pi_student - log pi_teacher)`` exactly, with ``log pi_student`` floored at
    ``STUDENT_LOGPROB_FLOOR``.

    With TT-OPD (``opd.top_k > 0``, see ``topk_weights``), ``sampled_in_topk`` turns the term off for a sampled
    token that is one of its position's candidates, which then counts through its candidate weight. The rest of
    the top-K term reaches the worker as weights, not advantages.

    Returns the new advantages and metrics under ``opd/``. The metrics describe the sampled token alone.
    """
    if not (advantages.shape == action_log_probs.shape == teacher_logprobs.shape == loss_mask.shape):
        raise ValueError(
            "shape mismatch: "
            f"advantages={tuple(advantages.shape)} action_log_probs={tuple(action_log_probs.shape)} "
            f"teacher_logprobs={tuple(teacher_logprobs.shape)} loss_mask={tuple(loss_mask.shape)}"
        )
    mask = loss_mask.to(dtype=torch.bool)
    regulated = teacher_signal(action_log_probs, teacher_logprobs, opd)
    tail = mask if sampled_in_topk is None else mask & ~sampled_in_topk
    opd_term = torch.where(tail, opd.kl_coef * regulated.signal, torch.zeros_like(action_log_probs))
    new_advantages = advantages + opd_term

    mask_f = mask.to(dtype=advantages.dtype)
    reverse_kl = torch.where(
        mask, action_log_probs.clamp(min=STUDENT_LOGPROB_FLOOR) - teacher_logprobs, torch.zeros_like(action_log_probs)
    )
    metrics = {
        "opd/reverse_kl": masked_mean(reverse_kl, mask_f).item(),
        "opd/reverse_kl_abs_max": reverse_kl.abs().max().item() if reverse_kl.numel() else 0.0,
        "opd/adv_rl_abs_mean": masked_mean(advantages.abs(), mask_f).item(),
        "opd/adv_opd_abs_mean": masked_mean(opd_term.abs(), mask_f).item(),
        "opd/kl_coef": float(opd.kl_coef),
        "opd/divergence_clip_fraction": masked_mean(regulated.divergence_clipped.to(mask_f.dtype), mask_f).item(),
        "opd/advantage_clip_fraction": masked_mean(regulated.advantage_clipped.to(mask_f.dtype), mask_f).item(),
    }
    log_ratio = -reverse_kl[mask]
    if log_ratio.numel():
        # Percentiles of the raw log-ratio, to pick the advantage bounds from a run. NumPy, because torch.quantile
        # refuses inputs above 2**24 elements and a math step has about 25M tokens.
        low, high = np.quantile(log_ratio.float().cpu().numpy(), [0.01, 0.99])
        metrics["opd/log_ratio_p01"], metrics["opd/log_ratio_p99"] = float(low), float(high)
    return new_advantages, metrics


def row_cells(support: PackedRaggedTensor, response_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """The (batch row, response column) of every token row. Trajectory ``b``'s ``n_b`` token rows sit in its last
    ``n_b`` response columns, right-aligned like ``rollout_logprobs``."""
    segment_lengths = support.sequence_lengths.long()
    batch = torch.arange(len(segment_lengths), device=support.device).repeat_interleave(segment_lengths)
    columns = torch.arange(support.num_rows, device=support.device) - support.cu_seqlens[:-1].long()[batch]
    return batch, columns + response_len - segment_lengths[batch]


def member_rows(support: PackedRaggedTensor) -> torch.Tensor:
    """The token row of every member."""
    return torch.arange(support.num_rows, device=support.device).repeat_interleave(support.row_lengths.long())


def with_member_values(support: PackedRaggedTensor, values: torch.Tensor) -> PackedRaggedTensor:
    """``support`` with its member-value column replaced by ``values``."""
    columns = list(support.columns)
    columns[SampleSupportColumn.MEMBER_VALUES] = values.to(columns[SampleSupportColumn.MEMBER_VALUES].dtype)
    return PackedRaggedTensor(columns, support.row_offsets, support.cu_seqlens)


@dataclass(frozen=True)
class TopKWeights:
    member_weights: torch.Tensor
    """float32 ``[members]``: each member's weight in the worker's ``sum_v w_v log pi(v)``. Candidates get ``c_v``;
    slot 0, which holds the sampled token so the worker can anchor its logits, gets 0."""
    token_weights: torch.Tensor
    """``[batch, response]``: the position's ``C = sum_v c_v``, for metrics."""
    sampled_in_topk: torch.Tensor
    """bool ``[batch, response]``: the sampled token is one of its position's candidates."""
    student_mass: torch.Tensor
    """``[batch, response]``: the student's probability mass on the candidates."""
    teacher_mass: torch.Tensor
    """``[batch, response]``: the teacher's probability mass on the candidates."""


def topk_weights(support: PackedRaggedTensor, loss_mask: torch.Tensor, opd: OPDConfig) -> TopKWeights:
    """TT-OPD's top-K term, split for the trainer and the worker.

    ``support`` holds scored token rows ``[sampled, candidates...]``: the sampler's logprob in the logprob column
    (unused for slot 0) and the teacher's in the member-value column. Each candidate ``v`` gets
    ``c_v = kl_coef * a(v) * p(v)``, with ``a`` regulated by ``teacher_signal`` and ``p(v)`` the sampler's. The worker
    adds ``-sum_v c_v log p(v)`` at the current policy, with no importance ratio: nothing in the sum was sampled at
    this position, so the top-K term stays exact on every update of a batch.
    """
    ids = support.columns[SampleSupportColumn.TOKEN_IDS].long()
    student = support.columns[SampleSupportColumn.LOGPROBS]
    teacher = support.columns[SampleSupportColumn.MEMBER_VALUES]
    row_of_member = member_rows(support)
    first_member = support.row_offsets[:-1].long()[row_of_member]
    is_candidate = torch.arange(support.num_members, device=support.device) != first_member

    signal = teacher_signal(student, teacher, opd).signal
    coefficients = torch.where(is_candidate, opd.kl_coef * signal * student.exp(), torch.zeros_like(signal))

    def row_sum(values: torch.Tensor) -> torch.Tensor:
        return values.new_zeros(support.num_rows).index_add(0, row_of_member, values)

    row_sums = row_sum(coefficients)

    batch, column = row_cells(support, loss_mask.shape[1])

    def on_tokens(row_values: torch.Tensor) -> torch.Tensor:
        grid = torch.zeros(loss_mask.shape, dtype=row_values.dtype, device=row_values.device)
        grid[batch, column] = row_values
        return grid

    hits = row_of_member[is_candidate & (ids == ids[first_member])]
    sampled_in_topk = torch.zeros(loss_mask.shape, dtype=torch.bool, device=support.device)
    sampled_in_topk[batch[hits], column[hits]] = True
    return TopKWeights(
        member_weights=coefficients,
        token_weights=on_tokens(row_sums),
        sampled_in_topk=sampled_in_topk,
        student_mass=on_tokens(row_sum(torch.where(is_candidate, student.exp(), torch.zeros_like(student)))),
        teacher_mass=on_tokens(row_sum(torch.where(is_candidate, teacher.exp(), torch.zeros_like(teacher)))),
    )


def topk_metrics(topk: TopKWeights, loss_mask: torch.Tensor) -> Dict[str, float]:
    """Coverage of the top-K term over trainable tokens."""
    mask = loss_mask.to(dtype=topk.token_weights.dtype)
    return {
        "opd/topk_mass_mean": masked_mean(topk.student_mass, mask).item(),
        "opd/teacher_topk_mass_mean": masked_mean(topk.teacher_mass, mask).item(),
        "opd/sampled_in_topk_fraction": masked_mean(topk.sampled_in_topk.to(mask.dtype), mask).item(),
        "opd/topk_weight_abs_mean": masked_mean(topk.token_weights.abs(), mask).item(),
    }
