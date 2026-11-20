"""Pure helpers for on-policy distillation, kept free of trainer state so they can be tested."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
import torch

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


def apply_opd_to_advantages(
    advantages: torch.Tensor,
    action_log_probs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    opd: OPDConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """``advantages + kl_coef * a`` on trainable tokens, with ``a = log pi_teacher - log pi_student``.

    The per-token term is the k1 (sampled-token) estimate of the reverse KL to the teacher; as a
    detached per-token coefficient it is the exact policy-gradient signal for minimizing that KL
    (only k1 has this property, so ``kl_estimator_type`` is deliberately not consulted). Applied
    *after* the advantage estimator so it composes with any of them: with zero task rewards the
    estimator emits zeros and this is pure on-policy distillation; with task rewards it is the
    additive recipe.

    ``opd``'s stabilization settings act on ``a`` alone, before ``kl_coef``, so their bounds are in nats and a
    task-reward advantage passes through untouched: the divergence clip, then either the advantage bounds or log
    compression (alternatives, see ``validate_opd_cfg``). With none set this is
    ``advantages - kl_coef * (log pi_student - log pi_teacher)`` exactly, with ``log pi_student`` floored at
    ``STUDENT_LOGPROB_FLOOR``.

    Returns the new advantages and metrics under ``opd/``.
    """
    if not (advantages.shape == action_log_probs.shape == teacher_logprobs.shape == loss_mask.shape):
        raise ValueError(
            "shape mismatch: "
            f"advantages={tuple(advantages.shape)} action_log_probs={tuple(action_log_probs.shape)} "
            f"teacher_logprobs={tuple(teacher_logprobs.shape)} loss_mask={tuple(loss_mask.shape)}"
        )
    mask = loss_mask.to(dtype=torch.bool)
    action_log_probs = action_log_probs.clamp(min=STUDENT_LOGPROB_FLOOR)
    reverse_kl = torch.where(mask, action_log_probs - teacher_logprobs, torch.zeros_like(action_log_probs))
    signal = -reverse_kl
    divergence_clipped = torch.zeros_like(mask)
    if opd.divergence_clip is not None:
        divergence_clipped = mask & (action_log_probs.exp() * reverse_kl > opd.divergence_clip)
        signal = torch.where(divergence_clipped, torch.zeros_like(signal), signal)
    advantage_clipped = torch.zeros_like(mask)
    if opd.advantage_clip_min is not None or opd.advantage_clip_max is not None:
        clamped = signal.clamp(min=opd.advantage_clip_min, max=opd.advantage_clip_max)
        advantage_clipped = clamped != signal
        signal = clamped
    if opd.advantage_log_compress:
        signal = signal.sign() * signal.abs().log1p()
    opd_term = opd.kl_coef * signal
    new_advantages = advantages + opd_term

    mask_f = mask.to(dtype=advantages.dtype)
    metrics = {
        "opd/reverse_kl": masked_mean(reverse_kl, mask_f).item(),
        "opd/reverse_kl_abs_max": reverse_kl.abs().max().item() if reverse_kl.numel() else 0.0,
        "opd/adv_rl_abs_mean": masked_mean(advantages.abs(), mask_f).item(),
        "opd/adv_opd_abs_mean": masked_mean(opd_term.abs(), mask_f).item(),
        "opd/kl_coef": float(opd.kl_coef),
        "opd/divergence_clip_fraction": masked_mean(divergence_clipped.to(mask_f.dtype), mask_f).item(),
        "opd/advantage_clip_fraction": masked_mean(advantage_clipped.to(mask_f.dtype), mask_f).item(),
    }
    log_ratio = -reverse_kl[mask]
    if log_ratio.numel():
        # Percentiles of the raw log-ratio, to pick the advantage bounds from a run. NumPy, because torch.quantile
        # refuses inputs above 2**24 elements and a math step has about 25M tokens.
        low, high = np.quantile(log_ratio.float().cpu().numpy(), [0.01, 0.99])
        metrics["opd/log_ratio_p01"], metrics["opd/log_ratio_p99"] = float(low), float(high)
    return new_advantages, metrics
