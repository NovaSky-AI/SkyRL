"""Translate SkyRL GRPO batches to Fireworks/Tinker datum layouts.

SkyRL stores full token sequences with left padding while response-level
tensors are right aligned.  Fireworks' built-in ``importance_sampling`` loss
expects next-token-shifted arrays spanning the model input.  Keeping the shape
conversion pure makes it testable without opening a Fireworks session.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch

if TYPE_CHECKING:
    import tinker


@dataclass(frozen=True)
class GRPODatumSpec:
    """Provider-independent contents of one Fireworks GRPO datum.

    Every tuple has ``len(model_input_token_ids)`` entries.  Prompt and masked
    response positions have zero advantage, so they contribute no policy
    gradient to Fireworks' importance-sampling loss.
    """

    model_input_token_ids: tuple[int, ...]
    target_tokens: tuple[int, ...]
    rollout_logprobs: tuple[float, ...]
    advantages: tuple[float, ...]

    def __post_init__(self) -> None:
        expected = len(self.model_input_token_ids)
        lengths = {
            "target_tokens": len(self.target_tokens),
            "rollout_logprobs": len(self.rollout_logprobs),
            "advantages": len(self.advantages),
        }
        mismatched = {
            name: length for name, length in lengths.items() if length != expected
        }
        if mismatched:
            raise ValueError(
                f"GRPO datum fields must all have length {expected}, got {mismatched}"
            )


def _require_matrix(batch: TrainingInputBatch, name: str) -> torch.Tensor:
    value = batch.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Fireworks GRPO requires tensor field {name!r}")
    if value.ndim != 2:
        raise ValueError(
            f"Fireworks GRPO field {name!r} must be rank 2, got shape {tuple(value.shape)}"
        )
    if value.shape[0] != batch.batch_size:
        raise ValueError(
            f"Fireworks GRPO field {name!r} has batch dimension {value.shape[0]}, expected {batch.batch_size}"
        )
    return value.detach().cpu()


def _right_aligned_length(
    mask: torch.Tensor, *, field_name: str, row_index: int
) -> int:
    present = [bool(value) for value in mask.tolist()]
    count = sum(present)
    expected = [False] * (len(present) - count) + [True] * count
    if present != expected:
        raise ValueError(f"{field_name}[{row_index}] must be right aligned")
    return count


def _unpadded_tokens(
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    row_index: int,
) -> list[int]:
    present = [bool(value) for value in attention_mask.tolist()]
    count = sum(present)
    expected = [False] * (len(present) - count) + [True] * count
    if present != expected:
        raise ValueError(
            f"attention_mask[{row_index}] must describe contiguous left padding"
        )
    return [int(token) for token in sequences[attention_mask.bool()].tolist()]


def training_batch_to_grpo_datum_specs(
    batch: TrainingInputBatch,
    *,
    max_seq_len: int | None = None,
) -> list[GRPODatumSpec]:
    """Convert a SkyRL policy mini-batch into shifted GRPO datum specs.

    The caller must run SkyRL's GRPO advantage estimator and loss-reduction
    scaling first.  This function does no reward or advantage normalization; it
    only removes padding, applies ``loss_mask``, and shifts the token layout.

    Args:
        batch: A policy ``TrainingInputBatch`` containing sequences, masks,
            rollout logprobs, and normalized advantages.
        max_seq_len: Optional Fireworks model-input limit.  The checked length
            is ``len(prompt + response) - 1``, matching the submitted
            ``tinker.ModelInput``.

    Returns:
        One :class:`GRPODatumSpec` per input row, preserving order.
    """

    if batch.batch_size == 0:
        return []

    sequences = _require_matrix(batch, "sequences")
    attention_mask = _require_matrix(batch, "attention_mask")
    response_mask = _require_matrix(batch, "response_mask")
    loss_mask = _require_matrix(batch, "loss_mask")
    advantages = _require_matrix(batch, "advantages")
    rollout_logprobs = _require_matrix(batch, "rollout_logprobs")

    response_width = response_mask.shape[1]
    for name, value in (
        ("loss_mask", loss_mask),
        ("advantages", advantages),
        ("rollout_logprobs", rollout_logprobs),
    ):
        if value.shape[1] != response_width:
            raise ValueError(
                f"Fireworks GRPO response field {name!r} has width {value.shape[1]}, "
                f"expected {response_width}"
            )

    specs: list[GRPODatumSpec] = []
    for row_index in range(batch.batch_size):
        tokens = _unpadded_tokens(
            sequences[row_index], attention_mask[row_index], row_index=row_index
        )
        response_len = _right_aligned_length(
            response_mask[row_index], field_name="response_mask", row_index=row_index
        )
        if response_len == 0:
            raise ValueError(f"Fireworks GRPO sample {row_index} has an empty response")

        prompt_len = len(tokens) - response_len
        if prompt_len < 1:
            raise ValueError(
                f"Fireworks GRPO sample {row_index} must contain at least one prompt token; "
                f"got {len(tokens)} total tokens and {response_len} response tokens"
            )

        model_input_token_ids = tokens[:-1]
        if max_seq_len is not None and len(model_input_token_ids) > max_seq_len:
            raise ValueError(
                f"Fireworks GRPO sample {row_index} has model-input length {len(model_input_token_ids)}, "
                f"exceeding max_seq_len={max_seq_len}"
            )

        response_slice = slice(response_width - response_len, response_width)
        response_loss_mask = [
            float(value) for value in loss_mask[row_index, response_slice].tolist()
        ]
        response_advantages = [
            float(value) for value in advantages[row_index, response_slice].tolist()
        ]
        response_logprobs = [
            float(value)
            for value in rollout_logprobs[row_index, response_slice].tolist()
        ]

        masked_advantages: list[float] = []
        masked_logprobs: list[float] = []
        for token_index, (mask_value, advantage, logprob) in enumerate(
            zip(response_loss_mask, response_advantages, response_logprobs, strict=True)
        ):
            if mask_value not in (0.0, 1.0):
                raise ValueError(
                    f"loss_mask[{row_index}] contains {mask_value} at response index {token_index}; "
                    "Fireworks GRPO requires a binary loss mask"
                )
            if mask_value == 0.0:
                masked_advantages.append(0.0)
                masked_logprobs.append(0.0)
                continue
            if not math.isfinite(advantage):
                raise ValueError(
                    f"advantages[{row_index}] contains a non-finite value at trainable response index {token_index}"
                )
            if not math.isfinite(logprob):
                raise ValueError(
                    f"rollout_logprobs[{row_index}] contains a non-finite value at trainable response index "
                    f"{token_index}"
                )
            masked_advantages.append(advantage)
            masked_logprobs.append(logprob)

        prompt_prediction_count = prompt_len - 1
        specs.append(
            GRPODatumSpec(
                model_input_token_ids=tuple(model_input_token_ids),
                target_tokens=tuple(
                    [0] * prompt_prediction_count + tokens[prompt_len:]
                ),
                rollout_logprobs=tuple(
                    [0.0] * prompt_prediction_count + masked_logprobs
                ),
                advantages=tuple([0.0] * prompt_prediction_count + masked_advantages),
            )
        )

    return specs


def _to_tinker_datum(spec: GRPODatumSpec) -> Any:
    try:
        import tinker
    except ImportError as exc:  # pragma: no cover - depends on optional installation
        raise ImportError(
            "Fireworks GRPO datum construction requires the 'tinker' package; "
            "install SkyRL with the Fireworks/Tinker extra"
        ) from exc

    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(list(spec.model_input_token_ids)),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(
                data=list(spec.target_tokens), dtype="int64"
            ),
            "logprobs": tinker.TensorData(
                data=list(spec.rollout_logprobs), dtype="float32"
            ),
            "advantages": tinker.TensorData(
                data=list(spec.advantages), dtype="float32"
            ),
        },
    )


def build_tinker_grpo_datums(
    batch: TrainingInputBatch,
    *,
    max_seq_len: int | None = None,
) -> list["tinker.Datum"]:
    """Build concrete Tinker datums for Fireworks ``importance_sampling``."""

    return [
        _to_tinker_datum(spec)
        for spec in training_batch_to_grpo_datum_specs(batch, max_seq_len=max_seq_len)
    ]
