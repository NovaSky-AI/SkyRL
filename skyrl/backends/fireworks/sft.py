"""Translate SkyRL SFT batches to Fireworks cross-entropy requests."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
from loguru import logger

from skyrl.backends.fireworks.training_backend import FireworksPolicyDispatch
from skyrl.backends.skyrl_train.distributed.dispatch import WorkerOutput
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch

if TYPE_CHECKING:
    import tinker


@dataclass(frozen=True)
class SFTDatumSpec:
    """Provider-independent contents of one Fireworks SFT datum."""

    model_input_token_ids: tuple[int, ...]
    target_tokens: tuple[int, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        expected = len(self.model_input_token_ids)
        lengths = {"target_tokens": len(self.target_tokens), "weights": len(self.weights)}
        mismatched = {name: length for name, length in lengths.items() if length != expected}
        if mismatched:
            raise ValueError(f"SFT datum fields must all have length {expected}, got {mismatched}")


def _matrix(batch: TrainingInputBatch, name: str) -> torch.Tensor:
    value = batch.get(name)
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"Fireworks SFT requires rank-2 tensor field {name!r}")
    if value.shape[0] != batch.batch_size:
        raise ValueError(f"Fireworks SFT field {name!r} has the wrong batch dimension")
    return value.detach().cpu()


def _tokens(sequences: torch.Tensor, attention_mask: torch.Tensor, row_index: int) -> list[int]:
    present = attention_mask.bool()
    if torch.any(present[:-1] & ~present[1:]):
        raise ValueError(f"attention_mask[{row_index}] must describe contiguous left padding")
    return [int(token) for token in sequences[present].tolist()]


def _tensor_data_values(value: Any) -> list[float]:
    """Convert a tensor-data value to a list of floats."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    elif hasattr(value, "to_numpy"):
        value = value.to_numpy()
        if hasattr(value, "tolist"):
            value = value.tolist()
    else:
        if isinstance(value, Mapping):
            value = value.get("data")
        else:
            value = getattr(value, "data", value)
        if hasattr(value, "tolist"):
            value = value.tolist()
    return [float(item) for item in value]


def _weighted_forward_loss(datums: list[Any], loss_fn_outputs: Any) -> float | None:
    """Compute weighted negative log likelihood from forward outputs."""
    if not loss_fn_outputs or len(loss_fn_outputs) != len(datums):
        return None

    total_loss = 0.0
    for datum, output in zip(datums, loss_fn_outputs):
        if not isinstance(output, Mapping) or "logprobs" not in output:
            return None
        loss_fn_inputs = getattr(datum, "loss_fn_inputs", None)
        if not isinstance(loss_fn_inputs, Mapping) or "weights" not in loss_fn_inputs:
            return None
        try:
            weights = _tensor_data_values(loss_fn_inputs["weights"])
            logprobs = _tensor_data_values(output["logprobs"])
        except (TypeError, ValueError):
            return None
        if len(weights) != len(logprobs):
            return None
        total_loss += sum(weight * -logprob for weight, logprob in zip(weights, logprobs))
    return total_loss


def training_batch_to_sft_datum_specs(
    batch: TrainingInputBatch,
    *,
    max_seq_len: int | None = None,
) -> list[SFTDatumSpec]:
    """Convert one collated SFT optimizer batch without changing loss weights."""

    if batch.batch_size == 0:
        return []
    sequences = _matrix(batch, "sequences")
    attention_mask = _matrix(batch, "attention_mask")
    loss_mask = _matrix(batch, "loss_mask").float()
    response_width = loss_mask.shape[1]
    specs: list[SFTDatumSpec] = []

    for row_index in range(batch.batch_size):
        tokens = _tokens(sequences[row_index], attention_mask[row_index], row_index)
        if len(tokens) < 2:
            raise ValueError(f"Fireworks SFT sample {row_index} must contain at least two tokens")
        model_input = tokens[:-1]
        if max_seq_len is not None and len(model_input) > max_seq_len:
            raise ValueError(
                f"Fireworks SFT sample {row_index} has model-input length {len(model_input)}, "
                f"exceeding max_seq_len={max_seq_len}"
            )

        action_weights = [float(value) for value in loss_mask[row_index].tolist()]
        if any(not math.isfinite(value) or value < 0 for value in action_weights):
            raise ValueError(f"loss_mask[{row_index}] must contain finite non-negative weights")
        if response_width > len(model_input):
            overflow = response_width - len(model_input)
            if any(action_weights[:overflow]):
                raise ValueError(f"loss_mask[{row_index}] supervises a token outside the sequence")
            action_weights = action_weights[overflow:]
        weights = [0.0] * (len(model_input) - len(action_weights)) + action_weights
        if not any(weights):
            continue
        specs.append(
            SFTDatumSpec(
                model_input_token_ids=tuple(model_input),
                target_tokens=tuple(tokens[1:]),
                weights=tuple(weights),
            )
        )

    if not specs:
        raise ValueError("Fireworks SFT batch has no supervised tokens")
    return specs


def build_tinker_sft_datums(
    batch: TrainingInputBatch,
    *,
    max_seq_len: int | None = None,
) -> list["tinker.Datum"]:
    """Build Tinker datums using the collator-normalized SFT weights."""

    try:
        import tinker
    except ImportError as exc:
        raise ImportError("Fireworks SFT requires the 'fireworks' extra") from exc

    return [
        tinker.Datum(
            model_input=tinker.ModelInput.from_ints(list(spec.model_input_token_ids)),
            loss_fn_inputs={
                "target_tokens": tinker.TensorData(data=list(spec.target_tokens), dtype="int64"),
                "weights": tinker.TensorData(data=list(spec.weights), dtype="float32"),
            },
        )
        for spec in training_batch_to_sft_datum_specs(batch, max_seq_len=max_seq_len)
    ]


class FireworksSFTDispatch(FireworksPolicyDispatch):
    """SFT model calls backed by Fireworks cross-entropy."""

    def _run_cross_entropy(self, batch: TrainingInputBatch, *, backward: bool) -> WorkerOutput:
        datums = build_tinker_sft_datums(batch, max_seq_len=self.fireworks_config.max_seq_len)
        operation = self.runtime.training_client.forward_backward if backward else self.runtime.training_client.forward
        result = operation(datums, "cross_entropy").result(timeout=self.fireworks_config.request_timeout_s)
        metrics = {key: float(value) for key, value in (getattr(result, "metrics", None) or {}).items()}
        if "loss:sum" in metrics:
            metrics.setdefault("final_loss" if backward else "loss", metrics["loss:sum"])
        elif not backward:
            loss = _weighted_forward_loss(datums, getattr(result, "loss_fn_outputs", None))
            if loss is None:
                logger.warning("Unable to compute Fireworks forward loss from per-token logprobs; omitting loss metric")
            else:
                metrics.setdefault("loss", loss)
        return WorkerOutput(
            loss_fn_output_type=str(getattr(result, "loss_fn_output_type", "scalar")),
            loss_fn_outputs=[],
            metrics=metrics,
        )

    def forward_backward(
        self,
        model: str,
        batch: TrainingInputBatch,
        loss_fn: Optional[str] = None,
        loss_fn_config: Optional[dict[str, Any]] = None,
        model_id: Optional[str] = None,
    ) -> WorkerOutput:
        self._require_policy(model)
        if loss_fn not in (None, "cross_entropy") or loss_fn_config is not None or model_id is not None:
            raise NotImplementedError("Fireworks SFT only supports cross_entropy without per-call overrides")
        return self._run_cross_entropy(batch, backward=True)

    def forward(
        self,
        model: str,
        batch: TrainingInputBatch,
        loss_fn: Optional[str] = None,
        loss_fn_config: Optional[dict[str, Any]] = None,
        model_id: Optional[str] = None,
    ) -> WorkerOutput:
        self._require_policy(model)
        if loss_fn not in (None, "cross_entropy") or loss_fn_config is not None or model_id is not None:
            raise NotImplementedError("Fireworks SFT only supports cross_entropy without per-call overrides")
        return self._run_cross_entropy(batch, backward=False)
