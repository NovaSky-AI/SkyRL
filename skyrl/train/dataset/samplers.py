"""Stateful samplers for :class:`~skyrl.train.sft_trainer.SFTTrainer`.

These samplers plug into ``torchdata.stateful_dataloader.StatefulDataLoader``
so the sampling position is captured in the dataloader's ``state_dict`` and
restored on resume. A sampler exposes ``state_dict``/``load_state_dict``; the
``StatefulDataLoader`` fast-forwards to the saved position when iteration
resumes after a checkpoint load.

Core ships :class:`StatefulSequentialSampler` (backing the ``sampler="sequential"``
config option) and :class:`DataMixingSampler` (weighted multi-source mixing, used
by default when multiple ``train_datasets`` are configured with ``sampler="random"``).
The ``sampler="custom"`` path loads a user-supplied sampler from
``SFTConfig.sampler_class_path`` via :func:`import_sampler_class`, instantiating
it as ``ClassName(tokenized, **sampler_kwargs)``. See
``examples/train/sft/curriculum_sampler.py`` for a reference custom sampler.
"""

from __future__ import annotations

import importlib
import math
from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Sequence

import torch
from loguru import logger
from torchdata.stateful_dataloader.sampler import RandomSampler
from torchdata.stateful_dataloader.stateful import Stateful

from skyrl.backends.skyrl_train.distributed.megatron.quantization_utils import (
    is_fp8_enabled,
)
from skyrl.train.dataset.collators import make_sft_sequence_packer

if TYPE_CHECKING:
    from skyrl.train.config.sft_config import SFTConfig
    from skyrl.train.dataset.sft_dataset import SFTDataset

__all__ = [
    "DataMixingSampler",
    "DPAlignedPackingBatchSampler",
    "StatefulSequentialSampler",
    "build_dp_aligned_packing_batch_sampler",
    "import_sampler_class",
]


def import_sampler_class(class_path: str) -> type:
    """Import a sampler class from a ``module.path.ClassName`` string.

    Args:
        class_path: Fully-qualified import path, e.g.
            ``"examples.train.sft.curriculum_sampler.CurriculumLearningSampler"``.
            The import runs inside a Ray task (which does not inherit the
            driver's ``PYTHONPATH``), so the module must be importable from the
            worker's ``sys.path`` -- e.g. a dotted path from the repo root when
            launching from it.

    Returns:
        The resolved class object.

    Raises:
        ValueError: If ``class_path`` is not a dotted ``module.ClassName`` path.
    """
    module_path, _, class_name = class_path.rpartition(".")
    if not module_path:
        raise ValueError(
            f"Invalid sampler_class_path '{class_path}'; expected a dotted path like " f"'my_module.MySampler'."
        )
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class DPAlignedPackingBatchSampler(torch.utils.data.Sampler[List[int]]):
    """Move contiguous batch boundaries to avoid DP-padding packed bins.

    The underlying sampler defines the sample order. Each emitted batch is a
    prefix of that order, and read-ahead indices remain in a small carry buffer,
    so no sample is dropped or reordered. Cumulative cardinality error is kept
    within the configured variation and the epoch still emits
    ``ceil(len(sampler) / batch_size)`` optimizer steps. The configured reset
    batch pays off any remaining sample debt at the fixed-step run boundary.
    """

    def __init__(
        self,
        sampler: torch.utils.data.Sampler[int],
        sequence_lengths: Sequence[int],
        batch_size: int,
        dp_size: int,
        allowed_variation: float,
        bin_capacity: int,
        tp_size: int,
        cp_size: int,
        fp8_enabled: bool = False,
        fp8_recipe: Optional[str] = None,
        cardinality_reset_batch: Optional[int] = None,
    ):
        if len(sampler) != len(sequence_lengths):
            raise ValueError(
                f"Sampler emits {len(sampler)} samples but sequence_lengths has {len(sequence_lengths)} entries."
            )
        if batch_size <= 0 or dp_size <= 0:
            raise ValueError(f"batch_size and dp_size must be positive, got {batch_size=} and {dp_size=}.")
        if not 0 < allowed_variation < 1:
            raise ValueError(f"allowed_variation must be in (0, 1), got {allowed_variation}.")

        self.sampler = sampler
        self.batch_size = batch_size
        self.dp_size = dp_size
        self.max_variation = math.floor(batch_size * allowed_variation)
        self.cardinality_reset_batch = cardinality_reset_batch
        if cardinality_reset_batch is not None and not 1 <= cardinality_reset_batch <= len(self):
            raise ValueError(f"cardinality_reset_batch must be in [1, {len(self)}], got {cardinality_reset_batch}.")

        self.packing_lengths = tuple(int(length) for length in sequence_lengths)
        self.packer = make_sft_sequence_packer(
            bin_capacity,
            tp_size,
            cp_size,
            fp8_enabled=fp8_enabled,
            fp8_recipe=fp8_recipe,
        )

    def __iter__(self) -> Iterator[List[int]]:
        return _DPAlignedPackingBatchSamplerIterator(self)

    def __len__(self) -> int:
        return math.ceil(len(self.sampler) / self.batch_size)

    def count_unpadded_bins(self, indices: Sequence[int]) -> int:
        """Return the real MFFD bin count before the collator pads for DP."""
        return len(self.packer.pack([self.packing_lengths[int(index)] for index in indices]))


class _DPAlignedPackingBatchSamplerIterator(Iterator[List[int]], Stateful):
    """Stateful rolling-prefix optimizer for one sampler epoch."""

    _BATCH_INDEX = "batch_index"
    _BUFFER = "buffer"
    _SAMPLES_EMITTED = "samples_emitted"
    _SAMPLER_STATE = "sampler_state"
    _SAMPLER_ITER_STATE = "sampler_iter_state"

    def __init__(self, batch_sampler: DPAlignedPackingBatchSampler):
        self.batch_sampler = batch_sampler
        self.sampler_iter = iter(batch_sampler.sampler)
        self.batch_index = 0
        self.samples_emitted = 0
        self.buffer: List[int] = []

    def __iter__(self) -> "_DPAlignedPackingBatchSamplerIterator":
        return self

    def _fill(self, target_size: int) -> None:
        while len(self.buffer) < target_size:
            try:
                self.buffer.append(next(self.sampler_iter))
            except StopIteration:
                return

    def _fill_to_end(self) -> None:
        while True:
            try:
                self.buffer.append(next(self.sampler_iter))
            except StopIteration:
                return

    def _select_batch_size(self, low: int, high: int, target: int) -> int:
        """Choose the closest prefix that avoids DP padding, if one exists.

        Prefixes are visited nearest to the debt-correcting target first, with
        larger batches winning equal-distance ties. If none is exactly aligned,
        prefer the prefix requiring the fewest dummy bins, then the closest.
        """
        candidates = sorted(range(low, high + 1), key=lambda size: (abs(size - target), -size))
        best_size = target
        best_score = (self.batch_sampler.dp_size, 0)
        for size in candidates:
            bin_count = self.batch_sampler.count_unpadded_bins(self.buffer[:size])
            dp_padding = -bin_count % self.batch_sampler.dp_size
            if dp_padding == 0:
                return size
            score = (dp_padding, abs(size - target))
            if score < best_score:
                best_score = score
                best_size = size
        return best_size

    def __next__(self) -> List[int]:
        total_batches = len(self.batch_sampler)
        if self.batch_index >= total_batches:
            raise StopIteration

        if self.batch_index == total_batches - 1:
            # The tail owns every remaining index so an epoch neither drops nor
            # carries examples into a newly shuffled sampler epoch.
            self._fill_to_end()
            if not self.buffer:
                raise StopIteration
            batch_size = len(self.buffer)
        else:
            nominal = self.batch_sampler.batch_size
            variation = self.batch_sampler.max_variation
            # Constrain both this batch and cumulative sample-count debt to
            # +/- max_variation. The target repays all existing debt now.
            cumulative_error = self.samples_emitted - self.batch_index * nominal
            low = max(nominal - variation, nominal - variation - cumulative_error)
            high = min(nominal + variation, nominal + variation - cumulative_error)
            target = min(max(nominal - cumulative_error, low), high)

            remaining_samples = len(self.batch_sampler.sampler) - self.samples_emitted
            # Reserve at least one sample for the non-tail continuation, then
            # trim again if the underlying iterator ends early.
            high = min(high, remaining_samples - 1)
            low = min(low, high)
            target = min(max(target, low), high)
            self._fill(high)
            high = min(high, len(self.buffer))
            low = min(low, high)
            target = min(max(target, low), high)
            if high <= 0:
                raise StopIteration
            if self.batch_index + 1 == self.batch_sampler.cardinality_reset_batch:
                # A fixed-step run must consume the same nominal sample prefix
                # as an unaligned run; repay debt at its final scheduled step.
                batch_size = target
            else:
                batch_size = self._select_batch_size(low, high, target)

        batch = self.buffer[:batch_size]
        del self.buffer[:batch_size]
        self.samples_emitted += batch_size
        self.batch_index += 1
        return batch

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            self._BATCH_INDEX: self.batch_index,
            self._BUFFER: self.buffer.copy(),
            self._SAMPLES_EMITTED: self.samples_emitted,
        }
        if isinstance(self.batch_sampler.sampler, Stateful):
            state[self._SAMPLER_STATE] = self.batch_sampler.sampler.state_dict()
        if isinstance(self.sampler_iter, Stateful):
            state[self._SAMPLER_ITER_STATE] = self.sampler_iter.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.batch_index = state[self._BATCH_INDEX]
        self.buffer = list(state[self._BUFFER])
        self.samples_emitted = state[self._SAMPLES_EMITTED]
        if self._SAMPLER_STATE in state:
            if not isinstance(self.batch_sampler.sampler, Stateful):
                raise TypeError("Checkpoint contains sampler state for a non-stateful sampler.")
            self.batch_sampler.sampler.load_state_dict(state[self._SAMPLER_STATE])
        self.sampler_iter = iter(self.batch_sampler.sampler)
        if self._SAMPLER_ITER_STATE in state:
            if not isinstance(self.sampler_iter, Stateful):
                raise TypeError("Checkpoint contains iterator state for a non-stateful sampler iterator.")
            self.sampler_iter.load_state_dict(state[self._SAMPLER_ITER_STATE])


def build_dp_aligned_packing_batch_sampler(
    tokenized: "SFTDataset",
    sampler: Optional[torch.utils.data.Sampler[int]],
    generator: torch.Generator,
    sft_cfg: "SFTConfig",
    dp_size: int,
) -> DPAlignedPackingBatchSampler:
    """Build the DP-aligned batch policy from an SFT configuration."""
    # A batch sampler always needs an explicit example sampler. Materialize the
    # normal single-dataset random path that DataLoader otherwise owns.
    if sampler is None:
        sampler = RandomSampler(tokenized, generator=generator)

    steps_per_epoch = math.ceil(len(sampler) / sft_cfg.batch_size)
    cardinality_reset_batch = None
    if steps_per_epoch:
        planned_steps = sft_cfg.num_steps if sft_cfg.num_steps is not None else sft_cfg.num_epochs * steps_per_epoch
        if sft_cfg.max_training_steps is not None:
            planned_steps = min(planned_steps, sft_cfg.max_training_steps)

        # Repay batch-cardinality debt on the run's final step within an epoch,
        # so a fixed-step run consumes the fixed-batching sample prefix.
        cardinality_reset_batch = planned_steps % steps_per_epoch or steps_per_epoch

    transformer_config_kwargs = sft_cfg.megatron_config.transformer_config_kwargs or {}
    return DPAlignedPackingBatchSampler(
        sampler=sampler,
        sequence_lengths=tokenized.sequence_lengths,
        batch_size=sft_cfg.batch_size,
        dp_size=dp_size,
        allowed_variation=sft_cfg.packing_batch_size_allowed_variation,
        bin_capacity=sft_cfg.resolved_bin_capacity(),
        tp_size=sft_cfg.megatron_config.tensor_model_parallel_size,
        cp_size=sft_cfg.megatron_config.context_parallel_size,
        fp8_enabled=is_fp8_enabled(transformer_config_kwargs.get("fp8")),
        fp8_recipe=transformer_config_kwargs.get("fp8_recipe"),
        cardinality_reset_batch=cardinality_reset_batch,
    )


class StatefulSequentialSampler(torch.utils.data.Sampler[int]):
    """Yield indices ``0..len-1`` in order, resumable across checkpoints.

    Unlike ``torch.utils.data.SequentialSampler``, this tracks an internal
    ``position`` cursor and exposes ``state_dict``/``load_state_dict`` so that
    a ``StatefulDataLoader`` can resume mid-epoch from the exact next index.
    The cursor resets to ``0`` once an epoch is exhausted, so a fresh iterator
    starts over from the beginning.
    """

    def __init__(self, data_source: Sequence):
        self.data_source = data_source
        self.position = 0

    def __iter__(self) -> Iterator[int]:
        while self.position < len(self.data_source):
            idx = self.position
            self.position += 1
            yield idx
        # Reset so the next epoch (a fresh ``iter()``) starts from the top.
        self.position = 0

    def __len__(self) -> int:
        return len(self.data_source)

    def state_dict(self) -> dict:
        return {"position": self.position}

    def load_state_dict(self, state: dict) -> None:
        self.position = state["position"]


class DataMixingSampler(torch.utils.data.Sampler[int]):
    """Weighted multi-source sampler built on ``WeightedRandomSampler``.

    The dataset is a concatenation of sources with sizes ``lengths`` (in order);
    ``weights`` gives a sampling weight per source. Per-example weights are set to
    ``weight_source / size_source`` so each *source* is sampled in proportion to
    its weight independent of its size.

    Each epoch draws a fresh plan of ``num_samples`` indices from a persistent
    generator seeded with ``seed``: exhausting the plan clears it, and the next
    ``iter()`` (the trainer re-creates the dataloader iterator at epoch
    boundaries) re-draws with the advanced generator state. ``state_dict``
    captures the cursor plus the generator state *as of the current plan's
    draw*, so a mid-epoch checkpoint resume reproduces the in-flight plan and
    all subsequent epochs match the uninterrupted run.

    Args:
        data_source: The (concatenated) training dataset; only its length is used.
        lengths: Size of each source, in the order they appear in the dataset.
        weights: Per-source sampling weight (need not sum to 1; relative scale matters).
        num_samples: Number of indices to emit per epoch. Defaults to
            ``len(data_source)``, so ``steps_per_epoch`` matches a single
            concatenated dataset of the same size.
        seed: Seed for the deterministic weighted draw.
        replacement: Whether to sample with replacement (passed through to
            ``WeightedRandomSampler``; mixing across sources generally wants True).
    """

    def __init__(
        self,
        data_source: Sequence,
        lengths: Sequence[int],
        weights: Sequence[float],
        num_samples: Optional[int] = None,
        seed: int = 0,
        replacement: bool = True,
    ):
        self.data_source = data_source
        n = len(data_source)
        if sum(lengths) != n:
            raise ValueError(f"DataMixingSampler: sum(lengths)={sum(lengths)} must equal len(data_source)={n}.")
        if len(weights) != len(lengths):
            raise ValueError(f"DataMixingSampler: weights ({len(weights)}) and lengths ({len(lengths)}) must align.")
        if any(length <= 0 for length in lengths):
            raise ValueError(f"DataMixingSampler: all lengths must be > 0, got {list(lengths)}.")
        if any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError(
                f"DataMixingSampler: weights must be non-negative with a positive sum, got {list(weights)}."
            )

        self.num_samples = num_samples if num_samples is not None else n
        if self.num_samples <= 0:
            raise ValueError(f"DataMixingSampler: num_samples must be > 0, got {self.num_samples}.")
        self.position = 0
        self.replacement = replacement

        # Spread each source's weight across its examples so the source-level
        # mixing proportion matches ``weights`` regardless of source size.
        per_example_weights: List[float] = []
        for length, weight in zip(lengths, weights):
            per_example_weights.extend([weight / length] * length)
        self._per_example_weights = per_example_weights

        self._generator = torch.Generator()
        self._generator.manual_seed(seed)
        # The current epoch's plan, drawn lazily. ``_plan_gen_state`` snapshots
        # the generator *before* the draw so a resumed sampler re-draws the
        # identical plan.
        self._plan: Optional[List[int]] = None
        self._plan_gen_state: Optional[torch.Tensor] = None

    def _ensure_plan(self) -> None:
        if self._plan is not None:
            return
        self._plan_gen_state = self._generator.get_state()
        weighted = torch.utils.data.WeightedRandomSampler(
            self._per_example_weights,
            num_samples=self.num_samples,
            replacement=self.replacement,
            generator=self._generator,
        )
        self._plan = list(weighted)

    def __iter__(self) -> Iterator[int]:
        self._ensure_plan()
        while self.position < len(self._plan):
            idx = self._plan[self.position]
            self.position += 1
            yield idx
        # Reset for the next epoch: clear the plan so the next ``iter()``
        # draws fresh indices with the advanced generator state.
        self.position = 0
        self._plan = None

    def __len__(self) -> int:
        return self.num_samples

    def state_dict(self) -> dict:
        # Mid-epoch: persist the pre-draw state so resume re-draws the current
        # plan. Between epochs: persist the advanced state so the next epoch's
        # draw matches the uninterrupted run.
        if self._plan is not None:
            generator_state = self._plan_gen_state
        else:
            generator_state = self._generator.get_state()
        return {"position": self.position, "generator_state": generator_state}

    def load_state_dict(self, state: dict) -> None:
        self.position = state["position"]
        if "generator_state" in state:
            self._generator.set_state(state["generator_state"])
        else:
            # Position-only state (no generator state): keep the freshly-seeded
            # generator and resume the cursor best-effort.
            logger.warning(
                "DataMixingSampler: checkpoint has no generator_state; "
                "resuming position only -- the restored plan may differ from the run that saved it."
            )
        self._plan = None
        self._plan_gen_state = None
