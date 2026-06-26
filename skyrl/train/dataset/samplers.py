"""Stateful samplers for :class:`~skyrl.train.sft_trainer.SFTTrainer`.

These samplers plug into ``torchdata.stateful_dataloader.StatefulDataLoader``
so the sampling position is captured in the dataloader's ``state_dict`` and
restored on resume. Each sampler exposes ``state_dict``/``load_state_dict``;
the ``StatefulDataLoader`` fast-forwards to the saved position when iteration
resumes after a checkpoint load.

Two reference samplers are provided:

- :class:`StatefulSequentialSampler` -- deterministic in-order iteration with
  resume support (e.g. for ordered/curriculum-pre-sorted datasets).
- :class:`CurriculumLearningSampler` -- a progressive, difficulty-staged
  sampler that unlocks harder subsets of the data as training advances.

Custom samplers can live anywhere importable; point
``SFTConfig.sampler_class_path`` at a ``module.path.ClassName`` and the trainer
will instantiate it as ``ClassName(tokenized, **sampler_kwargs)``.
"""

from __future__ import annotations

import importlib
import random
from typing import Iterator, List, Optional, Sequence

import torch

__all__ = [
    "StatefulSequentialSampler",
    "CurriculumLearningSampler",
    "import_sampler_class",
]


def import_sampler_class(class_path: str) -> type:
    """Import a sampler class from a ``module.path.ClassName`` string.

    Args:
        class_path: Fully-qualified import path, e.g.
            ``"skyrl.train.dataset.samplers.CurriculumLearningSampler"``.

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


class CurriculumLearningSampler(torch.utils.data.Sampler[int]):
    """Progressive difficulty-staged sampler.

    The dataset is assumed to be a concatenation of difficulty-ordered subsets
    (easy first, hard last), e.g. via ``ConcatDataset([easy, medium, hard])``.
    ``lengths`` gives the size of each subset. Training is split into one stage
    per subset; at stage ``k`` the sampler draws from the *cumulative* pool of
    subsets ``0..k`` (so easier examples keep being revisited as harder ones are
    unlocked). Within a stage, indices are drawn via repeated shuffled passes
    over the unlocked pool (sampling without replacement within each pass).

    The full index plan of ``num_samples`` entries is materialized up front from
    ``seed``, so iteration is fully deterministic and a single ``position``
    cursor is sufficient for checkpoint/resume.

    Args:
        data_source: The training dataset (used only for its length when
            ``lengths`` is not given).
        lengths: Size of each difficulty subset, in curriculum order. When
            ``None``, the whole dataset is treated as a single stage.
        num_samples: Total number of indices to emit across the run. Set this to
            ``num_steps * batch_size`` to cover the entire training schedule in a
            single pass (so the sampler's curriculum state survives epoch
            boundaries). Defaults to ``len(data_source)``.
        seed: Seed for the deterministic shuffle plan.
    """

    def __init__(
        self,
        data_source: Sequence,
        lengths: Optional[Sequence[int]] = None,
        num_samples: Optional[int] = None,
        seed: int = 0,
    ):
        self.data_source = data_source
        n = len(data_source)
        if lengths is None:
            lengths = [n]
        if sum(lengths) != n:
            raise ValueError(
                f"CurriculumLearningSampler: sum(lengths)={sum(lengths)} must equal " f"len(data_source)={n}."
            )
        if any(length <= 0 for length in lengths):
            raise ValueError(f"CurriculumLearningSampler: all lengths must be > 0, got {list(lengths)}.")
        self.lengths: List[int] = list(lengths)
        self.num_samples = num_samples if num_samples is not None else n
        if self.num_samples <= 0:
            raise ValueError(f"CurriculumLearningSampler: num_samples must be > 0, got {self.num_samples}.")
        self.seed = seed
        self.position = 0
        self._plan: List[int] = self._build_plan()

    def _build_plan(self) -> List[int]:
        """Materialize the full deterministic index sequence for the run."""
        rng = random.Random(self.seed)
        num_stages = len(self.lengths)

        # Cumulative offsets: subset ``k`` spans indices [offsets[k], offsets[k+1]).
        offsets = [0]
        for length in self.lengths:
            offsets.append(offsets[-1] + length)

        # Split the total budget roughly evenly across stages; the last stage
        # absorbs any remainder.
        base = self.num_samples // num_stages
        plan: List[int] = []
        for stage in range(num_stages):
            # Stage ``stage`` unlocks subsets 0..stage.
            pool = list(range(offsets[stage + 1]))
            count = base if stage < num_stages - 1 else self.num_samples - base * (num_stages - 1)
            emitted = 0
            while emitted < count:
                shuffled = pool[:]
                rng.shuffle(shuffled)
                take = min(len(shuffled), count - emitted)
                plan.extend(shuffled[:take])
                emitted += take
        return plan

    def __iter__(self) -> Iterator[int]:
        while self.position < len(self._plan):
            idx = self._plan[self.position]
            self.position += 1
            yield idx
        self.position = 0

    def __len__(self) -> int:
        return len(self._plan)

    def state_dict(self) -> dict:
        return {"position": self.position}

    def load_state_dict(self, state: dict) -> None:
        self.position = state["position"]
