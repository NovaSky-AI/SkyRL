"""Reference custom sampler: curriculum learning for ``SFTTrainer``.

This is an *example* of a user-supplied stateful sampler plugged into
``SFTTrainer`` via the ``sampler=custom`` config path -- it is intentionally
NOT part of the core library. Point ``sampler_class_path`` at this class via a
dotted path importable from the repo root (run from the repo root; no
``__init__.py`` needed thanks to namespace packages), passing the sampler
config as overrides on the base SFT example script::

    bash examples/train/sft/run_sft_megatron.sh \
        sampler=custom \
        sampler_class_path=examples.train.sft.curriculum_sampler.CurriculumLearningSampler \
        'sampler_kwargs={lengths: [34, 33, 33], num_samples: 40, seed: 42}'

(``lengths`` must sum to the dataset size -- 100 for the script's
``train[:100]`` split -- and ``num_samples`` should be
``num_steps * batch_size``.)

Note: the import happens inside a Ray task, which does not inherit the driver's
``PYTHONPATH`` -- so the path must resolve from the worker's ``sys.path``
(which includes the repo root when launching from it).

A custom sampler only needs ``__iter__``/``__len__`` plus
``state_dict``/``load_state_dict`` to be checkpointable by the trainer's
``StatefulDataLoader``. ``SFTTrainer`` instantiates it as
``CurriculumLearningSampler(tokenized, **sampler_kwargs)``.
"""

from __future__ import annotations

import random
from typing import Iterator, List, Optional, Sequence

import torch


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
