"""Reference custom sampler: weighted data mixing for ``SFTTrainer``.

Like ``curriculum_sampler.py``, this is an *example* of a user-supplied stateful
sampler plugged into ``SFTTrainer`` via ``sampler=custom`` -- it is intentionally
NOT part of the core library. Point ``sampler_class_path`` at this class (with
the directory on ``PYTHONPATH``) to use it::

    PYTHONPATH=examples/train/sft python -m skyrl.train.main_sft \
        sampler=custom \
        sampler_class_path=data_mixing_sampler.DataMixingSampler \
        'sampler_kwargs={lengths: [8000, 2000], weights: [0.5, 0.5], num_samples: 800, seed: 42}'

It mixes a concatenation of sources (``ConcatDataset([src_a, src_b, ...])``)
according to per-source ``weights``, using torch's native
``WeightedRandomSampler`` for the weighted draw. Because each source's weight is
divided across its examples, the *source-level* mixing proportion matches
``weights`` regardless of how many examples each source has.
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Sequence

import torch


class DataMixingSampler(torch.utils.data.Sampler[int]):
    """Weighted multi-source sampler built on ``WeightedRandomSampler``.

    The dataset is a concatenation of sources with sizes ``lengths`` (in order);
    ``weights`` gives a sampling weight per source. Per-example weights are set to
    ``weight_source / size_source`` so each *source* is sampled in proportion to
    its weight independent of its size. A deterministic plan of ``num_samples``
    indices is materialized up front from ``seed`` via
    ``torch.utils.data.WeightedRandomSampler``, so iteration is reproducible and a
    single ``position`` cursor is sufficient for checkpoint/resume.

    Args:
        data_source: The (concatenated) training dataset; only its length is used.
        lengths: Size of each source, in the order they appear in the dataset.
        weights: Per-source sampling weight (need not sum to 1; relative scale matters).
        num_samples: Total number of indices to emit. Set to
            ``num_steps * batch_size`` to cover the whole run in one pass (keeps
            the sampler state intact across epoch boundaries and makes resume
            bit-exact across the entire run). Defaults to ``len(data_source)``.
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

        # Spread each source's weight across its examples so the source-level
        # mixing proportion matches ``weights`` regardless of source size.
        per_example_weights: List[float] = []
        for length, weight in zip(lengths, weights):
            per_example_weights.extend([weight / length] * length)

        generator = torch.Generator()
        generator.manual_seed(seed)
        weighted = torch.utils.data.WeightedRandomSampler(
            per_example_weights,
            num_samples=self.num_samples,
            replacement=replacement,
            generator=generator,
        )
        self._plan: List[int] = list(weighted)

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
