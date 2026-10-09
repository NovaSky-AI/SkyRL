"""Dataset abstractions for SFT training.

``SFTTrainer.load_dataset`` returns a :class:`SFTDataset` regardless of the
ingestion path: :class:`TextDataset` for tokenize-on-load sources,
:class:`~skyrl.train.dataset.pretokenized.MemoryMappedDataset` for
pretokenized stores, and :class:`ConcatSFTDataset` when multiple sources are
configured. All are map-style (samplers, ``StatefulDataLoader`` prefetching
and resume, and the collators are agnostic to which one they receive) and
expose ``sequence_lengths`` so dataset statistics never require materializing
rows.
"""

import abc
import bisect
from typing import Iterable, Sequence

import torch.utils.data


class SFTDataset(torch.utils.data.Dataset, abc.ABC):
    """Base map-style dataset for SFT training.

    Rows are the trainer's normalized example dicts (``input_ids`` /
    ``attention_mask`` / ``num_actions`` / window ``loss_mask`` plus
    pass-through columns).
    """

    @property
    @abc.abstractmethod
    def sequence_lengths(self) -> Sequence[int]:
        """Tokenized length of every example (after truncation/dropping)."""
        raise NotImplementedError

    def __getitems__(self, indices: list) -> list:
        """Batched fetch (the entry point torch's fetcher prefers). Row-wise
        by default; subclasses override when a batch-at-once call amortizes
        real work (one arrow gather + one transform invocation for the mmap
        dataset; and e.g. one coalesced ranged read for network-backed ones)."""
        return [self[i] for i in indices]


class TextDataset(SFTDataset):
    """In-memory dataset of tokenized examples (the tokenize-on-load path).

    Wraps the ``list[dict]`` produced by ``SFTTrainer._load_and_tokenize``
    when caching is disabled. With caching enabled (the default), the trainer
    instead serves the tokenized arrow cache memory-mapped through
    ``MemoryMappedDataset``, so this fully-materialized form is only resident
    for ``disable_cache=True`` runs.
    """

    def __init__(self, examples: list):
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx):
        return self._examples[idx]

    @property
    def sequence_lengths(self) -> list[int]:
        return [len(ex["input_ids"]) for ex in self._examples]


class ConcatSFTDataset(SFTDataset, torch.utils.data.ConcatDataset):
    """Concatenation of :class:`SFTDataset` sources, in config order.

    A map-style view (no row materialization); global indices span the
    sources back to back, which is what ``DataMixingSampler`` mixes over.
    """

    def __init__(self, datasets: Iterable[SFTDataset]):
        torch.utils.data.ConcatDataset.__init__(self, datasets)

    @property
    def dataset_lengths(self) -> list[int]:
        """Size of each source, in order (configures weighted mixing)."""
        return [len(dataset) for dataset in self.datasets]

    def __getitems__(self, indices: list) -> list:
        """Batched fetch across sources, preserving the requested order.

        Torch's fetcher only checks the *top-level* dataset for
        ``__getitems__``, so without this a concat degrades every source to
        row-wise ``__getitem__`` -- a minor overhead for in-memory/mmap
        sources, but defeating for sources whose batched path amortizes real
        work (e.g. fetch-over-network stores). Indices are grouped per source
        and served through each source's batched entry point.
        """
        by_source: dict[int, list[tuple[int, int]]] = {}
        for pos, index in enumerate(indices):
            index = int(index)
            if index < 0:
                index += len(self)
            source = bisect.bisect_right(self.cumulative_sizes, index)
            local = index - (self.cumulative_sizes[source - 1] if source > 0 else 0)
            by_source.setdefault(source, []).append((pos, local))
        rows: list = [None] * len(indices)
        for source, items in by_source.items():
            fetched = self.datasets[source].__getitems__([local for _, local in items])
            for (pos, _), row in zip(items, fetched):
                rows[pos] = row
        return rows

    @property
    def sequence_lengths(self) -> list[int]:
        lengths: list[int] = []
        for dataset in self.datasets:
            lengths.extend(int(v) for v in dataset.sequence_lengths)
        return lengths
