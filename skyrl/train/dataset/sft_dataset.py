"""Dataset abstractions for SFT training.

``SFTTrainer.load_dataset`` returns a :class:`SFTDataset` regardless of the
ingestion path: :class:`TextDataset` for tokenize-on-load sources,
:class:`~skyrl.train.dataset.pretokenized.PretokenizedDataset` for
pretokenized stores, and :class:`ConcatSFTDataset` when multiple sources are
configured. All are map-style (samplers, ``StatefulDataLoader`` prefetching
and resume, and the collators are agnostic to which one they receive) and
expose ``sequence_lengths`` so dataset statistics never require materializing
rows.
"""

import abc
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


class TextDataset(SFTDataset):
    """In-memory dataset of tokenized examples (the tokenize-on-load path).

    Wraps the ``list[dict]`` produced by ``SFTTrainer._load_and_tokenize``.
    Rows are fully materialized; making this path lazy is a possible
    follow-up, independent of the interface.
    """

    def __init__(self, examples: list):
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx):
        return self._examples[idx]

    def __getitems__(self, indices: list) -> list:
        return [self._examples[i] for i in indices]

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

    @property
    def sequence_lengths(self) -> list[int]:
        lengths: list[int] = []
        for dataset in self.datasets:
            lengths.extend(int(v) for v in dataset.sequence_lengths)
        return lengths
