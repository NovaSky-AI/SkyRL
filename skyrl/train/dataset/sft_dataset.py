"""Dataset abstractions for SFT training.

``SFTTrainer.load_dataset`` returns a :class:`SFTDataset` regardless of the
ingestion path: :class:`TextDataset` for tokenize-on-load sources,
:class:`LazyVLMDataset` for VLM sources tokenized per fetch
(``lazy_vlm_dataset``), :class:`~skyrl.train.dataset.pretokenized.PretokenizedDataset`
for pretokenized stores, and :class:`ConcatSFTDataset` when multiple sources
are configured. All are map-style (samplers, ``StatefulDataLoader`` prefetching
and resume, and the collators are agnostic to which one they receive) and
expose ``sequence_lengths`` so dataset statistics never require materializing
rows.
"""

import abc
import bisect
from typing import Any, Callable, Iterable, Optional, Sequence

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

    Wraps the ``list[dict]`` produced by ``SFTTrainer._load_and_tokenize``.
    Rows are fully materialized; VLM sources can opt into
    :class:`LazyVLMDataset` instead (``lazy_vlm_dataset``), since their
    ``pixel_values`` dominate memory.
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


class LazyVLMDataset(SFTDataset):
    """Map-style dataset for VLM chat examples that tokenizes (and runs the
    image processor) per fetch instead of on load.

    The tokenize-on-load path keeps every example's ``pixel_values`` in memory
    (:class:`TextDataset`), which for image datasets outgrows host memory long
    before the dataset is large: a few thousand tokens of Qwen-style image
    patches weigh megabytes per row. This dataset keeps only the arrow-backed
    (memory-mapped) source dataset, the source-row indices of the examples
    that survived the up-front tokenization pass (``None`` results and
    over-length VLM rows are dropped exactly as on the eager path) and their
    token lengths. Fetching index ``i`` re-runs ``tokenize_fn`` on source row
    ``indices[i]``, so the rows are identical to the eager path's.

    ``tokenize_fn`` must be deterministic (a row that tokenized on load must
    tokenize on fetch) and picklable, e.g. a :func:`functools.partial` over
    :func:`~skyrl.train.sft_trainer.tokenize_chat_example` with the tokenizer
    and processor bound, so the dataset can be shipped to spawned dataloader
    workers (``dataloader_num_workers > 0``).
    """

    def __init__(
        self,
        dataset: Any,
        indices: Sequence[int],
        lengths: Sequence[int],
        tokenize_fn: Callable[[dict], Optional[dict]],
    ):
        indices = [int(i) for i in indices]
        lengths = [int(n) for n in lengths]
        if len(indices) != len(lengths):
            raise ValueError(f"indices ({len(indices)}) and lengths ({len(lengths)}) differ in length")
        num_rows = len(dataset)
        if indices and not (0 <= min(indices) and max(indices) < num_rows):
            raise ValueError(f"indices must lie in [0, {num_rows}); got min={min(indices)}, max={max(indices)}")
        self._dataset = dataset
        self._indices = indices
        self._lengths = lengths
        self._tokenize_fn = tokenize_fn

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx):
        idx = int(idx)
        if idx < 0:
            idx += len(self._indices)
        if not 0 <= idx < len(self._indices):
            raise IndexError(f"index {idx} out of range for LazyVLMDataset of length {len(self._indices)}")
        return self._fetch(self._indices[idx])

    def __getitems__(self, indices: list) -> list:
        """Batched fetch: one gather on the arrow-backed source for the whole
        batch (``dataset[[rows]]`` returns columns), then one ``tokenize_fn``
        call per row. Sources without batched indexing (plain sequences) fall
        back to row-wise fetches."""
        rows = []
        for idx in indices:
            idx = int(idx)
            if idx < 0:
                idx += len(self._indices)
            if not 0 <= idx < len(self._indices):
                raise IndexError(f"index {idx} out of range for LazyVLMDataset of length {len(self._indices)}")
            rows.append(self._indices[idx])
        if not hasattr(self._dataset, "column_names"):
            return [self._fetch(row) for row in rows]
        columns = self._dataset[rows]
        return [self._fetch(row, {k: v[j] for k, v in columns.items()}) for j, row in enumerate(rows)]

    def _fetch(self, row: int, source_row: Optional[dict] = None) -> dict:
        example = self._tokenize_fn(self._dataset[row] if source_row is None else source_row)
        if example is None:
            raise RuntimeError(
                f"Source row {row} tokenized to a row on load but to None on fetch; tokenize_fn must be deterministic"
            )
        return example

    @property
    def source_indices(self) -> list[int]:
        """Source-dataset row index of every example, in dataset order."""
        return list(self._indices)

    @property
    def sequence_lengths(self) -> list[int]:
        return list(self._lengths)


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
