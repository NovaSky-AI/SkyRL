"""Loading of pretokenized SFT datasets.

Some data pipelines tokenize offline (e.g. on a Spark/Ray data cluster). This
module lets the SFT trainer ingest such a dataset directly from a local file
or directory -- skipping online tokenization (``tokenize_chat_example`` and
``tokenize_sft_example`` in ``skyrl.train.sft_trainer``) entirely.

Supported on-disk formats (auto-detected):

- a HuggingFace ``Dataset.save_to_disk`` directory,
- Parquet file(s) (``.parquet``),
- JSON-lines file(s) (``.jsonl`` / ``.json``),
- raw Arrow IPC file(s) (``.arrow``).

Row schema (all rows must be stored unpadded; SkyRL pads at collation time):

- ``input_ids`` (list[int]): token ids for the full sequence.
- ``loss_mask`` (list[int], same length as ``input_ids``): 1 for tokens to
  compute loss on, 0 otherwise. Works for instruction-following data (1s on
  the response) and multi-turn conversational data (1s on every assistant
  turn, 0s in between).
- VLM data additionally carries ``pixel_values`` and ``image_grid_thw``
  (Qwen-style image tensors, stored as nested lists).

``num_actions`` (the trailing action-window length SkyRL's workers consume) is
inferred from the first nonzero ``loss_mask`` entry -- do not store it. Rows
are normalized to the trainer's internal representation (``input_ids`` /
``attention_mask`` / ``num_actions`` / window ``loss_mask``) and rows whose
loss window is empty (e.g. after ``max_length`` truncation) are dropped.

The loaded dataset is **memory-mapped, not materialized**: schema validation
and row dropping happen eagerly at load time via vectorized scans over the
arrow columns (so malformed stores still fail fast), while per-row
normalization runs lazily at access time. Memory is bounded by the OS page
cache rather than dataset size; the backing files must remain on disk for the
lifetime of the run.
"""

import os
from typing import Optional

import numpy as np
import pyarrow.compute as pc
from datasets import Dataset, concatenate_datasets, load_dataset
from loguru import logger

from skyrl.train.dataset.sft_dataset import SFTDataset

# Keys consumed (and re-emitted in normalized form) by the lazy row transform.
# ``num_actions`` and ``labels`` are consumed-but-dropped: ``num_actions`` is
# always re-inferred from ``loss_mask``, and HF-style ``labels`` are not a
# supported loss target (convert to a 0/1 ``loss_mask`` offline). All other
# columns pass through untouched (e.g. ``pixel_values`` / ``image_grid_thw``).
_CONSUMED_KEYS = frozenset({"input_ids", "attention_mask", "loss_mask", "num_actions", "labels"})

_VLM_KEYS = ("pixel_values", "image_grid_thw")

_PARQUET_EXTS = (".parquet",)
_JSON_EXTS = (".jsonl", ".json")
_ARROW_EXTS = (".arrow",)
_DATA_EXTS = _PARQUET_EXTS + _JSON_EXTS + _ARROW_EXTS

# Warn once per SFT run (process) when stores carry an attention_mask column
_warned_attention_mask_dropped = False

# Rows per vectorized validation chunk: bounds the numpy working set while
# scanning arbitrarily large stores.
_VALIDATION_CHUNK_ROWS = 200_000


# ---------------------------------------------------------------------------
# Format detection / loading
# ---------------------------------------------------------------------------


def _collect_data_files(root: str) -> dict[str, list[str]]:
    """Recursively collect data files under ``root``, grouped by format.

    Hidden files and directories (dotfiles) are skipped: `.ipynb_checkpoints/`
    holds stale copies that would silently duplicate rows, and macOS `._*`
    AppleDouble sidecars are not valid data files.
    """
    groups: dict[str, list[str]] = {"parquet": [], "json": [], "arrow": []}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in sorted(filenames):
            if name.startswith("."):
                continue
            full = os.path.join(dirpath, name)
            lower = name.lower()
            if lower.endswith(_PARQUET_EXTS):
                groups["parquet"].append(full)
            elif lower.endswith(_JSON_EXTS):
                groups["json"].append(full)
            elif lower.endswith(_ARROW_EXTS):
                groups["arrow"].append(full)
    return groups


def _load_arrow_files(files: list[str]) -> Dataset:
    parts = [Dataset.from_file(f) for f in files]
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


def _load_as_hf_dataset(local_path: str) -> Dataset:
    """Load a :class:`datasets.Dataset` from a local file or directory."""
    if os.path.isdir(local_path):
        # A ``Dataset.save_to_disk`` directory is identified by its state.json.
        if os.path.isfile(os.path.join(local_path, "state.json")):
            return Dataset.load_from_disk(local_path)
        groups = _collect_data_files(local_path)
        non_empty = {fmt: files for fmt, files in groups.items() if files}
        if not non_empty:
            raise ValueError(
                f"No supported data files found under '{local_path}'. "
                f"Expected a 'Dataset.save_to_disk' directory or files with one of: {_DATA_EXTS}"
            )
        if len(non_empty) > 1:
            raise ValueError(
                f"Found a mix of data formats under '{local_path}': "
                f"{ {fmt: len(files) for fmt, files in non_empty.items()} }. "
                "Store must contain a single format."
            )
        fmt, files = next(iter(non_empty.items()))
        if fmt == "arrow":
            return _load_arrow_files(files)
        return load_dataset(fmt, data_files=files, split="train")

    lower = local_path.lower()
    if lower.endswith(_PARQUET_EXTS):
        return load_dataset("parquet", data_files=local_path, split="train")
    if lower.endswith(_JSON_EXTS):
        return load_dataset("json", data_files=local_path, split="train")
    if lower.endswith(_ARROW_EXTS):
        return Dataset.from_file(local_path)
    raise ValueError(
        f"Unsupported pretokenized dataset file '{local_path}'. Expected one of: {_DATA_EXTS} "
        "or a 'Dataset.save_to_disk' directory."
    )


# ---------------------------------------------------------------------------
# Eager validation / row filtering (vectorized over arrow columns)
# ---------------------------------------------------------------------------


def _list_lengths(tbl, column: str) -> np.ndarray:
    """Per-row list lengths of a list column (nulls counted as length 0)."""
    return pc.fill_null(pc.list_value_length(tbl[column]), 0).to_numpy(zero_copy_only=False).astype(np.int64)


def _flat_values(tbl, column: str) -> np.ndarray:
    """Flattened values of a list column (null lists contribute no values,
    matching the length-0 convention of :func:`_list_lengths`)."""
    return pc.list_flatten(tbl[column]).to_numpy(zero_copy_only=False)


def _locate_row(lengths: np.ndarray, flat_idx: int) -> int:
    """Map an index into the flattened values back to its row within a chunk."""
    return int(np.searchsorted(np.cumsum(lengths), flat_idx, side="right"))


def sequence_lengths_from_arrow(dataset: Dataset) -> np.ndarray:
    """Per-row ``input_ids`` lengths from arrow offsets, in bounded chunks.

    For datasets already stored in the trainer's internal row form (e.g. the
    tokenized-dataset cache), lengths are the only load-time scan needed --
    validation and truncation already happened when the rows were produced.
    """
    arrow_ds = dataset.with_format("arrow")
    chunks = [
        _list_lengths(arrow_ds[start : start + _VALIDATION_CHUNK_ROWS], "input_ids")
        for start in range(0, len(dataset), _VALIDATION_CHUNK_ROWS)
    ]
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int64)


def _validate_and_plan(dataset: Dataset, max_length: Optional[int]) -> tuple[np.ndarray, np.ndarray, int]:
    """Eagerly validate the store and compute the keep/drop plan.

    Scans the arrow columns in bounded chunks (never materializing rows) so
    malformed stores fail fast at load time, exactly as the materializing
    loader did. Returns ``(keep_mask, sequence_lengths, num_vlm_dropped)``
    where ``keep_mask`` marks rows with a non-empty loss window (within
    ``max_length`` for text rows; over-length VLM rows are dropped, never
    truncated, mirroring the online VLM path).
    """
    columns = dataset.column_names
    has_vlm_columns = all(k in columns for k in _VLM_KEYS)
    lone_vlm_columns = [k for k in _VLM_KEYS if k in columns] if not has_vlm_columns else []

    arrow_ds = dataset.with_format("arrow")
    keep_chunks: list[np.ndarray] = []
    length_chunks: list[np.ndarray] = []
    num_vlm_dropped = 0

    for start in range(0, len(dataset), _VALIDATION_CHUNK_ROWS):
        tbl = arrow_ds[start : start + _VALIDATION_CHUNK_ROWS]
        ids_len = _list_lengths(tbl, "input_ids")
        mask_len = _list_lengths(tbl, "loss_mask")

        mismatch = ids_len != mask_len
        if mismatch.any():
            row = start + int(np.argmax(mismatch))
            bad = int(np.argmax(mismatch))
            raise ValueError(
                f"Row {row}: loss_mask length ({int(mask_len[bad])}) must equal len(input_ids) "
                f"({int(ids_len[bad])}). Window-form masks are not supported; store the full-sequence mask."
            )

        flat_mask = _flat_values(tbl, "loss_mask")
        if flat_mask.size:
            invalid = (flat_mask < 0) | (flat_mask > 1)
            if invalid.any():
                row = start + _locate_row(mask_len, int(np.argmax(invalid)))
                raise ValueError(f"Row {row}: loss_mask must contain only 0s and 1s.")

        if "attention_mask" in columns:
            att_flat = _flat_values(tbl, "attention_mask")
            if att_flat.size and (att_flat != 1).any():
                att_len = _list_lengths(tbl, "attention_mask")
                row = start + _locate_row(att_len, int(np.argmax(att_flat != 1)))
                raise ValueError(
                    f"Row {row}: attention_mask contains 0s. Pretokenized rows must be stored "
                    "unpadded (padding is applied at collation time)."
                )

        # A row is trainable iff a 1 exists in the loss mask before the
        # (optional) max_length truncation boundary.
        n = len(ids_len)
        starts = np.zeros(n, dtype=np.int64)
        np.cumsum(mask_len[:-1], out=starts[1:])
        has_loss = np.zeros(n, dtype=bool)
        if flat_mask.size:
            if max_length is not None:
                positions = np.arange(flat_mask.size, dtype=np.int64) - np.repeat(starts, mask_len)
                window = np.where(positions < max_length, flat_mask, 0)
            else:
                window = flat_mask
            nonempty = mask_len > 0
            if nonempty.any():
                # reduceat segments run between consecutive non-empty starts;
                # empty rows contribute no values, so segments stay aligned.
                has_loss[nonempty] = np.maximum.reduceat(window, starts[nonempty]) > 0

        if has_vlm_columns:
            pv_null = pc.is_null(tbl[_VLM_KEYS[0]]).to_numpy(zero_copy_only=False)
            grid_null = pc.is_null(tbl[_VLM_KEYS[1]]).to_numpy(zero_copy_only=False)
            pair_mismatch = pv_null != grid_null
            if pair_mismatch.any():
                row = start + int(np.argmax(pair_mismatch))
                raise ValueError(f"Row {row}: VLM rows must carry both {_VLM_KEYS}, found only one.")
            has_images = ~pv_null
        else:
            for column in lone_vlm_columns:
                present = pc.is_valid(tbl[column]).to_numpy(zero_copy_only=False)
                if present.any():
                    row = start + int(np.argmax(present))
                    raise ValueError(f"Row {row}: VLM rows must carry both {_VLM_KEYS}, found only one.")
            has_images = np.zeros(n, dtype=bool)

        # Over-length VLM rows are dropped, never truncated -- truncation would
        # cut image placeholder tokens and break image/text alignment.
        if max_length is not None:
            vlm_overlength = has_images & (ids_len > max_length)
        else:
            vlm_overlength = np.zeros(n, dtype=bool)
        num_vlm_dropped += int(vlm_overlength.sum())

        keep_chunks.append(has_loss & ~vlm_overlength)
        length_chunks.append(ids_len)

    keep = np.concatenate(keep_chunks) if keep_chunks else np.zeros(0, dtype=bool)
    lengths = np.concatenate(length_chunks) if length_chunks else np.zeros(0, dtype=np.int64)
    return keep, lengths, num_vlm_dropped


# ---------------------------------------------------------------------------
# Lazy row normalization
# ---------------------------------------------------------------------------


class _NormalizeTransform:
    """Lazy batch transform normalizing rows at access time.

    Rows reaching the transform are pre-validated and pre-filtered by
    :func:`_validate_and_plan`, so every row is well-formed and has a nonzero
    loss window; the transform only derives the internal representation. A
    class (not a closure) so the dataset pickles into spawn-based dataloader
    workers.

    TODO (sft): support consuming the full-sequence loss_mask in the workers
    directly instead of converting to the trailing action-window form
    (num_actions + window mask). The window representation is an RL legacy
    (prompt + trailing response); for SFT a position-aligned full-sequence
    mask is the more natural interface and would make this inference,
    the window slicing, and the collator's window padding unnecessary.
    """

    def __init__(self, max_length: Optional[int]):
        self.max_length = max_length

    def __call__(self, batch: dict) -> dict:
        max_length = self.max_length
        out = {k: v for k, v in batch.items() if k not in _CONSUMED_KEYS}
        input_ids_out, attention_out, num_actions_out, loss_mask_out = [], [], [], []
        for input_ids, loss_mask in zip(batch["input_ids"], batch["loss_mask"]):
            first = loss_mask.index(1)
            if max_length is not None and len(input_ids) > max_length:
                # Text-row truncation, mirroring the online tokenization path:
                # the prompt prefix is kept and the trailing window shrinks.
                # (Over-length VLM rows were dropped at load.)
                input_ids = input_ids[:max_length]
                loss_mask = loss_mask[:max_length]
            input_ids_out.append(input_ids)
            attention_out.append([1] * len(input_ids))
            num_actions_out.append(len(input_ids) - first)
            loss_mask_out.append(loss_mask[first:])
        out.update(
            {
                "input_ids": input_ids_out,
                "attention_mask": attention_out,
                "num_actions": num_actions_out,
                "loss_mask": loss_mask_out,
            }
        )
        return out


class MemoryMappedDataset(SFTDataset):
    """Map-style view over a validated, memory-mapped arrow store (a
    pretokenized dataset or the trainer's tokenized-dataset cache).

    Wraps an arrow-backed :class:`datasets.Dataset` (with the lazy
    normalization transform applied when rows need normalizing; stores
    already in the internal row form -- e.g. the tokenized-dataset cache --
    are wrapped without one) and strips ``None``-valued keys per row:
    mixed text+VLM stores materialize the image columns as ``None`` on text
    rows, and a ``None`` ``pixel_values`` key would break the collator's
    homogeneity check.

    Map-style datasets carry no iteration state, so checkpoint/resume is
    unchanged: ``StatefulDataLoader`` captures the sampler position exactly as
    it does for a plain list.
    """

    def __init__(self, dataset: Dataset, sequence_lengths: np.ndarray):
        self._dataset = dataset
        self._sequence_lengths = sequence_lengths

    @property
    def sequence_lengths(self) -> np.ndarray:
        """Post-truncation token count per row, computed from arrow offsets at
        load time (no row materialization); used for dataset statistics."""
        return self._sequence_lengths

    def __len__(self) -> int:
        return len(self._dataset)

    @staticmethod
    def _strip_none(row: dict) -> dict:
        return {k: v for k, v in row.items() if v is not None}

    def __getitem__(self, idx) -> dict:
        return self._strip_none(self._dataset[int(idx)])

    def __getitems__(self, indices: list) -> list[dict]:
        # Batched fetch used by the dataloader: one transform call per batch.
        batch = self._dataset[[int(i) for i in indices]]
        keys = list(batch.keys())
        return [self._strip_none({k: batch[k][j] for k in keys}) for j in range(len(indices))]

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_from_pretokenized(
    path: str,
    max_length: Optional[int] = None,
) -> MemoryMappedDataset:
    """Load a pretokenized SFT dataset from a local file or directory.

    The pretokenized counterpart of ``SFTTrainer._load_and_tokenize``: returns
    a map-style, memory-mapped dataset whose rows are the same normalized
    dicts the trainer's dataloaders and collators consume. Validation and row
    dropping are eager (vectorized arrow scans, no row materialization);
    normalization is lazy at access time, so memory stays bounded by the OS
    page cache instead of dataset size.

    Args:
        path: Local path to a file or directory in one of the supported
            formats (see module docstring).
        max_length: Optional sequence-length cap. Longer text rows are
            truncated (keeping the prompt prefix) and dropped if no loss tokens
            survive; longer VLM rows are always dropped with a warning,
            matching the online tokenization path.

    Returns:
        A :class:`MemoryMappedDataset` of normalized examples (``input_ids`` /
        ``attention_mask`` / ``num_actions`` / window ``loss_mask``, plus
        pass-through columns like ``pixel_values`` / ``image_grid_thw``) ready
        for the SFT collators.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pretokenized dataset path does not exist: {path}")

    dataset = _load_as_hf_dataset(path)
    logger.info(f"Loaded pretokenized dataset from '{path}': {len(dataset)} rows, columns={dataset.column_names}")

    columns = dataset.column_names
    if "input_ids" not in columns:
        raise ValueError(f"Pretokenized dataset at '{path}': missing required 'input_ids' column.")
    if "loss_mask" not in columns:
        raise ValueError(
            f"Pretokenized dataset at '{path}': missing required 'loss_mask' column "
            f"(full-sequence 0/1 mask, same length as input_ids)."
        )

    global _warned_attention_mask_dropped
    if "attention_mask" in columns and not _warned_attention_mask_dropped:
        _warned_attention_mask_dropped = True
        logger.warning(
            "Pretokenized dataset carries an 'attention_mask' column; its values are dropped and "
            "regenerated as all-ones (rows must be stored unpadded; padding is applied at collation time)."
        )

    keep, lengths, num_vlm_dropped = _validate_and_plan(dataset, max_length)
    num_kept = int(keep.sum())
    num_dropped = len(dataset) - num_kept

    if num_vlm_dropped:
        logger.warning(
            f"Dropping {num_vlm_dropped} VLM sample(s) longer than max_length={max_length}, "
            f"consider increasing max_length if you see this warning too much"
        )
    if num_dropped:
        logger.warning(
            f"Dropped {num_dropped}/{len(dataset)} pretokenized rows: empty loss window, "
            f"no loss tokens surviving max_length={max_length} truncation, or VLM rows over max_length."
        )
    if num_kept == 0:
        raise ValueError(f"Pretokenized dataset at '{path}' produced 0 usable examples.")

    if num_dropped:
        dataset = dataset.select(np.flatnonzero(keep))
        lengths = lengths[keep]
    if max_length is not None:
        lengths = np.minimum(lengths, max_length)

    dataset.set_transform(_NormalizeTransform(max_length))
    logger.info(f"Prepared {num_kept} pretokenized examples (memory-mapped)")
    return MemoryMappedDataset(dataset, lengths)
