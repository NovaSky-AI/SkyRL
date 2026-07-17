"""Loading of pretokenized SFT datasets from local or cloud storage.

Some data pipelines tokenize offline (e.g. on a Spark/Ray data cluster) and
publish the result to an object store. This module lets the SFT trainer ingest
such a store directly -- skipping online tokenization (``tokenize_chat_example``
and ``tokenize_sft_example`` in ``skyrl.train.sft_trainer``) entirely -- from:

- a local file or directory,
- an S3 path (``s3://bucket/prefix``),
- a GCS path (``gs://`` or ``gcs://``).

Cloud paths are materialized locally through
:mod:`skyrl.backends.skyrl_train.utils.io` (fsspec-backed, with the project's
S3 retry/credential-refresh handling) before loading.

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
"""

import hashlib
import os
import shutil
import tempfile
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from datasets import Dataset, concatenate_datasets, load_dataset
from loguru import logger

from skyrl.backends.skyrl_train.utils.io import io

# Keys consumed (and re-emitted in normalized form) by ``_normalize_row``.
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


# ---------------------------------------------------------------------------
# Cloud path materialization
# ---------------------------------------------------------------------------


def _cache_dir_for_path(cache_dir: str, path: str) -> str:
    """Local cache directory for a downloaded cloud dataset path."""
    key = hashlib.sha256(path.encode()).hexdigest()[:16]
    return os.path.join(cache_dir, "pretokenized", key)


def _download_to(path: str, target_dir: str) -> None:
    """Download a cloud file or directory into ``target_dir``.

    A cloud directory's contents land directly in ``target_dir``; a single
    cloud file lands at ``target_dir/<basename>``. The download goes to a
    sibling ``<target_dir>.tmp`` first and is atomically renamed so concurrent
    readers (e.g. multi-node runs sharing an NFS cache) never see a partial
    download.
    """
    temp_dir = target_dir + ".tmp"
    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    if io.isdir(path):
        io.download_directory(path, temp_dir)
    else:
        io.download_file(path, os.path.join(temp_dir, os.path.basename(path.rstrip("/"))))

    if os.path.isdir(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(os.path.dirname(target_dir), exist_ok=True)
    os.rename(temp_dir, target_dir)


@contextmanager
def _materialize_local(path: str, cache_dir: Optional[str], force_redownload: bool) -> Iterator[str]:
    """Yield a local path holding the dataset at ``path``.

    Local paths are yielded as-is. Cloud paths are downloaded either into a
    persistent cache directory (keyed by the remote path, reused across runs)
    or -- when ``cache_dir`` is ``None`` -- into a temporary directory that is
    removed once the rows have been materialized in memory.
    """
    if not io.is_cloud_path(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pretokenized dataset path does not exist: {path}")
        yield path
        return

    if cache_dir is not None:
        local_dir = _cache_dir_for_path(cache_dir, path)
        if os.path.isdir(local_dir) and not force_redownload:
            logger.info(f"Using cached download of '{path}' at {local_dir}")
        else:
            logger.info(f"Downloading pretokenized dataset '{path}' to {local_dir} ...")
            _download_to(path, local_dir)
        yield local_dir
        return

    with tempfile.TemporaryDirectory(prefix="skyrl_pretokenized_") as temp_dir:
        local_dir = os.path.join(temp_dir, "data")
        logger.info(f"Downloading pretokenized dataset '{path}' to temporary directory ...")
        _download_to(path, local_dir)
        yield local_dir


# ---------------------------------------------------------------------------
# Format detection / loading
# ---------------------------------------------------------------------------


def _collect_data_files(root: str) -> dict[str, list[str]]:
    """Recursively collect data files under ``root``, grouped by format."""
    groups: dict[str, list[str]] = {"parquet": [], "json": [], "arrow": []}
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
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
# Row normalization
# ---------------------------------------------------------------------------


def _validate_loss_mask(row: dict, row_idx: int, seq_len: int) -> list[int]:
    """Validate a row's full-sequence ``loss_mask`` and return it as ints."""
    loss_mask = row.get("loss_mask")
    if loss_mask is None:
        raise ValueError(
            f"Row {row_idx}: missing required 'loss_mask' column (full-sequence 0/1 mask, "
            f"same length as input_ids)."
        )
    loss_mask = [int(v) for v in loss_mask]
    if len(loss_mask) != seq_len:
        raise ValueError(
            f"Row {row_idx}: loss_mask length ({len(loss_mask)}) must equal len(input_ids) "
            f"({seq_len}). Window-form masks are not supported; store the full-sequence mask."
        )
    if any(v not in (0, 1) for v in loss_mask):
        raise ValueError(f"Row {row_idx}: loss_mask must contain only 0s and 1s.")
    return loss_mask


def _normalize_row(row: dict, row_idx: int, max_length: Optional[int]) -> Optional[dict]:
    """Normalize one pretokenized row to the trainer's internal representation.

    Infers ``num_actions`` from the first nonzero ``loss_mask`` entry: the
    action window spans from there to the end of the sequence, with the mask's
    interior 0s (e.g. user turns between assistant turns) preserved inside it.

    Returns ``None`` when the row should be dropped (empty loss window, fully
    truncated by ``max_length``, or a VLM row exceeding ``max_length``).
    """
    input_ids = row.get("input_ids")
    if input_ids is None:
        raise ValueError(f"Row {row_idx}: missing required 'input_ids' column.")
    input_ids = [int(t) for t in input_ids]
    seq_len = len(input_ids)
    if seq_len == 0:
        return None

    attention_mask = row.get("attention_mask")
    if attention_mask is not None and any(int(v) != 1 for v in attention_mask):
        raise ValueError(
            f"Row {row_idx}: attention_mask contains 0s. Pretokenized rows must be stored "
            "unpadded (padding is applied at collation time)."
        )

    loss_mask = _validate_loss_mask(row, row_idx, seq_len)
    first = next((i for i, v in enumerate(loss_mask) if v != 0), None)
    if first is None:
        # No trainable tokens; drop the row.
        return None
    num_actions = seq_len - first
    loss_mask = loss_mask[first:]

    # VLM rows: require both image keys, and mirror the online VLM path by
    # dropping (never truncating) rows over max_length -- truncation would cut
    # image placeholder tokens and break image/text alignment.
    has_images = any(row.get(k) is not None for k in _VLM_KEYS)
    if has_images:
        if any(row.get(k) is None for k in _VLM_KEYS):
            raise ValueError(f"Row {row_idx}: VLM rows must carry both {_VLM_KEYS}, found only one.")
        if max_length is not None and seq_len > max_length:
            return None
    elif max_length is not None and seq_len > max_length:
        # Truncate to max_length, mirroring the online tokenization path: the
        # prompt prefix is kept and the (trailing) action window shrinks.
        prompt_len = first
        input_ids = input_ids[:max_length]
        num_actions = max(max_length - prompt_len, 0)
        loss_mask = loss_mask[:num_actions]
        if num_actions <= 0 or sum(loss_mask) == 0:
            return None

    # Pass through extra columns (e.g. VLM tensors). None-valued columns are
    # dropped: mixed text+VLM stores materialize missing columns as None, and a
    # None 'pixel_values' key would break the collator's homogeneity check.
    normalized = {k: v for k, v in row.items() if k not in _CONSUMED_KEYS and v is not None}
    normalized.update(
        {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "num_actions": num_actions,
            "loss_mask": loss_mask,
        }
    )
    return normalized


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_from_pretokenized(
    path: str,
    max_length: Optional[int] = None,
    cache_dir: Optional[str] = None,
    force_redownload: bool = False,
) -> list[dict[str, Any]]:
    """Load a pretokenized SFT dataset from a local path or object store.

    The pretokenized counterpart of ``SFTTrainer._load_and_tokenize``: returns
    the same ``list[dict]`` representation the trainer's dataloaders and
    collators consume.

    Args:
        path: Local path, ``s3://``, ``gs://``, or ``gcs://`` URI pointing to a
            file or directory in one of the supported formats (see module
            docstring).
        max_length: Optional sequence-length cap. Longer text rows are
            truncated (keeping the prompt prefix) and dropped if no loss tokens
            survive; longer VLM rows are always dropped, matching the online
            tokenization path.
        cache_dir: Base directory for caching cloud downloads. ``None``
            downloads to a temporary directory instead (no reuse across runs).
        force_redownload: Re-download a cloud path even if a cached copy
            exists.

    Returns:
        List of normalized examples (``input_ids`` / ``attention_mask`` /
        ``num_actions`` / window ``loss_mask``, plus pass-through columns like
        ``pixel_values`` / ``image_grid_thw``) ready for the SFT collators.
    """
    with _materialize_local(path, cache_dir, force_redownload) as local_path:
        dataset = _load_as_hf_dataset(local_path)
        logger.info(f"Loaded pretokenized dataset from '{path}': {len(dataset)} rows, columns={dataset.column_names}")

        normalized: list[dict] = []
        num_dropped = 0
        for row_idx, row in enumerate(dataset):
            example = _normalize_row(row, row_idx, max_length)
            if example is None:
                num_dropped += 1
            else:
                normalized.append(example)

    if num_dropped:
        logger.warning(
            f"Dropped {num_dropped}/{len(dataset)} pretokenized rows: empty loss window, "
            f"no loss tokens surviving max_length={max_length} truncation, or VLM rows over max_length."
        )
    if not normalized:
        raise ValueError(f"Pretokenized dataset at '{path}' produced 0 usable examples.")
    logger.info(f"Prepared {len(normalized)} pretokenized examples")
    return normalized
