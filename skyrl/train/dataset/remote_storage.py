"""Materialize cloud-hosted (S3/GCS) datasets to local storage.

Shared by the SFT data paths: pretokenized stores
(:mod:`skyrl.train.dataset.pretokenized`) and text-format datasets
(``SFTTrainer._load_and_tokenize``) both accept ``s3://``, ``gs://``, and
``gcs://`` URIs, resolved through
:mod:`skyrl.backends.skyrl_train.utils.io` (fsspec-backed, with the
project's S3 retry/credential-refresh handling).

Downloads are cached under ``cache_dir`` keyed by the remote path (atomic
tmp-dir-then-rename, so concurrent readers -- e.g. multi-node runs sharing an
NFS cache -- never see a partial download). The cache is keyed by remote path
only: replacing a store's contents in place requires ``force_redownload``.
"""

import hashlib
import os
import shutil
import tempfile
from contextlib import contextmanager
from typing import Iterator, Optional

from loguru import logger

from skyrl.backends.skyrl_train.utils.io import io


def _cache_dir_for_path(cache_dir: str, path: str) -> str:
    """Local cache directory for a downloaded cloud dataset path."""
    key = hashlib.sha256(path.encode()).hexdigest()[:16]
    return os.path.join(cache_dir, "remote_datasets", key)


def _download_to(path: str, target_dir: str) -> None:
    """Download a cloud file or directory into ``target_dir``.

    A cloud directory's contents land directly in ``target_dir``; a single
    cloud file lands at ``target_dir/<basename>``. The download goes to a
    sibling ``<target_dir>.tmp`` first and is atomically renamed so concurrent
    readers never see a partial download.
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


def materialize_remote_dataset(
    path: str,
    cache_dir: Optional[str] = None,
    force_redownload: bool = False,
) -> str:
    """Return a *persistent* local copy of the dataset at ``path``.

    Local paths pass through unchanged (after an existence check). Cloud paths
    are downloaded into ``cache_dir`` (reused across runs), or into a
    process-lifetime temporary directory when ``cache_dir`` is ``None``.

    Use this when the resolved path outlives the caller -- e.g. the text
    tokenization path, whose worker subprocesses re-open the dataset by path.
    """
    if not io.is_cloud_path(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset path does not exist: {path}")
        return path

    if cache_dir is not None:
        local_dir = _cache_dir_for_path(cache_dir, path)
        if os.path.isdir(local_dir) and not force_redownload:
            logger.info(f"Using cached download of '{path}' at {local_dir}")
        else:
            logger.info(f"Downloading dataset '{path}' to {local_dir} ...")
            _download_to(path, local_dir)
        return local_dir

    local_dir = os.path.join(tempfile.mkdtemp(prefix="skyrl_remote_dataset_"), "data")
    logger.info(f"Downloading dataset '{path}' to temporary directory ...")
    _download_to(path, local_dir)
    return local_dir


@contextmanager
def materialize_local(path: str, cache_dir: Optional[str], force_redownload: bool) -> Iterator[str]:
    """Yield a local path holding the dataset at ``path``.

    Like :func:`materialize_remote_dataset`, but when ``cache_dir`` is ``None``
    the temporary download is removed at context exit. Use this when the data
    is fully consumed within the context (e.g. the pretokenized loader, which
    materializes rows in memory).
    """
    if not io.is_cloud_path(path) or cache_dir is not None:
        yield materialize_remote_dataset(path, cache_dir, force_redownload)
        return

    with tempfile.TemporaryDirectory(prefix="skyrl_remote_dataset_") as temp_dir:
        local_dir = os.path.join(temp_dir, "data")
        logger.info(f"Downloading dataset '{path}' to temporary directory ...")
        _download_to(path, local_dir)
        yield local_dir
