"""Drop a checkpoint's page cache after it has been loaded.

Freshly read checkpoint pages sit in the page cache as active file pages, and Ray's
memory monitor counts ``memory.current - inactive_file``: right after a large load the
cache alone can cross the kill threshold (ML4: 528 GiB/node). ``posix_fadvise(DONTNEED)``
drops a file's clean, unmapped pages; other readers just read from disk again.
"""

import functools
import os
import time
from collections.abc import Iterable

from loguru import logger

CHECKPOINT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")


def checkpoint_files(model_path: str, suffixes: tuple = CHECKPOINT_SUFFIXES) -> list:
    """Weight files under ``model_path`` (recursive), symlinks resolved, deduplicated."""
    if not os.path.isdir(model_path):
        return []
    out = {}
    for root, _, files in os.walk(model_path):
        for name in files:
            if name.endswith(suffixes):
                real = os.path.realpath(os.path.join(root, name))
                if os.path.isfile(real):
                    out.setdefault(real, None)
    return sorted(out)


def drop_page_cache(paths: Iterable[str]) -> tuple:
    """``posix_fadvise(DONTNEED)`` on each file. Returns (files dropped, bytes covered). No-op off POSIX."""
    if not hasattr(os, "posix_fadvise"):
        return 0, 0
    n = nbytes = 0
    for path in paths:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError as e:
            logger.warning(f"page cache drop: cannot open {path}: {e}")
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            nbytes += os.fstat(fd).st_size
            n += 1
        except OSError as e:
            logger.warning(f"page cache drop: fadvise failed on {path}: {e}")
        finally:
            os.close(fd)
    return n, nbytes


def drop_checkpoint_page_cache(model_path: str) -> tuple:
    t = time.time()
    n, nbytes = drop_page_cache(checkpoint_files(model_path))
    if n:
        logger.info(
            f"dropped page cache of {n} checkpoint files ({nbytes / 2**30:.1f} GiB) under {model_path} "
            f"in {time.time() - t:.1f}s"
        )
    return n, nbytes


def per_file_drop(iterator_fn, lag: int = 3):
    """Wrap vLLM's ``safetensors_weights_iterator``: iterate one file at a time and drop the page
    cache of finished files during the load, so the cache does not grow to the checkpoint size.

    A tensor yielded from a file keeps that file's whole mmap alive, and mapped pages cannot be
    dropped. The consumer still holds the previous file's last tensor while it asks for the next
    one, and the other TP workers on the node may still be inside the file, so a finished file is
    dropped once this worker has moved on to the next file, and again on each of the next ``lag``
    files; every file is dropped once more at the end. Prefetch (explicit, or vLLM's auto prefetch
    on a network FS) is passed through unchanged: it reads all files up front."""

    @functools.wraps(iterator_fn)
    def iterate(hf_weights_files, use_tqdm_on_load, safetensors_load_strategy=None, *args, **kwargs):
        from vllm.model_executor.model_loader import weight_utils as wu

        files = sorted(hf_weights_files, key=getattr(wu, "_natural_sort_key", None))
        fs = wu._get_fs_type(files) if hasattr(wu, "_get_fs_type") else ""
        if safetensors_load_strategy == "prefetch" or (
            safetensors_load_strategy is None and fs in ("nfs", "nfs4", "lustre")
        ):
            yield from iterator_fn(hf_weights_files, use_tqdm_on_load, safetensors_load_strategy, *args, **kwargs)
            return
        t = time.time()
        for i, path in enumerate(files):
            first = True
            for item in iterator_fn([path], False, safetensors_load_strategy, *args, **kwargs):
                yield item
                if first and i:  # the consumer has let go of the previous file's last tensor
                    drop_page_cache(files[max(0, i - lag) : i])
                first = False
        n, nbytes = drop_page_cache(files)
        logger.info(
            f"dropped page cache per file while loading {n} files ({nbytes / 2**30:.1f} GiB) "
            f"in {time.time() - t:.1f}s"
        )

    iterate._skyrl_per_file_drop = True
    return iterate


def patch_vllm_per_file_page_cache_drop() -> bool:
    """Install :func:`per_file_drop` on vLLM's default loader (import time, in every worker,
    before the model loads). ``SKYRL_PER_FILE_PAGE_CACHE_DROP=0`` disables it."""
    if os.environ.get("SKYRL_PER_FILE_PAGE_CACHE_DROP", "1") == "0" or not hasattr(os, "posix_fadvise"):
        return False
    try:
        from vllm.model_executor.model_loader import default_loader
    except ImportError:
        return False
    fn = getattr(default_loader, "safetensors_weights_iterator", None)
    if fn is None or getattr(fn, "_skyrl_per_file_drop", False):
        return False
    default_loader.safetensors_weights_iterator = per_file_drop(fn)
    return True
