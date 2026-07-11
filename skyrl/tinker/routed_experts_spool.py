"""On-disk handoff for Rollout Routing Replay (R3) rollout routing.

In non-colocated mode the Tinker API process forwards sample requests
directly to the SkyRL-Train-managed vLLM (see
``SkyRLTrainInferenceForwardingClient``), bypassing the engine subprocess.
The rollout routing vLLM returns therefore surfaces in the API process,
while ``forward_backward`` replays it from the training backend's
in-memory ``RoutedExpertsCache`` inside the engine subprocess. The two
processes always share a host (``api.py`` spawns the engine as a child),
so routing is handed off through a spool directory of content-addressed
``.npy`` files:

    <blake2b(model_id)[8 bytes].hex>-<blake2b(int64 token ids)[16 bytes].hex>.npy

The sequence digest is byte-identical to the backend cache key, so the
backend can look a file up from nothing but ``(model_id, training
sequence)``. Writers (the API's forwarding client) create files atomically
(temp file + rename); the reader (the backend) loads and deletes a file the
first time the trainer needs it and serves later epochs from its in-memory
cache. Files that never reach the trainer are pruned at weight-sync
boundaries with the same staleness semantics as the in-memory cache, and
the backend wipes the directory at startup and teardown.

This module is imported by both the API process and the training backend,
so it must stay free of heavy imports (no ray / torch / vllm).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import time
import uuid
from collections import deque
from typing import Sequence

import numpy as np

from skyrl.utils.log import logger

_TMP_PREFIX = ".tmp-"


def sequence_digest(token_ids: Sequence[int]) -> bytes:
    """16-byte blake2b digest of the token ids, hashed as int64 bytes.

    This is the R3 cache/spool key: fixed-size regardless of sequence length,
    and order- and value-exact. Must stay in sync with the backend's
    ``RoutedExpertsCache`` key (which delegates here).
    """
    buf = np.asarray(token_ids, dtype=np.int64).tobytes()
    return hashlib.blake2b(buf, digest_size=16).digest()


def _model_prefix(model_id: str) -> str:
    """Filename-safe 16-hex-char prefix identifying a model's spool files."""
    return hashlib.blake2b(model_id.encode("utf-8"), digest_size=8).hexdigest()


def resolve_spool_dir(explicit_dir: str | None, database_url: str) -> str:
    """Resolve the spool directory shared by the API and engine processes.

    Both processes call this with the same ``EngineConfig`` values (the API
    serializes its config into the engine's argv), so an explicit directory or
    the ``database_url``-derived default resolves identically on both sides
    without any runtime coordination. The default lives under the system temp
    dir; deployments whose temp dir is a small tmpfs can point
    ``routed_experts_spool_dir`` at real disk.
    """
    if explicit_dir:
        return explicit_dir
    tag = hashlib.sha256(database_url.encode("utf-8")).hexdigest()[:16]
    return os.path.join(tempfile.gettempdir(), f"skyrl-r3-spool-{tag}")


class RoutedExpertsSpool:
    """Content-addressed spool of per-sample rollout routing ``.npy`` files.

    The writer side (API process) only calls :meth:`write`. The reader side
    (training backend) calls :meth:`consume` on cache misses and owns the
    lifecycle: :meth:`on_weight_sync` prunes a model's abandoned files with
    the same staleness rule as the in-memory cache, :meth:`evict_model`
    drops a deleted adapter's files, and :meth:`wipe` clears everything at
    backend startup/teardown.
    """

    def __init__(self, root: str):
        self.root = root
        # Reader-side wall-clock timestamps of each model's weight syncs,
        # bounded to the staleness window (see on_weight_sync).
        self._sync_times: dict[str, deque] = {}

    def _path(self, model_id: str, token_ids: Sequence[int]) -> str:
        return os.path.join(self.root, f"{_model_prefix(model_id)}-{sequence_digest(token_ids).hex()}.npy")

    def write(self, model_id: str, token_ids: Sequence[int], npy_bytes: bytes) -> None:
        """Atomically publish one sample's routing as raw ``.npy`` bytes.

        The temp file lives in the spool directory itself so the final
        ``os.replace`` is an atomic same-filesystem rename: readers either see
        a complete file or none at all. Re-writing an existing key replaces
        the file (last write wins), mirroring the in-memory cache's refresh
        semantics.
        """
        os.makedirs(self.root, exist_ok=True)
        tmp_path = os.path.join(self.root, f"{_TMP_PREFIX}{uuid.uuid4().hex}")
        with open(tmp_path, "wb") as f:
            f.write(npy_bytes)
        os.replace(tmp_path, self._path(model_id, token_ids))

    def consume(self, model_id: str, token_ids: Sequence[int]) -> np.ndarray | None:
        """Load and delete one sample's routing; ``None`` when absent.

        The file is removed even when it fails to parse so a corrupt payload
        cannot poison every retry. Callers are expected to hold the loaded
        array in their own cache for repeat (multi-epoch) reads.
        """
        path = self._path(model_id, token_ids)
        try:
            with open(path, "rb") as f:
                arr = np.load(f, allow_pickle=False)
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.warning("R3 spool: dropping unreadable routing file %s: %s", path, e)
            arr = None
        try:
            os.unlink(path)
        except OSError:
            pass
        return arr

    def on_weight_sync(self, model_id: str, max_staleness: int) -> int:
        """Prune the model's files that outlived the staleness window.

        Mirrors ``RoutedExpertsCache``'s generation rule in the time domain: a
        file written during sync generation ``g`` is pruned at the sync that
        produces generation ``n`` when ``g <= n - 1 - max_staleness``, i.e.
        when its mtime predates the sync that produced generation
        ``n - max_staleness``. We keep the last ``max_staleness + 1`` sync
        timestamps per model; once the window is full, its oldest entry is
        exactly that cutoff. Stale temp files (crashed writes) are swept with
        the same cutoff regardless of model — any temp file older than a full
        sync interval is garbage.

        Returns the number of files removed.
        """
        window = self._sync_times.setdefault(model_id, deque(maxlen=max_staleness + 1))
        window.append(time.time())
        if len(window) < window.maxlen:
            return 0
        cutoff = window[0]

        prefix = f"{_model_prefix(model_id)}-"
        removed = 0
        try:
            names = os.listdir(self.root)
        except FileNotFoundError:
            return 0
        for name in names:
            if not (name.startswith(prefix) or name.startswith(_TMP_PREFIX)):
                continue
            path = os.path.join(self.root, name)
            try:
                if os.path.getmtime(path) < cutoff:
                    os.unlink(path)
                    removed += 1
            except OSError:
                continue
        if removed:
            logger.info(
                "R3 spool: weight sync for %s pruned %d never-trained routing file(s) from %s.",
                model_id,
                removed,
                self.root,
            )
        return removed

    def evict_model(self, model_id: str) -> int:
        """Drop all spool files (and sync history) for a deleted model."""
        self._sync_times.pop(model_id, None)
        prefix = f"{_model_prefix(model_id)}-"
        removed = 0
        try:
            names = os.listdir(self.root)
        except FileNotFoundError:
            return 0
        for name in names:
            if not name.startswith(prefix):
                continue
            try:
                os.unlink(os.path.join(self.root, name))
                removed += 1
            except OSError:
                continue
        return removed

    def wipe(self) -> None:
        """Remove the spool directory and reset sync history.

        Called at backend startup (leftovers from a previous run can never be
        replayed) and teardown. Writers recreate the directory on demand.
        """
        self._sync_times.clear()
        shutil.rmtree(self.root, ignore_errors=True)
