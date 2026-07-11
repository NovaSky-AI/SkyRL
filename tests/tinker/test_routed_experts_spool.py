"""CPU tests for the R3 on-disk spool (`skyrl.tinker.routed_experts_spool`).

The spool hands rollout routing from the Tinker API process (which forwards
sample requests directly to vLLM in non-colocated mode) to the training
backend in the engine subprocess. These tests cover the content-addressed
key, atomic write/consume roundtrip, weight-sync staleness pruning, and the
per-model eviction/wipe lifecycle. No heavy deps (ray/torch/vllm) needed. Run:
  uv run --extra dev --extra jax pytest tests/tinker/test_routed_experts_spool.py
"""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

from skyrl.tinker import routed_experts_spool as spool_module
from skyrl.tinker.routed_experts_spool import (
    RoutedExpertsSpool,
    resolve_spool_dir,
    sequence_digest,
)


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _routing(seq_len: int = 4, num_layers: int = 2, topk: int = 3) -> np.ndarray:
    return np.arange(seq_len * num_layers * topk, dtype=np.int32).reshape(seq_len, num_layers, topk)


def test_sequence_digest_is_exact_and_fixed_size():
    assert len(sequence_digest([1, 2, 3])) == 16
    assert sequence_digest([1, 2, 3]) == sequence_digest([1, 2, 3])
    assert sequence_digest([1, 2, 3]) != sequence_digest([3, 2, 1])
    # Hashed as int64 bytes: matches a manual digest.
    manual = hashlib.blake2b(np.asarray([1, 2, 3], dtype=np.int64).tobytes(), digest_size=16).digest()
    assert sequence_digest([1, 2, 3]) == manual


def test_resolve_spool_dir_explicit_and_derived():
    assert resolve_spool_dir("/data/spool", "sqlite:///a.db") == "/data/spool"
    derived = resolve_spool_dir(None, "sqlite:///a.db")
    # Deterministic across calls (API and engine resolve independently).
    assert derived == resolve_spool_dir(None, "sqlite:///a.db")
    assert derived != resolve_spool_dir(None, "sqlite:///b.db")
    assert derived.startswith(tempfile.gettempdir())


def test_write_consume_roundtrip_deletes_file(tmp_path):
    spool = RoutedExpertsSpool(str(tmp_path / "spool"))
    routing = _routing()
    tokens = [10, 11, 12, 20, 21]

    spool.write("model_a", tokens, _npy_bytes(routing))
    assert len(os.listdir(spool.root)) == 1

    got = spool.consume("model_a", tokens)
    np.testing.assert_array_equal(got, routing)
    # Consumed exactly once: the file is gone and no temp files linger.
    assert os.listdir(spool.root) == []
    assert spool.consume("model_a", tokens) is None


def test_consume_misses_are_none():
    # Directory that was never created (no writes yet).
    spool = RoutedExpertsSpool(os.path.join(tempfile.gettempdir(), "skyrl-r3-spool-nonexistent-dir"))
    assert spool.consume("m", [1, 2]) is None


def test_write_same_key_last_wins(tmp_path):
    spool = RoutedExpertsSpool(str(tmp_path))
    tokens = [1, 2, 3]
    spool.write("m", tokens, _npy_bytes(_routing()))
    newer = _routing() + 1
    spool.write("m", tokens, _npy_bytes(newer))
    assert len([n for n in os.listdir(spool.root) if not n.startswith(".tmp-")]) == 1
    np.testing.assert_array_equal(spool.consume("m", tokens), newer)


def test_consume_drops_corrupt_file(tmp_path):
    spool = RoutedExpertsSpool(str(tmp_path))
    tokens = [1, 2, 3]
    spool.write("m", tokens, b"not an npy payload")
    assert spool.consume("m", tokens) is None
    # Corrupt file removed so it cannot poison retries.
    assert os.listdir(spool.root) == []


def test_keys_are_scoped_per_model_and_sequence(tmp_path):
    spool = RoutedExpertsSpool(str(tmp_path))
    routing_a, routing_b = _routing(), _routing() + 1
    spool.write("model_a", [1, 2], _npy_bytes(routing_a))
    spool.write("model_b", [1, 2], _npy_bytes(routing_b))

    np.testing.assert_array_equal(spool.consume("model_a", [1, 2]), routing_a)
    np.testing.assert_array_equal(spool.consume("model_b", [1, 2]), routing_b)
    assert spool.consume("model_a", [2, 1]) is None


@pytest.fixture
def fake_clock(monkeypatch):
    """Replace the spool module's time source with a controllable clock."""
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(spool_module, "time", SimpleNamespace(time=lambda: clock.now))
    return clock


def test_weight_sync_prunes_by_staleness_window(tmp_path, fake_clock):
    """Files written during sync generation g die at the sync producing
    generation g + max_staleness + 1 — same rule as the in-memory cache."""
    spool = RoutedExpertsSpool(str(tmp_path))
    spool.write("m", [1, 2], _npy_bytes(_routing()))
    path = spool._path("m", [1, 2])
    os.utime(path, (fake_clock.now, fake_clock.now))  # written at t=1000 (gen 0)

    fake_clock.now = 1100.0
    assert spool.on_weight_sync("m", max_staleness=1) == 0  # sync 1: window not full
    assert os.path.exists(path)

    fake_clock.now = 1200.0
    assert spool.on_weight_sync("m", max_staleness=1) == 1  # sync 2: gen 0 <= 2 - 1 - 1
    assert not os.path.exists(path)


def test_weight_sync_keeps_files_written_after_recent_sync(tmp_path, fake_clock):
    spool = RoutedExpertsSpool(str(tmp_path))
    fake_clock.now = 1000.0
    spool.on_weight_sync("m", max_staleness=1)  # sync 1 at t=1000

    spool.write("m", [1, 2], _npy_bytes(_routing()))
    path = spool._path("m", [1, 2])
    os.utime(path, (1050.0, 1050.0))  # written during gen 1

    fake_clock.now = 1100.0
    assert spool.on_weight_sync("m", max_staleness=1) == 0  # sync 2: cutoff t=1000, file newer
    assert os.path.exists(path)

    fake_clock.now = 1200.0
    assert spool.on_weight_sync("m", max_staleness=1) == 1  # sync 3: cutoff t=1100
    assert not os.path.exists(path)


def test_zero_staleness_prunes_at_first_sync(tmp_path, fake_clock):
    spool = RoutedExpertsSpool(str(tmp_path))
    spool.write("m", [1, 2], _npy_bytes(_routing()))
    os.utime(spool._path("m", [1, 2]), (900.0, 900.0))

    fake_clock.now = 1000.0
    assert spool.on_weight_sync("m", max_staleness=0) == 1
    assert not os.path.exists(spool._path("m", [1, 2]))


def test_weight_sync_is_scoped_per_model_but_sweeps_stale_tmp(tmp_path, fake_clock):
    spool = RoutedExpertsSpool(str(tmp_path))
    spool.write("model_a", [1, 2], _npy_bytes(_routing()))
    spool.write("model_b", [1, 2], _npy_bytes(_routing()))
    # A crashed write leaves a temp file behind.
    stale_tmp = os.path.join(spool.root, ".tmp-deadbeef")
    with open(stale_tmp, "wb") as f:
        f.write(b"partial")
    for path in (spool._path("model_a", [1, 2]), spool._path("model_b", [1, 2]), stale_tmp):
        os.utime(path, (900.0, 900.0))

    fake_clock.now = 1000.0
    spool.on_weight_sync("model_a", max_staleness=0)

    assert not os.path.exists(spool._path("model_a", [1, 2]))
    assert os.path.exists(spool._path("model_b", [1, 2]))  # other model untouched
    assert not os.path.exists(stale_tmp)  # crashed writes are garbage for any model


def test_weight_sync_on_missing_dir_is_noop(tmp_path, fake_clock):
    spool = RoutedExpertsSpool(str(tmp_path / "never-created"))
    assert spool.on_weight_sync("m", max_staleness=0) == 0


def test_evict_model_and_wipe(tmp_path):
    spool = RoutedExpertsSpool(str(tmp_path))
    spool.write("model_a", [1, 2], _npy_bytes(_routing()))
    spool.write("model_a", [3, 4], _npy_bytes(_routing()))
    spool.write("model_b", [1, 2], _npy_bytes(_routing()))

    assert spool.evict_model("model_a") == 2
    assert spool.consume("model_a", [1, 2]) is None
    assert spool.consume("model_b", [1, 2]) is not None

    spool.write("model_b", [5, 6], _npy_bytes(_routing()))
    spool.wipe()
    assert not os.path.exists(spool.root)
    # Writers recreate the directory on demand after a wipe.
    spool.write("model_b", [5, 6], _npy_bytes(_routing()))
    assert spool.consume("model_b", [5, 6]) is not None
