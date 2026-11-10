"""The consumer's packed pulls stay below 2 GiB (NIXL LIBFABRIC reads of 2-4 GiB hang), CPU-only."""

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm", reason="the SkyRL sharded_rdt engines import vllm at module scope")

pytestmark = pytest.mark.vllm

from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.sharded_rdt_engine import (  # noqa: E402
    DEFAULT_MAX_PULL_GB,
    _max_pull_bytes,
    split_scatters_by_bytes,
)

MiB = 1 << 20


def _sc(i, nbytes):
    return SimpleNamespace(i=i, nbytes=nbytes)


def _packed(run):
    size = 0
    for sc in run:
        size = ((size + 15) & ~15) + sc.nbytes
    return size


def test_runs_keep_order_and_respect_the_cap():
    scatters = [_sc(i, n) for i, n in enumerate([300 * MiB, 300 * MiB, 7, 500 * MiB, 200 * MiB, 9, 600 * MiB])]
    runs = split_scatters_by_bytes(scatters, 1024 * MiB)
    assert [sc.i for run in runs for sc in run] == list(range(len(scatters)))
    assert all(_packed(run) <= 1024 * MiB for run in runs)
    assert [[sc.i for sc in run] for run in runs] == [[0, 1, 2], [3, 4, 5], [6]]


def test_a_chunk_under_the_cap_is_untouched():
    scatters = [_sc(i, 10 * MiB) for i in range(5)]
    assert split_scatters_by_bytes(scatters, 1024 * MiB) == [scatters]


def test_an_oversized_scatter_gets_its_own_run():
    scatters = [_sc(0, 10), _sc(1, 1500 * MiB), _sc(2, 10)]
    runs = split_scatters_by_bytes(scatters, 1024 * MiB)
    assert [[sc.i for sc in run] for run in runs] == [[0], [1], [2]]


def test_a_chunk_that_hung_is_cut_below_2gib():
    # A 2,562,893,824-byte pull (one MoE layer on the MXFP8 wire at TP4) hung before the cap.
    scatters = [_sc(i, 2_562_893_824 // 1557) for i in range(1557)]
    runs = split_scatters_by_bytes(scatters, _max_pull_bytes())
    assert len(runs) == 3
    assert all(_packed(run) < (1 << 31) for run in runs)


def test_knob(monkeypatch):
    monkeypatch.delenv("SKYRL_RDT_MAX_PULL_GB", raising=False)
    assert _max_pull_bytes() == int(DEFAULT_MAX_PULL_GB * (1 << 30))
    monkeypatch.setenv("SKYRL_RDT_MAX_PULL_GB", "0.5")
    assert _max_pull_bytes() == 512 * MiB
    for bad in ("2", "3.5", "0"):
        monkeypatch.setenv("SKYRL_RDT_MAX_PULL_GB", bad)
        with pytest.raises(ValueError, match="must be in"):
            _max_pull_bytes()
