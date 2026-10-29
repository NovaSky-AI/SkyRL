import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (the H2D restore is the trigger)")


def _anon_gib() -> float:
    for line in open("/proc/self/smaps_rollup"):
        if line.startswith("Anonymous:"):
            return int(line.split()[1]) / 2**20
    raise RuntimeError("no Anonymous in smaps_rollup")


def test_restored_frozen_param_stays_page_cache(tmp_path, monkeypatch):
    from skyrl.backends.skyrl_train.distributed.megatron import megatron_utils as mu

    monkeypatch.setattr(mu, "_FROZEN_OFFLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(mu, "_frozen_offload_failed", False)
    p = torch.nn.Parameter(torch.randint(0, 255, (2 * 2**30,), dtype=torch.uint8, device="cuda"), requires_grad=False)
    want = p.data.clone()
    assert mu._offload_frozen_param_to_file("frozen.weight", p)
    p.data = torch.empty(0, dtype=torch.uint8, device="cuda")
    before = _anon_gib()
    restored = p._offload_cpu_data.to(torch.cuda.current_device(), non_blocking=True)
    torch.cuda.synchronize()
    assert _anon_gib() - before < 0.5
    assert torch.equal(restored, want)
    assert not os.listdir(tmp_path / "rank0")


def _rss_file_gib() -> float:
    """Mapped file pages of this process (RssShmem too: the offload directory may be a tmpfs)."""
    kib = 0
    for line in open("/proc/self/status"):
        if line.startswith(("RssFile:", "RssShmem:")):
            kib += int(line.split()[1])
    return kib / 2**20


def test_restore_maps_the_offload_copy_before_copying(tmp_path, monkeypatch):
    from skyrl.backends.skyrl_train.distributed.megatron import megatron_utils as mu

    monkeypatch.setattr(mu, "_FROZEN_OFFLOAD_DIR", str(tmp_path))
    monkeypatch.setattr(mu, "_frozen_offload_failed", False)
    p = torch.nn.Parameter(torch.randint(0, 255, (2**30,), dtype=torch.uint8, device="cuda"), requires_grad=False)
    want = p.data.clone()
    assert mu._offload_frozen_param_to_file("frozen.weight", p)
    p.data = torch.empty(0, dtype=torch.uint8, device="cuda")
    before = _rss_file_gib()
    mu._populate_offload_mapping(p)
    assert _rss_file_gib() - before > 0.9
    assert torch.equal(p._offload_cpu_data.to(torch.cuda.current_device()), want)
