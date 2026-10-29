"""The NVFP4 QAT size-class grouping and the LeRoPE inverse frequencies run without waiting on the
device, and produce exactly what the synchronizing versions produced."""

import itertools

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (host syncs are the subject)")
pytest.importorskip("megatron.core")

from skyrl.backends.skyrl_train.patches.megatron.ml4 import modules  # noqa: E402
from skyrl.backends.skyrl_train.patches.megatron.ml4.ml4_train import (  # noqa: E402
    ops,
    qat,
)


def _by_size_class_reference(counts, n_cols, like, fn):
    """The previous implementation: per-class device index tensors and boolean masks."""
    out = like.new_empty((sum(counts), n_cols), dtype=torch.float32)
    starts = list(itertools.accumulate([0] + counts[:-1]))
    for groups, P in qat._size_classes(counts):
        gi = torch.tensor(groups, device=like.device)
        st = torch.tensor([starts[g] for g in groups], device=like.device)[:, None]
        cn = torch.tensor([counts[g] for g in groups], device=like.device)[:, None]
        ar = torch.arange(P, device=like.device)[None]
        valid = ar < cn
        idx = torch.where(valid, st + ar, st)
        out[idx[valid]] = fn(gi, idx)[valid]
    return out


@pytest.mark.parametrize(
    "counts",
    [[1], [3, 17, 16, 1, 40, 2], [5] * 4 + [300, 129, 128, 64, 33], list(range(1, 70))],
    ids=["one-row", "mixed", "big-classes", "many-classes"],
)
def test_by_size_class_bitwise_and_sync_free(counts):
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(sum(counts), 24, device="cuda", generator=g)
    w = torch.randn(len(counts), 24, 40, device="cuda", generator=g, dtype=torch.bfloat16)
    scale = torch.rand(len(counts), device="cuda", generator=g)

    def fn(gi, idx):
        return torch.bmm(x[idx].to(torch.bfloat16), w[gi], out_dtype=torch.float32) * scale[gi][:, None, None]

    want = _by_size_class_reference(counts, 40, x, fn)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        got = qat._by_size_class(counts, 40, x, fn)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.equal(got, want)


ROPE = {
    "rope_type": "yarn",
    "rope_theta": 10000.0,
    "factor": 4.0,
    "original_max_position_embeddings": 4096,
    "beta_fast": 32,
    "beta_slow": 1,
    "attention_factor": 1.0,
    "mscale": 1,
    "mscale_all_dim": 1,
}


def test_inv_freq_cached_bitwise_and_sync_free(monkeypatch):
    monkeypatch.setattr(modules, "_INV_FREQ", {})
    alpha = torch.randn(32, device="cuda") * 0.1
    want = ops.rope_inv_freq(ROPE, 64, alpha).to("cuda")
    assert torch.equal(modules._inv_freq_on(torch.device("cuda"), ROPE, 64, alpha), want)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        again = modules._inv_freq_on(torch.device("cuda"), ROPE, 64, alpha)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.equal(again, want)
    alpha.add_(0.5)
    assert torch.equal(
        modules._inv_freq_on(torch.device("cuda"), ROPE, 64, alpha), ops.rope_inv_freq(ROPE, 64, alpha).to("cuda")
    )
