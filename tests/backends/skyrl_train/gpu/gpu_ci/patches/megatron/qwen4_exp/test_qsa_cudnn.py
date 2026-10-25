"""Qwen3.8 QSA sparse attention: the cuDNN frontend kernels against the chunked SDPA path.

``qsa_sparse_attention`` runs a pack through cuDNN's ``qsa_forward`` / ``qsa_backward`` on
Blackwell (NVIDIA/cudnn-frontend#1765). This checks its output and q / k / v gradients against
the chunked SDPA path on a padded pack mixing sparse-regime sequences (top-512 blocks), short
dense ones and padding, for 24Q/2KV (TP=1) and 12Q/1KV (TP=2).

Needs an SM100/SM103 GPU and a cudnn-frontend with ``cudnn.ops.qsa_forward`` (>= 1.32; until it
is on PyPI, the GitHub dev wheel); skipped otherwise. Run with:
uv run --isolated --extra dev --extra megatron --with <nvidia_cudnn_frontend-1.32*.whl> pytest -s \
    tests/backends/skyrl_train/gpu/gpu_ci/patches/megatron/qwen4_exp/test_qsa_cudnn.py
"""

import pytest
import torch

pytestmark = pytest.mark.megatron

R, TOPK, D = 4, 512, 256


def _cudnn_qsa_available():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        from cudnn.ops import qsa_forward  # noqa: F401
    except (ImportError, AttributeError):
        return False
    return True


def _selection(lengths, pad_to, seed):
    from skyrl.backends.skyrl_train.patches.megatron.qwen4_exp.modules import (
        QSASelection,
        QSASequenceSelection,
    )

    gen = torch.Generator(device="cuda").manual_seed(seed)
    seqs, start = [], 0
    for n in lengths:
        mask = None
        if n > TOPK * R + R - 1:  # sparse regime: top-512 of the complete blocks, as the indexer
            nb = n // R
            visible = (torch.arange(n, device="cuda") + 1) // R
            complete = torch.arange(nb, device="cuda") < visible[:, None]
            scores = torch.rand(n, nb, device="cuda", generator=gen).masked_fill(~complete, -1)
            top = scores.topk(TOPK, dim=-1).indices
            mask = torch.where((visible > TOPK)[:, None], torch.zeros_like(complete).scatter_(-1, top, True), complete)
        seqs.append(QSASequenceSelection(start, n, mask))
        start += -(-n // pad_to) * pad_to
    return QSASelection(seqs, R), start


@pytest.mark.skipif(not _cudnn_qsa_available(), reason="needs SM100/SM103 and cudnn.ops.qsa_forward")
@pytest.mark.parametrize("heads", [(24, 2), (12, 1)], ids=["24q2kv", "12q1kv"])
def test_cudnn_qsa_matches_sdpa(heads, monkeypatch):
    from skyrl.backends.skyrl_train.patches.megatron.qwen4_exp import modules

    hq, hkv = heads
    selection, total = _selection([5000, 1500, 997, 2600], pad_to=16, seed=0)
    torch.manual_seed(0)
    q, k, v = (
        torch.randn(total, h, D, device="cuda", dtype=torch.bfloat16, requires_grad=True) for h in (hq, hkv, hkv)
    )
    grad = torch.randn(total, hq * D, device="cuda", dtype=torch.bfloat16)
    scale = D**-0.5

    def run():
        out = modules.qsa_sparse_attention(q, k, v, selection, scale)
        return (out, *torch.autograd.grad(out, (q, k, v), grad))

    assert modules._cudnn_qsa_usable(q, k, v, R)
    got = run()
    monkeypatch.setattr(modules, "_QSA_CUDNN", False)
    monkeypatch.setattr(modules, "_QSA_FLEX", False)
    expected = run()

    real = torch.zeros(total, dtype=torch.bool, device="cuda")
    for s in selection.sequences:
        real[s.start : s.start + s.length] = True
    for name, a, b in zip(("out", "dq", "dk", "dv"), got, expected):
        err = (a.float() - b.float())[real].abs().max().item()
        ref = b.float()[real].abs().max().item()
        assert err <= 1e-2 * ref, f"{name}: max|cudnn - sdpa| {err:.3g} vs max|sdpa| {ref:.3g}"
        assert torch.all(a[~real] == 0), f"{name}: nonzero on padding rows"
