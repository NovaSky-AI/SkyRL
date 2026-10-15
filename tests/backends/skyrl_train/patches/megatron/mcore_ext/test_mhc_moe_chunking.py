"""``chunked_mlp``: running the MLP/MoE over sequence chunks matches one call.

``HyperConnectionTransformerLayer`` replaces megatron-core's ``_forward_mlp``, which is where
``mlp_chunks_for_training`` is applied, so the mHC layer chunks through ``chunked_mlp`` instead.
Checked with a toy top-k MoE (router + per-expert linears, per-token like the real one): outputs
and input / parameter gradients match the unchunked call, the padding mask is split with the
tokens, and uneven or oversized chunk counts behave.

Pure tensor math, so it runs on CPU and lives outside ``gpu/``.
"""

import pytest
import torch

pytest.importorskip("megatron.core", reason="requires the megatron extra")

from skyrl.backends.skyrl_train.patches.megatron.mcore_ext.mhc_transformer_layer import (  # noqa: E402
    chunked_mlp,
)

# Runs in the CPU megatron job (`-m megatron`); without the marker that job deselects it.
pytestmark = pytest.mark.megatron

H, E, K = 16, 6, 2


class ToyMoE(torch.nn.Module):
    """Top-k routed experts; masked tokens contribute zero. Returns ``(output, bias)`` like MoELayer."""

    def __init__(self):
        super().__init__()
        self.router = torch.nn.Linear(H, E, bias=False)
        self.experts = torch.nn.ModuleList(torch.nn.Linear(H, H) for _ in range(E))
        self.calls = []

    def forward(self, x, padding_mask=None):
        self.calls.append((tuple(x.shape), None if padding_mask is None else tuple(padding_mask.shape)))
        probs, idx = torch.topk(torch.softmax(self.router(x), dim=-1), K, dim=-1)
        out = torch.zeros_like(x)
        for k in range(K):
            for e in range(E):
                sel = (idx[..., k] == e).unsqueeze(-1)
                out = out + sel * probs[..., k : k + 1] * self.experts[e](x)
        if padding_mask is not None:  # [b, s], True = padding
            out = out * (~padding_mask).transpose(0, 1).unsqueeze(-1)
        return out, None


def _run(mlp, x, mask, num_chunks):
    x = x.clone().requires_grad_(True)
    mlp.zero_grad()
    out, bias = chunked_mlp(mlp, x, mask, num_chunks)
    (out * torch.arange(out.numel()).view_as(out).sin()).sum().backward()
    return out.detach(), bias, x.grad, {n: p.grad.clone() for n, p in mlp.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("seq, num_chunks", [(32, 8), (30, 4), (5, 8), (32, 1)])
@pytest.mark.parametrize("with_mask", [False, True])
def test_chunked_matches_single_call(seq, num_chunks, with_mask):
    torch.manual_seed(0)
    mlp = ToyMoE()
    x = torch.randn(seq, 2, H)
    mask = (torch.rand(2, seq) < 0.2) if with_mask else None

    ref = _run(mlp, x, mask, 1)
    mlp.calls.clear()
    got = _run(mlp, x, mask, num_chunks)

    torch.testing.assert_close(got[0], ref[0], rtol=0, atol=1e-6)
    assert got[1] is None
    torch.testing.assert_close(got[2], ref[2], rtol=0, atol=1e-6)
    assert got[3].keys() == ref[3].keys()
    for name in ref[3]:
        torch.testing.assert_close(got[3][name], ref[3][name], rtol=1e-5, atol=1e-6)

    # One call per chunk (capped at one token per chunk); masks split along the sequence with the tokens.
    expected = min(num_chunks, seq)
    assert len(mlp.calls) == expected
    assert sum(shape[0] for shape, _ in mlp.calls) == seq
    for shape, mask_shape in mlp.calls:
        assert mask_shape == (None if mask is None else (2, shape[0]))


class IdRecordingMLP(torch.nn.Module):
    """Records the ``input_ids`` each call receives (hash-routed MoE layers take token ids)."""

    def __init__(self):
        super().__init__()
        self.input_ids = []

    def forward(self, x, padding_mask=None, input_ids=None):
        self.input_ids.append(input_ids)
        return x, None


@pytest.mark.parametrize("num_chunks", [1, 4])
def test_chunked_mlp_splits_input_ids_with_tokens(num_chunks):
    seq, b = 8, 2
    x = torch.randn(seq, b, H)
    input_ids = torch.arange(b * seq).view(b, seq)  # [b, s], while x is [s, b, h]
    mlp = IdRecordingMLP()

    chunked_mlp(mlp, x, None, num_chunks, input_ids=input_ids)

    assert len(mlp.input_ids) == num_chunks
    torch.testing.assert_close(torch.cat(mlp.input_ids, dim=1), input_ids)
    assert all(ids.shape == (b, seq // num_chunks) for ids in mlp.input_ids)


def test_chunked_mlp_omits_input_ids_when_unset():
    # MLPs that do not take ``input_ids`` (dense and ordinary MoE layers) must not receive it.
    out, _ = chunked_mlp(ToyMoE(), torch.randn(8, 2, H), None, 4)
    assert out.shape == (8, 2, H)
