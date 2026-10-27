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
    _ChunkedRouterReplay,
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
        self.calls.append((tuple(x.shape), padding_mask))
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


@pytest.mark.parametrize("seq, num_chunks", [(32, 8), (30, 4), (5, 8), (10, 8), (1, 8), (32, 1)])
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

    # Exactly num_chunks calls, each non-empty, on every rank (EP collectives must match): a
    # sequence shorter than that is padded with masked dummy tokens. Masks split with the tokens.
    assert len(mlp.calls) == num_chunks
    assert all(shape[0] >= 1 for shape, _ in mlp.calls)
    assert sum(shape[0] for shape, _ in mlp.calls) == max(seq, num_chunks)
    pad = max(0, num_chunks - seq)
    if not pad:
        for shape, call_mask in mlp.calls:
            assert (call_mask is None) if mask is None else (call_mask.shape == (2, shape[0]))
    else:
        # The dummy tokens are masked, also when the caller passed no mask.
        real = mask if mask is not None else torch.zeros(2, seq, dtype=torch.bool)
        expected = torch.cat([real, torch.ones(2, pad, dtype=torch.bool)], dim=1)
        assert torch.equal(torch.cat([call_mask for _, call_mask in mlp.calls], dim=1), expected)


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


class ReplayRouter(torch.nn.Module):
    """Router that goes through megatron-core's ``RouterReplay`` like ``TopKRouter`` does."""

    def __init__(self):
        super().__init__()
        from megatron.core.transformer.moe.router_replay import RouterReplay

        self.weight = torch.nn.Linear(H, E, bias=False)
        self.router_replay = RouterReplay()

    def forward(self, x):
        scores = torch.softmax(self.weight(x.reshape(-1, H)), dim=-1)

        def default(scores, topk, num_groups=None, group_topk=None):
            return torch.topk(scores, topk, dim=-1)

        return self.router_replay.get_replay_topk(scores, K, None, None, default)


class ReplayMoE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.router = ReplayRouter()
        self.seen = []

    def forward(self, x, padding_mask=None):
        _, idx = self.router(x)
        self.seen.append(idx.clone())
        return x * 1.0, None


def test_router_replay_pad_without_real_tokens():
    # Nothing to copy from an empty micro-batch: the dummies replay distinct experts 0..topk-1.
    padded = _ChunkedRouterReplay._pad(torch.empty(0, K, dtype=torch.long), 3)
    assert torch.equal(padded, torch.arange(K).expand(3, -1))


@pytest.mark.parametrize("seq", [30, 3])
@pytest.mark.parametrize("batch", [1, 2])
def test_chunked_router_replay(batch, seq):
    """R3: each chunk replays its own tokens' indices, forward and recompute; recording is whole."""
    from megatron.core.transformer.moe.router_replay import (
        RouterReplay,
        RouterReplayAction,
    )

    torch.manual_seed(0)
    num_chunks = 4
    x = torch.randn(seq, batch, H)
    mlp = ReplayMoE()
    real = seq * batch
    rr = mlp.router.router_replay
    target = torch.randint(0, E, (seq * batch, K))
    try:
        # Forward replay: the concatenated per-chunk indices are the micro-batch's target.
        rr.clear_indices()
        rr.set_target_indices(target)
        rr.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
        chunked_mlp(mlp, x, None, num_chunks)
        assert len(mlp.seen) == num_chunks
        assert torch.equal(torch.cat(mlp.seen)[:real], target)
        assert rr.target_topk_idx is target
        # Recompute (backward replay) pops the same micro-batch entry, split per chunk.
        mlp.seen.clear()
        rr.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)
        chunked_mlp(mlp, x, None, num_chunks)
        assert len(mlp.seen) == num_chunks
        assert torch.equal(torch.cat(mlp.seen)[:real], target)
        assert rr.replay_backward_list == []
        # Recording keeps every chunk, not just the last one.
        mlp.seen.clear()
        rr.clear_indices()
        rr.set_router_replay_action(RouterReplayAction.RECORD)
        chunked_mlp(mlp, x, None, num_chunks)
        assert torch.equal(rr.get_recorded_indices(), torch.cat(mlp.seen)[:real])
        assert rr.get_recorded_indices().shape[0] == real
    finally:
        rr.clear_indices()
        rr.clear_router_replay_action()
        RouterReplay.global_router_replay_instances.remove(rr)


class RecomputeReplayMoE(torch.nn.Module):
    """Like ``MoELayer`` under selective ``"moe"`` recompute: checkpoints its own call."""

    def __init__(self):
        super().__init__()
        self.router = ReplayRouter()
        self.moe_layer_recompute = True
        self.config = type("Config", (), {"fp8": None, "fp4": None})()
        self.calls = []

    def forward(self, x, padding_mask=None):
        def custom_forward(x):
            probs, idx = self.router(x)
            self.calls.append((x.detach().clone(), idx.clone()))
            return x * probs.sum(dim=-1).view(x.shape[0], x.shape[1], 1)

        if self.moe_layer_recompute and self.training:
            return torch.utils.checkpoint.checkpoint(custom_forward, x, use_reentrant=True), None
        return custom_forward(x), None


@pytest.fixture
def cpu_checkpoint(monkeypatch):
    """megatron-core's checkpoint saves CUDA RNG state; use torch's on CPU."""
    from skyrl.backends.skyrl_train.patches.megatron.mcore_ext import (
        mhc_transformer_layer,
    )

    monkeypatch.setattr(
        mhc_transformer_layer,
        "_checkpoint",
        lambda mlp, fn, chunk, mask: torch.utils.checkpoint.checkpoint(fn, chunk, mask, use_reentrant=True),
    )


@pytest.fixture
def replay_moe():
    from megatron.core.transformer.moe.router_replay import RouterReplay

    mlp = RecomputeReplayMoE().train()
    yield mlp
    rr = mlp.router.router_replay
    rr.clear_indices()
    rr.clear_router_replay_action()
    RouterReplay.global_router_replay_instances.remove(rr)


@pytest.mark.parametrize("seq", [30, 3])
@pytest.mark.parametrize("batch", [1, 2])
def test_chunked_router_replay_with_moe_recompute(cpu_checkpoint, replay_moe, batch, seq):
    """Selective "moe" recompute reruns each chunk's call on its own in backward, not the chunk
    loop; each rerun must still replay its own chunk's slice."""
    from megatron.core.transformer.moe.router_replay import RouterReplayAction

    torch.manual_seed(0)
    mlp, num_chunks, real = replay_moe, 4, seq * batch
    rr = mlp.router.router_replay
    target = torch.randint(0, E, (real, K))
    x = torch.randn(seq, batch, H, requires_grad=True)

    rr.set_target_indices(target)
    rr.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    out, _ = chunked_mlp(mlp, x, None, num_chunks)
    forward_calls = list(mlp.calls)
    assert len(forward_calls) == num_chunks
    assert torch.equal(torch.cat([idx for _, idx in forward_calls])[:real], target)
    assert rr.target_topk_idx is target and mlp.moe_layer_recompute
    # No rerun pops the micro-batch's entry, so it is dropped up front.
    assert rr.replay_backward_list == []

    mlp.calls.clear()
    rr.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)
    out.sum().backward()
    # One rerun per chunk (the MoE's own checkpoint is replaced, not nested), each on its slice.
    assert len(mlp.calls) == num_chunks
    expected = {tuple(xc.flatten().tolist()): idx for xc, idx in forward_calls}
    for xc, idx in mlp.calls:
        assert torch.equal(idx, expected[tuple(xc.flatten().tolist())])
    assert rr.router_replay_action == RouterReplayAction.REPLAY_BACKWARD
    assert x.grad is not None


def test_moe_recompute_keeps_earlier_micro_batches_entries(cpu_checkpoint, replay_moe):
    """Only this micro-batch's entry is dropped; earlier ones (pipeline 1F1B) stay queued."""
    from megatron.core.transformer.moe.router_replay import RouterReplayAction

    rr = replay_moe.router.router_replay
    earlier = torch.randint(0, E, (5, K))
    rr.set_target_indices(earlier)
    rr.set_target_indices(torch.randint(0, E, (8, K)))
    rr.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    chunked_mlp(replay_moe, torch.randn(8, 1, H, requires_grad=True), None, 4)
    assert len(rr.replay_backward_list) == 1 and rr.replay_backward_list[0] is earlier


@pytest.mark.parametrize("action", ["REPLAY_FORWARD", "REPLAY_BACKWARD"])
def test_replay_indices_must_match_the_token_count(replay_moe, action):
    """Extra or missing rows raise, as the unchunked gather does, instead of being sliced off."""
    from megatron.core.transformer.moe.router_replay import RouterReplayAction

    rr = replay_moe.router.router_replay
    rr.set_router_replay_action(RouterReplayAction[action])
    for rows in (10, 6):
        rr.clear_indices()
        rr.set_target_indices(torch.randint(0, E, (rows, K)))
        with pytest.raises(ValueError, match=f"have {rows} rows, but this MoE layer routes 8"):
            _ChunkedRouterReplay(replay_moe, [4, 4])
    # Dummy tokens are not counted: 6 real tokens padded with 2.
    rr.clear_indices()
    rr.set_target_indices(torch.randint(0, E, (6, K)))
    _ChunkedRouterReplay(replay_moe, [4, 4], pad_tokens=2)


def test_quantile_balancing_rejects_padded_chunks():
    """Dummy tokens need a padding mask, which quantile balancing does not support yet."""
    mlp = ToyMoE()
    mlp.router.routing_type = "quantile_balancing"
    with pytest.raises(NotImplementedError, match="quantile-balancing"):
        chunked_mlp(mlp, torch.randn(3, 2, H), None, 4)
    assert mlp.calls == []
    # A sequence of at least num_chunks tokens needs no padding and runs as before.
    chunked_mlp(mlp, torch.randn(8, 2, H), None, 4)
    assert len(mlp.calls) == 4
