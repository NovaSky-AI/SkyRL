"""R3 (rollout routing replay) for ML4 on the FSDP path, CPU tiny model.

Routes arrive as SkyRL's trainer builds them: packed over each row's real tokens,
``[sum(real), num_hidden_layers, top_k]`` indexed by global layer number (layer 0 is
dense: all-zero rows, never read), plus ``router_padding_mask`` for tokens without a
captured route.
"""

import pytest
import torch

P, R = 8, 6
ROWS = [(8, 6), (5, 6), (3, 2)]  # (prompt len, response len), whole block left-padded


def _batch(vocab, num_layers, topk, num_experts, captured, seed=0):
    from skyrl.backends.skyrl_train.utils.packed_tensor import PackedTensor
    from skyrl.train.dataset.preprocess import make_router_padding_mask

    g = torch.Generator().manual_seed(seed)
    S = P + R
    seqs = torch.zeros(len(ROWS), S, dtype=torch.long)
    mask = torch.zeros_like(seqs)
    loss_mask = torch.zeros(len(ROWS), R)
    segments = []
    for i, (p, r) in enumerate(ROWS):
        seqs[i, S - p - r :] = torch.randint(1, vocab, (p + r,), generator=g)
        mask[i, S - p - r :] = 1
        loss_mask[i, R - r :] = 1
        routes = torch.zeros(p + r, num_layers, topk, dtype=torch.uint8)
        for t in range(p + r):
            for layer in range(1, num_layers):
                routes[t, layer] = torch.randperm(num_experts, generator=g)[:topk].to(torch.uint8)
        segments.append(routes)
    routes = PackedTensor.from_segments(segments)
    padding = make_router_padding_mask(mask, captured)
    return seqs, mask, loss_mask, routes, padding


@pytest.fixture
def setup(ml4_model_dir):
    from conftest import load_ml4_wrapper
    from fake_ckpt import TINY_PARAMS

    wrapper, _ = load_ml4_wrapper(ml4_model_dir)
    # The wrapper loads the model through auto_map, a separate copy of ml4_train's classes.
    ML4MoE = type(next(iter(_layers(wrapper).values())))
    wrapper.eval()
    cfg = wrapper.model.config
    seen = {}

    def spy(self, x, *args):
        w, ids = orig(self, x, *args)
        seen[self] = (x.detach(), w.detach(), ids.detach())
        return w, ids

    orig = ML4MoE.route
    ML4MoE.route = spy
    try:
        yield wrapper, cfg, TINY_PARAMS["vocab_size"], seen, ML4MoE
    finally:
        ML4MoE.route = orig


def _layers(wrapper):
    from skyrl.backends.skyrl_train.utils.fsdp_router_replay import find_replay_routers

    return find_replay_routers(wrapper.model)


def test_replay_forces_rollout_experts(setup):
    wrapper, cfg, vocab, seen, _ = setup
    real_lens = [p + r for p, r in ROWS]
    captured = [real_lens[0], real_lens[1] - 1, real_lens[2] - 1]  # last sampled token: no route
    seqs, mask, loss_mask, routes, padding = _batch(
        vocab, cfg.num_hidden_layers, cfg.num_experts_per_tok, cfg.n_routed_experts, captured
    )
    routers = _layers(wrapper)
    assert sorted(routers) == list(range(1, cfg.num_hidden_layers))  # layer 0 is dense

    with torch.no_grad():
        base = wrapper(seqs, R, mask)
        got = wrapper(
            seqs,
            R,
            mask,
            loss_mask=loss_mask,
            rollout_expert_indices=routes,
            router_padding_mask=padding,
            enable_router_replay=True,
        )
    assert not torch.allclose(got, base)

    real = mask.bool()
    replayed = real & ~padding
    for layer, module in routers.items():
        ids = seen[module][2]
        ids = ids.view(*mask.shape, -1)
        dense = torch.full_like(ids, -1)
        dense[real] = routes.values[:, layer].long()
        # Every captured real token (prompt included) uses the rollout's experts in order.
        assert torch.equal(ids[replayed], dense[replayed])
        # Pads and uncaptured tokens keep the router's own top-k.
        native = module.replay_native_topk_ids.view(*mask.shape, -1)
        assert torch.equal(ids[~replayed], native[~replayed])

    metrics = wrapper.last_router_replay_metrics
    assert metrics["router_replay/row_coverage"] == pytest.approx(replayed.sum().item() / real.sum().item())
    assert 0.0 < metrics["router_replay/native_mismatch_frac"] <= 1.0

    with torch.no_grad():  # a forward without replay clears it
        again = wrapper(seqs, R, mask)
    torch.testing.assert_close(again, base)
    assert all(m.replay_topk_ids is None for m in routers.values())


def test_weights_match_megatron_replay(setup):
    """Megatron REPLAY_FORWARD: ids replayed, probs = router scores gathered at them, then the
    model's own normalization: for ML4 (softmax, norm_topk_prob) softmax over the gathered
    BF16-rounded logits, times routed_scaling_factor. Gradients reach the gate."""
    _, cfg, _, _, ML4MoE = setup
    moe = ML4MoE(cfg).float()
    torch.nn.init.normal_(moe.gate.weight, std=0.5)
    x = torch.randn(10, cfg.hidden_size)
    replay = torch.stack([torch.randperm(cfg.n_routed_experts)[: cfg.num_experts_per_tok] for _ in range(10)])
    replay[3] = -1  # not replayed: own top-k
    moe.replay_topk_ids = replay
    w, ids = moe.route(x)

    # Router logits as vLLM computes them : FP32 GEMM rounded to BF16; the replay
    # weights are recomputed from these trainer logits. Own top-k: ties to the lower expert index.
    logits = (x @ moe.gate.weight.T).to(torch.bfloat16).float()
    own = logits.sort(dim=-1, descending=True, stable=True).indices[:, : cfg.num_experts_per_tok]
    expect_ids = torch.where(replay >= 0, replay, own)
    assert torch.equal(ids, expect_ids)
    expect_w = torch.softmax(logits.gather(-1, expect_ids), -1) * cfg.routed_scaling_factor
    torch.testing.assert_close(w, expect_w)

    (w * torch.randn_like(w)).sum().backward()
    assert moe.gate.weight.grad is not None and moe.gate.weight.grad.abs().sum() > 0


def test_missing_required_row_raises(setup):
    wrapper, cfg, vocab, _, _ = setup
    real_lens = [p + r for p, r in ROWS]
    seqs, mask, loss_mask, routes, padding = _batch(
        vocab, cfg.num_hidden_layers, cfg.num_experts_per_tok, cfg.n_routed_experts, real_lens
    )
    padding[1, P + 2] = True  # a response token that later loss-active targets attend to
    with pytest.raises(ValueError, match="no rollout route"):
        wrapper(
            seqs,
            R,
            mask,
            loss_mask=loss_mask,
            rollout_expert_indices=routes,
            router_padding_mask=padding,
            enable_router_replay=True,
        )


def test_duplicate_expert_row_raises(setup):
    wrapper, cfg, vocab, _, _ = setup
    real_lens = [p + r for p, r in ROWS]
    seqs, mask, loss_mask, routes, padding = _batch(
        vocab, cfg.num_hidden_layers, cfg.num_experts_per_tok, cfg.n_routed_experts, real_lens
    )
    routes.values[4, 3] = 0  # layer 3 not recorded for this token: all zeros (0 is a valid id)
    with pytest.raises(ValueError, match="repeat an expert id"):
        wrapper(
            seqs,
            R,
            mask,
            loss_mask=loss_mask,
            rollout_expert_indices=routes,
            router_padding_mask=padding,
            enable_router_replay=True,
        )


def test_missing_routes_raise_when_enabled(setup):
    wrapper, _, vocab, _, _ = setup
    seqs = torch.randint(1, vocab, (1, 6))
    with pytest.raises(ValueError, match="no rollout expert indices"):
        wrapper(seqs, 2, torch.ones_like(seqs), enable_router_replay=True)
