"""CPU unit tests for HuggingFace MoE Rollout Routing Replay (R3).

Run with:
uv run --extra dev pytest tests/backends/skyrl_train/test_hf_router_replay.py
"""

import re
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers import OlmoeConfig, OlmoeForCausalLM

from skyrl.backends.skyrl_train.utils.hf_router_replay import (
    HFRouterReplayContext,
    align_replay_indices,
    install_router_replay_hooks,
)

NUM_LAYERS = 2
NUM_EXPERTS = 8
TOPK = 2
VOCAB_SIZE = 100


def _make_tiny_olmoe(norm_topk_prob: bool = False) -> OlmoeForCausalLM:
    config = OlmoeConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOPK,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=64,
        norm_topk_prob=norm_topk_prob,
    )
    return OlmoeForCausalLM(config).eval()


def _capture_dispatched_experts(model: nn.Module) -> dict[int, torch.Tensor]:
    """Record the ``top_k_index`` each MoE block hands to its experts, keyed by
    global transformer-layer index."""
    captured: dict[int, torch.Tensor] = {}

    def make_pre_hook(layer_idx: int):
        def pre_hook(module: nn.Module, args: tuple) -> None:
            captured[layer_idx] = args[1].detach().clone()

        return pre_hook

    for name, module in model.named_modules():
        if name.endswith("mlp.experts"):
            layer_idx = int(re.search(r"layers\.(\d+)\.", name).group(1))
            module.register_forward_pre_hook(make_pre_hook(layer_idx))
    return captured


def _random_replay_indices(batch: int, seq: int) -> torch.Tensor:
    return torch.randint(0, NUM_EXPERTS, (batch, seq, NUM_LAYERS, TOPK), dtype=torch.int16)


def test_install_router_replay_hooks_counts() -> None:
    model = _make_tiny_olmoe()
    ctx = HFRouterReplayContext()
    handles = install_router_replay_hooks(model, ctx)
    assert len(handles) == NUM_LAYERS


@pytest.mark.parametrize(
    "norm_topk_prob",
    [pytest.param(False, id="no_norm"), pytest.param(True, id="norm")],
)
def test_dispatched_experts_match_replay(norm_topk_prob: bool) -> None:
    torch.manual_seed(0)
    model = _make_tiny_olmoe(norm_topk_prob=norm_topk_prob)
    ctx = HFRouterReplayContext()
    install_router_replay_hooks(model, ctx)
    captured = _capture_dispatched_experts(model)

    batch, seq = 2, 5
    input_ids = torch.randint(0, VOCAB_SIZE, (batch, seq))
    per_layer = align_replay_indices(
        _random_replay_indices(batch, seq), num_layers=NUM_LAYERS, nnz_indices=None, sp_size=1
    )
    ctx.set(per_layer)
    with torch.no_grad():
        model(input_ids)

    for layer_idx in range(NUM_LAYERS):
        assert torch.equal(captured[layer_idx], per_layer[layer_idx])


def test_inactive_context_is_noop() -> None:
    torch.manual_seed(0)
    model = _make_tiny_olmoe()
    ctx = HFRouterReplayContext()
    install_router_replay_hooks(model, ctx)
    captured = _capture_dispatched_experts(model)

    batch, seq = 2, 5
    input_ids = torch.randint(0, VOCAB_SIZE, (batch, seq))
    per_layer = align_replay_indices(
        _random_replay_indices(batch, seq), num_layers=NUM_LAYERS, nnz_indices=None, sp_size=1
    )
    # Context never armed -> the hook is a no-op and routing is the model's own.
    with torch.no_grad():
        model(input_ids)
    assert not torch.equal(captured[0], per_layer[0])


def test_replaying_natural_selection_is_faithful() -> None:
    """Feeding the model its OWN top-k selection back reproduces both the
    dispatched experts and the logits, proving the fp32 score recompute matches
    the router's native math."""
    torch.manual_seed(0)
    model = _make_tiny_olmoe()
    ctx = HFRouterReplayContext()
    install_router_replay_hooks(model, ctx)
    captured = _capture_dispatched_experts(model)

    batch, seq = 2, 5
    input_ids = torch.randint(0, VOCAB_SIZE, (batch, seq))
    with torch.no_grad():
        natural_logits = model(input_ids).logits
    natural_selection = [captured[layer_idx].clone() for layer_idx in range(NUM_LAYERS)]

    ctx.set(natural_selection)
    with torch.no_grad():
        replayed_logits = model(input_ids).logits

    for layer_idx in range(NUM_LAYERS):
        assert torch.equal(captured[layer_idx], natural_selection[layer_idx])
    torch.testing.assert_close(replayed_logits, natural_logits)


def test_gradient_flows_through_gate() -> None:
    torch.manual_seed(0)
    model = _make_tiny_olmoe()
    model.train()
    ctx = HFRouterReplayContext()
    install_router_replay_hooks(model, ctx)

    batch, seq = 2, 5
    input_ids = torch.randint(0, VOCAB_SIZE, (batch, seq))
    per_layer = align_replay_indices(
        _random_replay_indices(batch, seq), num_layers=NUM_LAYERS, nnz_indices=None, sp_size=1
    )
    ctx.set(per_layer)
    model(input_ids).logits.sum().backward()

    gate_grads = [p.grad for name, p in model.named_parameters() if name.endswith("mlp.gate.weight")]
    assert len(gate_grads) == NUM_LAYERS
    for grad in gate_grads:
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_align_replay_indices_packed() -> None:
    """The packing path gathers with ``nnz_indices`` (dropping pad tokens) and
    yields long tensors in the router's flattened token order."""
    batch, seq = 2, 4
    rii = _random_replay_indices(batch, seq)
    # Drop one pad token per sequence: keep flat indices 0,1,2 and 4,5,6 (pad 3,7).
    nnz_indices = torch.tensor([0, 1, 2, 4, 5, 6], dtype=torch.long)

    per_layer = align_replay_indices(rii, num_layers=NUM_LAYERS, nnz_indices=nnz_indices, sp_size=1)

    flat = rii.reshape(-1, NUM_LAYERS, TOPK)[nnz_indices]
    for layer_idx in range(NUM_LAYERS):
        assert per_layer[layer_idx].dtype == torch.int64
        assert per_layer[layer_idx].shape == (nnz_indices.numel(), TOPK)
        assert torch.equal(per_layer[layer_idx], flat[:, layer_idx, :].long())


@pytest.mark.parametrize(
    ("rollout_expert_indices", "num_layers", "match"),
    [
        pytest.param(_random_replay_indices(2, 4), NUM_LAYERS + 1, "layers", id="layer_count_mismatch"),
        pytest.param(_random_replay_indices(2, 4)[..., 0], NUM_LAYERS, "4D", id="non_4d"),
    ],
)
def test_align_replay_indices_rejects_malformed(
    rollout_expert_indices: torch.Tensor, num_layers: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        align_replay_indices(rollout_expert_indices, num_layers=num_layers, nnz_indices=None, sp_size=1)


def test_install_raises_without_router() -> None:
    dense = nn.Sequential(nn.Linear(4, 4))
    with pytest.raises(NotImplementedError, match="no compatible softmax"):
        install_router_replay_hooks(dense, HFRouterReplayContext())


def test_router_without_norm_topk_prob_attr() -> None:
    """Qwen3.5-MoE's router is softmax top-k and always normalizes, but drops the
    ``norm_topk_prob`` attribute; the hook must still match it and normalize."""
    qwen3_5 = pytest.importorskip("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
    config = SimpleNamespace(num_experts_per_tok=TOPK, num_experts=NUM_EXPERTS, hidden_size=64)
    router = qwen3_5.Qwen3_5MoeTopKRouter(config)
    assert not hasattr(router, "norm_topk_prob"), "test premise: Qwen3.5 router omits the attr"

    layer = nn.Module()
    layer.mlp = nn.Module()
    layer.mlp.gate = router
    model = nn.Module()
    model.layers = nn.ModuleList([layer])

    ctx = HFRouterReplayContext()
    assert len(install_router_replay_hooks(model, ctx)) == 1  # matched despite no norm_topk_prob

    tokens = 5
    replayed = torch.randint(0, NUM_EXPERTS, (tokens, TOPK))
    ctx.set([replayed])
    _, scores, idx = router(torch.randn(tokens, 64))
    assert torch.equal(idx, replayed)
    torch.testing.assert_close(scores.float().sum(dim=-1), torch.ones(tokens))
