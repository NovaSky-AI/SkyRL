"""Routing replay (R3) through the looped Inkling2 block.

The inference engine reports one routing per layer run, in execution order (prelude, core x passes, coda);
``replay_utils`` groups them per router (``[passes, tokens, topk]``) and the loop block replays pass ``p``'s slice in
the layer's ``p``-th run, also in activation-recompute reruns. With top-2 of 8 experts (so routing matters):

- replaying the model's own routing (recorded per router call) reproduces its logits exactly;
- swapping two passes' routings changes them (the slices are consumed per pass, not in call order);
- a train step under replay with full recompute (one checkpoint per layer run) gives the gradients of the step
  without recompute (up to their sensitivity to rounding), and leaves no entry in Megatron's recompute FIFO.

Run with:
uv run --isolated --extra dev --extra megatron pytest -s tests/backends/skyrl_train/gpu/gpu_ci/patches/megatron/inkling2/test_loop_replay.py
"""

import pytest
import ray
import torch

pytestmark = pytest.mark.megatron


@ray.remote(num_gpus=1)
def _replay():
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.moe.router_replay import (
        RouterReplay,
        RouterReplayAction,
    )

    import skyrl.backends.skyrl_train.workers.megatron  # noqa: F401  (FA4 import guard)
    from skyrl.backends.skyrl_train.patches.megatron.inkling2.modules import (
        layer_runs,
    )
    from skyrl.backends.skyrl_train.utils.replay_utils import (
        _group_runs_by_layer,
        clear_router_replay,
    )
    from tests.backends.skyrl_train.gpu.gpu_ci.patches.megatron.inkling2.test_model_vs_reference import (
        UNPADDED_V,
        _build_megatron,
        _init_single_rank_megatron,
        _rounding_sensitivity,
        grad_rel,
    )

    _init_single_rank_megatron()
    torch.manual_seed(0)
    model = _build_megatron("flex", looped=True, topk=2, moe_enable_routing_replay=True, moe_router_fusion=False)
    cfg = model.config
    runs = layer_runs(cfg)
    lens = [23, 41]
    ids = torch.cat([torch.randint(0, UNPADDED_V, (n,), device="cuda") for n in lens]).view(1, -1)
    cu = torch.tensor([0, lens[0], sum(lens)], dtype=torch.int32, device="cuda")
    psp = PackedSeqParams(
        qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=max(lens), max_seqlen_kv=max(lens)
    )

    # The model's own routing, per router call (= per layer run, in execution order).
    calls = []
    hooks = [
        layer.mlp.router.register_forward_hook(lambda m, a, o: calls.append(o[1].nonzero()[:, 1].view(-1, 2)))
        for layer in model.decoder.layers
    ]
    model.eval()
    with torch.no_grad():
        natural = model(ids, None, None, packed_seq_params=psp).float()
    for h in hooks:
        h.remove()
    stats = {"calls": len(calls), "runs": len(runs)}

    def replay(per_run, action=RouterReplayAction.REPLAY_FORWARD):
        clear_router_replay()
        RouterReplay.set_replay_data(_group_runs_by_layer([t.to(torch.int32) for t in per_run], runs))
        RouterReplay.set_global_router_replay_action(action)

    with torch.no_grad():
        replay(calls)
        replayed = model(ids, None, None, packed_seq_params=psp).float()
        stats["own_routing_max_diff"] = (replayed - natural).abs().max().item()
        swapped = list(calls)
        p1 = [r for r, (i, p) in enumerate(runs) if p == 1]
        p2 = [r for r, (i, p) in enumerate(runs) if p == 2]
        for a, b in zip(p1, p2):
            swapped[a], swapped[b] = calls[b], calls[a]
        replay(swapped)
        stats["swapped_passes_max_diff"] = (model(ids, None, None, packed_seq_params=psp).float() - natural).abs().max().item()

    def train_step(recompute: bool):
        model.train()
        model.zero_grad(set_to_none=True)
        cfg.recompute_granularity = "full" if recompute else None
        cfg.recompute_method, cfg.recompute_num_layers = "uniform", 1
        replay(calls)
        logits = model(ids, None, None, packed_seq_params=psp)
        loss = torch.log_softmax(logits.float(), -1)[0, :-1].gather(-1, ids[0, 1:, None]).sum()
        RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)
        loss.backward()
        leftover = sum(len(r.replay_backward_list) for r in RouterReplay.global_router_replay_instances)
        clear_router_replay()
        cfg.recompute_granularity = None
        return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}, leftover

    plain, stats["leftover_plain"] = train_step(False)
    stats["rounding_sensitivity"] = _rounding_sensitivity(cfg, lambda: train_step(False)[0], plain)
    recomputed, stats["leftover_recompute"] = train_step(True)
    stats["recompute_rel"] = grad_rel(recomputed, plain)
    stats["same_params"] = set(recomputed) == set(plain)
    return stats


def test_looped_routing_replay(ray_init_fixture):
    stats = ray.get(_replay.remote())
    print(stats)
    assert stats["calls"] == stats["runs"]
    assert stats["own_routing_max_diff"] == 0.0
    assert stats["swapped_passes_max_diff"] > 0.1
    assert stats["leftover_plain"] == 0 and stats["leftover_recompute"] == 0
    assert stats["same_params"]
    # Replaying a wrong pass's slice in a rerun moves the gradients by O(1); see _rounding_sensitivity for the rest.
    assert stats["recompute_rel"] < 2 * stats["rounding_sensitivity"] + 1e-3
