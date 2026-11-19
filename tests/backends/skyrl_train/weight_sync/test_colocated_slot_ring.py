"""CPU tests for the colocated slot-ring packer and local-shard plan (weight_sync/colocated/slot_ring.py)."""

import pytest
import torch

from skyrl.backends.skyrl_train.weight_sync.colocated.slot_ring import (
    LocalShardPlan,
    build_local_shard_plan,
    expert_id,
    iter_packed_chunks,
    layer_key,
)


def _stream(num_layers=3, num_experts=4, numel=8):
    for layer in range(num_layers):
        yield f"model.layers.{layer}.self_attn.q_proj.weight", torch.full((numel,), layer, dtype=torch.bfloat16)
        for e in range(num_experts):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                yield (
                    f"model.layers.{layer}.mlp.experts.{e}.{proj}.weight",
                    torch.full((numel,), 100 * layer + e, dtype=torch.bfloat16),
                )
    yield "model.norm.weight", torch.ones(numel, dtype=torch.bfloat16)
    yield "lm_head.weight", torch.ones(numel, dtype=torch.bfloat16)


def _run(rank, world, *, cap, n=2, layer_aligned=False, plan=None):
    buffers = [torch.zeros(cap, dtype=torch.uint8) for _ in range(n)]
    out = []
    for chunk in iter_packed_chunks(
        _stream(), buffers, rank=rank, world_size=world, layer_aligned=layer_aligned, plan=plan
    ):
        # Unpack and check bytes against names before the slot is reused.
        off = 0
        tensors = []
        for name, size, shape, dt in zip(chunk.names, chunk.tensor_sizes, chunk.shapes, chunk.dtype_names):
            t = buffers[chunk.slot][off : off + size].view(getattr(torch, dt)).view(shape).clone()
            tensors.append((name, t))
            off += size
        out.append((chunk.slot, tensors))
    return out


def test_helpers():
    assert layer_key("model.layers.12.mlp.experts.3.up_proj.weight") == 12
    assert layer_key("lm_head.weight") is None
    assert expert_id("model.layers.12.mlp.experts.3.up_proj.weight") == 3
    assert expert_id("model.layers.12.mlp.shared_expert.up_proj.weight") is None


@pytest.mark.parametrize("layer_aligned", [False, True])
def test_full_stream_round_trips_bytes(layer_aligned):
    chunks = _run(0, 1, cap=16 * 6, layer_aligned=layer_aligned)
    got = [x for _, ts in chunks for x in ts]
    want = list(_stream())
    assert [n for n, _ in got] == [n for n, _ in want]
    for (_, a), (_, b) in zip(got, want):
        assert torch.equal(a, b)
    # Slots alternate.
    assert [s for s, _ in chunks] == [i % 2 for i in range(len(chunks))]
    if layer_aligned:
        for _, ts in chunks:
            assert len({layer_key(n) for n, _ in ts}) == 1


def test_layer_aligned_cuts_at_layer_boundaries_even_with_room():
    chunks = _run(0, 1, cap=1 << 20, layer_aligned=True)
    # 3 layers + one non-layer chunk (norm, lm_head).
    assert len(chunks) == 4


def test_local_shard_chunks_agree_across_ranks_and_filter_experts():
    world = 4
    # Two experts per worker; ranks 0/1 serve {0,1}, ranks 2/3 serve {2,3}.
    plan = LocalShardPlan(
        rank_experts=[frozenset({0, 1}), frozenset({0, 1}), frozenset({2, 3}), frozenset({2, 3})],
        rank_worker=[(0, 0), (0, 1), (1, 0), (1, 1)],
        server_worker_indices=[[0, 1], [0, 1]],
    )
    per_rank = [_run(r, world, cap=16 * 4, plan=plan) for r in range(world)]
    counts = {len(c) for c in per_rank}
    assert len(counts) == 1, f"chunk counts differ across ranks: {counts}"
    for r, chunks in enumerate(per_rank):
        names = [n for _, ts in chunks for n, _ in ts]
        experts = {expert_id(n) for n in names} - {None}
        assert experts == set(plan.rank_experts[r])
        # Non-expert tensors go to everyone.
        assert "lm_head.weight" in names and "model.layers.0.self_attn.q_proj.weight" in names
        want = {n: t for n, t in _stream() if plan.wants(r, n)}
        got = {n: t for _, ts in chunks for n, t in ts}
        assert got.keys() == want.keys()
        assert all(torch.equal(got[n], want[n]) for n in want)
    # Fewer chunks than the full stream needs: local bytes are smaller.
    full = _run(0, 1, cap=16 * 4)
    assert len(per_rank[0]) < len(full)


def test_oversized_tensor_raises():
    with pytest.raises(ValueError, match="larger than"):
        _run(0, 1, cap=8)


def test_build_local_shard_plan():
    infos = [
        [
            {"gpu_uuid": "g0", "worker_index": 0, "local_experts": [0, 1], "consistent": True},
            {"gpu_uuid": "g1", "worker_index": 1, "local_experts": [2, 3], "consistent": True},
        ]
    ]
    plan, reason = build_local_shard_plan(infos, ["g1", "g0"])
    assert plan is not None, reason
    assert plan.rank_worker == [(0, 1), (0, 0)]
    assert plan.rank_experts == [frozenset({2, 3}), frozenset({0, 1})]
    # No EP on vLLM: nothing to filter.
    no_ep = [[dict(i, local_experts=None) for i in infos[0]]]
    assert build_local_shard_plan(no_ep, ["g0", "g1"])[0] is None
    # A GPU without a worker.
    assert build_local_shard_plan(infos, ["g0", "g9"])[0] is None
    bad = [[dict(infos[0][0], consistent=False), infos[0][1]]]
    assert build_local_shard_plan(bad, ["g0", "g1"])[0] is None
