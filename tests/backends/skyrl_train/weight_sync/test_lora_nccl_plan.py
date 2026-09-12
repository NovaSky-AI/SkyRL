import hashlib

import pytest
import torch

from skyrl.backends.skyrl_train.weight_sync.lora_nccl import (
    LoRANcclConsumerRoute,
    build_lora_nccl_plan,
    build_lora_nccl_plan_receipt,
    pack_lora_nccl_bucket,
    unpack_lora_nccl_bucket,
)
from skyrl.backends.skyrl_train.weight_sync.lora_transport.consumer_plan import (
    LoRAConsumerPlan,
    LoRAConsumerPull,
)
from skyrl.backends.skyrl_train.weight_sync.lora_transport.contracts import (
    LoRASourceSlice,
)

LAYOUT_DIGEST = hashlib.sha256(b"fixed-layout").hexdigest()


def _consumer_plan(*pulls):
    return LoRAConsumerPlan(
        source_layout_digest=LAYOUT_DIGEST,
        receiver_plan=object(),
        pulls=tuple(pulls),
        copies=(),
    )


def _pull(source_rank, key, starts, stops):
    return LoRAConsumerPull(
        source_rank,
        LoRASourceSlice(key, tuple(starts), tuple(stops)),
    )


def test_plan_routes_only_requested_slices_and_has_stable_metadata():
    rank0 = _consumer_plan(
        _pull(0, "q.lora_A", (0, 0), (2, 4)),
        _pull(1, "expert.lora_B", (0, 0), (2, 2)),
    )
    rank1 = _consumer_plan(_pull(1, "q.lora_A", (2, 0), (4, 4)))

    plan = build_lora_nccl_plan({1: rank1, 0: rank0}, 32)
    repeated = build_lora_nccl_plan({0: rank0, 1: rank1}, 32)

    assert plan.plan_digest == repeated.plan_digest
    assert plan.inference_ranks == (0, 1)
    assert [(bucket.source_rank, bucket.inference_rank, bucket.source_bytes) for bucket in plan.buckets] == [
        (0, 0, 32),
        (1, 0, 16),
        (1, 1, 32),
    ]
    assert [(group.source_rank, group.inference_ranks) for group in plan.source_groups] == [
        (0, (0,)),
        (1, (0, 1)),
    ]
    assert plan.source_bytes == rank0.source_bytes + rank1.source_bytes


def test_plan_splits_routes_without_broadcasting_to_other_consumers():
    pulls = (
        _pull(0, "a", (0, 0), (2, 2)),
        _pull(0, "b", (0, 0), (2, 2)),
        _pull(0, "c", (0, 0), (2, 2)),
    )
    plan = build_lora_nccl_plan({3: _consumer_plan(*pulls)}, 32)

    assert [len(bucket.pulls) for bucket in plan.buckets] == [2, 1]
    assert {bucket.inference_rank for bucket in plan.buckets} == {3}
    assert all(bucket.source_rank == 0 for bucket in plan.buckets)


def test_plan_digest_and_buckets_do_not_depend_on_pull_order():
    pulls = (
        _pull(0, "c", (0, 0), (2, 2)),
        _pull(0, "a", (0, 0), (2, 2)),
        _pull(0, "b", (0, 0), (2, 2)),
    )

    plan = build_lora_nccl_plan({0: _consumer_plan(*pulls)}, 32)
    reordered = build_lora_nccl_plan({0: _consumer_plan(*reversed(pulls))}, 32)

    assert plan.plan_digest == reordered.plan_digest
    assert plan.buckets == reordered.buckets
    assert [pull.source_slice.key for pull in plan.buckets[0].pulls] == ["a", "b"]
    assert [pull.source_slice.key for pull in plan.buckets[1].pulls] == ["c"]


def test_pack_and_unpack_preserve_exact_fp32_slices():
    pull_a = _pull(2, "a", (1, 1), (3, 4))
    pull_b = _pull(2, "b", (0,), (3,))
    plan = build_lora_nccl_plan(
        {0: _consumer_plan(pull_a, pull_b)},
        64,
    )
    sources = {
        "a": torch.arange(20, dtype=torch.float32).view(4, 5),
        "b": torch.tensor([-0.0, float("nan"), 7.0], dtype=torch.float32),
    }

    packed = pack_lora_nccl_bucket(plan.buckets[0], sources)
    unpacked = unpack_lora_nccl_bucket(plan.buckets[0], packed)

    assert torch.equal(unpacked[pull_a], sources["a"][1:3, 1:4])
    assert torch.equal(
        unpacked[pull_b].view(torch.int32),
        sources["b"].view(torch.int32),
    )
    destination = unpacked[pull_a].to(torch.bfloat16)
    packed.add_(100)
    assert torch.equal(
        destination,
        sources["a"][1:3, 1:4].to(torch.bfloat16),
    )


@pytest.mark.parametrize(
    "plans,buffer_size,error",
    [
        ({}, 32, "at least one consumer"),
        ({0: _consumer_plan(_pull(0, "a", (0,), (1,)))}, 0, "must be positive"),
        (
            {
                0: _consumer_plan(_pull(0, "a", (0,), (1,))),
                1: LoRAConsumerPlan(
                    source_layout_digest=hashlib.sha256(b"other").hexdigest(),
                    receiver_plan=object(),
                    pulls=(_pull(0, "a", (0,), (1,)),),
                    copies=(),
                ),
            },
            32,
            "one source layout",
        ),
        (
            {0: _consumer_plan(_pull(0, "a", (0, 0), (4, 4)))},
            32,
            "exceeding buffer size",
        ),
    ],
)
def test_plan_rejects_unsupported_or_ambiguous_inputs(plans, buffer_size, error):
    with pytest.raises(ValueError, match=error):
        build_lora_nccl_plan(plans, buffer_size)


def test_plan_receipt_counts_replication_and_edges():
    shared = _pull(0, "shared", (0, 0), (2, 4))
    plan = build_lora_nccl_plan(
        {
            0: _consumer_plan(shared),
            1: _consumer_plan(shared),
        },
        64,
    )

    receipt = build_lora_nccl_plan_receipt(plan)

    assert receipt.source_group_count == 1
    assert receipt.edge_count == 2
    assert receipt.bucket_count == 2
    assert receipt.pull_count == 2
    assert receipt.unique_pull_count == 1
    assert receipt.unique_source_bytes == 32
    assert receipt.transmitted_bytes == 64
    assert receipt.replication_bytes == 32
    assert [(edge.source_rank, edge.inference_rank, edge.transmitted_bytes) for edge in receipt.edges] == [
        (0, 0, 32),
        (0, 1, 32),
    ]


def test_plan_receipt_rejects_ambiguous_partial_overlap():
    plan = build_lora_nccl_plan(
        {
            0: _consumer_plan(_pull(0, "shared", (0, 0), (2, 4))),
            1: _consumer_plan(_pull(0, "shared", (1, 0), (3, 4))),
        },
        64,
    )

    with pytest.raises(ValueError, match="partially overlapping"):
        build_lora_nccl_plan_receipt(plan)


def test_plan_receipt_scales_across_independent_source_keys():
    pulls = tuple(_pull(0, f"source_{index:05d}", (0,), (1,)) for index in range(4096))
    plan = build_lora_nccl_plan({0: _consumer_plan(*pulls)}, 4096 * 4)

    receipt = build_lora_nccl_plan_receipt(plan)

    assert receipt.pull_count == 4096
    assert receipt.unique_pull_count == 4096
    assert receipt.unique_source_bytes == 4096 * 4
    assert receipt.replication_bytes == 0


def test_consumer_route_round_trips_into_global_plan():
    consumer_plan = _consumer_plan(
        _pull(1, "z", (0,), (2,)),
        _pull(0, "a", (0,), (1,)),
    )

    route = LoRANcclConsumerRoute.from_consumer_plan(3, consumer_plan)
    restored = LoRANcclConsumerRoute.from_json_dict(route.to_json_dict())
    plan = build_lora_nccl_plan({3: restored}, 32)

    assert restored == route
    assert [pull.source_slice.key for pull in restored.pulls] == ["a", "z"]
    assert plan.inference_ranks == (3,)


def test_consumer_route_rejects_mapping_rank_mismatch():
    route = LoRANcclConsumerRoute.from_consumer_plan(
        3,
        _consumer_plan(_pull(0, "a", (0,), (1,))),
    )

    with pytest.raises(ValueError, match="mapping key"):
        build_lora_nccl_plan({2: route}, 32)
