import hashlib

import pytest

from skyrl.backends.skyrl_train.weight_sync import lora_nccl
from skyrl.backends.skyrl_train.weight_sync.lora_nccl import (
    LoRANcclConsumerRoute,
    LoRANcclRendezvous,
    build_lora_nccl_plan,
    open_lora_nccl_receiver_session,
    open_lora_nccl_source_session,
)
from skyrl.backends.skyrl_train.weight_sync.lora_transport.consumer_plan import (
    LoRAConsumerPlan,
    LoRAConsumerPull,
)
from skyrl.backends.skyrl_train.weight_sync.lora_transport.contracts import (
    LoRASourceSlice,
)

LAYOUT_DIGEST = hashlib.sha256(b"fixed-layout").hexdigest()


class _Communicator:
    def __init__(self):
        self.destroy_count = 0

    def send(self, tensor, dst, stream=None):
        raise NotImplementedError

    def recv(self, tensor, src, stream=None):
        raise NotImplementedError

    def destroy(self):
        self.destroy_count += 1


def _plan():
    rank0 = LoRANcclConsumerRoute(
        0,
        LAYOUT_DIGEST,
        (
            LoRAConsumerPull(0, LoRASourceSlice("a", (0,), (2,))),
            LoRAConsumerPull(2, LoRASourceSlice("c", (0,), (1,))),
        ),
    )
    rank2 = LoRANcclConsumerRoute(
        2,
        LAYOUT_DIGEST,
        (LoRAConsumerPull(0, LoRASourceSlice("b", (0,), (3,))),),
    )
    return build_lora_nccl_plan({0: rank0, 2: rank2}, 32)


def _rendezvous():
    return LoRANcclRendezvous.from_plan(
        _plan(),
        "adapter",
        "10.0.0.4",
        41000,
    )


def test_rendezvous_round_trip_preserves_sparse_shared_peer_mapping():
    rendezvous = _rendezvous()
    restored = LoRANcclRendezvous.from_json_dict(rendezvous.to_json_dict())

    assert restored == rendezvous
    assert restored.source_ranks == (0, 2)
    assert restored.inference_ranks == (0, 2)
    assert restored.world_size == 4
    assert restored.get_source_peer_rank(0) == 0
    assert restored.get_source_peer_rank(2) == 1
    assert restored.get_inference_peer_rank(0) == 2
    assert restored.get_inference_peer_rank(2) == 3


def test_source_and_receiver_join_one_shared_static_group():
    plan = _plan()
    rendezvous = _rendezvous()
    calls = []

    def factory(address, port, rank, world_size, device):
        calls.append((address, port, rank, world_size, str(device)))
        return _Communicator()

    source_session = open_lora_nccl_source_session(
        plan.source_groups[0],
        rendezvous,
        "cpu",
        factory,
    )
    route = LoRANcclConsumerRoute(
        0,
        LAYOUT_DIGEST,
        (
            LoRAConsumerPull(0, LoRASourceSlice("a", (0,), (2,))),
            LoRAConsumerPull(2, LoRASourceSlice("c", (0,), (1,))),
        ),
    )
    consumer_plan = LoRAConsumerPlan(
        LAYOUT_DIGEST,
        type("ReceiverPlan", (), {"modules": ()})(),
        route.pulls,
        (),
    )
    receiver_session = open_lora_nccl_receiver_session(
        consumer_plan,
        0,
        rendezvous,
        "cpu",
        factory,
    )

    assert calls == [
        ("10.0.0.4", 41000, 0, 4, "cpu"),
        ("10.0.0.4", 41000, 2, 4, "cpu"),
    ]
    assert source_session._peer_by_inference_rank == {0: 2, 2: 3}
    assert receiver_session._peer_by_source_rank == {0: 0, 2: 1}
    source_session.close()
    receiver_session.close()


def test_receiver_constructor_failure_destroys_shared_group(monkeypatch):
    communicator = _Communicator()

    def fail_session(*args, **kwargs):
        raise RuntimeError("injected session failure")

    monkeypatch.setattr(lora_nccl.rendezvous, "LoRANcclReceiverSession", fail_session)
    consumer_plan = LoRAConsumerPlan(
        LAYOUT_DIGEST,
        type("ReceiverPlan", (), {"modules": ()})(),
        (LoRAConsumerPull(0, LoRASourceSlice("a", (0,), (2,))),),
        (),
    )
    with pytest.raises(RuntimeError, match="injected session failure"):
        open_lora_nccl_receiver_session(
            consumer_plan,
            0,
            _rendezvous(),
            "cpu",
            lambda *_: communicator,
        )

    assert communicator.destroy_count == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_ranks", (0, 0), "source ranks"),
        ("inference_ranks", (), "inference ranks"),
        ("master_port", 0, "valid master port"),
    ],
)
def test_rendezvous_rejects_invalid_shared_group(field, value, message):
    values = _rendezvous().to_json_dict()
    values[field] = value

    with pytest.raises(ValueError, match=message):
        LoRANcclRendezvous.from_json_dict(values)
