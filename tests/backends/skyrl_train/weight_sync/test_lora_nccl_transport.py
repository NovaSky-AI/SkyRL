import hashlib
from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.weight_sync.lora_nccl import (
    LoRANcclReceiverSession,
    LoRANcclSourceSession,
    build_lora_nccl_plan,
)
from skyrl.backends.skyrl_train.weight_sync.lora_transport.consumer_plan import (
    LoRAConsumerCopy,
    LoRAConsumerPlan,
    LoRAConsumerPull,
)
from skyrl.backends.skyrl_train.weight_sync.lora_transport.contracts import (
    LoRASourceSlice,
    LoRAUpdateRequest,
)

LAYOUT_DIGEST = hashlib.sha256(b"fixed-layout").hexdigest()


class _MailboxCommunicator:
    def __init__(self, mailbox, *, fail_send=False, fail_receive=False):
        self.mailbox = mailbox
        self.fail_send = fail_send
        self.fail_receive = fail_receive
        self.destroy_count = 0
        self.buffer_pointers = []

    def send(self, tensor, dst, stream=None):
        if self.fail_send:
            raise RuntimeError("injected send failure")
        self.buffer_pointers.append(tensor.untyped_storage().data_ptr())
        self.mailbox.append((dst, tensor.clone()))

    def recv(self, tensor, src, stream=None):
        if self.fail_receive:
            raise RuntimeError("injected receive failure")
        assert src == 0
        dst, payload = self.mailbox.pop(0)
        assert dst == 1
        tensor.copy_(payload)

    def destroy(self):
        self.destroy_count += 1


def _request(digest=LAYOUT_DIGEST):
    return LoRAUpdateRequest("adapter", 4, digest, "float32")


def _consumer_plan():
    pull_a = LoRAConsumerPull(0, LoRASourceSlice("a", (1, 1), (3, 4)))
    pull_b = LoRAConsumerPull(0, LoRASourceSlice("b", (0, 0), (4, 2)))
    module = SimpleNamespace(
        module_name="model.proj",
        factor_shapes=(((2, 3), (4, 2)),),
    )
    receiver_plan = SimpleNamespace(modules=(module,))
    return LoRAConsumerPlan(
        source_layout_digest=LAYOUT_DIGEST,
        receiver_plan=receiver_plan,
        pulls=(pull_a, pull_b),
        copies=(
            LoRAConsumerCopy(0, module.module_name, 0, 0, (0, 0), (2, 3)),
            LoRAConsumerCopy(1, module.module_name, 0, 1, (0, 0), (4, 2)),
        ),
    )


def _sessions(communicator, *, buffer_size=32):
    consumer_plan = _consumer_plan()
    plan = build_lora_nccl_plan({0: consumer_plan}, buffer_size)
    sender = LoRANcclSourceSession(
        plan.source_groups[0],
        plan.plan_digest,
        LAYOUT_DIGEST,
        communicator,
        {0: 1},
        "cpu",
    )
    receiver = LoRANcclReceiverSession(
        consumer_plan,
        0,
        plan.plan_digest,
        buffer_size,
        communicator,
        {0: 0},
        "cpu",
    )
    return plan, sender, receiver


def test_persistent_sessions_stream_noncontiguous_fp32_slices_into_independent_bf16_factors():
    mailbox = []
    communicator = _MailboxCommunicator(mailbox)
    plan, sender, receiver = _sessions(communicator)
    sources = {
        "a": torch.arange(20, dtype=torch.float32).view(4, 5),
        "b": torch.arange(8, dtype=torch.float32).view(4, 2) + 100,
    }
    expected_a = sources["a"][1:3, 1:4].to(torch.bfloat16)
    expected_b = sources["b"].to(torch.bfloat16)

    send_receipt = sender.send(_request(), sources)
    sources["a"].add_(1000)
    sources["b"].zero_()
    factors, receive_receipt = receiver.receive(_request())

    actual_a = factors["model.proj"][0][0]
    actual_b = factors["model.proj"][1][0]
    torch.testing.assert_close(actual_a, expected_a, rtol=0, atol=0)
    torch.testing.assert_close(actual_b, expected_b, rtol=0, atol=0)
    assert actual_a.untyped_storage().data_ptr() not in communicator.buffer_pointers
    assert actual_b.untyped_storage().data_ptr() not in communicator.buffer_pointers
    assert len(set(communicator.buffer_pointers)) == 1
    assert send_receipt.fp32_bytes == receive_receipt.fp32_bytes == plan.source_bytes
    assert send_receipt.bucket_count == receive_receipt.bucket_count == 2
    assert send_receipt.direction == "send"
    assert receive_receipt.direction == "receive"


def test_source_session_rejects_layout_change_before_send():
    communicator = _MailboxCommunicator([])
    _, sender, _ = _sessions(communicator)

    with pytest.raises(ValueError, match="changed the source layout"):
        sender.send(_request(hashlib.sha256(b"other").hexdigest()), {})

    assert communicator.mailbox == []
    assert communicator.destroy_count == 0


def test_send_failure_destroys_poisoned_session_once():
    communicator = _MailboxCommunicator([], fail_send=True)
    _, sender, _ = _sessions(communicator)
    sources = {
        "a": torch.arange(20, dtype=torch.float32).view(4, 5),
        "b": torch.arange(8, dtype=torch.float32).view(4, 2),
    }

    with pytest.raises(RuntimeError, match="injected send failure"):
        sender.send(_request(), sources)
    with pytest.raises(RuntimeError, match="session is unusable"):
        sender.send(_request(), sources)
    sender.close()

    assert communicator.destroy_count == 1


def test_receive_failure_destroys_poisoned_session_once():
    communicator = _MailboxCommunicator([], fail_receive=True)
    _, _, receiver = _sessions(communicator)

    with pytest.raises(RuntimeError, match="injected receive failure"):
        receiver.receive(_request())
    with pytest.raises(RuntimeError, match="session is unusable"):
        receiver.receive(_request())
    receiver.close()

    assert communicator.destroy_count == 1


def test_receiver_rejects_missing_or_extra_source_peers():
    consumer_plan = _consumer_plan()
    plan = build_lora_nccl_plan({0: consumer_plan}, 32)

    with pytest.raises(ValueError, match="do not match"):
        LoRANcclReceiverSession(
            consumer_plan,
            0,
            plan.plan_digest,
            32,
            _MailboxCommunicator([]),
            {},
            "cpu",
        )
