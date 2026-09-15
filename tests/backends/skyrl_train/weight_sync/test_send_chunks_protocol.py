"""Every sender must implement the full ``send_chunks`` protocol.

``megatron_worker`` passes ``derive_metadata_from_chunks`` to whichever sender
strategy is active, so a sender that omits it from its signature raises
``TypeError`` on every Megatron sync — even with FP8 off.
"""

import asyncio
import inspect

import pytest

from skyrl.backends.skyrl_train.weight_sync.broadcast_strategy import (
    BroadcastWeightTransferSender,
)
from skyrl.backends.skyrl_train.weight_sync.cuda_ipc_strategy import (
    CudaIpcWeightTransferSender,
)
from skyrl.backends.skyrl_train.weight_sync.delta_strategy import (
    DeltaWeightTransferSender,
)

SENDER_CLASSES = (
    BroadcastWeightTransferSender,
    CudaIpcWeightTransferSender,
    DeltaWeightTransferSender,
)


@pytest.mark.parametrize("sender_cls", SENDER_CLASSES, ids=lambda c: c.__name__)
def test_send_chunks_accepts_protocol_arguments(sender_cls):
    signature = inspect.signature(sender_cls.send_chunks)
    signature.bind_partial(
        object(),
        chunks=[],
        weight_metadata=None,
        derive_metadata_from_chunks=False,
    )


def test_delta_sender_rejects_serialized_fp8_chunks():
    """Delta checkpoints cannot represent serialized-FP8 wire chunks."""

    with pytest.raises(ValueError, match="serialized FP8"):
        asyncio.run(
            DeltaWeightTransferSender.send_chunks(
                object(),
                chunks=[],
                derive_metadata_from_chunks=True,
            )
        )
