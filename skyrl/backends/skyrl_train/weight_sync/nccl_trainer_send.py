# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Trainer-side NCCL send, vendored from vLLM 0.26.0.

REMOVAL: ``NCCLWeightTransferEngine.trainer_send_weights`` and the engine's
``trainer_init`` staticmethod were deleted in vLLM 0.28.0, which replaced them
with a trainer-side engine abstraction (``NCCLTrainerWeightTransferEngine``,
dispatched through ``WeightTransferTrainerFactory``). SkyRL still drives the send
itself -- the sender owns the ``/collective_rpc`` round trip in
``broadcast_strategy.py`` -- so this module keeps 0.26.0's behaviour available
under the new pin. Delete it when SkyRL's sender layer moves onto vLLM's
trainer-send engines; that migration also retires ``WeightTransferStrategy``,
``WeightTransferSender`` and ``NewInferenceWorkerWrap``.

``packed_nccl_broadcast_producer`` (the real work) and ``nccl_common.trainer_init``
(the rendezvous) both survive in 0.28.0 unchanged, so this is a thin re-wrap
rather than a copy of the transfer itself.
"""

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, Optional, Tuple

import torch

if TYPE_CHECKING:
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

__all__ = ["nccl_trainer_send_weights", "nccl_trainer_init"]


def nccl_trainer_init(init_info: Any) -> "PyNcclCommunicator":
    """Open the trainer-side (rank 0) endpoint of the weight-transfer group.

    Was ``NCCLWeightTransferEngine.trainer_init`` in 0.26.0, which was itself a
    ``staticmethod`` alias of this helper. 0.28.0 dropped the alias but kept the
    helper, so this just forwards.

    Args:
        init_info: object or dict carrying ``master_address``, ``master_port``
            and ``world_size``.
    """
    # Lazy import: vllm is a Linux-only optional dependency (see
    # .claude/docs/weight_sync.md), so this module stays importable without it.
    from vllm.distributed.weight_transfer.nccl_common import trainer_init

    return trainer_init(init_info)


def nccl_trainer_send_weights(
    iterator: Iterator[Tuple[str, torch.Tensor]],
    group: "PyNcclCommunicator",
    *,
    src: int = 0,
    packed: bool = True,
    post_iter_func: Optional[Callable[[Tuple[str, torch.Tensor]], torch.Tensor]] = None,
    stream: Optional[torch.cuda.Stream] = None,
    packed_buffer_size_bytes: Optional[int] = None,
    packed_num_buffers: Optional[int] = None,
) -> None:
    """Broadcast dense weights from the trainer to the vLLM workers.

    Vendored from ``NCCLWeightTransferEngine.trainer_send_weights`` (vLLM 0.26.0),
    with the ``trainer_args`` dict/dataclass flattened into keyword arguments --
    the dataclass (``NCCLTrainerSendWeightsArgs``) was deleted alongside the method.

    ``packed`` must match what the receiving worker was told at init. As of
    vLLM 0.28.0 the worker reads it from ``NCCLWeightTransferInitInfo.packed``
    (set once during ``/init_weight_transfer_engine``) rather than from the
    per-round update info, so the two sides can no longer disagree per round --
    see ``BroadcastInitInfo.to_api_payload``.

    Args:
        iterator: ``(name, tensor)`` pairs to send.
        group: the ``PyNcclCommunicator`` from ``nccl_trainer_init``.
        src: source rank; the trainer is rank 0.
        packed: batch tensors into packed buffers instead of one broadcast each.
        post_iter_func: maps each pair to the tensor to send. Defaults to taking
            the tensor.
        stream: CUDA stream for the unpacked path. Defaults to the current
            stream. Ignored when ``packed`` (the producer makes its own).
        packed_buffer_size_bytes: packed buffer size. Defaults to vLLM's
            ``DEFAULT_PACKED_BUFFER_SIZE_BYTES``, which is also what the worker's
            init info defaults to -- leave unset so the two cannot drift.
        packed_num_buffers: packed buffer count, likewise defaulting to vLLM's
            ``DEFAULT_PACKED_NUM_BUFFERS``.
    """
    from vllm.distributed.weight_transfer.packed_tensor import (
        DEFAULT_PACKED_BUFFER_SIZE_BYTES,
        DEFAULT_PACKED_NUM_BUFFERS,
        packed_nccl_broadcast_producer,
    )

    if post_iter_func is None:
        post_iter_func = lambda item: item[1]  # noqa: E731

    if packed:
        packed_nccl_broadcast_producer(
            iterator=iterator,
            group=group,
            src=src,
            post_iter_func=post_iter_func,
            buffer_size_bytes=(
                DEFAULT_PACKED_BUFFER_SIZE_BYTES if packed_buffer_size_bytes is None else packed_buffer_size_bytes
            ),
            num_buffers=(DEFAULT_PACKED_NUM_BUFFERS if packed_num_buffers is None else packed_num_buffers),
        )
    else:
        send_stream = stream or torch.cuda.current_stream()
        for item in iterator:
            group.broadcast(post_iter_func(item), src=src, stream=send_stream)
