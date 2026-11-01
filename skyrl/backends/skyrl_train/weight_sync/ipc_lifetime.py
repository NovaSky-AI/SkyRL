"""Fail-closed lifetime protection for native vLLM packed IPC producers."""

from threading import Lock
from typing import Any


class FailedIPCProducerRegistry:
    """Process-lifetime ownership of exported buffers with uncertain consumers.

    This registry intentionally has no release operation: uncertain receivers must
    be fenced before the producer process exits. Retention survives engine garbage
    collection, but is not itself a receiver fence or protection against process exit.
    """

    def __init__(self) -> None:
        self._producers: tuple[Any, ...] = ()
        self._lock = Lock()

    def retain(self, producer: Any) -> None:
        with self._lock:
            self._producers = (*self._producers, producer)

    def __len__(self) -> int:
        with self._lock:
            return len(self._producers)

    def __contains__(self, producer: Any) -> bool:
        with self._lock:
            return any(retained is producer for retained in self._producers)


FAILED_IPC_PRODUCERS = FailedIPCProducerRegistry()


class FailClosedIPCLifetimeMixin:
    """Keep vLLM's suspended producer alive when a transfer cannot be proven safe.

    Only packed producers need this narrow override. Upstream vLLM has no hook
    exposing their lifetime; remove the override when such a hook is available.
    Private signatures and the copied loop are pinned by contract tests.
    """

    def send_weights(self) -> None:
        if self.client.fail_closed_transfers is not True:
            return super().send_weights()
        if getattr(self, "_skyrl_ipc_failed", False):
            raise RuntimeError("IPC transfer outcome is uncertain; restart required")
        if not self.packed:
            raise RuntimeError("Fail-closed IPC transfer requires packed buffers")
        try:
            super().send_weights()
        except BaseException:
            self._skyrl_ipc_failed = True
            producer = getattr(self, "_skyrl_ipc_producer", None)
            if producer is not None:
                FAILED_IPC_PRODUCERS.retain(producer)
            raise

    def _send_packed(self, source) -> None:
        if self.client.fail_closed_transfers is not True:
            return super()._send_packed(source)
        from vllm.distributed.weight_transfer.ipc_engine import packed_ipc_producer

        producer = packed_ipc_producer(
            iterator=iter(source),
            gpu_uuid=self.gpu_uuid,
            post_iter_func=lambda item: item[1],
            buffer_size_bytes=self.packed_buffer_size_bytes,
        )
        self._skyrl_ipc_producer = producer
        for chunk in producer:
            ipc_handle = self._all_gather_and_merge_handles([chunk.ipc_handle])[0]
            self._do_send(
                names=chunk.names,
                dtype_names=chunk.dtype_names,
                shapes=chunk.shapes,
                ipc_handles=ipc_handle,
                tensor_sizes=chunk.tensor_sizes,
            )
            self._post_send_sync()
        self._skyrl_ipc_producer = None
