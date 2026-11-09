"""Overlap each microbatch's host-side build and host-to-device copy with the previous one's compute.

``MicrobatchPrefetcher`` yields ``stage(host[i])`` for each host microbatch, in order. ``stage``
builds a microbatch on the host and copies it to the device; ``prefetch()``, called right after the
current microbatch's compute is launched, runs it for the next microbatch on a side stream, so the
copy proceeds while the GPU computes. Consuming a staged microbatch makes the current stream wait
for its copy on the GPU (``wait_event``) and records the stream on its tensors (``record_stream``)
so the caching allocator does not reuse them early. Nothing blocks the host.

Single-threaded and per pass: ``stage`` must be pure host work plus the copy (no collectives), and the
prefetcher holds nothing once exhausted.
"""

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import torch

from skyrl.utils import stage_trace

_COPY_STREAMS: Dict[int, torch.cuda.Stream] = {}


def _copy_stream(device: torch.device) -> torch.cuda.Stream:
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _COPY_STREAMS:
        _COPY_STREAMS[index] = torch.cuda.Stream(device=index)
    return _COPY_STREAMS[index]


@dataclass
class _Staged:
    item: Any
    tensors: List[torch.Tensor]
    copied: Optional[torch.cuda.Event]


class MicrobatchPrefetcher:
    """Iterate ``stage(host[i])``, staging microbatch ``i + 1`` while microbatch ``i`` computes.

    ``tensors(item)`` lists the device tensors a staged item holds. With ``enabled=False``, every
    microbatch is staged when it is consumed, which is what the computation reads either way.
    """

    def __init__(
        self,
        host: Sequence[Any],
        stage: Callable[[Any], Any],
        tensors: Callable[[Any], Iterable[torch.Tensor]],
        device: torch.device,
        enabled: bool = True,
    ) -> None:
        self._host = host
        self._stage_fn = stage
        self._tensors = tensors
        self._device = torch.device(device)
        self._enabled = enabled
        self._next = 0
        self._staged: Optional[_Staged] = None

    def __iter__(self) -> "MicrobatchPrefetcher":
        return self

    def __len__(self) -> int:
        return len(self._host)

    def __next__(self) -> Any:
        if self._next >= len(self._host):
            raise StopIteration
        if self._staged is None:
            # The first microbatch, or one consumed before its prefetch: stage it now, while the
            # computation waits (``mb.stage_exposed``, apart from the overlapped ``mb.stage``).
            self._stage(self._next, exposed=True)
        staged, self._staged = self._staged, None
        self._next += 1
        if staged.copied is not None:
            consumer = torch.cuda.current_stream(self._device)
            consumer.wait_event(staged.copied)
            for tensor in staged.tensors:
                tensor.record_stream(consumer)
        return staged.item

    def prefetch(self) -> None:
        """Stage the next microbatch. Call right after launching the current one's compute."""
        if self._enabled and self._staged is None and self._next < len(self._host):
            self._stage(self._next)

    def _stage(self, index: int, exposed: bool = False) -> None:
        t0 = time.perf_counter()
        try:
            if self._device.type != "cuda":
                self._staged = _Staged(self._stage_fn(self._host[index]), [], None)
                stage_trace.record("mb.stage_exposed" if exposed else "mb.stage", t0, time.perf_counter(), index=index)
                return
            stream = _copy_stream(self._device)
            with torch.cuda.stream(stream):
                # Device memory comes from the side stream's pool; ``record_stream`` hands it over.
                item = self._stage_fn(self._host[index])
                copied = torch.cuda.Event()
                copied.record(stream)
        except Exception as error:
            error.add_note(f"while staging microbatch {index}")
            raise
        tensors = [tensor for tensor in self._tensors(item) if tensor.is_cuda]
        self._staged = _Staged(item, tensors, copied)
        stage_trace.record("mb.stage_exposed" if exposed else "mb.stage", t0, time.perf_counter(), index=index)
