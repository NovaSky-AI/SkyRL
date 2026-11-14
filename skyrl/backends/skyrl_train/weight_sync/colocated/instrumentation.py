"""Instrumentation for colocated weight sync: per-phase memory and time, and weight checksums.

Everything here is debug-only and cheap enough to call once per phase. Nothing
imports vLLM at module scope, so the trainer side can use it too.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator

import torch

from skyrl.env_vars import SKYRL_COLOCATED_SYNC_TRAINER_CAP_GIB


@contextmanager
def trainer_memory_cap() -> Iterator[None]:
    """Cap this trainer process's CUDA allocations during the send (``SKYRL_COLOCATED_SYNC_TRAINER_CAP_GIB``)."""
    cap = SKYRL_COLOCATED_SYNC_TRAINER_CAP_GIB
    if cap is None or not torch.cuda.is_available():
        yield
        return
    _, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap * 1024**3 / total))
    try:
        yield
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0)


def tensor_checksum(t: torch.Tensor) -> tuple:
    """Order-sensitive integer checksum of a tensor's bytes, computed on its device.

    Integer arithmetic wraps deterministically, so two tensors with identical
    bytes always agree and a single flipped bit (almost) always disagrees.
    """
    if t.numel() == 0:
        return (0, 0, tuple(t.shape), str(t.dtype))
    b = t.detach().contiguous().view(-1).view(torch.uint8)
    pad = (-b.numel()) % 8
    if pad:
        b = torch.cat([b, b.new_zeros(pad)])
    x = b.view(torch.int64)
    idx = torch.arange(x.numel(), device=x.device, dtype=torch.int64)
    w = (idx % 65521) + 1
    s1 = int(x.sum().item())
    s2 = int((x * w).sum().item())
    return (s1, s2, tuple(t.shape), str(t.dtype))


def module_checksums(module: torch.nn.Module) -> Dict[str, tuple]:
    out: Dict[str, tuple] = {}
    seen = set()
    for kind, it in (("p", module.named_parameters(remove_duplicate=False)), ("b", module.named_buffers())):
        for name, t in it:
            key = f"{kind}:{name}"
            if key in seen or t is None:
                continue
            seen.add(key)
            out[key] = tensor_checksum(t)
    return out


def device_memory_snapshot(tag: str = "") -> Dict[str, Any]:
    """Allocator + device memory for this process's current device, in GiB."""
    if not torch.cuda.is_available():
        return {"tag": tag}
    gib = 1024**3
    free, total = torch.cuda.mem_get_info()
    return {
        "tag": tag,
        "allocated": torch.cuda.memory_allocated() / gib,
        "reserved": torch.cuda.memory_reserved() / gib,
        "max_allocated": torch.cuda.max_memory_allocated() / gib,
        "max_reserved": torch.cuda.max_memory_reserved() / gib,
        "device_used": (total - free) / gib,
        "device_free": free / gib,
    }


def cumem_snapshot() -> Dict[str, Any]:
    """vLLM CuMemAllocator usage by tag and sleep state, in GiB (inside a vLLM worker)."""
    try:
        from vllm.device_allocator.cumem import CuMemAllocator
    except Exception:  # noqa: BLE001
        return {}
    alloc = CuMemAllocator.get_instance()
    gib = 1024**3
    by_tag: Dict[str, Dict[str, float]] = {}
    for data in alloc.pointer_to_data.values():
        tag = data.tag
        size = data.handle[1]
        asleep = bool(getattr(data, "is_asleep", False))
        d = by_tag.setdefault(tag, {"awake": 0.0, "asleep": 0.0, "count": 0})
        d["asleep" if asleep else "awake"] += size / gib
        d["count"] += 1
    return by_tag


class _DevicePeakPoller:
    """Polls ``mem_get_info`` on a thread: the peak *device* use, which includes every
    process sharing the GPU (the colocated vLLM worker), unlike torch's own peaks."""

    def __init__(self, interval_s: float = 0.005) -> None:
        import threading

        self._device = torch.cuda.current_device()
        self._peak = 0
        self._stop = threading.Event()
        self._interval = interval_s
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            free, total = torch.cuda.mem_get_info(self._device)
            self._peak = max(self._peak, total - free)
            self._stop.wait(self._interval)

    def stop(self) -> float:
        self._stop.set()
        self._thread.join()
        return self._peak / 1024**3


class PhaseRecorder:
    """Collects per-phase memory/time and counters on one process."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.phases: list = []
        self.counters: Dict[str, float] = {}

    @contextmanager
    def phase(self, tag: str) -> Iterator[None]:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        poller = _DevicePeakPoller() if torch.cuda.is_available() else None
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            snap = device_memory_snapshot(tag)
            snap["seconds"] = time.perf_counter() - t0
            if poller is not None:
                snap["device_peak_used"] = poller.stop()
            self.phases.append(snap)

    def add(self, key: str, value: float) -> None:
        self.counters[key] = self.counters.get(key, 0.0) + value


class _TimedSource:
    """Proxy over a ``WeightSource`` that times the export generator (gather) and counts bytes."""

    def __init__(self, source: Any, recorder: PhaseRecorder) -> None:
        self._source = source
        self._rec = recorder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._source, name)

    @property
    def metadata(self):  # noqa: D401 - passthrough
        return self._source.metadata

    def iter_full(self):
        it = iter(self._source.iter_full())
        while True:
            t0 = time.perf_counter()
            try:
                name, tensor = next(it)
            except StopIteration:
                self._rec.add("export_seconds", time.perf_counter() - t0)
                return
            self._rec.add("export_seconds", time.perf_counter() - t0)
            self._rec.add("export_bytes", tensor.numel() * tensor.element_size())
            self._rec.add("export_tensors", 1)
            yield name, tensor


@contextmanager
def instrument_send(engine: Any, recorder: PhaseRecorder) -> Iterator[None]:
    """Time the export generator and every control-plane round trip of one send.

    Only rank 0's client issues round trips; every rank times its export.
    """
    source = getattr(engine, "source", None)
    client = getattr(engine, "client", None)
    wrapped = []
    if source is not None:
        engine.source = _TimedSource(source, recorder)
    for method in ("update_weights", "update_weights_per_server"):
        orig = getattr(client, method, None) if client is not None else None
        if orig is None:
            continue

        def timed(update_info, _orig=orig):
            t0 = time.perf_counter()
            try:
                return _orig(update_info)
            finally:
                recorder.add("round_trips", 1)
                recorder.add("round_trip_seconds", time.perf_counter() - t0)

        setattr(client, method, timed)
        wrapped.append(method)
    try:
        yield
    finally:
        if source is not None:
            engine.source = source
        for method in wrapped:
            delattr(client, method)
        ring = getattr(engine, "skyrl_last_send_stats", None)
        if ring:
            for k, v in ring.items():
                recorder.counters[f"ring_{k}"] = v
