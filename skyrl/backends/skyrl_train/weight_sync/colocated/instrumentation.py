"""Instrumentation for colocated weight sync: memory probes, timing and weight checksums.

Everything here is debug-only and cheap enough to call once per phase. Nothing
imports vLLM at module scope, so the trainer side can use it too.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

import torch

DEBUG_ENV = "SKYRL_COLOCATED_SYNC_DEBUG"
# Test-only: cap this trainer process's CUDA allocations during the send, in GiB, to
# reproduce "vLLM weights fill the GPU" OOMs on a small model.
TRAINER_CAP_ENV = "SKYRL_COLOCATED_SYNC_TRAINER_CAP_GIB"


@contextmanager
def trainer_memory_cap() -> Iterator[None]:
    cap = os.environ.get(TRAINER_CAP_ENV)
    if not cap or not torch.cuda.is_available():
        yield
        return
    _, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(min(1.0, float(cap) * 1024**3 / total))
    try:
        yield
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0)


def debug_enabled() -> bool:
    return os.environ.get(DEBUG_ENV, "0") not in ("", "0", "false", "False")


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
    """Collects per-phase memory/time on one process; dumped as one JSON line."""

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

    def dump(self, extra: Optional[Dict[str, Any]] = None) -> str:
        rec = {"recorder": self.name, "phases": self.phases, "counters": self.counters}
        if extra:
            rec.update(extra)
        return json.dumps(rec, default=str)


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
    for method in ("update_weights",):
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


def weight_allocation_map(model: torch.nn.Module) -> Dict[str, Any]:
    """Dump the layer -> ``weights``-allocation map of a loaded vLLM model (inside a worker).

    For every ``model.layers.<N>`` prefix: the CuMem allocations its parameters and
    buffers overlap, their bytes, and how many of those allocations are shared with
    another layer: how finely a per-layer wake of the weights pool could map it.
    """
    import bisect
    import re

    from vllm.device_allocator.cumem import CuMemAllocator

    alloc = CuMemAllocator.get_instance()
    segs = sorted((ptr, ptr + d.handle[1]) for ptr, d in alloc.pointer_to_data.items() if d.tag == "weights")
    starts = [s for s, _ in segs]

    def seg_of(p: int):
        i = bisect.bisect_right(starts, p) - 1
        return segs[i] if i >= 0 and segs[i][0] <= p < segs[i][1] else None

    layer_re = re.compile(r"(?:^|\.)layers\.(\d+)\.")
    by_layer: Dict[str, set] = {}
    outside = 0
    for name, t in list(model.named_parameters()) + list(model.named_buffers()):
        if t is None or t.device.type != "cuda" or t.numel() == 0:
            continue
        m = layer_re.search(name)
        key = m.group(1) if m else "other"
        seg = seg_of(t.data_ptr())
        if seg is None:
            outside += t.numel() * t.element_size()
            continue
        by_layer.setdefault(key, set()).add(seg)
    owners: Dict[tuple, set] = {}
    for k, ss in by_layer.items():
        for s in ss:
            owners.setdefault(s, set()).add(k)
    shared = {s for s, ks in owners.items() if len(ks) > 1}
    gib = 1024**3
    layers = {
        k: {
            "allocations": len(ss),
            "GiB": sum(e - s for s, e in ss) / gib,
            "shared_allocations": len(ss & shared),
        }
        for k, ss in by_layer.items()
    }
    return {
        "weights_allocations": len(segs),
        "weights_GiB": sum(e - s for s, e in segs) / gib,
        "shared_allocations": len(shared),
        "shared_GiB": sum(e - s for s, e in shared) / gib,
        "tensor_bytes_outside_weights_pool_GiB": outside / gib,
        "layers": layers,
    }


def benchmark_create_and_map(sizes_mb=(64, 256), counts=(1, 10, 100), restore: bool = True) -> list:
    """Time CuMem ``create_and_map`` (+ optional pinned H2D restore) of fresh allocations.

    Allocates through vLLM's ``weights``-style pool, sleeps them (discard or CPU
    backup), then times re-mapping ``count`` of them directly with ``create_and_map``
    -- what a per-layer wake would do -- without ``CuMemAllocator.wake_up``'s
    ``gc.collect()`` / ``empty_cache()``.
    """
    from vllm.device_allocator.cumem import (
        CuMemAllocator,
        create_and_map,
        libcudart,
        unmap_and_release,
    )

    alloc = CuMemAllocator.get_instance()
    results = []
    for size_mb in sizes_mb:
        for count in counts:
            nbytes = size_mb * 1024**2
            with alloc.use_memory_pool(tag="skyrl_bench"):
                ts = [torch.empty(nbytes, dtype=torch.uint8, device="cuda") for _ in range(count)]
            ptrs = [t.data_ptr() for t in ts]
            datas = [alloc.pointer_to_data[p] for p in ptrs if p in alloc.pointer_to_data]
            backups = []
            for d in datas:
                if restore:
                    cpu = torch.empty(d.handle[1], dtype=torch.uint8, pin_memory=True)
                    libcudart.cudaMemcpy(cpu.data_ptr(), d.handle[2], d.handle[1])
                    backups.append(cpu)
                unmap_and_release(d.handle)
                d.is_asleep = True
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for i, d in enumerate(datas):
                create_and_map(d.handle)
                d.is_asleep = False
                if restore:
                    libcudart.cudaMemcpy(d.handle[2], backups[i].data_ptr(), d.handle[1])
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            results.append(
                {
                    "size_MiB": size_mb,
                    "count": count,
                    "restore": restore,
                    "seconds": dt,
                    "GiB_per_s": count * nbytes / 1024**3 / dt if dt else None,
                    "ms_per_alloc": dt / max(1, len(datas)) * 1e3,
                }
            )
            del ts, backups
            torch.cuda.empty_cache()
    return results
