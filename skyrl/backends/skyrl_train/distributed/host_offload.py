"""Exact-size pinned host offload for GPU tensors (FSDP2 shards, optimizer state).

``module.to("cpu", non_blocking=True)`` allocates every tensor through torch's
pinned caching host allocator, which rounds each block up to a power of two and
keeps freed blocks cached: an offload costs up to ~2x its size in host memory
and none of it is returned after the backload (ML4 FSDP2 shards: 1.75x, kept).

Here every GPU storage of a module (or optimizer) is copied into one exact-size
anonymous mapping registered with ``cudaHostRegister`` and the GPU storage is
resized to 0 in place. Tensor objects, views and DTensor wrappers (FSDP2's
padded ``_sharded_param_data`` and the narrowed local tensor share a storage)
stay as they are; onload resizes each storage back, copies the bytes in and
unregisters and unmaps the host buffer. Same technique as the vLLM level-1
sleep backup (``vllm/device_allocator/cumem.py``). If registration fails the
buffer stays pageable, with a warning.
"""

import ctypes
import mmap
import weakref
from collections.abc import Iterable

import torch
from loguru import logger

_ALIGN = 64

# Owner (module or optimizer) -> its host buffer while offloaded.
_OFFLOADED: "weakref.WeakKeyDictionary[object, _HostOffload]" = weakref.WeakKeyDictionary()


def _clear_cuda_last_error() -> None:
    """Reset the runtime's last error after a failed cudaHostRegister/Unregister.

    torch does not expose ``cudaGetLastError``; a stale error would surface at the next
    kernel-launch check. Call it from the libcudart already loaded into the process.
    """
    path = None
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                if "libcudart" in line:
                    path = line.split()[-1]
                    break
    except OSError:
        pass
    if path is None:
        logger.warning("host offload: libcudart not found in /proc/self/maps; CUDA last error not cleared")
        return
    ctypes.CDLL(path).cudaGetLastError()


class _HostBuffer:
    def __init__(self, nbytes: int, pin: bool):
        self.nbytes = nbytes
        kwargs = {"flags": mmap.MAP_PRIVATE} if hasattr(mmap, "MAP_PRIVATE") else {}
        self._mmap = mmap.mmap(-1, max(nbytes, 1), **kwargs)
        self.tensor = torch.frombuffer(self._mmap, dtype=torch.uint8, count=max(nbytes, 1))[:nbytes]
        self.pinned = False
        if pin and nbytes > 0:
            rc = torch.cuda.cudart().cudaHostRegister(self.tensor.data_ptr(), nbytes, 0)
            self.pinned = int(rc) == 0
            if not self.pinned:
                _clear_cuda_last_error()
                logger.warning(
                    f"host offload: cudaHostRegister failed ({rc}) for {nbytes / 2**30:.2f} GiB; "
                    "the buffer stays pageable"
                )

    def release(self) -> None:
        if self.pinned:
            rc = torch.cuda.cudart().cudaHostUnregister(self.tensor.data_ptr())
            if int(rc) != 0:
                _clear_cuda_last_error()
                logger.warning(f"host offload: cudaHostUnregister failed ({rc})")
            self.pinned = False
        self.tensor = None
        try:
            self._mmap.close()
        except BufferError:
            # A view of the buffer is still referenced somewhere; the mapping is freed with it.
            logger.warning("host offload: host buffer still referenced, not unmapped now")


class _HostOffload:
    def __init__(self, storages: list, device_type: str, pin: bool):
        self.storages = storages
        self.offsets = []
        off = 0
        for s in storages:
            self.offsets.append(off)
            off += (s.nbytes() + _ALIGN - 1) // _ALIGN * _ALIGN
        self.sizes = [s.nbytes() for s in storages]
        self.buffer = _HostBuffer(off, pin=pin and device_type == "cuda")
        self.device_type = device_type

    def _byte_view(self, storage):
        return torch.empty(0, dtype=torch.uint8, device=storage.device).set_(storage)

    def offload(self) -> None:
        host = self.buffer.tensor
        nb = self.buffer.pinned
        for s, off, n in zip(self.storages, self.offsets, self.sizes):
            host[off : off + n].copy_(self._byte_view(s), non_blocking=nb)
        if self.device_type == "cuda":
            torch.cuda.synchronize()
        for s in self.storages:
            s.resize_(0)

    def onload(self) -> None:
        host = self.buffer.tensor
        nb = self.buffer.pinned
        for s, off, n in zip(self.storages, self.offsets, self.sizes):
            s.resize_(n)
            self._byte_view(s).copy_(host[off : off + n], non_blocking=nb)
        if self.device_type == "cuda":
            torch.cuda.synchronize()
        del host
        self.buffer.release()
        self.storages = []


def _local(t: torch.Tensor) -> torch.Tensor:
    from torch.distributed.tensor import DTensor

    return t._local_tensor if isinstance(t, DTensor) else t


def _unique_storages(tensors: Iterable[torch.Tensor | None], device_type: str) -> list:
    seen = {}
    for t in tensors:
        if t is None:
            continue
        t = _local(t)
        if t.device.type != device_type:
            continue
        s = t.untyped_storage()
        if s.nbytes() == 0:
            continue
        if not s.resizable():
            raise RuntimeError(f"host offload: storage of a {tuple(t.shape)} {t.dtype} tensor is not resizable")
        seen.setdefault(s.data_ptr(), s)
    return list(seen.values())


def module_tensors(module: torch.nn.Module, frozen_only: bool = False) -> list:
    """Params, their grads and buffers (what ``Module.to`` moves). ``frozen_only`` leaves out the
    trainable params and their grads (LoRA adapters stay on the GPU)."""
    out = []
    for p in module.parameters():
        if frozen_only and p.requires_grad:
            continue
        out.append(p)
        out.append(p.grad)
    out.extend(module.buffers())
    return out


def optimizer_tensors(optimizer: torch.optim.Optimizer) -> list:
    return [v for state in optimizer.state.values() for v in state.values() if isinstance(v, torch.Tensor)]


def offload_to_host(owner, tensors: Iterable[torch.Tensor | None], device_type: str = "cuda", pin: bool = True) -> int:
    """Move every ``device_type`` storage behind ``tensors`` into one exact-size host buffer.

    The storages are resized to 0 in place; ``onload_from_host(owner)`` restores them. Returns
    the host bytes held (0 if nothing to offload or ``owner`` is already offloaded).
    """
    if owner in _OFFLOADED:
        return 0
    storages = _unique_storages(tensors, device_type)
    if not storages:
        return 0
    state = _HostOffload(storages, device_type, pin=pin and torch.cuda.is_available())
    state.offload()
    _OFFLOADED[owner] = state
    return state.buffer.nbytes


def onload_from_host(owner) -> int:
    """Restore the storages offloaded for ``owner`` and free its host buffer. Returns bytes restored."""
    state = _OFFLOADED.pop(owner, None)
    if state is None:
        return 0
    n = state.buffer.nbytes
    state.onload()
    return n


def offloaded_bytes(owner) -> int:
    state = _OFFLOADED.get(owner)
    return 0 if state is None else state.buffer.nbytes


def is_offloaded(owner) -> bool:
    return owner in _OFFLOADED


def _reshard(module: torch.nn.Module) -> None:
    try:
        from torch.distributed.fsdp import FSDPModule
    except ImportError:
        return
    for m in module.modules():
        if isinstance(m, FSDPModule):
            m.reshard()


def offload_module(
    module: torch.nn.Module, device_type: str = "cuda", pin: bool = True, frozen_only: bool = False
) -> int:
    # Same entry state as ``Module.to``: FSDP2's ``_apply`` reshards first.
    _reshard(module)
    return offload_to_host(module, module_tensors(module, frozen_only), device_type=device_type, pin=pin)


def onload_module(module: torch.nn.Module) -> int:
    return onload_from_host(module)


def offload_optimizer(optimizer: torch.optim.Optimizer, device_type: str = "cuda", pin: bool = True) -> int:
    return offload_to_host(optimizer, optimizer_tensors(optimizer), device_type=device_type, pin=pin)


def onload_optimizer(optimizer: torch.optim.Optimizer) -> int:
    return onload_from_host(optimizer)


__all__ = [
    "is_offloaded",
    "offload_module",
    "offload_optimizer",
    "offload_to_host",
    "offloaded_bytes",
    "onload_from_host",
    "onload_module",
    "onload_optimizer",
]
