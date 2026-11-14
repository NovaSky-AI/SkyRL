"""Megatron ``DistributedOptimizer`` whose fp32 main params and Adam moments live on node-local disk.

Enabled by ``megatron_config.optimizer_offload_to_disk``. The regular distributed optimizer keeps 12 B/param of
state per DP shard on the GPU (fp32 main params + two fp32 Adam moments); megatron-core's CPU offload moves it to host
RAM instead, where a large model can exhaust it. Here the state lives in a file on node-local disk and the optimizer
step streams it through the GPU one chunk at a time. Streaming optimizer state from NVMe this way follows Miles'
``--stream-optimizer-state-to-disk`` (https://github.com/radixark/miles).

The state is laid out the way megatron-core already lays out the distributed optimizer: each data-parallel rank
owns one contiguous range of every DDP bucket's flat param and grad buffers. A chunk is (part of) that range, and
its main params and moments are three flat tensors of the same length, stored in one file per rank and optimizer.
Per chunk, the step:

    reads main + moments (disk -> pinned host buffer -> GPU),
    runs Adam on the GPU with the matching slice of the fp32 grad buffer as the gradient,
    copies the main into the matching slice of the param buffer, and
    writes main + moments back,

so the GPU holds one chunk of state at a time. For native-fp32 params the main is the param buffer itself, and only
the moments stream.

Gradients are megatron-core's: with ``grad_reduce_in_fp32``, ``_copy_model_grads_to_main_grads`` sets
each main param's ``.grad`` to a view of the grad buffer, so grad norm, clipping and zero counting run unchanged, on
the same memory the chunks read their gradients from. Adam is the regular optimizer's class over slices of the
same values, so with fp32 moments the step is bit-identical to the regular optimizer.

Without ``grad_reduce_in_fp32`` the grad buffer of the bf16 params is bf16 (2 instead of 4 B/param on the GPU, which
decides whether a model's full fine-tuning fits at all). megatron-core would then give every main param an fp32
copy of its whole gradient shard; instead the (storage-less) main shards of the bf16 params are created bf16, so
their ``.grad`` is a view of the bf16 grad buffer, and each chunk's gradient is upcast to fp32 for its Adam step
only. The master copy on disk stays fp32. Gradients are then accumulated and reduced in bf16, as with any
``grad_reduce_in_fp32=False`` run.

How it plugs into megatron-core:

- ``OffloadedDistributedOptimizer`` subclasses ``DistributedOptimizer``. Its ``_build_model_and_main_param_groups``
  is generated from the installed megatron-core source with one line replaced, so the per-param fp32 main shards
  are created with empty storage instead of as full copies on the GPU. They keep empty storage; megatron-core only
  reads their ``.grad``.
- ``get_megatron_optimizer`` constructs ``DistributedOptimizer`` by name, so ``build_optimizer_offloaded_to_disk``
  points that name at the subclass for the duration of the call only.
- The step reuses ``DistributedOptimizer.step_with_ready_grads`` for the param all-gather that follows the update.

Checkpoints: each rank copies its state files into ``<ckpt>/optimizer_offload_to_disk/``; the optimizer's part of
Megatron's distributed checkpoint is a marker. A checkpoint is therefore resumable with the same parallelism and
chunk layout only, and only from a local path.
"""

import errno
import fcntl
import inspect
import json
import os
import shutil
import textwrap
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

import megatron.core.optimizer as mcore_optimizer
import torch
from loguru import logger
from megatron.core.optimizer import Adam, get_megatron_optimizer
from megatron.core.optimizer import distrib_optimizer as _distrib_optimizer_module
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

from skyrl.env_vars import SKYRL_OPTIMIZER_OFFLOAD_TO_DISK_CHUNK_NUMEL
from skyrl.train.config.config import MegatronOptimizerOffloadToDiskConfig

MOMENT_STORAGE_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
CHECKPOINT_SUBDIR = "optimizer_offload_to_disk"
_MEMORY_FILESYSTEMS = ("tmpfs", "ramfs")

# (start, end, optimizer param group index), in coordinates of a DDP buffer's flat param/grad data.
Run = Tuple[int, int, int]


def filesystem_type(path: str, mounts: str = "/proc/mounts") -> str:
    """The type of the filesystem ``path`` is on, from the longest matching mount point in ``mounts``."""
    path = os.path.realpath(path)
    best, fstype = "", ""
    with open(mounts) as f:
        for line in f:
            fields = line.split()
            if len(fields) < 3:
                continue
            mount_point = fields[1].replace("\\040", " ")
            if (path == mount_point or path.startswith(mount_point.rstrip("/") + "/")) and len(mount_point) > len(best):
                best, fstype = mount_point, fields[2]
    return fstype


def check_state_dir(path: str, mounts: str = "/proc/mounts") -> None:
    """Raise if ``path`` is on a memory-backed filesystem; warn if it is on a network filesystem."""
    existing = os.path.abspath(path)
    while not os.path.exists(existing):
        existing = os.path.dirname(existing)
    fstype = filesystem_type(existing, mounts)
    if fstype in _MEMORY_FILESYSTEMS:
        raise ValueError(f"optimizer_offload_to_disk.dir={path!r} is on {fstype}, which is host memory.")
    if fstype.startswith(("nfs", "cifs", "fuse", "lustre", "gpfs", "ceph")):
        logger.warning(
            f"optimizer_offload_to_disk.dir={path!r} is on {fstype}; the whole optimizer state is read and written "
            "every step."
        )


def split_shard(start: int, end: int, params: List[Run], limit: int) -> List[Tuple[int, int, List[Run]]]:
    """Split a rank's shard ``[start, end)`` of a DDP bucket into chunks of at most ``limit`` elements.

    ``params`` are the ranges of the params (or parts of params) in the shard, with their param group. Consecutive
    params of the same group merge into one run, including the zero padding between them. Returns
    ``(chunk_start, chunk_end, runs)`` per chunk, with the runs clipped to the chunk.
    """
    merged: List[Run] = []
    for p_start, p_end, group in sorted(params):
        assert start <= p_start < p_end <= end, f"param range [{p_start}, {p_end}) outside shard [{start}, {end})"
        if merged and merged[-1][2] == group:
            merged[-1] = (merged[-1][0], p_end, group)
        else:
            merged.append((p_start, p_end, group))
    chunks = []
    for c_start in range(start, end, limit):
        c_end = min(c_start + limit, end)
        runs = [(max(s, c_start), min(e, c_end), g) for s, e, g in merged if s < c_end and e > c_start]
        chunks.append((c_start, c_end, runs))
    return chunks


@dataclass
class _Chunk:
    buffer_index: int
    start: int
    end: int
    runs: List[Run]
    main_on_disk: bool
    """False for fp32 DDP buffers, whose main params are the param buffer itself."""
    offset: int = 0
    """File offset of the chunk's first segment."""

    @property
    def numel(self) -> int:
        return self.end - self.start


def _transfer_all(op, fd: int, offset: int, buf) -> None:
    """``os.preadv`` / ``os.pwritev`` all of ``buf`` at ``offset``, continuing after short transfers."""
    view = memoryview(buf).cast("B")
    while len(view):
        n = op(fd, [view], offset)
        if n <= 0:
            raise OSError(f"{op.__name__} transferred {n} bytes at offset {offset}")
        view, offset = view[n:], offset + n


class _StateFile:
    """A preallocated file of raw tensor bytes, read and written through a pinned host buffer."""

    def __init__(self, path: str, nbytes: int, staging_bytes: int):
        self.path = path
        self.fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
        try:
            # Reserving the blocks up front makes a full disk fail here rather than mid-step. Unwritten blocks read
            # as zeros, also on the ftruncate fallback.
            os.posix_fallocate(self.fd, 0, max(nbytes, 1))
        except OSError as e:
            if e.errno not in (errno.EOPNOTSUPP, errno.ENOTSUP, errno.EINVAL):
                raise
            os.ftruncate(self.fd, nbytes)
        self._staging = torch.empty(staging_bytes, dtype=torch.uint8, pin_memory=True)
        self._host = self._staging.numpy()

    def read(self, offset: int, out: torch.Tensor) -> None:
        """Fill the contiguous tensor ``out`` with the bytes at ``offset``."""
        data = out.view(-1).view(torch.uint8)
        step = self._staging.numel()
        for pos in range(0, data.numel(), step):
            n = min(step, data.numel() - pos)
            _transfer_all(os.preadv, self.fd, offset + pos, self._host[:n])
            data[pos : pos + n].copy_(self._staging[:n])

    def write(self, offset: int, tensor: torch.Tensor) -> int:
        """Write the bytes of the contiguous tensor ``tensor`` at ``offset``."""
        data = tensor.view(-1).view(torch.uint8)
        step = self._staging.numel()
        for pos in range(0, data.numel(), step):
            n = min(step, data.numel() - pos)
            self._staging[:n].copy_(data[pos : pos + n])
            _transfer_all(os.pwritev, self.fd, offset + pos, self._host[:n])
        return data.numel()

    def close(self) -> None:
        try:
            os.close(self.fd)
        except OSError:
            pass


class _HostState:
    """The first ``nbytes`` of a store's state in page-locked host memory (``storage="host"``).

    One pageable allocation registered with CUDA (``cudaHostRegister``) rather than ``pin_memory=True``, whose caching
    host allocator rounds each allocation up to a power of two (138 GiB would take 256 GiB).
    """

    def __init__(self, nbytes: int, file: "_StateFile"):
        """Allocate, fill from ``file``'s first ``nbytes`` and register."""
        self.nbytes = nbytes
        self.data = torch.empty(max(nbytes, 1), dtype=torch.uint8)
        host = self.data.numpy()
        step = 1 << 30
        for pos in range(0, nbytes, step):
            _transfer_all(os.preadv, file.fd, pos, host[pos : min(pos + step, nbytes)])
        err = torch.cuda.cudart().cudaHostRegister(self.data.data_ptr(), self.data.numel(), 0)
        if int(err) != 0:
            raise RuntimeError(f"cudaHostRegister of {nbytes / 1024**3:.1f} GiB failed: {err}")
        self._registered = True
        self._streams = None

    def streams(self):
        """The (host -> GPU, GPU -> host) copy streams, created on first use (on the current device)."""
        if self._streams is None:
            self._streams = (torch.cuda.Stream(), torch.cuda.Stream())
        return self._streams

    def view(self, offset: int, nbytes: int) -> torch.Tensor:
        return self.data[offset : offset + nbytes]

    def read(self, offset: int, out: torch.Tensor) -> None:
        data = out.view(-1).view(torch.uint8)
        data.copy_(self.view(offset, data.numel()))

    def write(self, offset: int, tensor: torch.Tensor) -> int:
        data = tensor.view(-1).view(torch.uint8)
        self.view(offset, data.numel()).copy_(data)
        return data.numel()

    def write_to(self, fd: int) -> None:
        """Write the held bytes at the start of ``fd``."""
        torch.cuda.synchronize()  # pending GPU -> host copies of the last step
        host = self.data.numpy()
        step = 1 << 30
        for pos in range(0, self.nbytes, step):
            _transfer_all(os.pwritev, fd, pos, host[pos : min(pos + step, self.nbytes)])

    def read_from(self, fd: int) -> None:
        """Fill the held bytes from the start of ``fd``."""
        torch.cuda.synchronize()
        host = self.data.numpy()
        step = 1 << 30
        for pos in range(0, self.nbytes, step):
            _transfer_all(os.preadv, fd, pos, host[pos : min(pos + step, self.nbytes)])

    def close(self) -> None:
        if self._registered:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaHostUnregister(self.data.data_ptr())
            self._registered = False
        self.data = torch.empty(0, dtype=torch.uint8)


def _mem_available_bytes(meminfo: str = "/proc/meminfo") -> Tuple[int, int]:
    """(MemAvailable, MemTotal) in bytes."""
    values = {}
    with open(meminfo) as f:
        for line in f:
            key, rest = line.split(":", 1)
            values[key] = int(rest.split()[0]) * 1024
    return values["MemAvailable"], values["MemTotal"]


def host_grant_bytes(need: int, available: int, total: int, reserve_fraction: float, ranks: int) -> int:
    """Bytes of host memory a store may take: an equal share, among the node's ``ranks``, of what was available
    beyond ``reserve_fraction`` of the node's memory; at most ``need``."""
    spare = available - int(reserve_fraction * total)
    return max(0, min(need, spare // max(1, ranks)))


# Bytes this process has already placed in host memory, per ledger: a rank's stores (dense and expert) share one
# per-rank budget, taken from a single per-node MemAvailable reading.
_HOST_PLACED: Dict[str, int] = {}


@contextmanager
def _node_ledger(path: str):
    """A node-local JSON ledger under an exclusive lock: ``{"node": [MemAvailable, MemTotal]}``."""
    fd = os.open(path + ".lock", os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        ledger = {}
        if os.path.exists(path):
            with open(path) as f:
                ledger = json.load(f)
        yield ledger
        with open(path, "w") as f:
            json.dump(ledger, f)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


class DiskOptimizerStateStore:
    """The on-disk state of one ``OffloadedDistributedOptimizer`` and the streamed step over it."""

    def __init__(self, dist_opt: DistributedOptimizer, settings: MegatronOptimizerOffloadToDiskConfig, index: int):
        from megatron.core.fp4_utils import is_nvfp4tensor
        from megatron.core.fp8_utils import is_float8tensor

        self.dist_opt = dist_opt
        self.index = index
        config, ddp_config = dist_opt.config, dist_opt.ddp_config
        if config.use_precision_aware_optimizer or config.optimizer_cpu_offload:
            raise ValueError(
                "optimizer_offload_to_disk requires use_precision_aware_optimizer=False and optimizer_cpu_offload=False."
            )
        if not config.bf16:
            raise ValueError("optimizer_offload_to_disk requires bf16 training.")
        if config.optimizer_cuda_graph:
            raise ValueError("optimizer_offload_to_disk does not support optimizer_cuda_graph.")
        if ddp_config.use_megatron_fsdp:
            raise ValueError("optimizer_offload_to_disk does not support Megatron-FSDP.")
        params = [p for group in dist_opt.model_float16_groups for p in group]
        # The fp8 and NVFP4 main -> param copies are DP collectives over the whole param set (quantize_param_shard),
        # which cannot be split into chunks.
        if (
            config.reuse_grad_buf_for_mxfp8_param_ag
            or ddp_config.fp8_param_gather
            or getattr(ddp_config, "fp4_param_gather", False)
            or any(is_float8tensor(p) or is_nvfp4tensor(p) for p in params)
        ):
            raise ValueError("optimizer_offload_to_disk does not support fp8 or fp4 params.")
        if not isinstance(dist_opt.optimizer, Adam):
            raise ValueError(f"optimizer_offload_to_disk supports Adam only, got {type(dist_opt.optimizer).__name__}.")
        self.moment_dtype = MOMENT_STORAGE_DTYPES[settings.moment_storage_dtype]
        self.steps = 0

        self.chunks = self._build_chunks(SKYRL_OPTIMIZER_OFFLOAD_TO_DISK_CHUNK_NUMEL)
        nbytes = 0
        for chunk in self.chunks:
            chunk.offset = nbytes
            nbytes += self._chunk_nbytes(chunk)

        self.nbytes = nbytes
        self.dir = os.path.join(settings.dir, self.relative_dir)
        os.makedirs(self.dir, exist_ok=True)
        # The file always holds the state; with storage="host" a prefix of the chunks moves to page-locked host
        # memory at the first step (_place_in_host), after the trainer's start-up allocations are gone.
        self.file = _StateFile(os.path.join(self.dir, "state.bin"), nbytes, settings.staging_buffer_mb * 1024 * 1024)
        self.host: Optional[_HostState] = None
        self.host_chunks = 0
        self.host_bytes = 0
        self.settings = settings
        self._place_host_pending = settings.storage == "host"
        self.ledger = None if _build_context is None else _build_context.ledger
        self.local_ranks = 1 if _build_context is None else _build_context.local_ranks
        logger.info(
            f"optimizer_offload_to_disk: {len(self.chunks)} chunks, {nbytes / 1024**3:.1f} GiB at {self.file.path}, "
            f"moments stored as {settings.moment_storage_dtype}"
        )

    @property
    def relative_dir(self) -> str:
        """This store's location under any base directory, scratch or checkpoint.

        ``index`` counts the non-stub optimizers built in one ``get_megatron_optimizer`` call, which
        disambiguates the dense and expert optimizers of a chain.
        """
        return os.path.join(f"rank{torch.distributed.get_rank():05d}", f"opt{self.index}")

    def _build_chunks(self, limit: int) -> List[_Chunk]:
        dist_opt = self.dist_opt
        chunks = []
        for buffer_index, buffer in enumerate(dist_opt.buffers):
            if buffer.param_data is None or buffer.param_data.dtype not in (torch.bfloat16, torch.float32):
                raise ValueError("optimizer_offload_to_disk requires bf16 or fp32 DDP param buffers.")
            assert buffer.grad_data.dtype in (torch.float32, buffer.param_dtype), buffer.grad_data.dtype
            dp_rank, dp_size = buffer.data_parallel_group.rank(), buffer.data_parallel_group.size()
            (bucket_ranges,) = dist_opt.gbuf_ranges[buffer_index].values()
            for bucket, bucket_range in zip(buffer.buckets, bucket_ranges, strict=True):
                shard_size = bucket.grad_data.numel() // dp_size
                shard_start = bucket.offset + dp_rank * shard_size
                params = [
                    (ranges["gbuf_world"].start, ranges["gbuf_world"].end, dist_opt.model_param_group_index_map[p][0])
                    for p, ranges in bucket_range["param_map"].items()
                ]
                for start, end, runs in split_shard(shard_start, shard_start + shard_size, params, limit):
                    if runs:
                        chunks.append(_Chunk(buffer_index, start, end, runs, buffer.param_dtype != torch.float32))
        return chunks

    def _segments(self, chunk: _Chunk) -> List[Tuple[str, torch.dtype]]:
        main = [("main", torch.float32)] if chunk.main_on_disk else []
        return main + [("exp_avg", self.moment_dtype), ("exp_avg_sq", self.moment_dtype)]

    def _chunk_nbytes(self, chunk: _Chunk) -> int:
        return sum(chunk.numel * dtype.itemsize for _, dtype in self._segments(chunk))

    def _param_data(self, chunk: _Chunk) -> torch.Tensor:
        return self.dist_opt.buffers[chunk.buffer_index].param_data[chunk.start : chunk.end]

    def _grad_data(self, chunk: _Chunk) -> torch.Tensor:
        return self.dist_opt.buffers[chunk.buffer_index].grad_data[chunk.start : chunk.end]

    def _read(self, offset: int, out: torch.Tensor) -> None:
        if offset < self.host_bytes:
            self.host.read(offset, out)
        else:
            self.file.read(offset, out)

    def _write(self, offset: int, tensor: torch.Tensor) -> int:
        if offset < self.host_bytes:
            return self.host.write(offset, tensor)
        return self.file.write(offset, tensor)

    def _place_in_host(self) -> None:
        """Move the longest prefix of chunks that fits this store's share of the node's spare memory from the file
        into page-locked host memory (``storage="host"``; once, at the first step).

        The share is an equal split, among the node's ranks, of MemAvailable beyond ``host_reserve_fraction`` of
        MemTotal (kept for the colocated trainer's offloaded params, the inference engines and the page cache),
        capped by ``host_max_gb_per_rank``. MemAvailable is read once per node, by the first rank to get here (a
        node-local ledger under a lock), before any rank allocates; every rank then allocates and fills its share
        in parallel. A rank's later stores get what its earlier ones left of its share, so no store's placement
        depends on how far the other ranks have got.
        """
        self._place_host_pending = False
        placed = 0
        if self.ledger is None:
            available, total = _mem_available_bytes()
        else:
            with _node_ledger(self.ledger) as ledger:
                if "node" not in ledger:
                    ledger["node"] = list(_mem_available_bytes())
                available, total = ledger["node"]
            placed = _HOST_PLACED.get(self.ledger, 0)
        need = placed + self.nbytes
        if self.settings.host_max_gb_per_rank is not None:
            need = min(need, int(self.settings.host_max_gb_per_rank * 1024**3))
        share = host_grant_bytes(need, available, total, self.settings.host_reserve_fraction, self.local_ranks)
        self._allocate_host(max(0, share - placed))
        if self.ledger is not None:
            _HOST_PLACED[self.ledger] = placed + self.host_bytes
        logger.info(
            f"optimizer_offload_to_disk (rank {torch.distributed.get_rank()}, opt{self.index}): {self.host_chunks} of "
            f"{len(self.chunks)} chunks ({self.host_bytes / 1024**3:.1f} of {self.nbytes / 1024**3:.1f} GiB) in host "
            f"memory; node MemAvailable was {available / 1024**3:.0f} of {total / 1024**3:.0f} GiB"
        )

    def _allocate_host(self, grant: int) -> None:
        k, nbytes = 0, 0
        while k < len(self.chunks) and nbytes + self._chunk_nbytes(self.chunks[k]) <= grant:
            nbytes += self._chunk_nbytes(self.chunks[k])
            k += 1
        if k == 0:
            return
        self.host = _HostState(nbytes, self.file)
        self.host_chunks, self.host_bytes = k, nbytes

    def load_chunk(self, chunk: _Chunk) -> dict:
        """The chunk's fp32 main, exp_avg and exp_avg_sq on the GPU (the main is a param buffer view if not on disk)."""
        state = {} if chunk.main_on_disk else {"main": self._param_data(chunk)}
        offset = chunk.offset
        for name, dtype in self._segments(chunk):
            stored = torch.empty(chunk.numel, dtype=dtype, device=torch.cuda.current_device())
            self._read(offset, stored)
            state[name] = stored.float()
            offset += chunk.numel * dtype.itemsize
        return state

    def _store_chunk(self, chunk: _Chunk, state: dict) -> None:
        offset = chunk.offset
        for name, dtype in self._segments(chunk):
            self._write(offset, state[name].to(dtype))
            offset += chunk.numel * dtype.itemsize

    def _unpack(self, chunk: _Chunk, stored: torch.Tensor) -> dict:
        """The chunk's state as fp32 GPU tensors from ``stored``, its segments' bytes on the GPU (fp32 segments are
        views of ``stored``)."""
        state = {} if chunk.main_on_disk else {"main": self._param_data(chunk)}
        pos = 0
        for name, dtype in self._segments(chunk):
            n = chunk.numel * dtype.itemsize
            state[name] = stored[pos : pos + n].view(dtype).float()
            pos += n
        return state

    def _pack(self, chunk: _Chunk, state: dict, stored: torch.Tensor) -> None:
        """Write ``state`` back into ``stored`` in the stored dtypes (a no-op for fp32 views of it)."""
        pos = 0
        for name, dtype in self._segments(chunk):
            n = chunk.numel * dtype.itemsize
            view = stored[pos : pos + n].view(dtype)
            if view.data_ptr() != state[name].data_ptr():
                view.copy_(state[name])
            pos += n

    def _adam(self, chunk: _Chunk, state: dict):
        """The regular optimizer's Adam over this chunk's runs, with each run's param group hyperparameters."""
        grads = self._grad_data(chunk)
        if grads.dtype != torch.float32:
            grads = grads.float()  # bf16 grad buffer: this chunk's gradient in fp32, for its step only
        params_by_group: dict = {}
        moments = {}
        for start, end, group_index in chunk.runs:
            a, b = start - chunk.start, end - chunk.start
            param = torch.nn.Parameter(state["main"][a:b], requires_grad=False)
            param.grad = grads[a:b]
            moments[param] = {"exp_avg": state["exp_avg"][a:b], "exp_avg_sq": state["exp_avg_sq"][a:b]}
            params_by_group.setdefault(group_index, []).append(param)
        groups = []
        for group_index, params in params_by_group.items():
            group = {k: v for k, v in self.dist_opt.optimizer.param_groups[group_index].items() if k != "params"}
            groups.append({**group, "params": params, "step": self.steps})
        adam = Adam(groups, adam_w_mode=self.dist_opt.config.decoupled_weight_decay)
        for param, param_state in moments.items():
            adam.state[param] = param_state
        return adam

    @torch.no_grad()
    def step(self) -> None:
        """Adam-step every chunk and copy its updated main into the param buffer.

        The grad buffer was reduced and clipped by megatron-core before this runs.
        """
        if self._place_host_pending:
            self._place_in_host()
        clock = time.perf_counter
        started = clock()
        if self.host_chunks:
            self._step_from_host(self.chunks[: self.host_chunks])
        host_s = clock() - started
        spent = {"read": 0.0, "adam": 0.0, "write": 0.0}
        nbytes = 0
        for chunk in self.chunks[self.host_chunks :]:
            t0 = clock()
            state = self.load_chunk(chunk)
            torch.cuda.synchronize()
            t1 = clock()
            self._adam(chunk, state).step()
            if chunk.main_on_disk:
                self._param_data(chunk).copy_(state["main"])
            torch.cuda.synchronize()
            t2 = clock()
            self._store_chunk(chunk, state)
            spent["read"] += t1 - t0
            spent["adam"] += t2 - t1
            spent["write"] += clock() - t2
            nbytes += self._chunk_nbytes(chunk)
            # Free this chunk before the next load allocates one, so the GPU holds one chunk at a time.
            del state
        self.steps += 1
        if torch.distributed.get_rank() == 0:
            torch.cuda.synchronize()
            logger.info(
                f"optimizer_offload_to_disk step (opt{self.index}): host {self.host_bytes / 1024**3:.1f} GiB in "
                f"{host_s:.1f}s, disk {nbytes / 1024**3:.1f} GiB: "
                + ", ".join(f"{k} {v:.1f}s" for k, v in spent.items())
            )

    @torch.no_grad()
    def _step_from_host(self, chunks: List[_Chunk]) -> None:
        """The step over the chunks in page-locked host memory, pipelined over three CUDA streams: chunk ``i + 1`` is copied in
        (``h2d``) and chunk ``i - 1`` out (``d2h``) while chunk ``i``'s Adam step runs, so the step takes about as
        long as its host <-> GPU traffic. The GPU holds two chunks' stored bytes plus one chunk's fp32 working set."""
        compute = torch.cuda.current_stream()
        h2d, d2h = self.host.streams()
        device = torch.cuda.current_device()

        def fetch(chunk):
            nbytes = self._chunk_nbytes(chunk)
            with torch.cuda.stream(h2d):
                stored = torch.empty(nbytes, dtype=torch.uint8, device=device)
                stored.copy_(self.host.view(chunk.offset, nbytes), non_blocking=True)
                ready = torch.cuda.Event()
                ready.record(h2d)
            return stored, ready

        # The step reads gradients and writes params on the compute stream: order it after what ran there before.
        h2d.wait_stream(compute)
        nxt = fetch(chunks[0]) if chunks else None
        for i, chunk in enumerate(chunks):
            stored, ready = nxt
            nxt = fetch(chunks[i + 1]) if i + 1 < len(chunks) else None
            compute.wait_event(ready)
            stored.record_stream(compute)
            state = self._unpack(chunk, stored)
            self._adam(chunk, state).step()
            if chunk.main_on_disk:
                self._param_data(chunk).copy_(state["main"])
            self._pack(chunk, state, stored)
            del state
            d2h.wait_stream(compute)
            with torch.cuda.stream(d2h):
                self.host.view(chunk.offset, stored.numel()).copy_(stored, non_blocking=True)
                stored.record_stream(d2h)
            del stored
        compute.wait_stream(d2h)

    @torch.no_grad()
    def copy_main_to_model_params(self) -> None:
        for chunk in self.chunks:
            if chunk.main_on_disk:
                main = torch.empty(chunk.numel, dtype=torch.float32, device=torch.cuda.current_device())
                self._read(chunk.offset, main)
                self._param_data(chunk).copy_(main)
                del main

    @torch.no_grad()
    def write_main_params(self) -> int:
        """Set the on-disk mains from the model params (the param buffer)."""
        written = 0
        for chunk in self.chunks:
            if chunk.main_on_disk:
                written += self._write(chunk.offset, self._param_data(chunk).float())
        return written

    def close(self) -> None:
        """Close the state file and delete this store's directory, then the rank and run directories if empty."""
        if self.host is not None:
            self.host.close()
            self.host, self.host_chunks, self.host_bytes = None, 0, 0
        if self.ledger is not None:
            for path in (self.ledger, self.ledger + ".lock"):
                try:
                    os.remove(path)
                except OSError:
                    pass  # another rank of the node removed it
        self.file.close()
        shutil.rmtree(self.dir, ignore_errors=True)
        rank_dir = os.path.dirname(self.dir)
        try:
            os.rmdir(rank_dir)
        except OSError:
            return  # another store of this rank still uses it
        _unlock_rank_dir(rank_dir)
        try:
            os.rmdir(os.path.dirname(rank_dir))
        except OSError:
            pass  # another rank still uses it

    def _manifest(self) -> dict:
        return {
            "moment_dtype": str(self.moment_dtype),
            "steps": self.steps,
            "chunks": [[c.buffer_index, c.start, c.end, c.main_on_disk] for c in self.chunks],
        }

    def save_to(self, base_dir: str) -> None:
        dirpath = os.path.join(base_dir, self.relative_dir)
        os.makedirs(dirpath, exist_ok=True)
        target = os.path.join(dirpath, "state.bin")
        shutil.copyfile(self.file.path, target)  # the host-held prefix in it is stale: overwritten next
        if self.host is not None:
            fd = os.open(target, os.O_WRONLY | os.O_CLOEXEC)
            try:
                self.host.write_to(fd)
            finally:
                os.close(fd)
        with open(os.path.join(dirpath, "manifest.json"), "w") as f:
            json.dump(self._manifest(), f)
        logger.info(f"optimizer_offload_to_disk: saved {len(self.chunks)} chunks to {dirpath}")

    @torch.no_grad()
    def load_from(self, base_dir: str) -> None:
        dirpath = os.path.join(base_dir, self.relative_dir)
        if not os.path.isfile(os.path.join(dirpath, "manifest.json")):
            raise FileNotFoundError(
                f"No offloaded optimizer state at {dirpath}. Resume requires a checkpoint saved with "
                "optimizer_offload_to_disk and the same parallelism."
            )
        with open(os.path.join(dirpath, "manifest.json")) as f:
            manifest = json.load(f)
        current = self._manifest()
        if manifest["moment_dtype"] != current["moment_dtype"]:
            raise ValueError(
                f"Offloaded optimizer state at {dirpath} stores moments as {manifest['moment_dtype']}, this run as "
                f"{current['moment_dtype']}; moment_storage_dtype must match the checkpoint."
            )
        if manifest["chunks"] != current["chunks"]:
            raise ValueError(
                f"Offloaded optimizer state at {dirpath} has a different chunk layout than this run; resume needs "
                "the same parallelism, DDP bucket size and SKYRL_OPTIMIZER_OFFLOAD_TO_DISK_CHUNK_NUMEL."
            )
        source = os.path.join(dirpath, "state.bin")
        expected = sum(self._chunk_nbytes(c) for c in self.chunks)
        if os.path.getsize(source) != expected:
            raise ValueError(f"{source} has {os.path.getsize(source)} bytes, expected {expected}.")
        shutil.copyfile(source, self.file.path)
        if self.host is not None:
            self.host.read_from(self.file.fd)
        self.steps = manifest["steps"]
        logger.info(f"optimizer_offload_to_disk: loaded {len(self.chunks)} chunks from {dirpath}")


_ANCHOR = "shard_main_param = shard_model_param.clone().float()"
# ``_skyrl_main_dtype`` is set in the generated method's globals for each build: fp32, or the model param's dtype
# when the grad buffer is (``grad_reduce_in_fp32=False``), so ``.grad`` can be a view of it.
_DEFERRED = (
    "shard_main_param = torch.empty_like(shard_model_param, dtype=_skyrl_main_dtype or torch.float32)",
    "shard_main_param.untyped_storage().resize_(0)",
)


def _deferred_build_model_and_main_param_groups() -> classmethod:
    """``DistributedOptimizer._build_model_and_main_param_groups`` creating fp32 main shards with empty storage.

    Generated from the installed megatron-core source by replacing the one line that allocates a main shard as a
    copy of the model shard, so the rest of the method tracks megatron-core. The tensors keep their identity,
    shape and attributes. Raises if a megatron-core bump moves the line, rather than silently allocating every
    main shard on the GPU.
    """
    original = DistributedOptimizer.__dict__["_build_model_and_main_param_groups"]
    assert isinstance(original, classmethod)
    func = original.__func__
    lines = textwrap.dedent(inspect.getsource(func)).splitlines()
    hits = [i for i, line in enumerate(lines) if line.strip() == _ANCHOR]
    if len(hits) != 1:
        raise RuntimeError(
            f"optimizer_offload_to_disk: expected exactly one {_ANCHOR!r} in "
            f"DistributedOptimizer._build_model_and_main_param_groups, found {len(hits)}. "
            "The installed megatron-core changed this method; update optimizer_offload_to_disk.py for it."
        )
    at = hits[0]
    indent = lines[at][: len(lines[at]) - len(lines[at].lstrip())]
    lines[at : at + 1] = [indent + line for line in _DEFERRED]
    while lines[0].lstrip().startswith("@"):
        lines.pop(0)  # rewrapped as a classmethod below

    namespace: dict = {}
    code = compile("\n".join(lines) + "\n", f"<optimizer_offload_to_disk: {inspect.getsourcefile(func)}>", "exec")
    # A copy of the module's globals, so ``_skyrl_main_dtype`` does not leak into megatron-core's namespace.
    method_globals = {**vars(_distrib_optimizer_module), "_skyrl_main_dtype": None}
    exec(code, method_globals, namespace)
    generated = namespace[func.__name__]
    generated.__qualname__ = f"OffloadedDistributedOptimizer.{func.__name__}"
    generated.__doc__ = func.__doc__
    return classmethod(generated)


@dataclass
class _BuildContext:
    settings: MegatronOptimizerOffloadToDiskConfig
    next_index: int = 0
    ledger: Optional[str] = None
    """Node-local ledger file for host-memory placement (``storage="host"``)."""
    local_ranks: int = 1
    """Ranks of this job on this node."""


_build_context: Optional[_BuildContext] = None


class OffloadedDistributedOptimizer(DistributedOptimizer):
    """A ``DistributedOptimizer`` whose fp32 main params and Adam moments live on disk (see the module docstring).

    Constructed by megatron-core's ``get_megatron_optimizer`` inside ``build_optimizer_offloaded_to_disk``, which
    first installs the generated ``_build_model_and_main_param_groups``.
    """

    def __init__(self, *args, **kwargs):
        context = _build_context
        assert context is not None, "construct through build_optimizer_offloaded_to_disk()"
        model_chunks = kwargs["model_chunks"] if "model_chunks" in kwargs else args[4]
        ddp_config = model_chunks[0].ddp_config
        build = type(self).__dict__["_build_model_and_main_param_groups"].__func__
        # bf16 grad buffer: bf16 main shards (see the module docstring); built below, by super().__init__.
        build.__globals__["_skyrl_main_dtype"] = None if ddp_config.grad_reduce_in_fp32 else torch.bfloat16
        try:
            super().__init__(*args, **kwargs)
        finally:
            build.__globals__["_skyrl_main_dtype"] = None
        self.disk_state: Optional[DiskOptimizerStateStore] = None
        if self.is_stub_optimizer:
            return
        self.disk_state = DiskOptimizerStateStore(self, context.settings, context.next_index)
        context.next_index += 1
        written = self.disk_state.write_main_params()
        logger.info(f"optimizer_offload_to_disk: wrote {written / 1024**3:.1f} GiB of main params")

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        if self.is_stub_optimizer:
            return super().step_with_ready_grads()
        self.disk_state.step()
        # The inner optimizer step and the main -> model param copy of MixedPrecisionOptimizer.step_with_ready_grads
        # just ran chunk by chunk. Marked as a stub, it skips both, and DistributedOptimizer.step_with_ready_grads
        # still runs its param all-gather.
        self.is_stub_optimizer = True
        try:
            return super().step_with_ready_grads()
        finally:
            self.is_stub_optimizer = False

    def _copy_model_grads_to_main_grads(self):
        if self.ddp_config.grad_reduce_in_fp32 or self.is_stub_optimizer:
            return super()._copy_model_grads_to_main_grads()
        # megatron-core's, without the ``.float()``: each main shard's ``.grad`` is a view of the grad buffer in the
        # main shard's own dtype (bf16 for bf16 params, whose grad buffer is bf16; fp32 for native-fp32 params).
        for model_groups, shard_main_groups in (
            (self.model_float16_groups, self.shard_fp32_from_float16_groups),
            (self.model_fp32_groups, self.shard_fp32_groups),
        ):
            for model_group, shard_main_group in zip(model_groups, shard_main_groups):
                for model_param, shard_main_param in zip(model_group, shard_main_group):
                    param_range = self._get_model_param_range_map(model_param)["param"]
                    assert param_range.size == shard_main_param.nelement()
                    shard_grad = model_param.main_grad.view(-1)[param_range.start : param_range.end]
                    shard_main_param.grad = shard_grad.to(shard_main_param.dtype)

    @torch.no_grad()
    def _copy_main_params_to_model_params(self):
        if self.is_stub_optimizer:
            return
        self.disk_state.copy_main_to_model_params()

    @torch.no_grad()
    def _copy_model_params_to_main_params(self, state_dict=None):
        if self.is_stub_optimizer:
            return
        if state_dict is not None:
            raise NotImplementedError("optimizer_offload_to_disk does not reload main params from a state dict.")
        self.disk_state.write_main_params()

    # The state is in the store's file, saved and loaded by save_to() / load_from(). The marker keeps the
    # optimizer's entry in Megatron's distributed checkpoint, which ChainedOptimizer.load_state_dict counts.
    def state_dict(self):
        return {"optimizer_offload_to_disk": True}

    def load_state_dict(self, state_dict):
        pass

    def sharded_state_dict(self, model_sharded_state_dict, is_loading: bool = False, **kwargs):
        return self.state_dict()


@contextmanager
def _construct_distributed_optimizer_as(cls) -> Iterator[None]:
    """Make megatron-core's ``get_megatron_optimizer``, which looks ``DistributedOptimizer`` up by name, build ``cls``."""
    original = mcore_optimizer.DistributedOptimizer
    mcore_optimizer.DistributedOptimizer = cls
    try:
        yield
    finally:
        mcore_optimizer.DistributedOptimizer = original


_rank_dir_locks: dict = {}


def _lock_rank_dir(rank_dir: str) -> None:
    """Hold an exclusive lock on ``rank_dir`` for the life of the process, so two runs cannot share (and clear) it."""
    if rank_dir in _rank_dir_locks:
        return
    os.makedirs(os.path.dirname(rank_dir), exist_ok=True)
    fd = os.open(rank_dir + ".lock", os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise RuntimeError(
            f"{rank_dir} is in use by another process; give concurrent runs different trainer.run_name values."
        ) from None
    _rank_dir_locks[rank_dir] = fd


def _unlock_rank_dir(rank_dir: str) -> None:
    fd = _rank_dir_locks.pop(rank_dir, None)
    if fd is not None:
        try:
            os.unlink(rank_dir + ".lock")
        except OSError:
            pass
        os.close(fd)


def build_optimizer_offloaded_to_disk(model_chunks, config, settings: MegatronOptimizerOffloadToDiskConfig):
    """``get_megatron_optimizer`` with every ``DistributedOptimizer`` built as an ``OffloadedDistributedOptimizer``.

    ``settings.dir`` is this run's state directory; each rank uses ``<dir>/rank<global rank>/``."""
    global _build_context

    if not config.use_distributed_optimizer:
        raise ValueError("optimizer_offload_to_disk requires use_distributed_optimizer=True")
    check_state_dir(settings.dir)
    # Clear what an earlier run of this rank left behind (a killed run never reaches its cleanup); the rank
    # directory is this rank's alone.
    rank_dir = os.path.join(settings.dir, f"rank{torch.distributed.get_rank():05d}")
    _lock_rank_dir(rank_dir)
    shutil.rmtree(rank_dir, ignore_errors=True)
    os.makedirs(rank_dir, exist_ok=True)

    if any(not chunk.ddp_config.grad_reduce_in_fp32 for chunk in model_chunks):
        # bf16 grad buffer: the main grads megatron-core norms and clips are bf16 views, plus fp32 ones for
        # native-fp32 params, and its multi-tensor kernels need a list of one dtype.
        from skyrl.backends.skyrl_train.patches.megatron.patch_clip_grads_mixed_dtype import (
            apply_clip_grads_mixed_dtype_patch,
        )

        apply_clip_grads_mixed_dtype_patch()

    if "_build_model_and_main_param_groups" not in OffloadedDistributedOptimizer.__dict__:
        OffloadedDistributedOptimizer._build_model_and_main_param_groups = _deferred_build_model_and_main_param_groups()

    _build_context = _BuildContext(settings)
    if settings.storage == "host":
        # Collective (every rank builds its optimizer): the ranks sharing this node, and a ledger name unique to this
        # launch, so a relaunch on the same node does not read a dead run's ledger.
        import socket

        hosts = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(hosts, socket.gethostname())
        nonce = [os.urandom(8).hex() if torch.distributed.get_rank() == 0 else None]
        torch.distributed.broadcast_object_list(nonce, src=0)
        _build_context.local_ranks = hosts.count(socket.gethostname())
        # In /dev/shm: node-local, and outside the state directory, which close() removes.
        _build_context.ledger = f"/dev/shm/skyrl_optimizer_host_ledger_{nonce[0]}.json"
    try:
        with _construct_distributed_optimizer_as(OffloadedDistributedOptimizer):
            optimizer = get_megatron_optimizer(config=config, model_chunks=model_chunks)
    finally:
        _build_context = None
    if not all(
        isinstance(opt, OffloadedDistributedOptimizer) for opt in getattr(optimizer, "chained_optimizers", [optimizer])
    ):
        raise RuntimeError(
            "get_megatron_optimizer did not build an OffloadedDistributedOptimizer: this megatron-core version no "
            "longer constructs DistributedOptimizer through the name build_optimizer_offloaded_to_disk replaces."
        )
    return optimizer


def disk_state_stores(optimizer) -> List[DiskOptimizerStateStore]:
    """The stores of ``optimizer``'s offloaded sub-optimizers; empty for any other optimizer."""
    optimizers = getattr(optimizer, "chained_optimizers", [optimizer])
    return [
        opt.disk_state
        for opt in optimizers
        if isinstance(opt, OffloadedDistributedOptimizer) and opt.disk_state is not None
    ]
