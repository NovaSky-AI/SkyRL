"""Gather many small FSDP2 ``DTensor`` shards with one collective per bucket.

``DTensor.full_tensor()`` is one all-gather per tensor. Over tens of thousands of LoRA
tensors that is tens of thousands of collectives, each paying the DTensor dispatch and
the NCCL launch/latency for a few hundred KiB. FSDP2 itself never gathers like that: it
copies the shards of a whole unit into one flat buffer and runs a single
``all_gather_into_tensor``. :func:`iter_full_tensors` does the same for an arbitrary
list of shards, bucket by bucket, and returns exactly the bytes ``full_tensor()`` returns.

Only ``Shard(0)`` on a 1-D mesh (FSDP2's layout) is batched. Anything else (plain
tensors, other placements, multi-dim meshes, empty tensors) goes through
``full_tensor()`` one by one, in order.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence

import torch
import torch.distributed as dist
from loguru import logger

#: Bytes of full (gathered) tensors per all-gather.
DEFAULT_GATHER_BUCKET_BYTES = 256 * 1024 * 1024


def _materialize(t: torch.Tensor) -> torch.Tensor:
    full_tensor = getattr(t, "full_tensor", None)
    return full_tensor() if callable(full_tensor) else t


def _batch_key(t: torch.Tensor) -> tuple | None:
    """Group key for a batchable FSDP2 shard, ``None`` if it must take ``full_tensor()``."""
    from torch.distributed.tensor import DTensor, Shard

    if not isinstance(t, DTensor) or t.numel() == 0 or t.dim() == 0:
        return None
    mesh = t.device_mesh
    if mesh.ndim != 1 or tuple(t.placements) != (Shard(0),):
        return None
    return (id(mesh), t.dtype, t.to_local().device)


def _gathered_bytes(t: torch.Tensor) -> int:
    """Receive-buffer bytes of one tensor: dim 0 padded to ``world * ceil(d0 / world)``."""
    world = t.device_mesh.size()
    rows = t.shape[0]
    return world * -(-rows // world) * (t.numel() // rows) * t.element_size()


def _gather_bucket(shards: Sequence[torch.Tensor]) -> Iterator[torch.Tensor]:
    """Full tensors of ``shards`` (same mesh, dtype, device; ``Shard(0)``) via one all-gather.

    The all-gather runs on the first ``next()``; each full tensor is cut out of the receive
    buffer only when it is consumed, so the bucket costs its receive buffer plus one tensor.

    ``Shard(0)`` splits dim 0 like ``torch.chunk``: rank ``r`` holds rows
    ``[r * c, min((r + 1) * c, d0))`` with ``c = ceil(d0 / world)``, so trailing ranks may
    hold fewer rows or none. Every rank contributes ``c`` rows per tensor (zero padded),
    the gathered ``(world, total)`` buffer is cut per tensor, and the padding is dropped.
    """
    mesh = shards[0].device_mesh
    group = mesh.get_group()
    world = mesh.size()
    rank = mesh.get_local_rank()
    dtype = shards[0].dtype
    local0 = shards[0].to_local()

    layout = []  # (offset, padded numel per rank, chunk rows, global shape)
    total = 0
    for t in shards:
        shape = tuple(t.shape)
        rows = shape[0]
        chunk = -(-rows // world)
        row_numel = t.numel() // rows
        expect_rows = max(0, min(chunk, rows - rank * chunk))
        local_rows = t.to_local().shape[0] if t.to_local().dim() else 0
        if local_rows != expect_rows:
            raise RuntimeError(
                f"unexpected Shard(0) layout: rank {rank} holds {local_rows} rows of {shape}, expected {expect_rows}"
            )
        layout.append((total, chunk * row_numel, chunk, shape))
        total += chunk * row_numel

    send = torch.zeros(total, dtype=dtype, device=local0.device)
    for t, (off, _, _, _) in zip(shards, layout):
        local = t.to_local().reshape(-1)
        send[off : off + local.numel()].copy_(local)
    recv = torch.empty(world * total, dtype=dtype, device=local0.device)
    dist.all_gather_into_tensor(recv, send, group=group)
    recv = recv.view(world, total)

    del send
    for off, padded, chunk, shape in layout:
        yield recv[:, off : off + padded].reshape(world * chunk, *shape[1:])[: shape[0]].contiguous()


def iter_full_tensors(
    tensors: Iterable[tuple[str, torch.Tensor]],
    bucket_bytes: int = DEFAULT_GATHER_BUCKET_BYTES,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(name, full tensor)`` for ``(name, shard)`` pairs, in order.

    Collective: every rank of the mesh must pass the same names, global shapes and dtypes
    in the same order (bucket boundaries depend only on those). A bucket is flushed when
    adding the next tensor would exceed ``bucket_bytes`` of gathered output, when the group
    key changes, and before any tensor that is not batchable. Yielded tensors are
    contiguous; one may be a view of the bucket's receive buffer (and keep it alive while held).
    The bucket budget counts the receive buffer, i.e. dim 0 padded to ``world * ceil(d0 / world)``.
    """
    pending: list[tuple[str, torch.Tensor]] = []
    pending_key = None
    pending_bytes = 0
    fallback = 0

    def flush() -> Iterator[tuple[str, torch.Tensor]]:
        nonlocal pending, pending_key, pending_bytes
        if not pending:
            return
        names = [n for n, _ in pending]
        fulls = _gather_bucket([t for _, t in pending])
        pending, pending_key, pending_bytes = [], None, 0
        yield from zip(names, fulls)

    for name, t in tensors:
        key = _batch_key(t)
        if key is None:
            yield from flush()
            fallback += 1
            yield name, _materialize(t)
            continue
        nbytes = _gathered_bytes(t)
        if pending and (key != pending_key or pending_bytes + nbytes > bucket_bytes):
            yield from flush()
        pending.append((name, t))
        pending_key = key
        pending_bytes += nbytes
    yield from flush()
    if fallback:
        logger.warning(f"iter_full_tensors: {fallback} tensors were not Shard(0) on a 1-D mesh; gathered one by one")
