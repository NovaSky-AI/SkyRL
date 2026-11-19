"""M2/M3: the colocated CUDA-IPC send path -- a ring of packed slots, an async
rank-0 control plane, optional layer-aligned chunks and receiver-local shards.

Replaces ``IPCTrainerWeightTransferEngine._send_packed`` when any of
``weight_sync_num_ipc_buffers > 1``, ``weight_sync_layer_aligned_chunks`` or
``weight_sync_local_shard`` is set. With all three off the vLLM path runs unchanged.

Pipeline (N slots, every trainer rank):

    for chunk k:
        pack chunk k into slot k % N          (the export -- gathers -- runs here)
        all-gather the slot's IPC handle (+ this rank's layout, for local shards)
        rank 0: join send(k-1), then post send(k) on a helper thread
        barrier                               (everyone now knows slot (k-1) % N is free)

so the blocking ``update_weights`` round trip for chunk k overlaps the export and
pack of chunk k+1. Sends stay strictly serialized (send(k) starts after send(k-1)
returns), so each worker still sees one update at a time; N > 2 adds nothing
beyond N = 2 under that rule.

Local shards (M2, Phase A): every trainer rank sits on the GPU of exactly one vLLM
worker. A tensor that names a routed expert (``.experts.<id>.``) is packed only by
the ranks whose worker serves that expert under vLLM expert parallelism;
everything else is packed by every rank, as today. Chunk boundaries must agree
across ranks, so every rank cuts with the same rule over the *per-rank* byte
counts, which it can compute for every rank from the shared plan: cut when any
rank's slot would overflow (or at a layer boundary). Rank 0 then ships one
payload per server: a list indexed by global worker rank (``GPUWorker.update_weights``
routes it), each entry carrying only that worker's names and its GPU's handle.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
_EXPERT_RE = re.compile(r"\.layers\.(\d+)\..*\.experts\.(\d+)\.")


def iter_source(source: Any) -> Iterator[Tuple[str, torch.Tensor]]:
    """Full ``(name, tensor)`` pairs of a ``WeightSource``: ``iter_full()`` on vLLM >= 0.33,
    plain iteration before."""
    iter_full = getattr(source, "iter_full", None)
    return iter_full() if callable(iter_full) else iter(source)


def layer_key(name: str) -> Optional[int]:
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def expert_id(name: str) -> Optional[int]:
    m = _EXPERT_RE.search(name)
    return int(m.group(2)) if m else None


@dataclass
class LocalShardPlan:
    """Which routed experts each trainer rank's colocated vLLM worker serves.

    ``rank_experts[r]`` is the set of global expert ids served by the worker on
    trainer rank ``r``'s GPU, or ``None`` if that worker serves every expert
    (no vLLM expert parallelism) -- then the rank packs every expert.
    ``rank_worker[r]`` is ``(server_index, worker_index)`` for that worker, where
    ``worker_index`` is the position ``GPUWorker.update_weights`` reads from a list
    payload (``data_parallel_index * world_size + rank``).
    """

    rank_experts: List[Optional[frozenset]]
    rank_worker: List[Tuple[int, int]]
    server_worker_indices: List[List[int]]

    def wants(self, rank: int, name: str) -> bool:
        experts = self.rank_experts[rank]
        if experts is None:
            return True
        e = expert_id(name)
        return e is None or e in experts


@dataclass
class _Chunk:
    slot: int
    names: List[str] = field(default_factory=list)
    shapes: List[List[int]] = field(default_factory=list)
    dtype_names: List[str] = field(default_factory=list)
    tensor_sizes: List[int] = field(default_factory=list)


def _sync(buf: torch.Tensor) -> None:
    # Drain queued copies so the consumer sees a fully written slot.
    if buf.is_cuda:
        torch.cuda.current_stream().synchronize()


def iter_packed_chunks(
    stream: Iterator[Tuple[str, torch.Tensor]],
    buffers: Sequence[torch.Tensor],
    *,
    rank: int,
    world_size: int,
    layer_aligned: bool,
    plan: Optional[LocalShardPlan],
) -> Iterator[_Chunk]:
    """Pack the stream into the slot ring; yield each chunk once its slot is written.

    Every rank must call this with the same stream (the export is collective) and
    gets the same number of chunks: the cut rule depends only on the stream and the
    shared plan. The caller must not resume the generator until the next slot is free.
    """
    n = len(buffers)
    cap = buffers[0].numel()
    slot = 0
    chunk = _Chunk(slot=slot)
    used = [0] * world_size  # bytes each rank has packed into the current chunk
    cur_layer: Any = object()
    for name, tensor in stream:
        flat = tensor.contiguous().view(-1).view(torch.uint8)
        nbytes = flat.numel()
        if nbytes > cap:
            raise ValueError(
                f"Tensor {name!r} is {nbytes} bytes, larger than the {cap}-byte IPC slot; "
                "raise generator.inference_engine.weight_transfer_threshold_cuda_ipc_GB."
            )
        takers = range(world_size) if plan is None else [r for r in range(world_size) if plan.wants(r, name)]
        layer = layer_key(name) if layer_aligned else None
        boundary = layer_aligned and layer != cur_layer and any(used)
        overflow = any(used[r] + nbytes > cap for r in takers)
        if boundary or overflow:
            _sync(buffers[0])
            yield chunk
            slot = (slot + 1) % n
            chunk = _Chunk(slot=slot)
            used = [0] * world_size
        cur_layer = layer
        if rank in takers:
            off = used[rank]
            buffers[slot][off : off + nbytes].copy_(flat)
            chunk.names.append(name)
            chunk.shapes.append(list(tensor.shape))
            chunk.dtype_names.append(str(tensor.dtype).split(".")[-1])
            chunk.tensor_sizes.append(nbytes)
        for r in takers:
            used[r] += nbytes
        del flat, tensor
    if any(used):
        _sync(buffers[0])
        yield chunk


class SlotRingSender:
    """Drives one send over the ring. Built per ``send_weights`` call on every rank."""

    def __init__(self, engine: Any, *, num_buffers: int, layer_aligned: bool, plan: Optional[LocalShardPlan]):
        self.engine = engine
        self.num_buffers = max(1, int(num_buffers))
        self.layer_aligned = layer_aligned
        self.plan = plan
        self.stats: Dict[str, float] = {"chunks": 0, "send_wait_seconds": 0.0, "max_chunk_bytes": 0}

    def send(self, source: Any) -> None:
        from torch.multiprocessing.reductions import reduce_tensor

        eng = self.engine
        dist = torch.distributed
        multi = dist.is_initialized() and dist.get_world_size() > 1
        rank = dist.get_rank() if multi else 0
        world = dist.get_world_size() if multi else 1
        size = eng.packed_buffer_size_bytes
        buffers = [torch.empty(size, dtype=torch.uint8, device="cuda") for _ in range(self.num_buffers)]
        handles = [reduce_tensor(b)[1] for b in buffers]
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cws-send") if eng.is_sender else None
        pending: Optional[Future] = None
        try:
            for chunk in iter_packed_chunks(
                iter_source(source),
                buffers,
                rank=rank,
                world_size=world,
                layer_aligned=self.layer_aligned,
                plan=self.plan,
            ):
                mine = {
                    "handle": {eng.gpu_uuid: handles[chunk.slot]},
                    "names": chunk.names,
                    "shapes": chunk.shapes,
                    "dtype_names": chunk.dtype_names,
                    "tensor_sizes": chunk.tensor_sizes,
                }
                gathered = [None] * world
                if multi:
                    dist.all_gather_object(gathered, mine)
                else:
                    gathered = [mine]
                self.stats["chunks"] += 1
                self.stats["max_chunk_bytes"] = max(self.stats["max_chunk_bytes"], sum(chunk.tensor_sizes))
                if eng.is_sender:
                    payloads = self._payloads(gathered)
                    if pending is not None:
                        t0 = time.perf_counter()
                        pending.result()
                        self.stats["send_wait_seconds"] += time.perf_counter() - t0
                    pending = pool.submit(self._ship, payloads)
                    if self.num_buffers == 1:
                        # One slot: the next pack overwrites it, so wait right here.
                        pending.result()
                        pending = None
                if multi:
                    # Rank 0 has joined send(k-1) (or send(k) with one slot), so the
                    # slot the next chunk packs into is free on every worker.
                    dist.barrier()
            if pending is not None:
                pending.result()
                pending = None
        finally:
            if pending is not None:
                # A failure on the trainer side: never leave a send in flight
                # against a buffer that is about to be freed.
                try:
                    pending.result()
                except Exception:  # noqa: BLE001
                    pass
            if pool is not None:
                pool.shutdown(wait=True)
            del buffers

    # ---- rank 0 only ----

    def _payloads(self, gathered: List[Dict[str, Any]]) -> Any:
        """One update_info for every server (a dict) or one per server (a per-worker list each)."""
        plan = self.plan
        if plan is None:
            merged: Dict[str, Any] = {}
            for g in gathered:
                merged.update(g["handle"])
            g0 = gathered[0]
            return {
                "names": g0["names"],
                "dtype_names": g0["dtype_names"],
                "shapes": g0["shapes"],
                "ipc_handles": merged,
                "tensor_sizes": g0["tensor_sizes"],
            }
        per_server: List[List[Any]] = [[{} for _ in range(max(idx) + 1)] for idx in plan.server_worker_indices]
        filled = set()
        for r, g in enumerate(gathered):
            s, w = plan.rank_worker[r]
            filled.add((s, w))
            per_server[s][w] = {
                "names": g["names"],
                "dtype_names": g["dtype_names"],
                "shapes": g["shapes"],
                "ipc_handles": g["handle"],
                "tensor_sizes": g["tensor_sizes"],
            }
        for s, idx in enumerate(plan.server_worker_indices):
            missing = [w for w in idx if (s, w) not in filled]
            if missing:
                raise RuntimeError(f"local-shard plan leaves workers {missing} of server {s} without a trainer rank")
        return per_server

    def _ship(self, payloads: Any) -> None:
        client = self.engine.client
        if isinstance(payloads, dict):
            client.update_weights(payloads)
        else:
            client.update_weights_per_server(payloads)


class MultiSlotImporter:
    """Receiver-side cache of every slot's IPC mapping for one transfer.

    vLLM's ``PackedBufferImporter`` caches one mapping and releases it when a
    different export arrives; with a ring of slots that would rebuild and release a
    mapping on every chunk and drive the producer-side refcount negative (see its
    docstring). Keyed by the rebuild args, each slot is opened once per transfer
    and released once at ``close()``.
    """

    def __init__(self) -> None:
        self._entries: Dict[tuple, torch.Tensor] = {}

    def rebuild(self, list_args: list) -> torch.Tensor:
        from torch.multiprocessing.reductions import rebuild_cuda_tensor

        key = tuple(list_args)
        t = self._entries.get(key)
        if t is None:
            t = rebuild_cuda_tensor(*list_args)
            self._entries[key] = t
        return t

    def close(self) -> None:
        self._entries.clear()


def build_local_shard_plan(
    worker_infos_per_server: List[List[Dict[str, Any]]], gpu_uuids: List[str]
) -> Tuple[Optional[LocalShardPlan], str]:
    """Map every trainer rank to the vLLM worker on its GPU and that worker's experts.

    ``worker_infos_per_server[s]`` is ``skyrl_local_shard_info`` from every worker of
    server ``s``; ``gpu_uuids[r]`` is trainer rank ``r``'s GPU. Returns
    ``(None, reason)`` to fall back to full tensors when any worker has no trainer
    rank, any rank no worker, or expert placement is not uniform across layers.
    """
    by_uuid: Dict[str, Tuple[int, int, Optional[frozenset]]] = {}
    server_idx: List[List[int]] = []
    for s, infos in enumerate(worker_infos_per_server):
        server_idx.append([int(i["worker_index"]) for i in infos])
        for info in infos:
            if not info.get("consistent", True):
                return None, "expert placement differs across MoE layers"
            experts = info.get("local_experts")
            by_uuid[info["gpu_uuid"]] = (
                s,
                int(info["worker_index"]),
                frozenset(experts) if experts is not None else None,
            )
    if len(gpu_uuids) != len(set(gpu_uuids)):
        return None, "two trainer ranks share a GPU"
    if set(by_uuid) != set(gpu_uuids):
        return None, "trainer ranks and vLLM workers do not pair one-to-one by GPU"
    plan = LocalShardPlan(
        rank_experts=[by_uuid[u][2] for u in gpu_uuids],
        rank_worker=[(by_uuid[u][0], by_uuid[u][1]) for u in gpu_uuids],
        server_worker_indices=server_idx,
    )
    if all(e is None for e in plan.rank_experts):
        return None, "vLLM serves every expert on every worker (no expert parallelism): nothing to filter"
    return plan, ""
