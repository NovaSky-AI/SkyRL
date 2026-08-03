# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Trainer-side engine for the sharded-RDT (pull-based NIXL) backend.

Symmetric to the NCCL/IPC trainer engines, but RDT is *pull-based*: the vLLM
workers initiate every transfer, dialing the trainer's Ray actors and pulling
the exact slice each worker consumes over NIXL. So unlike NCCL (which
broadcasts) this engine does not push anything from `send_weights`; instead it

  * owns a per-rank **producer server** — an internal Ray actor that exposes the
    NIXL serve surface (`rdt_produce_weights_batched`, `free_gather`,
    `reserve_serve_arena`) the worker engine calls by name, and
  * on each `send_weights`, gathers this rank's weights group-by-group from the
    `WeightSource`, shares each group into the server over CUDA IPC, and (on the
    sender) drives the inference-side `start/update/finish` handshake — the
    single empty `update_weights` unblocks the workers to pull.

Everything the old `RDTShardedProducer` example mixin held — gather cache, serve
rings, free ref-counting, arena registration — now lives on the server actor,
spawned and owned by this engine. Trainer processes need no mixin, no named
actors, and no `enable_tensor_transport` / `max_concurrency` actor options: any
process that can reach Ray and (for multi-rank) `torch.distributed` works.
"""

import contextlib
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import ray
import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor
from typing_extensions import Self
from vllm.logger import init_logger

from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_base import (
    ParamMeta,
    TrainerInitInfo,
    TrainerWeightTransferEngine,
    VLLMWeightSyncClient,
    WeightSource,
)
from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_common import (
    ALLOWED_OPS,
    arena_alloc_bytes,
    count_consumers,
    layerwise_groups,
)

if TYPE_CHECKING:
    pass

from skyrl.backends.skyrl_train.weight_sync.rdt_libfabric_shim import (
    ensure_ray_rdt_libfabric,
)

ensure_ray_rdt_libfabric()

logger = init_logger(__name__)

# How many gathered groups may be resident (and served) at once before the
# gather loop blocks. Matches the old RDTShardedProducer GATHER_LOOKAHEAD.
DEFAULT_GATHER_LOOKAHEAD = 2

# The actor method name the worker engine dials for the NIXL pull. Fixed by
# contract (ShardedRDTWeightTransferInitInfo.produce_method_name default).
PRODUCE_METHOD_NAME = "rdt_produce_weights_batched"


@dataclass
class ShardedRDTTrainerInitInfo(TrainerInitInfo):
    """Trainer init info for the sharded-RDT backend.

    Identical on every trainer rank except `rank` (kw-only, from the base;
    rank 0 is the sender). The trainer no longer supplies actor names — the
    engine generates a server-actor name per rank and all-gathers them across
    ranks — so this only carries the must-agree wire params, which the sender
    forwards verbatim onto the worker-side init info so the two sides can't
    drift.
    """

    backend: ClassVar[str] = "sharded_rdt"

    num_consumers: int
    """Total inference-worker (consumer) count across the whole fleet (TP*DP),
    for the M:N block assignment / free ref-count."""
    trainer_actor_namespace: str | None = None
    """Ray namespace the engine spawns its serve actors in. The inference
    workers (which run in their own EngineCore subprocess with its own
    ``ray.init``) resolve those actors by name, so this must be the namespace
    they can see. Forwarded to the worker-side init info."""
    num_rdt_buffers: int = 2
    """Serve/receive ring depth K (must match the worker)."""
    layerwise_split: int = 1
    """Chunk split S (forwarded to the worker; the producer mirrors its
    packed layout)."""
    arena_presize_gb: float = 0.0
    """Serve-arena pre-size floor in GiB (avoids NIXL desc-cache churn)."""
    nosync: bool = False
    """Scoped-sync serve: pack on a dedicated stream gated on gather events
    instead of a whole-device sync."""
    pack_check: bool = False
    """Emit per-blob checksums to /tmp/rdt_profile for offline diffing."""
    gather_lookahead: int = DEFAULT_GATHER_LOOKAHEAD
    """Resident gathered groups before the gather loop blocks."""


class _RDTProducerServer:
    """Per-rank NIXL serve surface for the sharded-RDT backend.

    Spawned by the engine as an internal Ray actor sharing the trainer rank's
    GPU (via CUDA IPC). Absorbs the serve half of the old `RDTShardedProducer`
    example mixin: a gather cache of rebuilt IPC tensors, per-consumer serve
    rings, free ref-counting, and the byte-exact packed serve. The engine feeds
    it gathered weights with `publish_group`; the workers pull with
    `rdt_produce_weights_batched` and free with `free_gather`.

    This is a plain class; the engine wraps it with `ray.remote(...)` at spawn
    so the actor options (name / tensor transport / concurrency / GPU pinning)
    live in one place.
    """

    def __init__(
        self,
        *,
        free_target: int,
        num_rdt_buffers: int,
        arena_presize_gb: float,
        nosync: bool,
        pack_check: bool,
        gather_lookahead: int,
    ) -> None:
        import gc

        self._device_index = torch.accelerator.current_device_index()
        # name -> rebuilt CUDA-IPC tensor (or view); guarded by _cache_cond.
        self._cache: dict[str, torch.Tensor] = {}
        self._cache_cond = threading.Condition()
        self._gather_error: BaseException | None = None

        # [RDT-FREE-REFCOUNT] Each assigned consumer fires free_gather for every
        # group; the group is actually freed (and reported back to the engine)
        # only on the free_target-th call.
        self._free_target = max(1, free_target)
        self._free_counts: dict[tuple, int] = {}
        self._free_lock = threading.Lock()

        # [RDT-BACKPRESSURE] Published-but-not-yet-freed group keys. publish_group
        # blocks while len(...) >= gather_lookahead; free_gather (the consumer
        # back-edge) drains it. Freed keys are handed back to the engine so it
        # drops its trainer-side refs to the shared storage.
        self._lookahead = max(1, gather_lookahead)
        self._inflight_keys: list[tuple] = []
        self._name_to_key: dict[str, tuple] = {}
        self._freed_pending: list[tuple] = []

        # [RDT-RING] Per-consumer ring of packed serve arenas, rotated per pull.
        self._nring = max(1, num_rdt_buffers)
        self._serve_rings: dict[int, list[torch.Tensor | None]] = {}
        self._serve_idx: dict[int, int] = {}
        self._serve_lock = threading.Lock()
        # registerMem on a shared NIXL agent is not concurrency-safe; serialize.
        self._reg_lock = threading.Lock()
        self._arena_presize = int(arena_presize_gb * (1 << 30))

        # [RDT-NOSYNC] Scoped-sync serve stream + per-name completion events.
        self._scoped_sync = nosync
        self._serve_stream = torch.cuda.Stream() if nosync else None
        self._cache_event: dict[str, torch.cuda.Event] = {}

        self._pack_check = pack_check

        # [RDT-EARLY-ACK] Rebuild worker: publish_group acks right after
        # admission and hands the CUDA-IPC rebuild (driver opens + per-name
        # views, ~36ms/group at 235B) to this thread. Keeps the rebuild OFF the
        # trainer's publish-window drain path; consumers already block on
        # _cache_cond until names appear, so serve correctness is unchanged.
        # Disable with SKYRL_RDT_EARLY_ACK=0 (rebuild inline, pre-existing
        # behavior) for A/B.
        import os as _os
        import queue as _queue

        self._early_ack = _os.environ.get("SKYRL_RDT_EARLY_ACK", "1") == "1"
        self._rebuild_q: "_queue.Queue" = _queue.Queue()
        if self._early_ack:
            self._rebuild_thread = threading.Thread(target=self._rebuild_worker, daemon=True)
            self._rebuild_thread.start()

        # profiling counters
        self._timing_lock = threading.Lock()
        self._produce_calls = self._produce_specs = self._produce_bytes = 0
        self._produce_wait_seconds = self._produce_slice_seconds = 0.0
        self._produce_method_seconds = 0.0
        # [RDT-LINK-TIMING] Decompose the publish -> serve -> free credit loop
        # into non-overlapping links so the slow one can be named (see
        # multi_node_rdt.md Part X.8). All wall-clock, this process only.
        self._publish_calls = 0
        self._publish_bp_wait_seconds = 0.0  # publish blocked on lookahead credit
        self._publish_rebuild_seconds = 0.0  # CUDA-IPC rebuild_cuda_tensor loop
        self._publish_open_seconds = 0.0  # first-encounter storage opens
        self._publish_open_count = 0
        self._publish_view_seconds = 0.0  # repeat-storage view rebuilds
        self._publish_time: dict[tuple, float] = {}  # key -> publish done (cache_cond)
        self._serve_done: dict[tuple, float] = {}  # key -> last produce done (cache_cond)
        self._serve_lag_waited_seconds = 0.0  # publish -> produce wake (produce waited)
        self._serve_lag_waited_calls = 0
        self._serve_lag_ready_seconds = 0.0  # publish -> produce arrival (consumer late)
        self._serve_lag_ready_calls = 0
        self._free_lag_seconds = 0.0  # last serve done -> freeing free_gather arrival
        self._free_lag_count = 0
        self._publish_to_free_seconds = 0.0  # publish done -> freed (downstream RTT)

        from skyrl.backends.skyrl_train.weight_sync._nixl_profile import (
            install_nixl_timing,
        )

        install_nixl_timing()  # fail-soft inside

        # Freeze the static post-init object graph so gen-2 GC never stops the
        # world mid-serve (measured straggler fix in the old producer).
        gc.collect()
        gc.freeze()

    def ping(self) -> int:
        return self._device_index

    def warmup_nixl(self) -> None:
        """Create this server's NIXL agent NOW, while the rank's GPU is quiet.

        The engine calls this at spawn time, before the server-name all-gather
        barrier — i.e. before any trainer rank can be spinning in a gather
        collective. Lazily creating the agent later (first reserve_serve_arena,
        during the sender's worker-init RPC) deadlocks on EFA: libfabric's
        fi_getinfo probes CUDA HMEM support with a cudaMalloc/cudaFree, and
        cudaFree blocks behind the co-resident trainer rank's persistent NCCL
        collective kernel — which cannot finish because the collective is
        waiting on the sender rank, which is waiting on the worker-init RPC,
        which is waiting on this server. Creating (and HMEM-warming) the agent
        up front breaks the cycle. The warmup buffer stays registered/alive so
        the agent and its CUDA-HMEM path stay initialized.
        """
        from ray.experimental import register_nixl_memory

        self._nixl_warmup_buf = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
        with self._reg_lock:
            register_nixl_memory(self._nixl_warmup_buf)

    # ---------------- engine-facing (per sync) ----------------

    def begin_sync(self) -> None:
        """Reset per-sync free/backpressure state. The driver awaits the
        previous sync's finish (which drains every consumer's frees) before the
        next begins, so nothing is in flight here."""
        with self._free_lock:
            self._free_counts.clear()
        with self._cache_cond:
            self._gather_error = None
            self._inflight_keys.clear()
            self._name_to_key.clear()
            self._freed_pending.clear()
            self._publish_time.clear()
            self._serve_done.clear()

    def publish_group(self, group_key: tuple, entries: tuple) -> list[tuple]:
        """Rebuild one gather group's CUDA-IPC tensors and publish to the serve
        cache. Blocks while `gather_lookahead` groups are already in flight so
        trainer memory stays bounded (the consumer's `free_gather` drains it).
        Returns the group keys freed since the last call so the engine can drop
        its refs to the shared storage.

        ``entries`` is ``(storages, views)``: ``storages`` maps a storage id to
        the ``reduce_tensor`` args of a whole-storage uint8 view (ONE CUDA-IPC
        export per storage), ``views`` maps each served name to
        ``(sid, dtype_name, shape, stride, storage_offset)`` — rebuilt here as
        ``as_strided`` views so the per-name cost is microseconds."""
        _t_bp0 = time.perf_counter()
        storages, views = entries
        with self._cache_cond:
            while len(self._inflight_keys) >= self._lookahead:
                if self._gather_error is not None:
                    break
                self._cache_cond.wait()
            if self._early_ack:
                # [RDT-EARLY-ACK] Claim the backpressure slot + name routing NOW
                # (so accounting is exact), defer the rebuild to the worker, and
                # ack immediately — the trainer's publish window drains without
                # eating the IPC-open latency.
                self._inflight_keys.append(group_key)
                for n in views:
                    self._name_to_key[n] = group_key
                self._publish_time[group_key] = time.perf_counter()
                freed = self._freed_pending
                self._freed_pending = []
        if self._early_ack:
            self._rebuild_q.put((group_key, entries))
            with self._timing_lock:
                self._publish_calls += 1
                self._publish_bp_wait_seconds += time.perf_counter() - _t_bp0
            return freed
        _t_bp1 = time.perf_counter()

        rebuilt: dict[str, torch.Tensor] = {}
        # [RDT-STORAGE-PUBLISH] One rebuild_cuda_tensor per unique STORAGE (the
        # IPC open + event sync), then cheap as_strided views per name. The
        # per-name rebuild_cuda_tensor loop this replaces cost ~32us/name of
        # pure Python on ~19k names/sync (plus the open amplification under
        # expandable segments, see _expandable_segments_disabled_for_sync).
        _open_s = _view_s = 0.0
        _opens = 0
        _t_r0 = time.perf_counter()
        bases: dict[int, torch.Tensor] = {}
        for sid, reduce_args in storages.items():
            list_args = list(reduce_args)
            # Index 6 of reduce_tensor's args is the exporter's device index;
            # rebuild on this server's device (same physical GPU as the rank).
            list_args[6] = self._device_index
            bases[sid] = rebuild_cuda_tensor(*list_args)
            _opens += 1
        _open_s = time.perf_counter() - _t_r0
        _t_v0 = time.perf_counter()
        for name, (sid, dtype_name, shape, stride, storage_offset) in views.items():
            typed = bases[sid].view(getattr(torch, dtype_name))
            rebuilt[name] = torch.as_strided(typed, shape, stride, storage_offset)
        _view_s = time.perf_counter() - _t_v0
        del bases
        _t_rb = time.perf_counter()

        ev = None
        if self._serve_stream is not None:
            ev = torch.cuda.Event()
            ev.record()
        with self._cache_cond:
            self._cache.update(rebuilt)
            if ev is not None:
                for n in rebuilt:
                    self._cache_event[n] = ev
            self._inflight_keys.append(group_key)
            for n in rebuilt:
                self._name_to_key[n] = group_key
            self._publish_time[group_key] = time.perf_counter()
            freed = self._freed_pending
            self._freed_pending = []
            self._cache_cond.notify_all()
        with self._timing_lock:
            self._publish_calls += 1
            self._publish_bp_wait_seconds += _t_bp1 - _t_bp0
            self._publish_rebuild_seconds += _t_rb - _t_bp1
            self._publish_open_seconds += _open_s
            self._publish_open_count += _opens
            self._publish_view_seconds += _view_s
        return freed

    def _rebuild_worker(self) -> None:
        """[RDT-EARLY-ACK] Drain deferred rebuilds in publish order. Runs the
        exact inline rebuild (IPC storage opens + per-name as_strided views),
        then publishes into the serve cache and wakes blocked produce calls.
        Errors surface through _gather_error so waiters raise instead of hang."""
        torch.cuda.set_device(self._device_index)  # fresh thread: bind the rank's GPU
        while True:
            item = self._rebuild_q.get()
            if item is None:
                return
            group_key, (storages, views) = item
            try:
                _t_r0 = time.perf_counter()
                bases: dict[int, torch.Tensor] = {}
                for sid, reduce_args in storages.items():
                    list_args = list(reduce_args)
                    list_args[6] = self._device_index
                    bases[sid] = rebuild_cuda_tensor(*list_args)
                _open_s = time.perf_counter() - _t_r0
                _t_v0 = time.perf_counter()
                rebuilt: dict[str, torch.Tensor] = {}
                for name, (sid, dtype_name, shape, stride, storage_offset) in views.items():
                    typed = bases[sid].view(getattr(torch, dtype_name))
                    rebuilt[name] = torch.as_strided(typed, shape, stride, storage_offset)
                _view_s = time.perf_counter() - _t_v0
                del bases
                ev = None
                if self._serve_stream is not None:
                    ev = torch.cuda.Event()
                    ev.record()
                with self._cache_cond:
                    self._cache.update(rebuilt)
                    if ev is not None:
                        for n in rebuilt:
                            self._cache_event[n] = ev
                    self._cache_cond.notify_all()
                with self._timing_lock:
                    self._publish_rebuild_seconds += time.perf_counter() - _t_r0
                    self._publish_open_seconds += _open_s
                    self._publish_open_count += len(storages)
                    self._publish_view_seconds += _view_s
            except BaseException as e:  # noqa: BLE001 - waiters must not hang
                with self._cache_cond:
                    self._gather_error = RuntimeError(f"[early-ack] rebuild failed for {group_key}: {e!r}")
                    self._cache_cond.notify_all()

    def end_sync(self) -> list[tuple]:
        """Block until every published group has been freed by its consumers;
        return the remaining freed keys so the engine drops its last refs."""
        with self._cache_cond:
            while self._inflight_keys:
                if self._gather_error is not None:
                    break
                self._cache_cond.wait()
            freed = self._freed_pending
            self._freed_pending = []
            return freed

    def set_gather_error(self, message: str) -> None:
        """Record a trainer-side gather failure so blocked serves / publishes
        stop waiting and surface it."""
        with self._cache_cond:
            self._gather_error = RuntimeError(message)
            self._cache_cond.notify_all()

    # ---------------- consumer-facing (called by name over Ray) ----------------

    def free_gather(self, names: list[str]) -> None:
        """Consumer back-edge: one consumer finished pulling this group. Ref-count
        to `free_target`; on the last, drop the cache entries, release one
        backpressure slot, and record the freed key for the engine."""
        key = self._name_to_key.get(names[0]) if names else None
        with self._free_lock:
            count = self._free_counts.get(tuple(names), 0) + 1
            self._free_counts[tuple(names)] = count
            do_free = count >= self._free_target
            if do_free:
                del self._free_counts[tuple(names)]
        if not do_free:
            return
        _now = time.perf_counter()
        with self._cache_cond:
            for name in names:
                self._cache.pop(name, None)
                self._cache_event.pop(name, None)
                self._name_to_key.pop(name, None)
            if key is not None and key in self._inflight_keys:
                self._inflight_keys.remove(key)
                self._freed_pending.append(key)
            _sd = self._serve_done.pop(key, None) if key is not None else None
            _pt = self._publish_time.pop(key, None) if key is not None else None
            self._cache_cond.notify_all()
        with self._timing_lock:
            if _sd is not None:
                self._free_lag_seconds += _now - _sd
                self._free_lag_count += 1
            if _pt is not None:
                self._publish_to_free_seconds += _now - _pt

    def reserve_serve_arena(self, consumer_id: int, nbytes: int) -> None:
        """Pre-allocate + NIXL-register this consumer's serve ring before any
        pull, while the fabric is idle (avoids registration races during the
        sync-0 RDMA churn under M:N fan-in). Idempotent; grows only if needed."""
        from ray.experimental import register_nixl_memory

        alloc = arena_alloc_bytes(nbytes, self._arena_presize)
        with self._serve_lock:
            rings = self._serve_rings.setdefault(consumer_id, [None] * self._nring)
            self._serve_idx.setdefault(consumer_id, 0)
        for i in range(self._nring):
            slot = rings[i]
            if slot is None or slot.numel() < alloc:
                t = torch.empty(alloc, dtype=torch.uint8, device="cuda:0")
                with self._reg_lock:
                    register_nixl_memory(t)
                rings[i] = t

    @ray.method(tensor_transport="nixl")
    def rdt_produce_weights_batched(self, specs: list, pack: bool = True, consumer_id: int = 0):
        """Serve one batched slice request over NIXL.

        Waits until the specs' names are cached, replays each spec's op chain
        (pure views into cached tensors, guarded by ALLOWED_OPS), byte-packs the
        slices 16B-aligned into this consumer's ring slot (mirroring the
        consumer's identical layout), and returns the one packed blob.
        `pack=False` serves one tensor per spec (the rare unbaked path).
        """
        t_m0 = time.perf_counter()
        needed = sorted({n for n, _ in specs})
        t_w0 = time.perf_counter()
        with self._cache_cond:
            while not all(n in self._cache for n in needed):
                if self._gather_error is not None:
                    raise RuntimeError(f"gather errored before {needed}: {self._gather_error!r}")
                self._cache_cond.wait()
            _grp_key = self._name_to_key.get(needed[0])
            _pub_t = self._publish_time.get(_grp_key) if _grp_key is not None else None
        wait_s = time.perf_counter() - t_w0
        # [RDT-LINK-TIMING] publish -> here: if we waited, this is the notify/wake
        # latency; if the group was already cached, it is how late this produce
        # ARRIVED after publish (consumer-side credit / issue lateness).
        if _pub_t is not None:
            _lag = time.perf_counter() - _pub_t
            with self._timing_lock:
                if wait_s > 0.001:
                    self._serve_lag_waited_seconds += _lag
                    self._serve_lag_waited_calls += 1
                else:
                    self._serve_lag_ready_seconds += _lag
                    self._serve_lag_ready_calls += 1

        t_s0 = time.perf_counter()
        sliced: list = []  # (byte_off, tensor)
        pack_cur = 0
        nbytes = 0
        for name, chain in specs:
            t = self._cache[name]
            for op, args, kw in chain:
                if op not in ALLOWED_OPS:
                    raise ValueError(f"{name!r}: disallowed op {op!r}")
                t = getattr(t, op)(*args, **dict(kw))
            off = (pack_cur + 15) & ~15
            pack_cur = off + t.numel() * t.element_size()
            sliced.append((off, t))
            nbytes += t.numel() * t.element_size()

        if not pack:
            out = [t.contiguous().clone() for _off, t in sliced]
            torch.accelerator.synchronize()
            self._bump_timing(t_m0, wait_s, t_s0, len(specs), nbytes, _grp_key)
            return out

        with self._serve_lock:
            rings = self._serve_rings.setdefault(consumer_id, [None] * self._nring)
            idx = self._serve_idx.get(consumer_id, 0)
            self._serve_idx[consumer_id] = (idx + 1) % self._nring
        arena = rings[idx]
        if arena is None or arena.numel() < pack_cur:
            from ray.experimental import register_nixl_memory

            alloc = arena_alloc_bytes(pack_cur, self._arena_presize)
            arena = torch.empty(alloc, dtype=torch.uint8, device="cuda:0")
            with self._reg_lock:
                register_nixl_memory(arena)
            rings[idx] = arena

        ss = self._serve_stream
        if ss is not None:
            for ev in {id(e): e for e in (self._cache_event.get(n) for n in needed) if e is not None}.values():
                ss.wait_event(ev)
        with torch.cuda.stream(ss):
            for off, t in sliced:
                nb = t.numel() * t.element_size()
                view = arena[off : off + nb].view(t.dtype).reshape(t.shape)
                view.copy_(t)
        if ss is not None:
            ss.synchronize()

        blob = arena[:pack_cur]
        if self._pack_check:
            self._log_pack_check(blob, pack_cur)
        self._bump_timing(t_m0, wait_s, t_s0, len(specs), nbytes, _grp_key)
        return [blob]

    # ---------------- profiling ----------------

    def _bump_timing(self, t_m0, wait_s, t_s0, nspecs, nbytes, grp_key=None) -> None:
        slice_s = time.perf_counter() - t_s0
        if grp_key is not None:
            # [RDT-LINK-TIMING] last serve of the group; free_gather measures
            # its arrival against this stamp.
            with self._cache_cond:
                self._serve_done[grp_key] = time.perf_counter()
        with self._timing_lock:
            self._produce_calls += 1
            self._produce_specs += nspecs
            self._produce_wait_seconds += wait_s
            self._produce_slice_seconds += slice_s
            self._produce_bytes += nbytes
            self._produce_method_seconds += time.perf_counter() - t_m0

    def _log_pack_check(self, blob: torch.Tensor, pack_cur: int) -> None:
        import json
        import os

        s = 0
        w = 32 << 20
        for i in range(0, pack_cur, w):
            s += int(blob[i : min(i + w, pack_cur)].sum(dtype=torch.int64))
        os.makedirs("/tmp/rdt_profile", exist_ok=True)
        with open("/tmp/rdt_profile/packcheck_prod.jsonl", "a") as f:
            f.write(json.dumps({"pid": os.getpid(), "bytes": pack_cur, "sum": s}) + "\n")

    def get_produce_timing(self) -> dict:
        import os

        # [RDT-LINK-TIMING] config echo: verifies env/flag propagation into THIS
        # server process (SKYRL_RDT_NOSYNC reach was previously unverified) and
        # whether the shim's extract patch is live.
        _extract_patched = False
        try:
            import ray.experimental.rdt.nixl_tensor_transport as _t

            _extract_patched = "ensure_ray_rdt_libfabric" in getattr(
                _t.NixlTensorTransport.extract_tensor_transport_metadata, "__qualname__", ""
            )
        except Exception:  # noqa: BLE001
            pass
        with self._timing_lock:
            return dict(
                calls=self._produce_calls,
                specs=self._produce_specs,
                wait_seconds=self._produce_wait_seconds,
                slice_seconds=self._produce_slice_seconds,
                bytes=self._produce_bytes,
                method_seconds=self._produce_method_seconds,
                publish_calls=self._publish_calls,
                publish_bp_wait_seconds=self._publish_bp_wait_seconds,
                publish_rebuild_seconds=self._publish_rebuild_seconds,
                publish_open_seconds=self._publish_open_seconds,
                publish_open_count=self._publish_open_count,
                publish_view_seconds=self._publish_view_seconds,
                serve_lag_waited_seconds=self._serve_lag_waited_seconds,
                serve_lag_waited_calls=self._serve_lag_waited_calls,
                serve_lag_ready_seconds=self._serve_lag_ready_seconds,
                serve_lag_ready_calls=self._serve_lag_ready_calls,
                free_lag_seconds=self._free_lag_seconds,
                free_lag_count=self._free_lag_count,
                publish_to_free_seconds=self._publish_to_free_seconds,
                cfg_lookahead=self._lookahead,
                cfg_nring=self._nring,
                cfg_free_target=self._free_target,
                cfg_scoped_sync=self._scoped_sync,
                cfg_nosync_env=os.environ.get("SKYRL_RDT_NOSYNC"),
                cfg_extract_sync_patched=_extract_patched,
            )

    def reset_produce_timing(self) -> None:
        with self._timing_lock:
            self._produce_calls = self._produce_specs = self._produce_bytes = 0
            self._produce_wait_seconds = self._produce_slice_seconds = 0.0
            self._produce_method_seconds = 0.0
            self._publish_calls = 0
            self._publish_bp_wait_seconds = self._publish_rebuild_seconds = 0.0
            self._publish_open_seconds = self._publish_view_seconds = 0.0
            self._publish_open_count = 0
            self._serve_lag_waited_seconds = self._serve_lag_ready_seconds = 0.0
            self._serve_lag_waited_calls = self._serve_lag_ready_calls = 0
            self._free_lag_seconds = self._publish_to_free_seconds = 0.0
            self._free_lag_count = 0

    def get_nixl_timing(self) -> dict:
        from skyrl.backends.skyrl_train.weight_sync import _nixl_profile

        return _nixl_profile.snapshot()

    def reset_nixl_timing(self) -> None:
        from skyrl.backends.skyrl_train.weight_sync import _nixl_profile

        _nixl_profile.reset()

    def shutdown(self) -> None:
        with self._cache_cond:
            self._cache.clear()
            self._cache_event.clear()
        with self._serve_lock:
            self._serve_rings.clear()


class ShardedRDTTrainerWeightTransferEngine(TrainerWeightTransferEngine[ShardedRDTTrainerInitInfo]):
    """Trainer-side engine for the pull-based sharded-RDT backend.

    Lives on every trainer rank. Owns a per-rank `_RDTProducerServer` actor
    (the NIXL serve surface). `send_weights` gathers this rank's weights
    group-by-group from the `WeightSource`, shares each group into the server
    over CUDA IPC, and — on the sender — drives the inference-side handshake so
    the workers pull. Non-sender ranks only gather (staying in the collective).
    """

    init_info_cls = ShardedRDTTrainerInitInfo

    def __init__(
        self,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource,
        is_sender: bool = True,
        init_info: ShardedRDTTrainerInitInfo,
    ) -> None:
        super().__init__(client=client, source=source, is_sender=is_sender)
        self._init_info = init_info
        self._server: Any = None  # the _RDTProducerServer actor handle
        self._server_name: str | None = None
        # Group-major metadata / partition, computed at trainer_init.
        self._meta: list[ParamMeta] = []
        self._groups: list[list[str]] = []
        # Strong refs to gathered tensors we've shared into the server, keyed by
        # group key. CUDA-IPC exports must outlive the importer, so we hold them
        # until the server reports the group freed. See send_weights.
        self._inflight: dict[tuple, dict[str, torch.Tensor]] = {}
        self._sync_timing: dict[str, float] = {}

    def _rpc(self, method: str, *args: Any) -> Any:
        """Call one of the server actor's methods and block for the result.
        The single seam through which the engine talks to its server, so tests
        can inject a local (non-Ray) fake server."""
        import ray

        return ray.get(getattr(self._server, method).remote(*args))

    def _publish_async(self, key, entries):
        """Fire publish_group WITHOUT blocking (the gather loop overlaps the
        publish with the next group's gather) and return a handle that
        ``_drop_when_ready`` resolves. Ray actor handle in production; a plain
        (non-Ray) fake server runs inline and returns its result directly."""
        method = self._server.publish_group
        remote = getattr(method, "remote", None)
        if remote is not None:
            return remote(key, entries)
        return method(key, entries)

    def _drop_when_ready(self, ref) -> None:
        import ray

        freed = ray.get(ref) if isinstance(ref, ray.ObjectRef) else ref
        self._drop_inflight(freed)

    # ---------------- construction ----------------

    @classmethod
    def trainer_init(
        cls,
        init_info: ShardedRDTTrainerInitInfo,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource | None = None,
    ) -> Self:
        if source is None:
            raise ValueError("Sharded RDT trainer weight transfer requires a WeightSource.")
        engine = cls(
            client=client,
            source=source,
            is_sender=init_info.is_sender,
            init_info=init_info,
        )

        engine._meta = list(source.metadata())
        names = [m.name for m in engine._meta]
        engine._groups = layerwise_groups(names)
        flat = [n for g in engine._groups for n in g]
        if flat != names:
            raise ValueError(
                "Sharded RDT requires a WeightSource whose metadata order is "
                "group-contiguous (pre / per-decoder-layer / post). Reorder the "
                "source so each model.layers.<N>.* block is contiguous."
            )

        world, rank = engine._world_and_rank()
        num_producers = world
        free_target = count_consumers(num_producers, init_info.num_consumers, rank)
        engine._spawn_server(free_target)

        # Every rank's server must exist before the sender's init RPC (the worker
        # init calls reserve_serve_arena back on ALL producer servers). The
        # all-gather of server names doubles as that barrier.
        server_names = engine._all_gather_server_names(world, rank)

        if engine.is_sender:
            worker_init = engine._build_worker_init_info(server_names)
            engine.client.init_weight_transfer_engine(asdict(worker_init))
        return engine

    def _world_and_rank(self) -> tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size(), torch.distributed.get_rank()
        return 1, self._init_info.rank

    def _all_gather_server_names(self, world: int, rank: int) -> list[str]:
        assert self._server_name is not None
        if world <= 1 or not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return [self._server_name]
        gathered: list[str | None] = [None] * world
        torch.distributed.all_gather_object(gathered, self._server_name)
        return [n for n in gathered if n is not None]

    def _spawn_server(self, free_target: int) -> None:
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        ii = self._init_info
        self._server_name = f"vllm_rdt_producer_{uuid.uuid4().hex[:12]}_rk{ii.rank}"
        node_id = ray.get_runtime_context().get_node_id()
        # Pin the server to this rank's physical GPU (num_gpus=0 so Ray doesn't
        # allocate a second one; CUDA_VISIBLE_DEVICES makes CUDA IPC to the
        # rank's gathered tensors possible — same device family as the IPC
        # backend). max_concurrency > 1: serves pulls while begin/publish/end
        # calls are in flight. enable_tensor_transport: NIXL serve.
        gpu_ids = ray.get_gpu_ids()
        # The server is the trainer rank's process twin: forward the env the
        # rank runs under (library paths etc.) so it imports torch/vllm the
        # same way, then pin it to the rank's physical GPU for CUDA IPC.
        import os

        env_vars = {
            k: os.environ[k]
            for k in (
                "LD_LIBRARY_PATH",
                "LD_PRELOAD",
                "NCCL_CUMEM_ENABLE",
                "VLLM_NCCL_SO_PATH",
                "PATH",
            )
            if k in os.environ
        }
        if gpu_ids:
            # The server is a num_gpus=0 actor (it SHARES the rank's GPU for
            # CUDA IPC, so it must not claim a second one). Ray would otherwise
            # set CUDA_VISIBLE_DEVICES="" and hide every GPU; tell it not to
            # touch the var (the pattern the weight-transfer tests use) and pin
            # the server to the rank's physical GPU ourselves.
            env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
            env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        runtime_env = {"env_vars": env_vars} if env_vars else {}
        server_cls = ray.remote(_RDTProducerServer).options(
            name=self._server_name,
            namespace=ii.trainer_actor_namespace,
            num_cpus=0,
            num_gpus=0,
            # Thread budget: up to K produce calls sit BLOCKED in cache-wait per
            # bound consumer, plus one backpressure-blocked publish_group, plus
            # free_gather / begin/end_sync must still get a thread promptly (a
            # queued free_gather stalls the whole credit loop). Keep real slack.
            max_concurrency=max(8, 2 * ii.num_rdt_buffers + 4),
            enable_tensor_transport=True,
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
            runtime_env=runtime_env,
        )
        self._server = server_cls.remote(
            free_target=free_target,
            num_rdt_buffers=ii.num_rdt_buffers,
            arena_presize_gb=ii.arena_presize_gb,
            nosync=ii.nosync,
            pack_check=ii.pack_check,
            gather_lookahead=ii.gather_lookahead,
        )
        ray.get(self._server.ping.remote())
        # Pre-barrier NIXL warmup: must complete before _all_gather_server_names
        # so no rank can be inside a gather collective yet (see warmup_nixl).
        ray.get(self._server.warmup_nixl.remote())

    def _build_worker_init_info(self, server_names: list[str]):
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_engine import (
            ShardedRDTWeightTransferInitInfo,
        )

        group_lens = [len(g) for g in self._groups]
        names = [m.name for m in self._meta]
        dtype_names = [str(m.dtype).split(".")[-1] for m in self._meta]
        shapes = [list(m.shape) for m in self._meta]
        return ShardedRDTWeightTransferInitInfo(
            trainer_actor_names=server_names,
            trainer_actor_namespace=self._init_info.trainer_actor_namespace,
            produce_method_name=PRODUCE_METHOD_NAME,
            names=names,
            dtype_names=dtype_names,
            shapes=shapes,
            group_lens=group_lens,
            num_consumers=self._init_info.num_consumers,
            num_rdt_buffers=self._init_info.num_rdt_buffers,
            layerwise_split=self._init_info.layerwise_split,
            arena_presize_gb=self._init_info.arena_presize_gb,
            pack_check=self._init_info.pack_check,
        )

    # ---------------- per-round ----------------

    def send_weights(self) -> None:
        assert self.source is not None
        if not self.is_sender:
            self._run_gather_loop(update_future=None)
            return

        wall0 = time.perf_counter()
        t0 = time.perf_counter()
        self.client.start_weight_update()
        self._sync_timing["start_seconds"] = time.perf_counter() - t0

        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_engine import (
            ShardedRDTWeightTransferUpdateInfo,
        )

        empty_update = asdict(ShardedRDTWeightTransferUpdateInfo())
        import time as _t

        with ThreadPoolExecutor(max_workers=1) as exe:
            # The workers block inside update_weights until they've pulled every
            # group, so it runs concurrently with the gather/publish loop.
            tu0 = time.perf_counter()
            future = exe.submit(self.client.update_weights, empty_update)
            _t0 = _t.time()
            self._run_gather_loop(update_future=future)
            _gather = _t.time() - _t0
            future.result()  # surface inference-side errors
            _tail = _t.time() - _t0 - _gather
            self._sync_timing["update_weights_seconds"] = time.perf_counter() - tu0

        tf0 = time.perf_counter()
        self.client.finish_weight_update()
        self._sync_timing["finish_seconds"] = time.perf_counter() - tf0
        self._sync_timing["wall_seconds"] = time.perf_counter() - wall0
        logger.info("[rdt-sync-timing] gather_publish_loop=%.2fs consumer_tail_after_gather=%.2fs", _gather, _tail)

    def _run_gather_loop(self, update_future) -> None:
        """Gather this rank's weights group-by-group and publish each into the
        server over CUDA IPC. `publish_group` blocks when the lookahead is full,
        so the loop self-paces to the consumers' pull rate. Runs on every rank;
        only the sender has an `update_future` to fail fast on."""
        gather0 = time.perf_counter()
        _t_next = _t_pub = _t_exp = 0.0
        assert self.source is not None  # guaranteed by trainer_init
        self._rpc("begin_sync")
        # [RDT-GROUP-SOURCE] Sources may expose iter_groups() — (names,
        # tensors) parallel lists per layerwise group — so the handoff is one
        # generator hop per GROUP instead of per tensor (~95 vs ~37k at 235B;
        # the per-tensor path cost ~0.9s/sync of pure Python). Falls back to
        # the per-name iterator for sources without it (FSDP, plain Megatron).
        _iter_groups = getattr(self.source, "iter_groups", None)
        git = _iter_groups() if _iter_groups is not None else None
        it = iter(self.source) if git is None else None
        # [RDT-ASYNC-PUBLISH] Publishes are fired without an inline ray.get and
        # harvested this window deep, so the publish RPC + server-side rebuild
        # overlap the NEXT group's gather/export instead of serializing the
        # loop. The server's lookahead backpressure still bounds resident
        # groups (a queued publish blocks server-side; worst case
        # lookahead + window - 1 groups are live at once).
        _PUBLISH_WINDOW = int(os.environ.get("SKYRL_RDT_PUBLISH_WINDOW", "2"))
        pending_publish: list = []

        try:
            for group in self._groups:
                key = tuple(group)
                _tn = time.perf_counter()
                if git is not None:
                    names, tensors = next(git)
                    if list(names) != list(group):
                        raise RuntimeError(
                            f"WeightSource group yielded {len(names)} names starting {names[:2]!r} "
                            f"but expected {len(group)} starting {group[:2]!r}; "
                            "iteration order must match metadata."
                        )
                else:
                    names, tensors = [], []
                    for expected in group:
                        name, tensor = next(it)
                        if name != expected:
                            raise RuntimeError(
                                f"WeightSource yielded {name!r} but expected "
                                f"{expected!r}; iteration order must match metadata."
                            )
                        names.append(name)
                        tensors.append(tensor)
                _te = time.perf_counter()
                _t_next += _te - _tn
                # [RDT-STORAGE-PUBLISH] Share each unique STORAGE once (one
                # cudaIpc export instead of one per name) and describe every
                # name as an as_strided view spec relative to its storage.
                storages: dict[int, tuple] = {}
                views: dict[str, tuple] = {}
                refs: dict[str, torch.Tensor] = {}
                for name, tensor in zip(names, tensors):
                    tensor = tensor.detach()
                    if not tensor.is_cuda:
                        tensor = tensor.cuda()
                    tensor = tensor.contiguous()
                    refs[name] = tensor  # keep the export alive
                    ust = tensor.untyped_storage()
                    sid = ust.data_ptr()
                    if sid not in storages:
                        base = torch.empty(0, dtype=torch.uint8, device=tensor.device)
                        base.set_(ust, 0, (ust.nbytes(),))
                        _rebuild, reduce_args = reduce_tensor(base)
                        storages[sid] = reduce_args
                    views[name] = (
                        sid,
                        str(tensor.dtype).split(".")[-1],
                        list(tensor.shape),
                        list(tensor.stride()),
                        tensor.storage_offset(),
                    )
                _t_exp += time.perf_counter() - _te
                del tensors
                # Hold our refs before publishing; drop them only when the
                # server reports the group freed (IPC export must outlive import).
                self._inflight[key] = refs
                _tp = time.perf_counter()
                pending_publish.append(self._publish_async(key, (storages, views)))
                while len(pending_publish) >= _PUBLISH_WINDOW:
                    self._drop_when_ready(pending_publish.pop(0))
                _t_pub += time.perf_counter() - _tp
                if update_future is not None and update_future.done():
                    # update_weights returned/failed early — surface now instead
                    # of blocking further publishes.
                    update_future.result()
            _tp = time.perf_counter()
            while pending_publish:
                self._drop_when_ready(pending_publish.pop(0))
            freed = self._rpc("end_sync")
            _t_pub += time.perf_counter() - _tp
            self._drop_inflight(freed)
        except BaseException as e:
            with contextlib.suppress(Exception):
                self._rpc("set_gather_error", repr(e))
            self._inflight.clear()
            raise
        finally:
            self._sync_timing["gather_seconds"] = time.perf_counter() - gather0
            self._sync_timing["source_next_seconds"] = _t_next
            self._sync_timing["publish_rpc_seconds"] = _t_pub
            self._sync_timing["export_seconds"] = _t_exp

    def _drop_inflight(self, freed_keys: list) -> None:
        for k in freed_keys:
            self._inflight.pop(tuple(k), None)

    # ---------------- misc ----------------

    def get_sync_timing(self) -> dict:
        """Coarse per-round timing (start / gather / update_weights / finish /
        wall seconds) — the replacement for the example CriticalPathProfiler's
        driver buckets. Producer/NIXL counters live on the server."""
        return dict(self._sync_timing)

    def get_produce_timing(self) -> dict:
        return self._rpc("get_produce_timing")

    def reset_produce_timing(self) -> None:
        self._rpc("reset_produce_timing")

    def get_nixl_timing(self) -> dict:
        return self._rpc("get_nixl_timing")

    def reset_nixl_timing(self) -> None:
        self._rpc("reset_nixl_timing")

    def shutdown(self) -> None:
        if self._server is None:
            return
        import ray

        with contextlib.suppress(Exception):
            ray.get(self._server.shutdown.remote())
            ray.kill(self._server)
        self._server = None
        self._inflight.clear()
