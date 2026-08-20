# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Trainer-side engine for the sharded-RDT (pull-based NIXL) backend.

RDT is pull-based, so unlike NCCL this engine broadcasts nothing. It owns a
per-rank producer server (an internal Ray actor exposing the NIXL serve surface
the worker engine dials by name), and on each ``send_weights`` gathers this
rank's weights group-by-group from the ``WeightSource``, shares each group into
the server over CUDA IPC, and — on the sender — drives the inference-side
start/update/finish handshake, whose single empty ``update_weights`` unblocks the
workers to pull.

All serve-side state lives on the server actor, so trainer processes need no
mixin, no named actors and no special actor options.

See docs/training/weight_transfer/sharded_rdt.md for the publish -> serve ->
free_group -> release lifecycle and the ownership model.

Vendored from the vLLM RDT fork (``vllm/distributed/weight_transfer/``); keep
edits in sync with it. The intended differences are the import paths, the
LIBFABRIC shim, and the profiling SkyRL keeps for optimization work.
"""

import contextlib
import os
import threading
import time
import uuid
from collections.abc import Callable, Collection
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar

import ray
import torch
from torch.multiprocessing.reductions import (
    StorageWeakRef,
    rebuild_cuda_tensor,
    reduce_tensor,
)
from typing_extensions import Self
from vllm.logger import init_logger

from skyrl.backends.skyrl_train.weight_sync.rdt_libfabric_shim import (
    ensure_ray_rdt_libfabric,
)
from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_base import (
    ParamMeta,
    TrainerInitInfo,
    TrainerWeightTransferEngine,
    VLLMWeightSyncClient,
    WeightSource,
    layerwise_groups,
)
from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_common import (
    ALLOWED_OPS,
    buffer_alloc_bytes,
    check_ray_rdt_version,
)

ensure_ray_rdt_libfabric()

logger = init_logger(__name__)

# Gathered-but-unfreed groups the gather loop may run ahead by before it stops
# gathering. A gathered group is published immediately (publish_group never
# blocks), so at most ``lookahead + 1`` groups are resident on the trainer — the
# memory bound larger models size against, pinned by the tests.
#
# At 1, group N+1 is gathered and serveable while the consumers pull N, so the
# boundary's free-barrier latency hides behind live pulls. Raise it only if one
# group's gather is slower than its pulls.
DEFAULT_GATHER_LOOKAHEAD = 1

# [RDT-STALL-WATCHDOG] Seconds with no publish, produce or free before the
# producer declares the sync dead. A consumer that dies mid-sync never signals
# ``free_group``, so the group is never released and the waits below block
# forever — which stops this rank iterating its WeightSource, a collective, and
# wedges every other trainer rank with no exception anywhere. This converts that
# into one real error.
#
# A liveness backstop, not a latency target: nominal gaps are sub-second, so this
# is ~100x margin. It rides the init info because the producer is a Ray actor and
# does not inherit the trainer's environment.
#
# SkyRL: resolved on the WORKER (``RdtWeightSyncSender``) from
# ``InferenceFaultToleranceConfig.stall_timeout_s`` or
# ``SKYRL_RDT_STALL_TIMEOUT_S`` and shipped in the init info -- the sidecar is a
# Ray actor inheriting the raylet's environment, not the worker's, so a
# launch-time override never reaches this process.
DEFAULT_STALL_TIMEOUT_S = 300.0
# How often a blocked waiter re-checks the progress stamp.
_STALL_POLL_S = 5.0

# The actor method the worker engine dials for the NIXL pull; fixed by contract.
PRODUCE_METHOD_NAME = "rdt_produce_weights_batched"


# [RDT-SHARE-SLOTS] Cross-deployment serve-slot sharing.
#
# Consumers whose ids differ by a multiple of ``workers_per_replica`` are the
# same worker of different inference deployments: identical parallel config,
# identical baked plan, identical chunk sequence, byte-identical pack layout. A
# serve ring per consumer therefore costs one full ring PER REPLICA on the
# producer's GPU (measured on the 235B 4-node shape: 5.0 GiB at one deployment,
# 12-14 GiB at two, 26-28 GiB at four -- on top of the ~44 GiB the trainer rank
# itself holds), and repeats the pack once per replica for identical bytes.
#
# NIXL reads are one-sided and non-destructive, so R readers can read ONE
# registered slot concurrently. What a producer cannot observe is when a reader
# has FINISHED, so the release edge comes from the consumer's ISSUE order, which
# it sends as ``seq`` (see the consumer's ``_Chunk``): the pipeline drains pull i
# before issuing i+K, so slot ``seq % K`` was last packed for ``seq - K``, whose
# read is over. Under sharing the same holds, because generation ``seq`` is only
# packed once every live sharer has arrived at it -- so each has drained
# ``seq - K``.
#
# Deriving the slot from a per-call counter on THIS side instead (execution
# order) is what the earlier version did, and it is wrong: Ray may start a
# consumer's K concurrent produce calls in any order, so the call that executes
# K-before another can be a pull that is still being read, and its slot gets
# repacked underneath the reader. Silent, and it shows up only as a logprob
# drift -- measured at 2x the healthy gap on the 4-node 235B run.
#
# Hence one rendezvous per generation, keyed by ``seq``: the group's live sharers
# meet there, the LAST to arrive packs, and all of them return that one blob.
#
# Two failure modes, both closed rather than left silent:
#   * sharers that DON'T pull the same chunks would rendezvous on signatures
#     nobody else ever sends -- a 300s stall watchdog death. So the plan is
#     compared at init instead (``reserve_serve_buffer``'s ``plan_digest``),
#     where it is one error naming both consumers before any byte moves.
#   * a sharer that dies mid-sync never arrives, so its group stalls -- exactly
#     what a dead consumer already does to the per-group free barrier. The next
#     sync's ``begin_sync`` narrows the live set and the group shrinks with it.


@dataclass
class _SharedPack:
    """One generation of one slot-sharing group: the rendezvous state for a
    single chunk, identified by the consumers' issue index. See
    ``[RDT-SHARE-SLOTS]``."""

    group: int
    slot: int
    """``seq % ring_depth``. Fixed at creation, so slot reuse follows the
    consumers' issue order rather than this side's execution order."""
    arrived: set[int] = field(default_factory=set)
    blob: Any = None
    nbytes: int = 0
    packing: bool = False
    done: bool = False
    error: BaseException | None = None


@dataclass
class ShardedRDTTrainerInitInfo(TrainerInitInfo):
    """Trainer init info for the sharded-RDT backend.

    Identical on every rank except ``rank`` (rank 0 is the sender). Carries only
    the must-agree wire params; the sender forwards them verbatim onto the
    worker-side init info so the two cannot drift. Server-actor names are
    generated per rank and all-gathered by the engine, not supplied here.
    """

    backend: ClassVar[str] = "sharded_rdt"

    num_consumers: int
    """Total inference-worker (consumer) count across the whole fleet
    (DP*TP*PP*PCP), for the M:N block assignment / free ref-count."""
    workers_per_replica: int = 0
    """Consumers per inference DEPLOYMENT (``num_consumers // num_replicas``).

    Fixes the slot-sharing groups: consumers whose ids differ by a multiple of
    this are the same worker of different deployments, so they bake identical
    plans and pull byte-identical chunks, and ONE registered serve slot can
    serve all of them (see ``[RDT-SHARE-SLOTS]``). 0 disables sharing, and so
    does the single-deployment value ``num_consumers``, which makes every group
    a singleton and the serve path byte-identical to the unshared one.
    """
    trainer_actor_namespace: str | None = None
    """Ray namespace the engine spawns its serve actors in. The inference
    workers (which run in their own EngineCore subprocess with its own
    ``ray.init``) resolve those actors by name, so this must be the namespace
    they can see. Forwarded to the worker-side init info."""
    num_rdt_buffers: int = 2
    """Serve/receive ring depth K (must match the worker)."""
    buffer_presize_gb: float = 0.0
    """Serve-buffer pre-size floor in GiB (avoids NIXL desc-cache churn)."""
    pack_check: bool = False
    """Emit per-blob checksums to /tmp/rdt_profile for offline diffing."""
    gather_lookahead: int = DEFAULT_GATHER_LOOKAHEAD
    """Gathered-but-unfreed groups the gather loop may run ahead by. Bounds
    trainer-resident memory at ``gather_lookahead + 1`` groups."""
    stall_timeout_s: float = DEFAULT_STALL_TIMEOUT_S
    """Seconds of no publish/serve/free progress before the producer fails the sync
    (see ``DEFAULT_STALL_TIMEOUT_S``). Resolved worker-side so the env override
    reaches it."""


class _RDTProducerServer:
    """Per-rank NIXL serve surface: an internal Ray actor sharing the trainer
    rank's GPU over CUDA IPC. Holds the gather cache, per-consumer serve rings,
    the per-group free barrier and the packed serve. The engine feeds it with
    ``publish_group``; workers pull with ``rdt_produce_weights_batched`` and
    signal with ``free_group``.

    A plain class — the engine wraps it with ``ray.remote(...)`` at spawn so the
    actor options live in one place.
    """

    def __init__(
        self,
        *,
        num_rdt_buffers: int,
        buffer_presize_gb: float,
        pack_check: bool,
        gather_lookahead: int,
        served_names: list[str] | None = None,
        stall_timeout_s: float = DEFAULT_STALL_TIMEOUT_S,
        workers_per_replica: int = 0,
    ) -> None:
        import gc

        self._device_index = torch.accelerator.current_device_index()
        # name -> rebuilt CUDA-IPC tensor (or view); guarded by _cache_cond.
        self._cache: dict[str, torch.Tensor] = {}
        self._cache_cond = threading.Condition()
        self._gather_error: BaseException | None = None

        # [RDT-STALL-WATCHDOG] Monotonic stamp of the last forward step of the
        # publish -> serve -> free loop. The waits below poll it instead of
        # blocking forever, so a dead consumer fails the sync with a real error.
        self._stall_timeout = float(stall_timeout_s)
        self._last_progress = time.monotonic()

        # Names this producer publishes, so a misrouted pull fails loudly
        # instead of blocking forever in the cache wait. None = serve anything.
        self._served_names = set(served_names) if served_names is not None else None

        # [RDT-FREE-BARRIER] Every live consumer signals free_group(gi) at every
        # owner, once per sync; the group frees when the count reaches
        # begin_sync's live total — one uniform target, no routed per-producer
        # ones. Signals may precede the publish, which completes them.
        self._live_count = 1
        self._free_counts: dict[int, int] = {}
        # gi -> the names this producer published for it (what release drops).
        self._group_names: dict[int, list[str]] = {}

        # [RDT-GATHER-CREDIT] Published-but-unfreed groups, plus freed ones not
        # yet handed back. The memory gate lives in the ENGINE's gather loop; this
        # side only accounts — free_group moves a group from _inflight_groups to
        # _freed_pending, which the engine collects via wait_freed / end_sync to
        # drop its storage refs. _lookahead is engine-enforced, unused here.
        self._lookahead = max(0, gather_lookahead)
        self._inflight_groups: list[int] = []
        self._name_to_group: dict[str, int] = {}
        self._freed_pending: list[int] = []

        # [RDT-RING] Ring of packed serve buffers, one ring per SLOT-SHARING
        # GROUP (one consumer per group unless several deployments share, see
        # [RDT-SHARE-SLOTS]), a slot handed out per generation.
        self._nring = max(1, num_rdt_buffers)
        self._serve_rings: dict[int, list[torch.Tensor | None]] = {}
        self._serve_lock = threading.Lock()
        # registerMem on a shared NIXL agent is not concurrency-safe; serialize.
        self._reg_lock = threading.Lock()
        self._buffer_presize = int(buffer_presize_gb * (1 << 30))
        self._serve_device = torch.device("cuda", self._device_index)

        # [RDT-SHARE-SLOTS] Slot-sharing state, all guarded by _cache_cond (the
        # rendezvous waits are on it, so the stall watchdog covers them too).
        # `_share_width` = consumers per deployment; 0 means every group is a
        # singleton and this degenerates to the unshared serve path.
        self._share_width = max(0, int(workers_per_replica))
        self._live_ids: set[int] | None = None
        self._sharing_active = False
        self._sharers: dict[int, frozenset] = {}  # group -> live members
        self._gens: dict[int, dict] = {}  # group -> {seq: _SharedPack}
        # group -> {cid: plan digest}. Init-time only, so NOT cleared per sync:
        # it is what proves the sharers of a group pull the same chunks.
        self._plan_digests: dict[int, dict[int, str]] = {}

        self._pack_check = pack_check
        # [RDT-PACK-DSTS] (sharing group, ring idx, packed layout) ->
        # (buffer data_ptr, destination views). Keyed on the buffer pointer too, so
        # a ring regrow invalidates rather than writing into a freed buffer. See
        # the serve path for why the layout, not the spec names, is the key.
        self._pack_dsts: dict[tuple, tuple[int, list[torch.Tensor]]] = {}
        # [RDT-EXPORT-RING] handle -> rebuilt base. The trainer reuses buffers
        # across groups, so the same handle arrives on many publishes; without
        # this we reopen it every group. Cleared per sync so no IPC mapping into
        # trainer memory outlives the sync that made it.
        self._base_cache: dict[tuple, torch.Tensor] = {}

        # profiling counters
        self._timing_lock = threading.Lock()
        self._produce_calls = self._produce_specs = self._produce_bytes = 0
        self._produce_wait_seconds = self._produce_slice_seconds = 0.0
        self._produce_method_seconds = 0.0
        # [RDT-LINK-TIMING] Decompose the publish -> serve -> free credit loop
        # into non-overlapping links so the slow one can be named (see
        # multi_node_rdt.md Part X.8). All wall-clock, this process only.
        self._publish_calls = 0
        self._credit_wait_seconds = 0.0  # engine's wait_freed blocked on a credit
        self._credit_wait_calls = 0
        self._publish_rebuild_seconds = 0.0  # CUDA-IPC rebuild_cuda_tensor loop
        self._publish_open_seconds = 0.0  # first-encounter storage opens
        self._publish_open_count = 0
        self._publish_view_seconds = 0.0  # repeat-storage view rebuilds
        self._publish_time: dict[int, float] = {}  # gi -> publish done (cache_cond)
        self._serve_done: dict[int, float] = {}  # gi -> last produce done (cache_cond)
        self._serve_lag_waited_seconds = 0.0  # publish -> produce wake (produce waited)
        self._serve_lag_waited_calls = 0
        self._serve_lag_ready_seconds = 0.0  # publish -> produce arrival (consumer late)
        self._serve_lag_ready_calls = 0
        self._free_lag_seconds = 0.0  # last serve done -> freeing free_group arrival
        self._free_lag_count = 0
        self._publish_to_free_seconds = 0.0  # publish done -> freed (downstream RTT)
        # [RDT-SHARE-SLOTS] serves answered from a co-replica's pack, and what
        # waiting for it cost. shared_serves == 0 means nothing was shared.
        self._shared_serves = 0
        self._share_wait_seconds = 0.0

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

    # ---------------- stall watchdog ----------------

    def _note_progress_locked(self) -> None:
        """Record that the credit loop moved. Caller holds ``_cache_cond``."""
        self._last_progress = time.monotonic()

    def _wait_for(self, blocked: Callable[[], bool], what: str) -> None:
        """``_cache_cond.wait()`` with a liveness bound.

        Waits while ``blocked()`` holds, returning early on a gather error. If
        nothing on this producer progresses for ``_stall_timeout``, self-fires
        ``set_gather_error``, which every waiter here already checks — so the rank
        unwinds through one path and the driver gets a real exception.

        The progress stamp is global to the producer, not per-waiter: a merely
        slow waiter is kept alive by its peers' progress. Caller holds
        ``_cache_cond``.
        """
        while blocked():
            if self._gather_error is not None:
                return
            self._cache_cond.wait(_STALL_POLL_S)
            if not blocked() or self._gather_error is not None:
                return
            stalled = time.monotonic() - self._last_progress
            if stalled >= self._stall_timeout:
                msg = (
                    f"RDT stall: no progress for {stalled:.0f}s while waiting for {what} "
                    f"(timeout {self._stall_timeout:.0f}s). A consumer most likely died mid-sync: "
                    f"{len(self._inflight_groups)} group(s) published and unfreed. "
                    "Set SKYRL_RDT_STALL_TIMEOUT_S to change the bound."
                )
                logger.error("[rdt-stall] %s", msg)
                # Same channel a gather failure uses: wakes every waiter here, and each
                # of them raises rather than returning a half-served result.
                self._gather_error = RuntimeError(msg)
                self._cache_cond.notify_all()
                return

    def warmup_nixl(self) -> None:
        """Create this server's NIXL agent now, while the rank's GPU is quiet.

        Called at spawn, before the server-name all-gather, so no rank can be
        spinning in a collective. Creating the agent lazily instead deadlocks on
        EFA-class fabrics (see "warmup_nixl breaks a startup deadlock" in the
        doc). The warmup buffer stays registered so the agent's CUDA-HMEM path
        stays initialized.
        """
        from ray.experimental import register_nixl_memory

        self._nixl_warmup_buf = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
        with self._reg_lock:
            register_nixl_memory(self._nixl_warmup_buf)

    # ---------------- engine-facing (per sync) ----------------

    def begin_sync(self, live_count: int, live_consumer_ids: list | None = None) -> None:
        """Reset per-sync free/backpressure state and set this sync's barrier
        target.

        ``live_count`` is how many consumers take part in THIS sync. Required, no
        default: a forgotten argument silently targeting 1 would free groups after
        the FIRST signal while others still pull — use-after-free, not an error.
        The driver awaits the previous sync's finish before the next begins, so
        nothing is in flight; a straggler signal would otherwise credit the wrong
        sync, which is why the consumer drains its signals before finishing.

        ``live_consumer_ids`` is that same live set enumerated, which is what
        sizes each slot-sharing group's rendezvous ([RDT-SHARE-SLOTS]) — a count
        cannot, since a group is a specific set of ids. ``None`` (what an engine
        that does not send it gets) leaves every group a singleton, i.e. sharing
        off for the sync.

        The packed-destination cache deliberately survives: the layout repeats
        every sync. So does ``_plan_digests``, which is init-time state.
        """
        with self._cache_cond:
            self._gather_error = None
            self._live_count = max(1, int(live_count))
            self._live_ids = set(int(c) for c in live_consumer_ids) if live_consumer_ids is not None else None
            # Sharing is only ON when the geometry says several deployments are
            # present; one deployment (width == live count) takes the singleton
            # path, which skips even the request signature.
            self._sharing_active = (
                self._share_width > 0 and self._live_ids is not None and len(self._live_ids) > self._share_width
            )
            self._sharers.clear()
            self._gens.clear()
            self._free_counts.clear()
            self._group_names.clear()
            self._inflight_groups.clear()
            self._name_to_group.clear()
            self._freed_pending.clear()
            self._publish_time.clear()
            self._serve_done.clear()
            self._base_cache.clear()
            self._note_progress_locked()

    def publish_group(self, group_idx: int, entries: tuple) -> None:
        """Rebuild one gather group's CUDA-IPC tensors into the serve cache.

        NEVER blocks: a gathered group is serveable immediately. The memory bound
        lives in the engine's gather loop, which stops GATHERING (not publishing)
        past ``gather_lookahead`` unfreed groups — see [RDT-GATHER-CREDIT].

        ``entries`` is ``(storages, views)``: one CUDA-IPC export per storage,
        plus per-name ``(sid, dtype_name, shape, stride, storage_offset)`` rebuilt
        here as ``as_strided`` views.

        Signals can arrive BEFORE their publish — a consumer pulling nothing of a
        group signals it as its plan starts — so a group whose barrier is already
        satisfied is released here. Freed groups reach the engine only through
        ``wait_freed`` / ``end_sync``: a freed notice riding an unharvested async
        publish result would wedge the loop.
        """
        _t_p0 = time.perf_counter()
        storages, views = entries

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
            cached = self._base_cache.get(reduce_args)
            if cached is not None:
                bases[sid] = cached
                continue
            list_args = list(reduce_args)
            # Index 6 of reduce_tensor's args is the exporter's device index;
            # rebuild on this server's device (same physical GPU as the rank).
            list_args[6] = self._device_index
            bases[sid] = self._base_cache[reduce_args] = rebuild_cuda_tensor(*list_args)
            _opens += 1
        _open_s = time.perf_counter() - _t_r0
        _t_v0 = time.perf_counter()
        for name, (sid, dtype_name, shape, stride, storage_offset) in views.items():
            typed = bases[sid].view(getattr(torch, dtype_name))
            rebuilt[name] = torch.as_strided(typed, shape, stride, storage_offset)
        _view_s = time.perf_counter() - _t_v0
        del bases
        _t_rb = time.perf_counter()

        with self._cache_cond:
            self._cache.update(rebuilt)
            self._inflight_groups.append(group_idx)
            self._group_names[group_idx] = list(rebuilt)
            for n in rebuilt:
                self._name_to_group[n] = group_idx
            self._publish_time[group_idx] = time.perf_counter()
            if self._free_counts.get(group_idx, 0) >= self._live_count:
                self._release_group_locked(group_idx)
            self._note_progress_locked()
            self._cache_cond.notify_all()
        with self._timing_lock:
            self._publish_calls += 1
            self._publish_rebuild_seconds += _t_rb - _t_p0
            self._publish_open_seconds += _open_s
            self._publish_open_count += _opens
            self._publish_view_seconds += _view_s

    def wait_freed(self) -> list[int]:
        """Block until at least one published group has been freed; return and
        clear the freed backlog. The engine's gather-credit gate calls this once
        its loop is ``gather_lookahead`` groups ahead, so this wait is where the
        trainer paces to the consumers' pull rate.

        Raises rather than returning empty when the sync errors or stalls: the
        engine is blocked here inside its gather loop, and an empty return would
        spin it straight back in.
        """
        _t0 = time.perf_counter()
        with self._cache_cond:
            self._wait_for(lambda: not self._freed_pending, "a freed-group credit")
            if not self._freed_pending:
                raise RuntimeError(f"gather errored while waiting for a freed-group credit: {self._gather_error!r}")
            freed = self._freed_pending
            self._freed_pending = []
        with self._timing_lock:
            self._credit_wait_seconds += time.perf_counter() - _t0
            self._credit_wait_calls += 1
        return freed

    def end_sync(self) -> list[int]:
        """Block until every published group has been freed by its consumers;
        return the remaining freed keys so the engine drops its last refs."""
        with self._cache_cond:
            self._wait_for(lambda: bool(self._inflight_groups), "every published group to be freed")
            freed = self._freed_pending
            self._freed_pending = []
            self._base_cache.clear()
            return freed

    def set_gather_error(self, message: str) -> None:
        """Record a trainer-side gather failure so blocked serves / publishes
        stop waiting and surface it."""
        with self._cache_cond:
            self._gather_error = RuntimeError(message)
            self._cache_cond.notify_all()

    # ---------------- consumer-facing (called by name over Ray) ----------------

    def _release_group_locked(self, group_idx: int) -> None:
        """Drop one group's cache entries and queue it for the engine (whose
        gather-credit gate blocks in ``wait_freed`` on exactly this). Caller
        holds ``_cache_cond``; shared by the last ``free_group`` and by
        ``publish_group`` completing an early-signaled group."""
        for name in self._group_names.pop(group_idx, ()):
            self._cache.pop(name, None)
            self._name_to_group.pop(name, None)
        self._free_counts.pop(group_idx, None)
        if group_idx in self._inflight_groups:
            self._inflight_groups.remove(group_idx)
            self._freed_pending.append(group_idx)
        _now = time.perf_counter()
        _sd = self._serve_done.pop(group_idx, None)
        _pt = self._publish_time.pop(group_idx, None)
        with self._timing_lock:
            if _sd is not None:
                self._free_lag_seconds += _now - _sd
                self._free_lag_count += 1
            if _pt is not None:
                self._publish_to_free_seconds += _now - _pt

    def free_group(self, group_idx: int) -> None:
        """Consumer back-edge: one consumer is done with group ``group_idx``,
        either because its last chunk landed or because it had nothing to pull.

        The per-group barrier counts one signal per live consumer against the
        ``begin_sync`` count; every consumer signals every owner, so the target is
        the same integer everywhere. The last signal drops the cache entries and
        queues the group as gather credit for ``wait_freed``. A signal arriving
        before its publish is only counted — ``publish_group`` completes it.
        """
        gi = int(group_idx)
        with self._cache_cond:
            count = self._free_counts.get(gi, 0) + 1
            self._free_counts[gi] = count
            if gi in self._group_names and count >= self._live_count:
                self._release_group_locked(gi)
            self._note_progress_locked()
            self._cache_cond.notify_all()

    def _new_serve_buffer(self, nbytes: int) -> torch.Tensor:
        """Allocate + NIXL-register one serve slot: the single allocation seam, so
        registration can never be skipped on one of the two paths that make
        buffers (the init-time reservation and the serve-path backstop)."""
        from ray.experimental import register_nixl_memory

        t = torch.empty(nbytes, dtype=torch.uint8, device=self._serve_device)
        with self._reg_lock:
            register_nixl_memory(t)
        return t

    def reserve_serve_buffer(self, consumer_id: int, nbytes: int, plan_digest: str | None = None) -> None:
        """Pre-allocate + NIXL-register this consumer's serve ring before any
        pull, while the fabric is idle (avoids registration races during the
        sync-0 RDMA churn under M:N fan-in). Idempotent; grows only if needed.

        The ring is keyed by SLOT-SHARING GROUP, so the sharers of one group
        reserve ONE ring between them ([RDT-SHARE-SLOTS]): R deployments cost one
        ring, not R. They all call this — same group, same size — so the whole
        body runs under ``_serve_lock``, or two concurrent first-callers would
        each allocate a ring and one would be dropped still registered.

        ``plan_digest`` is the caller's ordered chunk plan for THIS producer.
        Sharers must agree on it, and this is where a disagreement is caught,
        deliberately rather than at serve time: at serve time it is a rendezvous
        nobody completes (a 300s stall-watchdog death), here it is one error
        naming both consumers before a byte moves.

        Raises:
            RuntimeError: two consumers of one sharing group disagree on the
                chunks they pull from this producer.
        """
        sg = self._share_group(consumer_id)
        if plan_digest is not None:
            with self._cache_cond:
                seen = self._plan_digests.setdefault(sg, {})
                clash = next(((c, d) for c, d in seen.items() if d != plan_digest), None)
                if clash is None:
                    seen[consumer_id] = plan_digest
            if clash is not None:
                raise RuntimeError(
                    f"consumers {clash[0]} and {consumer_id} are in slot-sharing group {sg} but do "
                    f"not pull the same chunks from this producer ({clash[1]} != {plan_digest}). "
                    "Sharing a serve slot requires the deployments to be identical; check that "
                    "workers_per_replica matches the inference fleet's geometry."
                )
        alloc = buffer_alloc_bytes(nbytes, self._buffer_presize)
        with self._serve_lock:
            rings = self._serve_rings.setdefault(sg, [None] * self._nring)
            for i in range(self._nring):
                slot = rings[i]
                if slot is None or slot.numel() < alloc:
                    rings[i] = self._new_serve_buffer(alloc)

    # ---------------- [RDT-SHARE-SLOTS] the rendezvous ----------------

    def _share_group(self, consumer_id: int) -> int:
        """The slot-sharing group ``consumer_id`` belongs to — its index WITHIN
        its deployment, since that is what fixes the plan. Without a width the
        group is the consumer itself and nothing is ever shared."""
        return consumer_id % self._share_width if self._share_width > 0 else consumer_id

    def _sharers_of(self, sg: int) -> frozenset:
        """The LIVE consumers of group ``sg``: the rendezvous width. Derived from
        ``begin_sync``'s live set, so a degraded sync narrows the barrier instead
        of waiting forever on a dead replica. Caller holds ``_cache_cond``."""
        cached = self._sharers.get(sg)
        if cached is None:
            if not self._sharing_active or self._live_ids is None:
                cached = frozenset((sg,))
            else:
                cached = frozenset(c for c in self._live_ids if c % self._share_width == sg) or frozenset((sg,))
            self._sharers[sg] = cached
        return cached

    def _join_share_group(self, consumer_id: int, seq: int) -> tuple:
        """Register this call's arrival in its group's rendezvous for ``seq`` and
        return ``(generation, is_packer)``.

        Exactly one caller per generation gets ``is_packer=True`` -- whichever
        completes the arrival set -- and it is the only one that touches the GPU.
        The slot is ``seq % nring``: fixed by the consumers' issue order, so it
        needs no release accounting on this side (see ``[RDT-SHARE-SLOTS]``).
        """
        sg = self._share_group(consumer_id)
        with self._cache_cond:
            gens = self._gens.setdefault(sg, {})
            gen = gens.get(seq)
            if gen is None:
                gen = gens[seq] = _SharedPack(group=sg, slot=seq % self._nring)
            gen.arrived.add(consumer_id)
            if gen.packing or gen.done or not gen.arrived >= self._sharers_of(sg):
                return gen, False
            gen.packing = True
            return gen, True

    def _await_shared_pack(self, gen: _SharedPack) -> torch.Tensor:
        """Block until this generation's packer published its blob, and return
        it. A failed pack is re-raised here, so every sharer of a bad pack fails
        instead of reading a half-written slot."""
        _t0 = time.perf_counter()
        with self._cache_cond:
            self._wait_for(lambda: not gen.done and gen.error is None, "a co-replica's shared serve pack")
            if gen.error is not None:
                raise RuntimeError(f"the sharer packing this chunk failed: {gen.error!r}") from gen.error
            if not gen.done:
                raise RuntimeError(f"gather errored while awaiting a shared pack: {self._gather_error!r}")
            blob = gen.blob
        with self._timing_lock:
            self._shared_serves += 1
            self._share_wait_seconds += time.perf_counter() - _t0
        return blob

    def _serve_slot(self, sg: int, idx: int, need: int) -> torch.Tensor:
        """Group ``sg``'s ring slot ``idx``, grown if this chunk outgrew the
        reservation. Growing registers memory while the fabric is busy — the
        hazard ``reserve_serve_buffer`` exists to avoid — so it is a backstop:
        the reservation is sized from the same static plan as this pack. The slot
        is owned by one generation, so no other call can touch ``rings[idx]``."""
        with self._serve_lock:
            buffer = self._serve_rings.setdefault(sg, [None] * self._nring)[idx]
        if buffer is not None and buffer.numel() >= need:
            return buffer
        buffer = self._new_serve_buffer(buffer_alloc_bytes(need, self._buffer_presize))
        with self._serve_lock:
            self._serve_rings.setdefault(sg, [None] * self._nring)[idx] = buffer
        return buffer

    def _pack_shared(self, gen: _SharedPack, specs: list) -> tuple:
        """Replay the op chains into this generation's slot and publish the blob
        to the sharers waiting on it.

        Runs on exactly one call per generation and OUTSIDE ``_cache_cond``: the
        pack is GPU work and must not block publishes, frees or other groups'
        rendezvous.
        """
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

        buffer = self._serve_slot(gen.group, gen.slot, pack_cur)

        # [RDT-PACK-DSTS] The destination views are a pure function of the packed
        # layout, which is byte-identical every sync, so build them once per
        # (sharing group, ring slot, layout) and reuse — rebuilding per call cost
        # 5.2ms of a 7.5ms 384-spec 235B group.
        #
        # The key is the LAYOUT, not the spec names: a name can appear in two
        # requests with different op chains (per-ep_rank chunking splits one
        # name's copies across chunks), and serving the second through the first's
        # views would write the wrong bytes with nothing downstream to catch it.
        dst_key = (
            gen.group,
            gen.slot,
            tuple((off, t.dtype, t.shape) for off, t in sliced),
        )
        cached = self._pack_dsts.get(dst_key)
        if cached is None or cached[0] != buffer.data_ptr():
            dsts = []
            for off, t in sliced:
                nb = t.numel() * t.element_size()
                dsts.append(buffer[off : off + nb].view(t.dtype).reshape(t.shape))
            self._pack_dsts[dst_key] = (buffer.data_ptr(), dsts)
        else:
            dsts = cached[1]
        torch._foreach_copy_(dsts, [t for _off, t in sliced])

        blob = buffer[:pack_cur]
        with self._cache_cond:
            gen.blob = blob
            gen.nbytes = nbytes
            gen.done = True
            self._note_progress_locked()
            self._cache_cond.notify_all()
        return blob, nbytes

    def _fail_shared_pack(self, gen: _SharedPack, exc: BaseException) -> None:
        """Publish a pack failure so the waiting sharers raise instead of hanging
        on a generation that will never complete."""
        with self._cache_cond:
            gen.error = exc
            self._cache_cond.notify_all()

    @ray.method(tensor_transport="nixl")
    def rdt_produce_weights_batched(self, specs: list, consumer_id: int = 0, seq: int = -1):
        """Serve one batched slice request over NIXL.

        Waits until the specs' names are cached, then rendezvouses with the other
        live sharers of this consumer's slot-sharing group ([RDT-SHARE-SLOTS]).
        The last sharer to arrive replays each spec's op chain (pure views into
        cached tensors, guarded by ALLOWED_OPS) and byte-packs the slices
        16B-aligned into the group's ring slot, mirroring the consumer's identical
        layout; every sharer then returns that one packed blob, and R deployments
        cost one pack and one slot instead of R. With a single deployment every
        group is a singleton, so the arriving call is always its own packer and
        this is the unshared serve path exactly.

        Callers that want a single slice pass one spec and read the blob back with
        that slice's dtype/shape (see the consumer's ``PullSink``).
        """
        t_m0 = time.perf_counter()
        needed = sorted({n for n, _ in specs})
        if self._served_names is not None:
            unserved = [n for n in needed if n not in self._served_names]
            if unserved:
                # Without this the cache wait below would block forever: this
                # producer never gathers these names, so they can never arrive.
                raise RuntimeError(
                    f"pull routed to the wrong producer: {unserved[:3]} ({len(unserved)} names) are not served here"
                )
        t_w0 = time.perf_counter()
        with self._cache_cond:
            self._wait_for(lambda: not all(n in self._cache for n in needed), f"{len(needed)} name(s) to be published")
            if not all(n in self._cache for n in needed):
                # _wait_for only gives up on a gather error (its own stall included),
                # so the names are never coming.
                raise RuntimeError(f"gather errored before {needed}: {self._gather_error!r}")
            _grp_key = self._name_to_group.get(needed[0])
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

        if seq < 0:
            # The slot is derived from it, so a caller that does not send one
            # cannot be served safely (see [RDT-SHARE-SLOTS]).
            raise ValueError("rdt_produce_weights_batched requires the consumer's issue index `seq`")
        gen, is_packer = self._join_share_group(consumer_id, seq)
        if not is_packer:
            blob = self._await_shared_pack(gen)
            self._bump_timing(t_m0, wait_s, None, len(specs), gen.nbytes, _grp_key)
            return [blob]

        t_s0 = time.perf_counter()
        try:
            blob, nbytes = self._pack_shared(gen, specs)
        except BaseException as e:
            self._fail_shared_pack(gen, e)
            raise
        if self._pack_check:
            self._log_pack_check(blob, blob.numel())
        self._bump_timing(t_m0, wait_s, t_s0, len(specs), nbytes, _grp_key)
        return [blob]

    # ---------------- profiling ----------------

    def _bump_timing(self, t_m0, wait_s, t_s0, nspecs, nbytes, grp_key=None) -> None:
        # t_s0 None = this call was answered from another sharer's pack, so it
        # sliced nothing; its blocked time is in share_wait_seconds instead.
        slice_s = 0.0 if t_s0 is None else time.perf_counter() - t_s0
        # [RDT-LINK-TIMING] last serve of the group; free_group measures its arrival
        # against this stamp. Also the third of the four progress signals the stall
        # watchdog reads (publish / produce / free / begin_sync): a long sync whose
        # consumers are pulling steadily but slowly must never trip it.
        with self._cache_cond:
            if grp_key is not None:
                self._serve_done[grp_key] = time.perf_counter()
            self._note_progress_locked()
        with self._timing_lock:
            self._produce_calls += 1
            self._produce_specs += nspecs
            self._produce_wait_seconds += wait_s
            self._produce_slice_seconds += slice_s
            self._produce_bytes += nbytes
            self._produce_method_seconds += time.perf_counter() - t_m0

    def _log_pack_check(self, blob: torch.Tensor, pack_cur: int) -> None:
        import json

        s = 0
        w = 32 << 20
        for i in range(0, pack_cur, w):
            s += int(blob[i : min(i + w, pack_cur)].sum(dtype=torch.int64))
        os.makedirs("/tmp/rdt_profile", exist_ok=True)
        with open("/tmp/rdt_profile/packcheck_prod.jsonl", "a") as f:
            f.write(json.dumps({"pid": os.getpid(), "bytes": pack_cur, "sum": s}) + "\n")

    def get_produce_timing(self) -> dict:

        # [RDT-LINK-TIMING] config echo: verifies flag propagation into THIS
        # server process and whether the shim's extract patch is live.
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
                credit_wait_seconds=self._credit_wait_seconds,
                credit_wait_calls=self._credit_wait_calls,
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
                shared_serves=self._shared_serves,
                share_wait_seconds=self._share_wait_seconds,
                cfg_share_width=self._share_width,
                cfg_sharing_active=self._sharing_active,
                cfg_serve_rings=len(self._serve_rings),
                cfg_lookahead=self._lookahead,
                cfg_nring=self._nring,
                cfg_served_names=-1 if self._served_names is None else len(self._served_names),
                cfg_extract_sync_patched=_extract_patched,
            )

    def reset_produce_timing(self) -> None:
        with self._timing_lock:
            self._produce_calls = self._produce_specs = self._produce_bytes = 0
            self._produce_wait_seconds = self._produce_slice_seconds = 0.0
            self._produce_method_seconds = 0.0
            self._publish_calls = 0
            self._credit_wait_seconds = self._publish_rebuild_seconds = 0.0
            self._credit_wait_calls = 0
            self._publish_open_seconds = self._publish_view_seconds = 0.0
            self._publish_open_count = 0
            self._serve_lag_waited_seconds = self._serve_lag_ready_seconds = 0.0
            self._serve_lag_waited_calls = self._serve_lag_ready_calls = 0
            self._free_lag_seconds = self._publish_to_free_seconds = 0.0
            self._free_lag_count = 0
            self._shared_serves = 0
            self._share_wait_seconds = 0.0

    def get_nixl_timing(self) -> dict:
        from skyrl.backends.skyrl_train.weight_sync import _nixl_profile

        return _nixl_profile.snapshot()

    def reset_nixl_timing(self) -> None:
        from skyrl.backends.skyrl_train.weight_sync import _nixl_profile

        _nixl_profile.reset()

    def shutdown(self) -> None:
        with self._cache_cond:
            self._cache.clear()
        with self._cache_cond:
            # Generations hold blob views into the rings, so they go first.
            self._gens.clear()
            self._sharers.clear()
        with self._serve_lock:
            self._serve_rings.clear()
            # Must go with the rings: these are views INTO them, and the
            # data_ptr guard that normally invalidates them cannot tell a freed
            # buffer from a new one recycled at the same address.
            self._pack_dsts.clear()


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
        # Ownership resolved at trainer_init from the fleet's held names: the
        # distinct owner sets, the per-name index into them, this rank's groups
        # and its held set. Routing itself is consumer-side; the trainer only
        # ships the table.
        self._owner_sets: list[list[int]] = []
        self._name_owner_class: list[int] = []
        self._owned_idx: list[int] = []
        self._held_names: set[str] | None = None
        # Strong refs to gathered tensors we've shared into the server, keyed by
        # group index. CUDA-IPC exports must outlive the importer, so we hold
        # them until the server reports the group freed. See send_weights.
        self._inflight: dict[int, dict[str, torch.Tensor]] = {}
        self._sync_timing: dict[str, float] = {}
        # [RDT-EXPORT-RING] Reusable packed slots, one per resident group. See
        # _pack_group_for_export. Persistent across syncs: freeing and
        # reallocating them each sync fragmented the caching allocator badly
        # enough to grow reserved memory ~1.5 GB per sync.
        self._export_ring: list[torch.Tensor | None] = []
        self._export_ring_args: list[tuple | None] = []
        # [RDT-EXPORT-RING] Which LIVE group holds which slot, plus the slots
        # free to hand out. A modular counter is NOT safe here: the credit gate
        # bounds the NUMBER of unfreed groups, not their ORDER, and `wait_freed`
        # returns groups in barrier-completion order. If group N+1 frees before
        # group N, a counter hands group N+2 the slot group N is still being
        # served from, and the consumer silently receives the wrong bytes.
        # Tie each slot to its group and return it only when that group frees.
        self._slot_of_group: dict[int, int] = {}
        self._free_slots: list[int] = []

    def _rpc(self, method: str, *args: Any) -> Any:
        """Call one of the server actor's methods and block for the result.
        The single seam through which the engine talks to its server, so tests
        can inject a local (non-Ray) fake server."""
        import ray

        return ray.get(getattr(self._server, method).remote(*args))

    @staticmethod
    def _contig_stride(shape) -> tuple:
        stride: list[int] = []
        acc = 1
        for s in reversed(list(shape)):
            stride.append(acc)
            acc *= int(s)
        return tuple(reversed(stride))

    def _pack_group_for_export(self, held: list, slot_idx: int) -> tuple:
        """Copy ``held`` into ring slot ``slot_idx``; return ``(storages, views,
        refs)`` in the same shape the per-storage path returns, but with ONE
        storage. Views are built exactly as ``publish_group`` rebuilds them, so
        the two sides cannot disagree about layout.

        Slot safety is NOT the credit gate on its own. That gate bounds how MANY
        groups are unfreed, not the order they free in, and barriers complete out
        of order (``pre_free`` fires at plan start; a light group can finish
        before an earlier heavy one). Reusing slots by a modular counter
        therefore hands a live group's buffer to a later group and silently
        serves the wrong bytes — which is exactly what it did, see
        ``~/default/rdt_reward_collapse_investigation.md``.

        The caller instead takes ``slot_idx`` from a pool keyed by LIVE group
        (``_slot_of_group`` / ``_free_slots``), and ``_drop_inflight`` returns a
        slot only when its group is actually freed everywhere.
        """
        offsets: list[int] = []
        cur = 0
        for _name, t in held:
            cur = (cur + 15) & ~15  # 16B keeps every element size we ship aligned
            offsets.append(cur)
            cur += t.numel() * t.element_size()
        need = max(16, (cur + 15) & ~15)

        slot = self._export_ring[slot_idx]
        device = held[0][1].device
        if slot is None or slot.numel() < need or slot.device != device:
            self._export_ring[slot_idx] = self._export_ring_args[slot_idx] = None
            slot = torch.empty(need, dtype=torch.uint8, device=device)
            self._export_ring[slot_idx] = slot

        ust = slot.untyped_storage()
        sid = ust.data_ptr()
        reduce_args = self._export_ring_args[slot_idx]
        if reduce_args is None:
            base = torch.empty(0, dtype=torch.uint8, device=device)
            base.set_(ust, 0, (ust.nbytes(),))
            _rebuild, reduce_args = reduce_tensor(base)
            self._export_ring_args[slot_idx] = reduce_args

        views: dict[str, tuple] = {}
        refs: dict[str, torch.Tensor] = {}
        for (name, t), off in zip(held, offsets):
            shape, esz = tuple(t.shape), t.element_size()
            stride = self._contig_stride(shape)
            dst = torch.as_strided(slot.view(t.dtype), shape, stride, off // esz)
            dst.copy_(t)
            refs[name] = dst
            views[name] = (sid, str(t.dtype).split(".")[-1], list(shape), list(stride), off // esz)

        # [RDT-EXPORT-RING] The bytes must be ON THE DEVICE before publish_group
        # announces them. These `copy_`s are enqueued on THIS rank's stream, but
        # the sidecar reads the slot from ANOTHER PROCESS over CUDA IPC, and
        # there is no ordering between our stream and that process's. Without
        # this sync the producer packs whatever happens to be in the slot when it
        # gets there -- silently, and only for ring-packed tensors, since the
        # direct path ships tensors the source produced long before.
        #
        # This is the bug that collapsed training: partially-stale small weights
        # (k/v projections, norms, router) left generation plausible while the
        # rollout/train logprob gap blew up ~50x, which produced a huge first
        # gradient and destroyed the policy in one step. See
        # ~/default/rdt_reward_collapse_investigation.md.
        if slot.is_cuda:
            torch.cuda.current_stream(slot.device).synchronize()
        return {sid: reduce_args}, views, refs

    def _publish_async(self, group_idx: int, entries):
        """Fire publish_group WITHOUT waiting on the RPC (the gather loop
        overlaps the publish's server-side rebuild with the next group's
        gather) and return a handle that ``_await_publish`` resolves. Ray actor
        handle in production; a plain (non-Ray) fake server runs inline."""
        method = self._server.publish_group
        remote = getattr(method, "remote", None)
        if remote is not None:
            return remote(group_idx, entries)
        return method(group_idx, entries)

    def _await_publish(self, ref) -> None:
        """Resolve one async publish so a server-side rebuild error surfaces at
        window depth instead of at end_sync. Publishes carry nothing back —
        freed groups flow only through wait_freed/end_sync (one channel; see
        publish_group)."""
        import ray

        if isinstance(ref, ray.ObjectRef):
            ray.get(ref)

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

        # The VMM-based expandable_segments allocator makes IPC storage opens ~9x
        # slower on both export and rebuild, and CUDA-IPC publish is this engine's
        # hot path. Frameworks enabling it should disable it around send_weights.
        # Only the env var is visible; runtime changes are not introspectable.
        import os as _os

        if "expandable_segments:True" in _os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
            logger.warning(
                "Sharded-RDT trainer: PYTORCH_CUDA_ALLOC_CONF enables "
                "expandable_segments; CUDA-IPC weight publishing will be "
                "several times slower. Disable expandable segments around "
                "weight sync (allocate gather buffers in classic segments)."
            )

        engine._meta = list(source.metadata())
        names = [m.name for m in engine._meta]
        engine._groups = layerwise_groups(names)
        flat = [n for g in engine._groups for n in g]
        if flat != names:
            raise ValueError(
                "Sharded RDT requires a WeightSource whose metadata order is "
                "group-contiguous. Reorder the source so all names sharing a "
                "layer index are adjacent."
            )

        world, rank = engine._world_and_rank()
        engine._resolve_ownership(world, rank)
        engine._spawn_server(sorted(engine._held_names or []))

        # Every rank's server must exist before the sender's init RPC (the worker
        # init calls reserve_serve_buffer back on ALL producer servers). The
        # all-gather of server names doubles as that barrier.
        server_names = engine._all_gather_server_names(world, rank)
        # Retained so a RESTARTED consumer can be re-initialized without another
        # all-gather (see get_worker_init_payload). The uuid actor names stay valid
        # for the run, because producers never restart.
        engine._server_names = server_names

        if engine.is_sender:
            worker_init = engine._build_worker_init_info(server_names)
            engine.client.init_weight_transfer_engine(asdict(worker_init))
        return engine

    def get_worker_init_payload(self) -> dict:
        """The consumer-side init payload, rebuilt on demand. Pure — no collective.

        Reads only retained state, so a restarted inference engine can rejoin at a
        sync boundary: the driver can ask any time, including mid-run with every
        rank in its own training step. A collective here would deadlock, since the
        ranks are not at a matching one.

        Raises:
            RuntimeError: called before ``trainer_init`` cached the server names.
        """
        if getattr(self, "_server_names", None) is None:
            raise RuntimeError(
                "get_worker_init_payload requires trainer_init to have run (the "
                "producer server names are gathered there)."
            )
        return asdict(self._build_worker_init_info(self._server_names))

    def _world_and_rank(self) -> tuple[int, int]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_world_size(), torch.distributed.get_rank()
        return 1, self._init_info.rank

    def _resolve_ownership(self, world: int, rank: int) -> None:
        """Resolve which rank holds which name, and this rank's publish plan.

        A source may hold only part of the model — pipeline stages, expert
        parallelism, or any mix — so each rank declares its held names and the
        fleet all-gathers them. The consumers route per name, so the wire carries
        the transposed result: the distinct owner sets, and a per-name index into
        them.

        The masks are positional over metadata order, which is why the metadata
        digest is checked first: a rank whose names disagree would transpose into
        the wrong owners entirely.
        """
        assert self.source is not None  # guaranteed by trainer_init
        num_groups = len(self._groups)
        names = [m.name for m in self._meta]
        held = self.source.held_names()
        if held is None:
            held_set = set(names)
        else:
            held_set = {str(n) for n in held}
            unknown = sorted(held_set - set(names))
            if unknown:
                raise ValueError(
                    f"WeightSource.held_names() lists {len(unknown)} name(s) not " f"in metadata(), e.g. {unknown[:3]}."
                )
            if not held_set:
                raise ValueError(
                    "WeightSource.held_names() is empty; a rank with nothing to " "serve cannot take part in the gather"
                )

        # One collective carries this rank's holdings as a bitmask over metadata
        # order, plus the digest of the metadata it indexes into. The digest is
        # what makes partial ownership safe: only the sender's metadata reaches
        # the consumers, so a rank describing just its own share would silently
        # mis-serve the model.
        digest = self._meta_digest()
        mask = bytearray((len(names) + 7) // 8)
        for i, n in enumerate(names):
            if n in held_set:
                mask[i >> 3] |= 1 << (i & 7)
        per_rank = self._all_gather_owned(world, (digest, bytes(mask)))
        mismatched = [r for r, (d, *_rest) in enumerate(per_rank) if d != digest]
        if mismatched:
            raise ValueError(
                f"WeightSource.metadata() disagrees across trainer ranks "
                f"(rank {rank} digest {digest}, differing ranks {mismatched[:4]}). "
                "Every rank must describe the WHOLE model, even when it holds "
                "only some of it."
            )
        masks = [m for _d, m in per_rank]

        # Transpose to per-name owners, then dedup into classes. Numbering by
        # FIRST APPEARANCE in metadata order keeps it a pure function of
        # rank-identical inputs, which is what lets a rejoining consumer rebuild
        # the identical table from get_worker_init_payload.
        owner_sets: list[list[int]] = []
        class_of_owners: dict[tuple[int, ...], int] = {}
        name_owner_class: list[int] = []
        for i, n in enumerate(names):
            owners = tuple(r for r, m in enumerate(masks) if m[i >> 3] & (1 << (i & 7)))
            if not owners:
                raise ValueError(
                    f"no trainer rank holds {n!r}; every name in metadata() must "
                    "be held by at least one rank or it can never be served."
                )
            ci = class_of_owners.get(owners)
            if ci is None:
                ci = len(owner_sets)
                class_of_owners[owners] = ci
                owner_sets.append(list(owners))
            name_owner_class.append(ci)

        self._owner_sets = owner_sets
        self._name_owner_class = name_owner_class
        # Groups holding anything here: exactly what iteration must cover.
        self._owned_idx = [gi for gi in range(num_groups) if any(n in held_set for n in self._groups[gi])]
        self._held_names = held_set

    def _validate_held_yields(self, gi: int, names, tensors) -> None:
        """Stamps must be truthful against yields (the ABC contract's first
        invariant), and this is the one place both sit side by side. Without
        this check, a source that stamps a name as held but yields ``None`` for
        it produces a 300s stall-watchdog death (consumers route pulls here,
        the pull passes the served-names guard, and the cache wait never
        completes) instead of an immediate, named error. It also discharges
        the interchangeability invariant: if every rank's yields match the
        rank-identical stamps, same-coordinate ranks hold identical sets by
        construction. Cost: one set lookup per name per sync."""
        if self._held_names is None:
            return
        for name, tensor in zip(names, tensors):
            if (tensor is None) == (name in self._held_names):
                claim = "does not hold" if tensor is not None else "holds"
                have = "a real tensor" if tensor is not None else "None"
                raise RuntimeError(
                    f"held_names() disagrees with the yielded tensors: group {gi} "
                    f"name {name!r} yielded {have} but this rank {claim} it. A "
                    "WeightSource must yield a real tensor for every held name "
                    "and None for the rest (see WeightSource.held_names)."
                )

    def _meta_digest(self) -> str:
        """Stable digest of this rank's metadata (name order + count)."""
        import hashlib

        h = hashlib.sha256()
        h.update(f"{len(self._meta)}\n".encode())
        for m in self._meta:
            h.update(m.name.encode())
            h.update(b"\n")
        return h.hexdigest()[:16]

    def _all_gather_owned(self, world: int, mine: tuple) -> list[tuple]:
        """All-gather each rank's (metadata digest, held-name bitmask)."""
        if world <= 1 or not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return [mine]
        gathered: list[Any] = [None] * world
        torch.distributed.all_gather_object(gathered, mine)
        return gathered

    def _all_gather_server_names(self, world: int, rank: int) -> list[str]:
        assert self._server_name is not None
        if world <= 1 or not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return [self._server_name]
        gathered: list[str | None] = [None] * world
        torch.distributed.all_gather_object(gathered, self._server_name)
        return [n for n in gathered if n is not None]

    def _spawn_server(self, served_names: list[str]) -> None:
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        # The trainer is usually a separate install from the workers, so it gets
        # its own check rather than trusting the consumer side's.
        check_ray_rdt_version()

        ii = self._init_info
        self._server_name = f"vllm_rdt_producer_{uuid.uuid4().hex[:12]}_rk{ii.rank}"
        node_id = ray.get_runtime_context().get_node_id()
        # Pin the server to this rank's physical GPU: num_gpus=0 so Ray does not
        # allocate a second, CUDA_VISIBLE_DEVICES so CUDA IPC to the rank's
        # gathered tensors works. max_concurrency > 1 serves pulls while control
        # calls are in flight; enable_tensor_transport gives the NIXL serve.
        gpu_ids = ray.get_gpu_ids()
        # The server is the trainer rank's process twin: forward the env the
        # rank runs under (library paths etc.) so it imports torch/vllm the
        # same way, then pin it to the rank's physical GPU for CUDA IPC.

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
            # A num_gpus=0 actor sharing the rank's GPU: Ray would set
            # CUDA_VISIBLE_DEVICES="" and hide every GPU, so tell it not to touch
            # the var and pin the device ourselves.
            env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
            env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[0])
        runtime_env = {"env_vars": env_vars} if env_vars else {}
        server_cls = ray.remote(_RDTProducerServer).options(
            name=self._server_name,
            namespace=ii.trainer_actor_namespace,
            num_cpus=0,
            num_gpus=0,
            # Thread budget: under EP-local routing every consumer pulls from every
            # producer, and with issue-ahead each can park K produce calls in the
            # cache-wait — C*K blocked calls. A smaller pool queues the control
            # plane behind them and deadlocks: the parked produces wait for a
            # publish that can never get a thread (the sync-2 wedge at 235B tp8,
            # 16 parked vs 8 threads). Size for the worst case plus control-plane
            # slack.
            max_concurrency=ii.num_consumers * ii.num_rdt_buffers + 4,
            enable_tensor_transport=True,
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
            runtime_env=runtime_env,
        )
        self._server = server_cls.remote(
            served_names=served_names,
            num_rdt_buffers=ii.num_rdt_buffers,
            buffer_presize_gb=ii.buffer_presize_gb,
            pack_check=ii.pack_check,
            gather_lookahead=ii.gather_lookahead,
            stall_timeout_s=ii.stall_timeout_s,
            workers_per_replica=ii.workers_per_replica,
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
            owner_sets=self._owner_sets,
            name_owner_class=self._name_owner_class,
            num_consumers=self._init_info.num_consumers,
            num_rdt_buffers=self._init_info.num_rdt_buffers,
            buffer_presize_gb=self._init_info.buffer_presize_gb,
            pack_check=self._init_info.pack_check,
        )

    # ---------------- per-round ----------------

    def send_weights(self, live_consumer_ids: Collection[int] | None = None) -> None:
        """Gather this rank's weights and publish them for the consumers to pull.

        ``live_consumer_ids`` restricts the sync to the consumers still alive;
        ``None`` serves the whole provisioned set. The provisioned geometry is
        FROZEN for the run — a degraded sync only lowers the live count handed to
        ``begin_sync``. Every rank must get the SAME live set, since they share
        the gather collectives, so the caller computes it once for all of them.
        """
        assert self.source is not None
        if live_consumer_ids is None:
            live_ids = list(range(self._init_info.num_consumers))
            live_count = len(live_ids)
        else:
            live_ids = sorted(set(int(c) for c in live_consumer_ids))
            live_count = len(live_ids)
            logger.warning(
                "[rdt-degraded] serving %d/%d live consumers; every group's free " "barrier counts to the live total",
                live_count,
                self._init_info.num_consumers,
            )
        self._send_weights_inner(live_count, live_ids)

    def _send_weights_inner(self, live_count: int, live_ids: list[int]) -> None:
        if not self.is_sender:
            self._run_gather_loop(update_future=None, live_count=live_count, live_ids=live_ids)
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
            self._run_gather_loop(update_future=future, live_count=live_count, live_ids=live_ids)
            _gather = _t.time() - _t0
            future.result()  # surface inference-side errors
            _tail = _t.time() - _t0 - _gather
            self._sync_timing["update_weights_seconds"] = time.perf_counter() - tu0

        tf0 = time.perf_counter()
        self.client.finish_weight_update()
        self._sync_timing["finish_seconds"] = time.perf_counter() - tf0
        self._sync_timing["wall_seconds"] = time.perf_counter() - wall0
        logger.info("[rdt-sync-timing] gather_publish_loop=%.2fs consumer_tail_after_gather=%.2fs", _gather, _tail)

    def _run_gather_loop(self, update_future, live_count: int, live_ids: list[int] | None = None) -> None:
        """Gather this rank's weights group-by-group and publish each into the
        server over CUDA IPC. A gathered group is published — serveable —
        immediately; the loop gates BEFORE the next gather while more than
        `gather_lookahead` groups are unfreed (the per-group free barrier: a
        credit releases when every live consumer has signaled the group). So
        the loop self-paces to the consumers' pull rate with at most
        `gather_lookahead + 1` groups resident. Runs on every rank; only the
        sender has an `update_future` to fail fast on."""
        gather0 = time.perf_counter()
        _t_next = _t_pub = _t_exp = _t_credit = 0.0
        assert self.source is not None  # guaranteed by trainer_init
        # The live IDS ride along with the count: the free barrier only needs the
        # count, but the producer's slot-sharing rendezvous needs to know WHICH
        # consumers are live ([RDT-SHARE-SLOTS]).
        self._rpc("begin_sync", live_count, live_ids)
        # One generator resume per GROUP: `iter_groups` yields (names, tensors)
        # per owned group in metadata order. Every owner must reach a group in the
        # same order or their shared gather collective mismatches.
        groups = self.source.iter_groups()
        # Publishes fire without an inline ray.get and are harvested this window
        # deep, so the RPC and server-side rebuild overlap the NEXT group's
        # gather/export. The window also surfaces a server-side error at depth 2
        # rather than at end_sync.
        _PUBLISH_WINDOW = 2
        pending_publish: list = []
        # [RDT-GATHER-CREDIT] The memory bound: gate BEFORE gathering while more
        # than `bound` groups are unfreed, so at most bound + 1 are resident. The
        # count is exact — `_inflight` gains its entry right after each gather and
        # only wait_freed/end_sync shrinks it, so a group freed server-side but not
        # yet collected still holds its refs and still counts. The publish window
        # drains first, so every resident group is pullable while we wait.
        # 0 is legal and means NO pipelining: the loop waits for group N to be
        # freed by every live consumer before gathering N+1, so exactly one
        # group is resident and gather/publish/pull never overlap. It is a
        # deliberate ablation baseline, not a production setting -- the sync
        # then costs the full serialized (gather + publish + pull + free) RTT
        # per group. 1 (the default) overlaps N+1's gather with N's pulls.
        bound = max(0, self._init_info.gather_lookahead)

        # [RDT-EXPORT-RING] `bound + 1` slots is exactly the residency the credit
        # gate enforces. Small storages are packed into a slot and exported once
        # per slot instead of once per storage (518 -> 7 cudaIpcGetMemHandle
        # calls per rank per sync at 235B, each 0.5-1.5ms).
        use_ring = os.environ.get("SKYRL_RDT_EXPORT_RING", "1") not in ("0", "false", "False")
        ring_max_bytes = int(float(os.environ.get("SKYRL_RDT_EXPORT_RING_MAX_MB", "64")) * (1 << 20))
        nslots = bound + 1
        # data_ptr -> (storage weakref, reduce_args), this sync only. `storages`
        # is per-group, so a source that reuses a buffer across groups (the
        # expert-stack ring does) would otherwise re-export it every group. The
        # weakref is load-bearing: a freed allocation can be reissued at the same
        # address, and serving that from a stale handle is silently wrong bytes.
        handle_cache: dict[int, tuple] = {}
        # getattr: tests drive this loop on engines built without __init__.
        if use_ring and len(getattr(self, "_export_ring", [])) != nslots:
            self._export_ring = [None] * nslots
            self._export_ring_args = [None] * nslots
        # Slots are handed out per group and returned by _drop_inflight, so this
        # pool must start full every sync (the previous sync ended with every
        # group freed).
        self._slot_of_group = {}
        self._free_slots = list(range(nslots))

        try:
            for gi in self._owned_idx:
                group = self._groups[gi]
                _tc = time.perf_counter()
                while len(self._inflight) > bound:
                    while pending_publish:
                        self._await_publish(pending_publish.pop(0))
                    self._drop_inflight(self._rpc("wait_freed"))
                _tn = time.perf_counter()
                _t_credit += _tn - _tc
                names, tensors = next(groups)
                if list(names) != list(group):
                    raise RuntimeError(
                        f"WeightSource group yielded {len(names)} names starting {names[:2]!r} "
                        f"but expected {len(group)} starting {group[:2]!r}; "
                        "iteration order must match metadata."
                    )
                _te = time.perf_counter()
                _t_next += _te - _tn
                self._validate_held_yields(gi, names, tensors)
                # Share each unique STORAGE once and describe every name as an
                # as_strided view onto it. ``None`` means a name this rank does not
                # hold (a foreign expert); the source keeps it in the list so the
                # order check stays rank-uniform, and it is dropped here, before
                # the IPC export, matching the sidecar's served_names.
                storages: dict[int, tuple] = {}
                views: dict[str, tuple] = {}
                refs: dict[str, torch.Tensor] = {}
                small: list = []
                for name, tensor in zip(names, tensors):
                    if tensor is None:
                        continue
                    tensor = tensor.detach()
                    if not tensor.is_cuda:
                        tensor = tensor.cuda()
                    # Threshold on the STORAGE, not the tensor: the expert views
                    # are ~12 MB each but 48 of them share one ~400 MB stack the
                    # source already coalesced, so a per-tensor test would pack a
                    # whole layer and cost +1 group of peak memory.
                    if use_ring and tensor.untyped_storage().nbytes() <= ring_max_bytes:
                        small.append((name, tensor))
                        continue
                    tensor = tensor.contiguous()
                    refs[name] = tensor  # keep the export alive
                    ust = tensor.untyped_storage()
                    sid = ust.data_ptr()
                    if sid not in storages:
                        cached = handle_cache.get(sid)
                        if cached is not None and not cached[0].expired():
                            reduce_args = cached[1]
                        else:
                            base = torch.empty(0, dtype=torch.uint8, device=tensor.device)
                            base.set_(ust, 0, (ust.nbytes(),))
                            _rebuild, reduce_args = reduce_tensor(base)
                            handle_cache[sid] = (StorageWeakRef(ust), reduce_args)
                        storages[sid] = reduce_args
                    views[name] = (
                        sid,
                        str(tensor.dtype).split(".")[-1],
                        list(tensor.shape),
                        list(tensor.stride()),
                        tensor.storage_offset(),
                    )
                if small:
                    # Take a slot no LIVE group is using. The gate above already
                    # guarantees one exists (at most `bound` groups are unfreed
                    # and there are `bound + 1` slots), so this loop is a
                    # backstop, not the normal path — but it must drain publishes
                    # before blocking, exactly like the credit gate.
                    while not self._free_slots:
                        while pending_publish:
                            self._await_publish(pending_publish.pop(0))
                        self._drop_inflight(self._rpc("wait_freed"))
                    slot_idx = self._free_slots.pop(0)
                    self._slot_of_group[gi] = slot_idx
                    st, vw, rf = self._pack_group_for_export(small, slot_idx)
                    storages.update(st)
                    views.update(vw)
                    refs.update(rf)
                del small
                _t_exp += time.perf_counter() - _te
                del tensors
                # The loop above leaves its last iteration's `tensor` / `ust` /
                # `base` bound for the rest of this group's turn -- through the
                # credit gate and into the NEXT gather. `base` is a
                # whole-storage view, so that pins one tensor AND its entire
                # allocation past `_drop_inflight`, where `refs` was already
                # dropped. It is invisible to the `_inflight` accounting the
                # `lookahead + 1` residency bound is built on, so nothing
                # catches it. Assignment rather than `del`: all three are
                # unbound when the group had no direct-path tensor.
                tensor = ust = base = None
                if not refs:
                    # A group with nothing held here cannot occur (every group
                    # carries replicated names), but publishing an empty group
                    # would park a credit nobody's pull is waiting on; skip.
                    continue
                # Hold our refs before publishing; drop them only when the
                # server reports the group freed (IPC export must outlive import).
                self._inflight[gi] = refs
                _tp = time.perf_counter()
                pending_publish.append(self._publish_async(gi, (storages, views)))
                while len(pending_publish) >= _PUBLISH_WINDOW:
                    self._await_publish(pending_publish.pop(0))
                _t_pub += time.perf_counter() - _tp
                if update_future is not None and update_future.done():
                    # update_weights returned/failed early -- surface now instead of
                    # blocking further gathers.
                    update_future.result()
            _tp = time.perf_counter()
            while pending_publish:
                self._await_publish(pending_publish.pop(0))
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
            self._sync_timing["credit_wait_seconds"] = _t_credit
            # Keep the slots, drop the handles: the sidecar's base cache is
            # cleared per sync, so a handle must not outlive the sync either.
            self._export_ring_args = [None] * len(getattr(self, "_export_ring", []))

    def _drop_inflight(self, freed_keys: list) -> None:
        """Release a freed group's refs AND its export-ring slot.

        Returning the slot here — rather than assuming FIFO reuse — is what makes
        the ring safe: a group's slot becomes reusable exactly when every live
        consumer has finished with that group, whatever order the groups complete
        in.
        """
        free_slots = getattr(self, "_free_slots", None)
        slot_of_group = getattr(self, "_slot_of_group", None)
        for k in freed_keys:
            gi = int(k)
            self._inflight.pop(gi, None)
            if slot_of_group is None:
                continue
            slot = slot_of_group.pop(gi, None)
            if slot is not None and free_slots is not None and slot not in free_slots:
                free_slots.append(slot)

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
