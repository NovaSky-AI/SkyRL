"""Sharded RDT (Ray Direct Transport / NIXL) weight transfer strategy.

Unlike the broadcast (push) and CUDA-IPC strategies, the ``sharded_rdt`` backend
is **pull-based**: the vLLM inference workers pull only the slice each one
consumes from a *named* trainer Ray actor over Ray's ``tensor_transport="nixl"``,
driven by a plan the vLLM ``ShardedRDTWeightTransferEngine`` bakes once at init.
The receiver is that vLLM engine (built + baked on the inference workers via the
SkyRL worker extension); there is no SkyRL-side receiver.

Driving model (mirrors the vLLM ``sharded_rdt_trainer`` reference, adapted to run
in SkyRL's training workers): the consumer engine pre-builds its whole-model
chunk/free plan at init (``group_lens`` on the init info), so one EMPTY
``update_weights`` per sync makes every worker pull all of its slices, pipelined
over its receive-arena ring. On the training side rank 0 fires that single update
CONCURRENTLY with a gather loop that runs on every rank:

    start_weight_update
      concurrently:
        update_weights_rdt({})          # (rank 0) workers pull, pipelined
        gather loop (all ranks):        # collective full_tensor per group;
          for each layer-aligned group: # rank 0 caches + serves NIXL pulls,
            gather_layer(group)         # blocks on gather_lookahead backpressure
      finish_weight_update              # (rank 0) workers drain + finalize

The gather (``full_tensor``) is a collective every rank joins, so the per-group
collectives keep all ranks in lockstep with rank 0's serve/backpressure pace; the
concurrent ``update_weights_rdt`` is what drives the consumer pulls whose
``free_gather`` back-edge releases rank 0's gather-ahead slots.

All engine-specific wiring lives here, behind the generic ``WeightTransferStrategy``
/ ``WeightTransferSender`` hooks (``populate_init_info`` / ``initialize_receivers``
/ ``bind_trainer_worker``), so ``Worker.init_weight_sync_state`` stays backend-agnostic.
"""

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

import ray
import torch

from skyrl.backends.skyrl_train.weight_sync.base import WeightChunk
from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_common import (
    ALLOWED_OPS,
    arena_alloc_bytes,
    count_consumers,
    layerwise_groups,
)
from skyrl.backends.skyrl_train.weight_sync.transfer_strategy import (
    WeightSyncInitInfo,
    WeightTransferSender,
    WeightTransferStrategy,
)
from skyrl.train.utils.utils import str_to_torch_dtype

if TYPE_CHECKING:
    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
        RemoteInferenceClient,
    )
    from skyrl.train.config import InferenceEngineConfig

logger = logging.getLogger(__name__)

# Name the trainer rank-0 actor is registered under and that the vLLM engine
# resolves via ray.get_actor. Must match the name used when the master training
# actor is created (see PPORayActorGroup in workers/worker.py).
RDT_TRAINER_ACTOR_NAME = "skyrl_rdt_trainer"
RDT_PRODUCE_METHOD_NAME = "rdt_produce_weights_batched"

# Defaults for the must-agree wire knobs (mirror the vLLM sharded_rdt defaults).
_DEFAULT_NUM_RDT_BUFFERS = 2
_DEFAULT_LAYERWISE_SPLIT = 1
_DEFAULT_ARENA_PRESIZE_GB = 0.0
_DEFAULT_PACK_CHECK = False
# Max gathered-but-not-yet-freed groups the producer holds at once (backpressure).
_DEFAULT_GATHER_LOOKAHEAD = 2


@dataclass
class ShardedRdtInitInfo(WeightSyncInitInfo):
    """Initialization info for the sharded_rdt (NIXL pull) backend.

    Carries everything the vLLM ``ShardedRDTWeightTransferEngine`` needs to bake
    its replay plan, pre-build the static whole-model chunk plan (``group_lens``),
    resolve the trainer actor, and size its receive arenas. ``names``/
    ``dtype_names``/``shapes``/``group_lens`` are filled by ``populate_init_info``
    from the trainer's weight extractor (in group-major order) before init is sent.
    """

    trainer_actor_name: str
    trainer_actor_namespace: Optional[str]
    num_consumers: int
    model_dtype_str: str = "bfloat16"
    produce_method_name: str = RDT_PRODUCE_METHOD_NAME
    num_rdt_buffers: int = _DEFAULT_NUM_RDT_BUFFERS
    layerwise_split: int = _DEFAULT_LAYERWISE_SPLIT
    arena_presize_gb: float = _DEFAULT_ARENA_PRESIZE_GB
    pack_check: bool = _DEFAULT_PACK_CHECK
    gather_lookahead: int = _DEFAULT_GATHER_LOOKAHEAD
    names: List[str] = field(default_factory=list)
    dtype_names: List[str] = field(default_factory=list)
    shapes: List[List[int]] = field(default_factory=list)
    group_lens: List[int] = field(default_factory=list)

    @staticmethod
    def strategy_type() -> type:
        return ShardedRdtTransferStrategy

    def to_api_payload(self) -> Dict[str, Any]:
        """Return the dict for the vLLM ``ShardedRDTWeightTransferInitInfo``.

        Sent (identically to every server) via the client's
        ``init_weight_transfer_engine_rdt`` collective_rpc. Drops the SkyRL-only
        fields (``override_existing_receiver``, ``model_dtype_str``,
        ``gather_lookahead`` — a producer-only knob); the vLLM init info keys
        dtype per-param via ``dtype_names``.
        """
        return {
            "trainer_actor_name": self.trainer_actor_name,
            "trainer_actor_namespace": self.trainer_actor_namespace,
            "produce_method_name": self.produce_method_name,
            "num_consumers": self.num_consumers,
            "num_rdt_buffers": self.num_rdt_buffers,
            "layerwise_split": self.layerwise_split,
            "arena_presize_gb": self.arena_presize_gb,
            "pack_check": self.pack_check,
            "names": self.names,
            "dtype_names": self.dtype_names,
            "shapes": self.shapes,
            "group_lens": self.group_lens,
        }


class ShardedRdtWeightTransferSender(WeightTransferSender):
    """Drives the pull-based whole-model concurrent weight sync.

    Runs on every training rank. Only rank 0 talks to the inference client and
    serves NIXL pulls; all ranks participate in the ``gather_layer`` collectives.
    The owning training worker is injected via ``bind_trainer_worker`` (it hosts
    the producer serve surface + gather cache — see ``RdtProducerMixin``).
    """

    def __init__(
        self,
        init_info: ShardedRdtInitInfo,
        inference_client: "RemoteInferenceClient",
    ) -> None:
        self._init_info = init_info
        self._inference_client = inference_client
        # Set by bind_trainer_worker (called from init_weight_sync_state).
        self._worker: Any = None

    def bind_trainer_worker(self, worker: Any) -> None:
        """Keep a reference to the trainer worker that backs gather/serve + pulls."""
        self._worker = worker
        # Push the must-agree serve knobs onto the producer before any sync.
        worker._rdt_configure(
            num_consumers=self._init_info.num_consumers,
            num_rdt_buffers=self._init_info.num_rdt_buffers,
            arena_presize_gb=self._init_info.arena_presize_gb,
            pack_check=self._init_info.pack_check,
            gather_lookahead=self._init_info.gather_lookahead,
        )

    def _run_gather_loop(self, groups: List[List[str]], dtype: torch.dtype) -> None:
        """Gather this rank's weights group-by-group into the producer cache.

        Runs in a worker thread (see ``send_chunks``) so it executes CONCURRENTLY
        with rank 0's ``update_weights_rdt`` awaiting on the event loop. Every
        rank runs it: ``gather_layer`` is a collective (``full_tensor``), and on
        rank 0 it blocks on the gather-ahead backpressure that the consumers'
        ``free_gather`` back-edge releases. The per-group collective keeps all
        ranks paced with rank 0.
        """
        worker = self._worker
        # A worker thread does not inherit the main thread's current CUDA device;
        # set it so the gather collectives + casts land on this rank's GPU.
        if torch.cuda.is_available():
            torch.cuda.set_device(torch.cuda.current_device())
        for group_names in groups:
            worker.gather_layer(group_names, dtype)

    async def send_chunks(
        self,
        chunks: Iterable[WeightChunk],
        weight_metadata: Optional[Dict[str, list]] = None,
    ) -> None:
        """Drive the concurrent gather + pull sync.

        ``chunks`` is ignored — RDT does not push extracted chunks; it gathers
        live params layer-by-layer and the inference workers pull slices. Every
        rank must call this (``gather_layer`` is a collective).
        """
        if weight_metadata is None or "names" not in weight_metadata:
            raise ValueError(
                "sharded_rdt requires weight_metadata with 'names'. Call "
                "weight_extractor.get_weight_metadata() and pass it to send_chunks."
            )
        if self._worker is None:
            raise RuntimeError(
                "ShardedRdtWeightTransferSender not bound to a worker. "
                "init_weight_sync_state must call sender.bind_trainer_worker(self)."
            )

        # Gather groups must match the group-major partition baked into the init
        # info's group_lens; recomputing from the SAME name list guarantees it.
        groups = layerwise_groups(list(self._init_info.names))
        dtype = str_to_torch_dtype(self._init_info.model_dtype_str)
        is_rank0 = torch.distributed.is_initialized() and torch.distributed.get_rank() == 0

        self._worker._rdt_begin_sync()
        try:
            if is_rank0:
                await self._inference_client.start_weight_update(is_checkpoint_format=True)
                # The gather loop blocks on collectives + backpressure, so run it
                # in a thread while the single EMPTY update_weights_rdt awaits on
                # the event loop: the consumers' pulls (served on the actor
                # threadpool) release the gather-ahead slots. The consumer
                # pre-built its static plan at init, so the update carries no names.
                gather_fut = asyncio.create_task(asyncio.to_thread(self._run_gather_loop, groups, dtype))
                update_fut = asyncio.create_task(self._inference_client.update_weights_rdt({}))
                await self._join_concurrent(gather_fut, update_fut)
                self._worker._rdt_end_sync()
                await self._inference_client.finish_weight_update()
            else:
                await asyncio.to_thread(self._run_gather_loop, groups, dtype)
                self._worker._rdt_end_sync()
        except BaseException as e:  # noqa: BLE001 - unblock any waiter, then re-raise
            self._worker._rdt_set_error(repr(e))
            raise

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    async def _join_concurrent(self, gather_fut: "asyncio.Task", update_fut: "asyncio.Task") -> None:
        """Await both the gather loop and the update; if either fails, unblock the
        producer (so the other stops waiting) and surface the first error."""
        done, pending = await asyncio.wait({gather_fut, update_fut}, return_when=asyncio.FIRST_EXCEPTION)
        err = next((t.exception() for t in done if t.exception() is not None), None)
        if err is not None:
            # Wake any producer wait (backpressure / cache) so the still-running
            # coroutine can exit instead of hanging, then join it.
            self._worker._rdt_set_error(repr(err))
            for t in pending:
                try:
                    await t
                except BaseException:  # noqa: BLE001 - primary error re-raised below
                    pass
            raise err
        for t in pending:
            await t

    def teardown(self) -> None:
        self._worker = None


class ShardedRdtTransferStrategy(WeightTransferStrategy):
    """Factory for the sharded_rdt (NIXL pull) weight transfer strategy."""

    @staticmethod
    def create_init_info(
        ie_cfg: "InferenceEngineConfig", inference_world_size: Optional[int] = None
    ) -> ShardedRdtInitInfo:
        """Build the RDT init info.

        ``inference_world_size`` is the total inference-worker (consumer) count,
        used for the producer's free ref-count target and the engine's M:N block
        assignment. ``names``/``dtype_names``/``shapes``/``group_lens`` are filled
        in later by ``populate_init_info`` from the trainer's weight extractor.
        The actor namespace is the current Ray namespace so the vLLM engine's
        ``ray.get_actor`` resolves the same actor the training side names.
        """
        try:
            namespace = ray.get_runtime_context().namespace or None
        except Exception:  # noqa: BLE001
            namespace = None
        if not inference_world_size:
            raise ValueError(
                "sharded_rdt requires the inference world size (consumer count); " f"got {inference_world_size!r}."
            )
        return ShardedRdtInitInfo(
            override_existing_receiver=not ie_cfg.run_engines_locally,
            trainer_actor_name=RDT_TRAINER_ACTOR_NAME,
            trainer_actor_namespace=namespace,
            num_consumers=int(inference_world_size),
            model_dtype_str=ie_cfg.model_dtype,
            num_rdt_buffers=int(getattr(ie_cfg, "rdt_num_rdt_buffers", _DEFAULT_NUM_RDT_BUFFERS)),
            layerwise_split=int(getattr(ie_cfg, "rdt_layerwise_split", _DEFAULT_LAYERWISE_SPLIT)),
            arena_presize_gb=float(getattr(ie_cfg, "rdt_arena_presize_gb", _DEFAULT_ARENA_PRESIZE_GB)),
            pack_check=bool(getattr(ie_cfg, "rdt_pack_check", _DEFAULT_PACK_CHECK)),
            gather_lookahead=int(getattr(ie_cfg, "rdt_gather_lookahead", _DEFAULT_GATHER_LOOKAHEAD)),
        )

    @staticmethod
    def get_vllm_transfer_engine() -> type:
        """Return the vLLM weight-transfer engine class for this strategy."""
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_engine import (
            ShardedRDTWeightTransferEngine,
        )

        return ShardedRDTWeightTransferEngine

    @staticmethod
    def populate_init_info(init_info: WeightSyncInitInfo, *, weight_extractor: Any) -> None:
        """Fill group-major ``names``/``dtype_names``/``shapes``/``group_lens``.

        The vLLM engine bakes its replay plan over the full parameter-name list
        AND pre-builds the static chunk plan from ``group_lens`` at init, so both
        must be conveyed up front (before ``initialize_receivers``) in group-major
        order. The metadata is read without materializing tensors (state_dict
        shapes only), then reordered to the gather-group partition the sender
        drives (``layerwise_groups``).
        """
        if weight_extractor is None:
            raise RuntimeError("sharded_rdt requires a weight extractor on the trainer worker.")
        assert isinstance(init_info, ShardedRdtInitInfo)
        dtype = str_to_torch_dtype(init_info.model_dtype_str)
        metadata = weight_extractor.get_weight_metadata(dtype)
        names = list(metadata["names"])
        dtype_names = list(metadata["dtype_names"])
        shapes = [list(s) for s in metadata["shapes"]]

        # Reorder everything into group-major order (pre / per-layer / post) so
        # names is contiguous per group and group_lens partitions it exactly.
        idx = {n: i for i, n in enumerate(names)}
        groups = layerwise_groups(names)
        order = [idx[n] for g in groups for n in g]
        init_info.names = [names[i] for i in order]
        init_info.dtype_names = [dtype_names[i] for i in order]
        init_info.shapes = [shapes[i] for i in order]
        init_info.group_lens = [len(g) for g in groups]

    @staticmethod
    def initialize_receivers(init_info: WeightSyncInitInfo, inference_client: "RemoteInferenceClient"):
        """Initialize + bake the vLLM engine on the inference side.

        Routed through ``init_weight_transfer_engine_rdt`` (a collective_rpc to
        the worker extension) rather than the native NCCL init endpoint, because
        the worker must inject the model + bake under ``set_current_vllm_config``.
        """
        assert isinstance(init_info, ShardedRdtInitInfo)
        return inference_client.init_weight_transfer_engine_rdt(init_info.to_api_payload())

    @staticmethod
    def create_sender(
        init_info: ShardedRdtInitInfo,
        inference_client: "RemoteInferenceClient",
    ) -> ShardedRdtWeightTransferSender:
        """Create the sender on every training rank (no process group needed)."""
        return ShardedRdtWeightTransferSender(
            init_info=init_info,
            inference_client=inference_client,
        )


class RdtProducerMixin:
    """Trainer-side NIXL producer for the ``sharded_rdt`` weight-sync backend.

    This is the trainer half of the sharded_rdt strategy. It lives in the
    weight-sync layer but is *mixed into* the training worker actor, because its
    methods must be actor methods: the vLLM inference workers resolve the rank-0
    actor by name and call ``rdt_produce_weights_batched`` / ``free_gather`` /
    ``reserve_serve_arena`` over Ray's ``tensor_transport="nixl"``, and
    ``gather_layer`` runs the FSDP ``full_tensor()`` collective over the worker's
    live params. This ports the serve half of vLLM's ``_RDTProducerServer``
    (``sharded_rdt_trainer.py``), with the gather cache populated in-process by
    ``gather_layer`` instead of by CUDA-IPC ``publish_group``.

    Host-actor requirements (the worker that mixes this in must provide):
      * ``self.weight_extractor`` — the SAME extractor used for ``get_weight_metadata``;
        exposes ``model`` (whose ``state_dict()`` yields the per-rank shards keyed
        by the names ``get_weight_metadata`` returns, modulo ``weight_prefix``),
        ``weight_prefix``, and ``_gather_tensor(param)`` (all-gathers a sharded
        param to its full tensor). ``FSDPWeightExtractor`` satisfies all of these.
      * a ``torch.distributed`` process group (the FSDP group).

    Only the rank-0 actor — created named + with ``enable_tensor_transport=True``
    — caches gathered tensors and serves pulls; every rank participates in the
    ``gather_layer`` collective. State is lazily initialized so the mixin composes
    with the worker's own ``__init__`` (it has none of its own).
    """

    # ---------------- state ----------------

    def _rdt_state(self) -> None:
        """Lazily initialize the producer state (the mixin has no __init__)."""
        if getattr(self, "_rdt_inited", False):
            return
        # name -> gathered full tensor (rank 0 only); guarded by _rdt_cache_cond.
        self._rdt_cache: Dict[str, torch.Tensor] = {}
        self._rdt_cache_cond = threading.Condition()
        self._rdt_gather_error: Optional[BaseException] = None

        # [RDT-FREE-REFCOUNT] Each bound consumer fires free_gather for every
        # group; the group is actually freed only on the free_target-th call.
        self._rdt_free_target = 1
        self._rdt_free_counts: Dict[tuple, int] = {}
        self._rdt_free_lock = threading.Lock()

        # [RDT-BACKPRESSURE] Gathered-but-not-yet-freed group keys. gather_layer
        # blocks while len(...) >= gather_lookahead; free_gather drains it.
        self._rdt_lookahead = _DEFAULT_GATHER_LOOKAHEAD
        self._rdt_inflight_keys: List[tuple] = []
        self._rdt_name_to_key: Dict[str, tuple] = {}

        # [RDT-RING] Per-consumer ring of packed serve arenas, rotated per pull.
        self._rdt_nring = _DEFAULT_NUM_RDT_BUFFERS
        self._rdt_serve_rings: Dict[int, List[Optional[torch.Tensor]]] = {}
        self._rdt_serve_idx: Dict[int, int] = {}
        self._rdt_serve_lock = threading.Lock()
        # registerMem on a shared NIXL agent is not concurrency-safe; serialize.
        self._rdt_reg_lock = threading.Lock()
        self._rdt_arena_presize = 0
        self._rdt_pack_check = False
        # Device the gathered tensors + serve arenas live on (a Ray actor with 1
        # GPU sees it as index 0; captured on the main thread for the serve
        # threadpool, which does not inherit the current device).
        self._rdt_device = torch.cuda.current_device() if torch.cuda.is_available() else None
        self._rdt_inited = True

    def _rdt_is_rank0(self) -> bool:
        return torch.distributed.is_initialized() and torch.distributed.get_rank() == 0

    def _rdt_configure(
        self,
        *,
        num_consumers: int,
        num_rdt_buffers: int,
        arena_presize_gb: float,
        pack_check: bool,
        gather_lookahead: int,
        num_producers: int = 1,
        producer_rank: int = 0,
    ) -> None:
        """Set the must-agree serve knobs before a sync. Called by the sender via
        the worker; idempotent. With a single rank-0 producer the free target is
        the full consumer count (every consumer pulls from and frees rank 0)."""
        self._rdt_state()
        self._rdt_free_target = max(1, count_consumers(num_producers, num_consumers, producer_rank))
        self._rdt_nring = max(1, num_rdt_buffers)
        self._rdt_arena_presize = int(arena_presize_gb * (1 << 30))
        self._rdt_pack_check = bool(pack_check)
        self._rdt_lookahead = max(1, gather_lookahead)

    # ---------------- sync lifecycle (called by the sender on the worker) -------

    def _rdt_begin_sync(self) -> None:
        """Reset per-sync free/backpressure state. The previous sync's finish
        drains every consumer's frees, so nothing is in flight here."""
        self._rdt_state()
        with self._rdt_free_lock:
            self._rdt_free_counts.clear()
        with self._rdt_cache_cond:
            self._rdt_gather_error = None
            self._rdt_inflight_keys.clear()
            self._rdt_name_to_key.clear()
            self._rdt_cache.clear()
            self._rdt_cache_cond.notify_all()

    def _rdt_end_sync(self) -> None:
        """Block until every gathered group has been freed by its consumers, so
        gathered storage is released before the next sync begins (only rank 0
        gathers/serves, so other ranks return immediately)."""
        if not self._rdt_is_rank0():
            return
        self._rdt_state()
        with self._rdt_cache_cond:
            while self._rdt_inflight_keys:
                if self._rdt_gather_error is not None:
                    break
                self._rdt_cache_cond.wait()

    def _rdt_set_error(self, message: str) -> None:
        """Record a sync failure so blocked serves / gathers stop waiting."""
        self._rdt_state()
        with self._rdt_cache_cond:
            self._rdt_gather_error = RuntimeError(message)
            self._rdt_cache_cond.notify_all()

    # ---------------- gather (collective; publishes into the cache) -------------

    def gather_layer(self, names: list, dtype: torch.dtype) -> None:
        """Collectively all-gather one layer-aligned group of params.

        Every FSDP rank must call this with the SAME ``names`` in the SAME ORDER
        (``full_tensor()`` is a collective). Rank 0 first blocks while
        ``gather_lookahead`` groups are already gathered-and-unfreed (the
        backpressure the consumers' ``free_gather`` releases), THEN joins the
        collective and caches each gathered tensor (cast to the inference
        ``dtype``) under its (possibly prefixed) name; other ranks discard.
        """
        self._rdt_state()
        rank0 = self._rdt_is_rank0()
        # Backpressure BEFORE the collective: rank 0 waits for a free; other ranks
        # then wait for rank 0 at the collective (natural pace propagation).
        if rank0:
            with self._rdt_cache_cond:
                while len(self._rdt_inflight_keys) >= self._rdt_lookahead:
                    if self._rdt_gather_error is not None:
                        raise self._rdt_gather_error
                    self._rdt_cache_cond.wait()
                if self._rdt_gather_error is not None:
                    raise self._rdt_gather_error

        prefix = getattr(self.weight_extractor, "weight_prefix", "") or ""
        # Use the SAME model handle the metadata came from: get_weight_metadata
        # reads ``weight_extractor.model.state_dict()`` (the inner HF model),
        # whereas the worker's ``self.model`` may be a wrapper prefixed otherwise.
        sd = self.weight_extractor.model.state_dict()
        gathered: Dict[str, torch.Tensor] = {}
        try:
            for name in names:
                raw = name[len(prefix) :] if prefix and name.startswith(prefix) else name
                param = sd[raw]
                full = self.weight_extractor._gather_tensor(param).to(dtype).detach().contiguous()
                if rank0:
                    gathered[name] = full
                else:
                    del full
        except BaseException as e:
            with self._rdt_cache_cond:
                self._rdt_gather_error = e
                self._rdt_cache_cond.notify_all()
            raise

        if not rank0:
            return
        key = tuple(names)
        with self._rdt_cache_cond:
            self._rdt_cache.update(gathered)
            self._rdt_inflight_keys.append(key)
            for n in gathered:
                self._rdt_name_to_key[n] = key
            self._rdt_cache_cond.notify_all()

    # ---------------- consumer-facing (called by name over Ray) -----------------

    def free_gather(self, names: list) -> None:
        """Consumer back-edge: one consumer finished pulling this group. Ref-count
        to ``free_target``; on the last, drop the cache entries and release one
        backpressure slot."""
        self._rdt_state()
        key = self._rdt_name_to_key.get(names[0]) if names else None
        tnames = tuple(names)
        with self._rdt_free_lock:
            count = self._rdt_free_counts.get(tnames, 0) + 1
            self._rdt_free_counts[tnames] = count
            do_free = count >= self._rdt_free_target
            if do_free:
                del self._rdt_free_counts[tnames]
        if not do_free:
            return
        with self._rdt_cache_cond:
            for name in names:
                self._rdt_cache.pop(name, None)
                self._rdt_name_to_key.pop(name, None)
            if key is not None and key in self._rdt_inflight_keys:
                self._rdt_inflight_keys.remove(key)
            self._rdt_cache_cond.notify_all()

    def reserve_serve_arena(self, consumer_id: int, nbytes: int) -> None:
        """Pre-allocate + NIXL-register this consumer's serve ring before any
        pull, while the fabric is idle (avoids registration races during the
        sync-0 RDMA churn under M:N fan-in). Idempotent; grows only if needed."""
        from ray.experimental import register_nixl_memory

        self._rdt_state()
        alloc = arena_alloc_bytes(nbytes, self._rdt_arena_presize)
        dev = self._rdt_device if self._rdt_device is not None else 0
        with self._rdt_serve_lock:
            rings = self._rdt_serve_rings.setdefault(consumer_id, [None] * self._rdt_nring)
            self._rdt_serve_idx.setdefault(consumer_id, 0)
        for i in range(self._rdt_nring):
            slot = rings[i]
            if slot is None or slot.numel() < alloc:
                t = torch.empty(alloc, dtype=torch.uint8, device=torch.device("cuda", dev))
                with self._rdt_reg_lock:
                    register_nixl_memory(t)
                rings[i] = t

    @ray.method(tensor_transport="nixl")
    def rdt_produce_weights_batched(self, specs: list, pack: bool = True, consumer_id: int = 0):
        """Serve one batched slice request over NIXL (rank 0 only).

        Waits until the specs' names are cached (the gather loop caches the owning
        group before/while the consumer pulls), replays each spec's op chain (pure
        views into cached tensors, guarded by ALLOWED_OPS), byte-packs the slices
        16B-aligned into this consumer's ring slot (mirroring the consumer's
        identical layout), and returns the one packed blob. ``pack=False`` serves
        one tensor per spec (the rare unbaked path). Ported from the vLLM
        ``_RDTProducerServer.rdt_produce_weights_batched``.
        """
        self._rdt_state()
        needed = sorted({n for n, _ in specs})
        with self._rdt_cache_cond:
            while not all(n in self._rdt_cache for n in needed):
                if self._rdt_gather_error is not None:
                    raise RuntimeError(f"gather errored before {needed}: {self._rdt_gather_error!r}")
                self._rdt_cache_cond.wait()

        sliced: List[tuple] = []  # (byte_off, tensor)
        pack_cur = 0
        for name, chain in specs:
            t = self._rdt_cache[name]
            for op, args, kw in chain:
                if op not in ALLOWED_OPS:
                    raise ValueError(f"{name!r}: disallowed op {op!r}")
                t = getattr(t, op)(*args, **dict(kw))
            off = (pack_cur + 15) & ~15
            pack_cur = off + t.numel() * t.element_size()
            sliced.append((off, t))

        if not pack:
            out = [t.contiguous().clone() for _off, t in sliced]
            torch.accelerator.synchronize()
            return out

        dev = self._rdt_device if self._rdt_device is not None else 0
        with self._rdt_serve_lock:
            rings = self._rdt_serve_rings.setdefault(consumer_id, [None] * self._rdt_nring)
            idx = self._rdt_serve_idx.get(consumer_id, 0)
            self._rdt_serve_idx[consumer_id] = (idx + 1) % self._rdt_nring
        arena = rings[idx]
        if arena is None or arena.numel() < pack_cur:
            from ray.experimental import register_nixl_memory

            alloc = arena_alloc_bytes(pack_cur, self._rdt_arena_presize)
            arena = torch.empty(alloc, dtype=torch.uint8, device=torch.device("cuda", dev))
            with self._rdt_reg_lock:
                register_nixl_memory(arena)
            rings[idx] = arena

        for off, t in sliced:
            nb = t.numel() * t.element_size()
            view = arena[off : off + nb].view(t.dtype).reshape(t.shape)
            view.copy_(t)
        # Ensure the packed blob is fully written before NIXL reads it.
        torch.accelerator.synchronize()
        return [arena[:pack_cur]]

    def rdt_shutdown(self) -> None:
        """Drop producer state + serve rings (their NIXL registration is pinned
        for the process lifetime; this just releases our strong refs)."""
        if not getattr(self, "_rdt_inited", False):
            return
        with self._rdt_cache_cond:
            self._rdt_cache.clear()
            self._rdt_inflight_keys.clear()
            self._rdt_name_to_key.clear()
            self._rdt_cache_cond.notify_all()
        with self._rdt_serve_lock:
            self._rdt_serve_rings.clear()
            self._rdt_serve_idx.clear()
