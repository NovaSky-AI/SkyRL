"""Sharded RDT (Ray Direct Transport / NIXL) weight transfer strategy.

Unlike the broadcast (push) and CUDA-IPC strategies, the ``sharded_rdt`` backend
is **pull-based**: the vLLM inference workers pull only the slice each one
consumes from a *named* trainer Ray actor over Ray's ``tensor_transport="nixl"``,
driven by a plan the vLLM ``ShardedRDTWeightTransferEngine`` bakes once at init.
The receiver is that vLLM engine (built + baked on the inference workers via the
SkyRL worker extension); there is no SkyRL-side receiver.

This module is a THIN ADAPTER over the vendored vLLM RDT trainer engine
(``sharded_rdt_trainer.ShardedRDTTrainerWeightTransferEngine``): all the
trainer-side machinery — the per-rank ``_RDTProducerServer`` sidecar actor
(gather cache, packed serve rings, free ref-counting, arena registration), the
group-by-group gather/publish loop over CUDA IPC, and the concurrent
start/update/finish handshake — lives in the vendored files, VERBATIM from the
``vllm-rdt-weight-sync`` fork, so a future vLLM upgrade is just deleting the
vendored files and repointing imports. The adapter supplies the three pieces the
vendored engine takes from its host:

  * a ``VLLMWeightSyncClient`` (``_SyncInferenceClient``) that bridges the
    engine's four synchronous control-plane calls onto SkyRL's async
    ``RemoteInferenceClient`` RDT routes;
  * a ``WeightSource`` (``_FsdpWeightSource``) over the FSDP policy model that
    yields group-contiguous ``(name, full bf16 tensor)`` pairs; and
  * the driving glue (``ShardedRdtWeightTransferSender``) that runs
    ``trainer_init`` / ``send_weights`` off the worker's event loop.

All engine-specific wiring lives here, behind the generic
``WeightTransferStrategy`` / ``WeightTransferSender`` hooks, so
``Worker.init_weight_sync_state`` stays backend-agnostic.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Optional

import ray
import torch

from skyrl.backends.skyrl_train.weight_sync.base import WeightChunk
from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_base import (
    ParamMeta,
    WeightSource,
)
from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_common import (
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
    """Config-derived init info for the sharded_rdt (NIXL pull) backend.

    Only the knobs that ``ShardedRDTTrainerInitInfo`` needs to spawn the sidecar
    producer server and that the sender forwards onto the vLLM consumer engine's
    init info. Unlike the push backends, the parameter names/shapes/dtypes and
    their gather-group partition are NOT carried here — the vendored trainer
    engine reads them from the ``WeightSource`` at ``trainer_init`` and builds
    the consumer-side init info itself.
    """

    num_consumers: int
    trainer_actor_namespace: Optional[str]
    model_dtype_str: str = "bfloat16"
    num_rdt_buffers: int = _DEFAULT_NUM_RDT_BUFFERS
    layerwise_split: int = _DEFAULT_LAYERWISE_SPLIT
    arena_presize_gb: float = _DEFAULT_ARENA_PRESIZE_GB
    pack_check: bool = _DEFAULT_PACK_CHECK
    gather_lookahead: int = _DEFAULT_GATHER_LOOKAHEAD

    @staticmethod
    def strategy_type() -> type:
        return ShardedRdtTransferStrategy


class _SyncInferenceClient:
    """Synchronous ``VLLMWeightSyncClient`` over SkyRL's async ``RemoteInferenceClient``.

    The vendored trainer engine drives the inference side through four
    synchronous calls (``init_weight_transfer_engine`` / ``start_weight_update`` /
    ``update_weights`` / ``finish_weight_update``); SkyRL's client exposes the
    RDT control plane as coroutines on the worker's event loop. Each call is
    scheduled onto that loop with ``run_coroutine_threadsafe`` and awaited — the
    engine only ever calls these from a worker thread (see
    ``ShardedRdtWeightTransferSender``), never from the loop itself, so this never
    deadlocks and the aiohttp session stays pinned to one loop.
    """

    def __init__(self, client: "RemoteInferenceClient", loop: asyncio.AbstractEventLoop) -> None:
        self._client = client
        self._loop = loop

    def _run(self, coro) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def init_weight_transfer_engine(self, init_info: Dict[str, Any]) -> None:
        # Routed to the worker extension (bake under set_current_vllm_config),
        # NOT the native NCCL init endpoint.
        self._run(self._client.init_weight_transfer_engine_rdt(init_info))

    def start_weight_update(self) -> None:
        self._run(self._client.start_weight_update(is_checkpoint_format=True))

    def update_weights(self, update_info: Dict[str, Any]) -> None:
        self._run(self._client.update_weights_rdt(update_info))

    def finish_weight_update(self) -> None:
        self._run(self._client.finish_weight_update())


class _FsdpWeightSource(WeightSource):
    """``WeightSource`` over the FSDP policy model for the sidecar trainer engine.

    Yields ``(name, full tensor)`` pairs in group-contiguous (pre / per-layer /
    post) order, cast to the inference dtype, using the worker's
    ``FSDPWeightExtractor`` so the names (incl. ``weight_prefix``) and the
    all-gather match exactly what the consumer engine baked its plan over.

    ``metadata()`` reads state_dict shapes only (no gather); iteration all-gathers
    each parameter (``full_tensor()``) and is therefore a collective every trainer
    rank must run in lockstep — which the vendored engine's ``send_weights``
    guarantees (all ranks iterate the source).
    """

    def __init__(self, weight_extractor: Any, dtype: torch.dtype) -> None:
        self._extractor = weight_extractor
        self._dtype = dtype
        meta = weight_extractor.get_weight_metadata(dtype)
        names = list(meta["names"])
        shapes = [list(s) for s in meta["shapes"]]
        # Reorder into group-major order so layerwise_groups(names) partitions the
        # list exactly and each model.layers.<N>.* block is contiguous (the order
        # the trainer engine validates + the gather loop drives).
        idx = {n: i for i, n in enumerate(names)}
        order = [idx[n] for g in layerwise_groups(names) for n in g]
        self._names = [names[i] for i in order]
        self._shapes = [shapes[i] for i in order]

    def metadata(self) -> List[ParamMeta]:
        return [ParamMeta(name, self._dtype, tuple(shape)) for name, shape in zip(self._names, self._shapes)]

    def __iter__(self) -> Iterator[tuple]:
        # A worker thread does not inherit the main thread's current CUDA device;
        # set it so the gather collectives + casts land on this rank's GPU.
        if torch.cuda.is_available():
            torch.cuda.set_device(torch.cuda.current_device())
        prefix = getattr(self._extractor, "weight_prefix", "") or ""
        # Use the SAME model handle the metadata came from: get_weight_metadata
        # reads ``weight_extractor.model.state_dict()`` (the inner HF model),
        # whereas the worker's ``self.model`` may be a wrapper prefixed otherwise.
        sd = self._extractor.model.state_dict()
        for name in self._names:
            raw = name[len(prefix) :] if prefix and name.startswith(prefix) else name
            full = self._extractor._gather_tensor(sd[raw]).to(self._dtype).detach().contiguous()
            yield name, full


class ShardedRdtWeightTransferSender(WeightTransferSender):
    """Drives the vendored sidecar trainer engine for the pull-based RDT sync.

    Runs on every training rank. Holds no transfer logic of its own — it builds
    the ``WeightSource`` + sync client and delegates to
    ``ShardedRDTTrainerWeightTransferEngine.trainer_init`` (rendezvous, spawn the
    per-rank producer server, and on rank 0 bake the inference side) and
    ``send_weights`` (the concurrent gather/publish + start/update/finish drive).

    Both engine calls block (Ray + torch.distributed collectives, and rank 0's
    synchronous inference-client calls), so they run off the worker's event loop
    via ``asyncio.to_thread``; the sync client bridges the engine's control-plane
    calls back onto the loop. ``trainer_init`` is done lazily on the first
    ``send_chunks`` so it runs in a thread (it needs rank 0 to call the client,
    which the loop must be free to service).
    """

    def __init__(
        self,
        init_info: ShardedRdtInitInfo,
        inference_client: "RemoteInferenceClient",
    ) -> None:
        self._init_info = init_info
        self._inference_client = inference_client
        self._worker: Any = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._sync_client: Optional[_SyncInferenceClient] = None
        self._engine: Any = None

    def bind_trainer_worker(self, worker: Any) -> None:
        """Capture the worker + event loop; build the sync client.

        Called on every rank from ``init_weight_sync_state`` (on the loop). The
        heavy ``trainer_init`` is deferred to the first ``send_chunks`` (which
        runs it in a thread), so this stays non-blocking.
        """
        self._worker = worker
        self._loop = asyncio.get_running_loop()
        self._sync_client = _SyncInferenceClient(self._inference_client, self._loop)

    def _trainer_init_blocking(self) -> Any:
        """Build the source + trainer init info and rendezvous. Runs in a worker
        thread (blocks on Ray spawn, an all-gather collective, and — on rank 0 —
        the inference-side bake)."""
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_trainer import (
            ShardedRDTTrainerInitInfo,
            ShardedRDTTrainerWeightTransferEngine,
        )

        if self._worker is None or self._worker.weight_extractor is None:
            raise RuntimeError(
                "ShardedRdtWeightTransferSender not bound to a worker with a "
                "weight_extractor. init_weight_sync_state must build the extractor "
                "and call sender.bind_trainer_worker(self)."
            )
        dtype = str_to_torch_dtype(self._init_info.model_dtype_str)
        source = _FsdpWeightSource(self._worker.weight_extractor, dtype)

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        trainer_init_info = ShardedRDTTrainerInitInfo(
            rank=rank,
            num_consumers=self._init_info.num_consumers,
            trainer_actor_namespace=self._init_info.trainer_actor_namespace,
            num_rdt_buffers=self._init_info.num_rdt_buffers,
            layerwise_split=self._init_info.layerwise_split,
            arena_presize_gb=self._init_info.arena_presize_gb,
            pack_check=self._init_info.pack_check,
            gather_lookahead=self._init_info.gather_lookahead,
        )
        return ShardedRDTTrainerWeightTransferEngine.trainer_init(
            trainer_init_info,
            client=self._sync_client,
            source=source,
        )

    async def send_chunks(
        self,
        chunks: Iterable[WeightChunk],
        weight_metadata: Optional[Dict[str, list]] = None,
    ) -> None:
        """Drive one weight sync via the vendored trainer engine.

        ``chunks``/``weight_metadata`` are ignored — RDT does not push extracted
        chunks; the engine gathers live params group-by-group and the inference
        workers pull their slices. Every rank must call this (the gather is a
        collective). The first call also rendezvouses (bake + spawn servers).
        """
        if self._worker is None:
            raise RuntimeError(
                "ShardedRdtWeightTransferSender not bound to a worker. "
                "init_weight_sync_state must call sender.bind_trainer_worker(self)."
            )
        if self._engine is None:
            self._engine = await asyncio.to_thread(self._trainer_init_blocking)
        await asyncio.to_thread(self._engine.send_weights)

    def teardown(self) -> None:
        engine = self._engine
        self._engine = None
        self._worker = None
        if engine is not None:
            engine.shutdown()


class ShardedRdtTransferStrategy(WeightTransferStrategy):
    """Factory for the sharded_rdt (NIXL pull) weight transfer strategy."""

    @staticmethod
    def create_init_info(
        ie_cfg: "InferenceEngineConfig", inference_world_size: Optional[int] = None
    ) -> ShardedRdtInitInfo:
        """Build the RDT init info (config-derived knobs only).

        ``inference_world_size`` is the total inference-worker (consumer) count,
        used for the producer's free ref-count target and the engine's M:N block
        assignment. The actor namespace is the current Ray namespace so the
        sidecar producer servers are spawned where the vLLM engine's
        ``ray.get_actor`` (in its own EngineCore ``ray.init``) can resolve them.
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
            num_consumers=int(inference_world_size),
            trainer_actor_namespace=namespace,
            model_dtype_str=ie_cfg.model_dtype,
            num_rdt_buffers=int(getattr(ie_cfg, "rdt_num_rdt_buffers", _DEFAULT_NUM_RDT_BUFFERS)),
            layerwise_split=int(getattr(ie_cfg, "rdt_layerwise_split", _DEFAULT_LAYERWISE_SPLIT)),
            arena_presize_gb=float(getattr(ie_cfg, "rdt_arena_presize_gb", _DEFAULT_ARENA_PRESIZE_GB)),
            pack_check=bool(getattr(ie_cfg, "rdt_pack_check", _DEFAULT_PACK_CHECK)),
            gather_lookahead=int(getattr(ie_cfg, "rdt_gather_lookahead", _DEFAULT_GATHER_LOOKAHEAD)),
        )

    @staticmethod
    def get_vllm_transfer_engine() -> type:
        """Return the vLLM (consumer) weight-transfer engine class for this strategy."""
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_engine import (
            ShardedRDTWeightTransferEngine,
        )

        return ShardedRDTWeightTransferEngine

    @staticmethod
    def initialize_receivers(init_info: WeightSyncInitInfo, inference_client: "RemoteInferenceClient"):
        """No-op: the inference-side engine is initialized + baked by the trainer
        engine's ``trainer_init`` (via ``init_weight_transfer_engine_rdt``), which
        the sender runs on the first sync. The generic init flow awaits this hook,
        so return a completed coroutine rather than the native NCCL init."""

        async def _noop() -> None:
            return None

        return _noop()

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
