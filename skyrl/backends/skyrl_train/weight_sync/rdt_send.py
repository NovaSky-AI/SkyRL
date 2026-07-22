"""Trainer-side driver for the sharded-RDT (NIXL pull) weight-sync backend.

RDT is deliberately kept OUT of SkyRL's ``WeightTransferStrategy`` /
``WeightTransferSender`` abstraction. That abstraction encodes trainer-side send
logic for the legacy *push* backends (NCCL broadcast, CUDA IPC), which extract
weights on the worker and hand chunks to a sender. RDT instead already matches
vLLM's *new* trainer-send model — a ``WeightSource`` + a
``TrainerWeightTransferEngine`` + a ``VLLMWeightSyncClient`` — where the engine
owns the whole round trip and the inference workers pull. Forcing that through
``send_chunks(chunks, metadata)`` meant ignoring both args and bolting
RDT-only hooks onto the shared base classes.

So we bypass the abstraction: ``Worker.init_weight_sync_state`` builds a
``RdtWeightSyncSender`` for the RDT backend, and ``broadcast_to_inference_engines``
calls ``.send()`` directly (skipping extraction + ``send_chunks`` entirely). This
is an intermediate state: when vLLM ships trainer-send for NCCL/IPC too
(see ``vllm-trainer-send-pr3``), those backends collapse into this same shape and
the ``WeightTransferStrategy`` layer is deleted — at which point this file's
``_SyncInferenceClient`` + ``_FsdpWeightSource`` become the canonical glue and the
vendored ``sharded_rdt_*`` files are dropped in favor of ``vllm.distributed.weight_transfer``.

FSDP only: ``_FsdpWeightSource`` reads the FSDP ``FSDPWeightExtractor``. Megatron
RDT support needs its own ``WeightSource`` (a follow-up).
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional

import torch

from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_base import (
    ParamMeta,
    WeightSource,
)
from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_common import (
    layerwise_groups,
)
from skyrl.train.utils.utils import str_to_torch_dtype

if TYPE_CHECKING:
    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
        RemoteInferenceClient,
    )
    from skyrl.train.config import InferenceEngineConfig

logger = logging.getLogger(__name__)

# Defaults for the must-agree wire knobs (mirror the vLLM sharded_rdt defaults).
_DEFAULT_NUM_RDT_BUFFERS = 2
_DEFAULT_LAYERWISE_SPLIT = 1
_DEFAULT_ARENA_PRESIZE_GB = 0.0
_DEFAULT_PACK_CHECK = False
# Max gathered-but-not-yet-freed groups the producer holds at once (backpressure).
_DEFAULT_GATHER_LOOKAHEAD = 2


class _SyncInferenceClient:
    """Synchronous ``VLLMWeightSyncClient`` over SkyRL's async ``RemoteInferenceClient``.

    The vendored trainer engine drives the inference side through four
    synchronous calls (``init_weight_transfer_engine`` / ``start_weight_update`` /
    ``update_weights`` / ``finish_weight_update``); SkyRL's client exposes the
    RDT control plane as coroutines on the worker's event loop. Each call is
    scheduled onto that loop with ``run_coroutine_threadsafe`` and awaited — the
    engine only ever calls these from a worker thread (see ``RdtWeightSyncSender``),
    never from the loop itself, so this never deadlocks and the aiohttp session
    stays pinned to one loop.
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


class RdtWeightSyncSender:
    """Drives sharded-RDT weight sync through the vendored vLLM trainer-send
    engine, bypassing SkyRL's ``WeightTransferStrategy`` / ``WeightTransferSender``.

    Built once per worker in ``init_weight_sync_state``; ``send()`` is called from
    ``broadcast_to_inference_engines`` each RL step. The heavy ``trainer_init``
    (spawn the per-rank ``_RDTProducerServer`` sidecar + rank-0 bake) and every
    ``send_weights`` block on Ray + ``torch.distributed`` collectives, so both run
    off the worker's event loop via ``asyncio.to_thread``; ``_SyncInferenceClient``
    bridges the engine's control-plane calls back onto the loop.
    """

    def __init__(
        self,
        inference_client: "RemoteInferenceClient",
        ie_cfg: "InferenceEngineConfig",
        inference_world_size: int,
        trainer_actor_namespace: Optional[str],
    ) -> None:
        if not inference_world_size:
            raise ValueError(
                f"sharded_rdt requires the inference world size (consumer count); got {inference_world_size!r}."
            )
        self._client = inference_client
        self._ie_cfg = ie_cfg
        self._world_size = int(inference_world_size)
        self._namespace = trainer_actor_namespace
        self._engine: Any = None

    async def send(self, weight_extractor: Any) -> None:
        """Sync weights once. The first call rendezvouses (spawn sidecar servers +
        rank-0 bake); every rank must call it (the gather is a collective)."""
        if weight_extractor is None:
            raise RuntimeError(
                "sharded_rdt weight sync requires the worker's weight_extractor " "(built in init_weight_sync_state)."
            )
        if self._engine is None:
            loop = asyncio.get_running_loop()
            sync_client = _SyncInferenceClient(self._client, loop)
            self._engine = await asyncio.to_thread(self._trainer_init_blocking, weight_extractor, sync_client)
        await asyncio.to_thread(self._engine.send_weights)

    def _trainer_init_blocking(self, weight_extractor: Any, sync_client: _SyncInferenceClient) -> Any:
        """Build the WeightSource + trainer init info and rendezvous. Runs in a
        worker thread (blocks on the Ray spawn, an all-gather collective, and —
        on rank 0 — the inference-side bake)."""
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_trainer import (
            ShardedRDTTrainerInitInfo,
            ShardedRDTTrainerWeightTransferEngine,
        )

        dtype = str_to_torch_dtype(self._ie_cfg.model_dtype)
        source = _FsdpWeightSource(weight_extractor, dtype)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        init_info = ShardedRDTTrainerInitInfo(
            rank=rank,
            num_consumers=self._world_size,
            trainer_actor_namespace=self._namespace,
            num_rdt_buffers=int(getattr(self._ie_cfg, "rdt_num_rdt_buffers", _DEFAULT_NUM_RDT_BUFFERS)),
            layerwise_split=int(getattr(self._ie_cfg, "rdt_layerwise_split", _DEFAULT_LAYERWISE_SPLIT)),
            arena_presize_gb=float(getattr(self._ie_cfg, "rdt_arena_presize_gb", _DEFAULT_ARENA_PRESIZE_GB)),
            pack_check=bool(getattr(self._ie_cfg, "rdt_pack_check", _DEFAULT_PACK_CHECK)),
            gather_lookahead=int(getattr(self._ie_cfg, "rdt_gather_lookahead", _DEFAULT_GATHER_LOOKAHEAD)),
        )
        return ShardedRDTTrainerWeightTransferEngine.trainer_init(
            init_info,
            client=sync_client,
            source=source,
        )

    def teardown(self) -> None:
        engine = self._engine
        self._engine = None
        if engine is not None:
            engine.shutdown()
