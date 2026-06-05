"""Sharded RDT (Ray Direct Transport / NIXL) weight transfer strategy.

Unlike the broadcast (push) and CUDA-IPC strategies, the ``sharded_rdt`` backend
is **pull-based**: the vLLM inference workers pull only the slice each one
consumes from a *named* trainer Ray actor over Ray's ``tensor_transport="nixl"``,
driven layer-by-layer, replaying a plan baked once at init time. The receiver is
the vLLM ``ShardedRDTWeightTransferEngine`` (built by vLLM's factory in the
GPUWorker); there is no SkyRL-side receiver.

The sender therefore does NOT ship bytes. On the training side, rank 0 (a named,
tensor-transport-enabled Ray actor) drives the per-layer-group loop:

    start_weight_update
      for each layer-aligned group of param names:
        gather_layer(group)            # collective full_tensor() on all ranks;
                                       # rank 0 caches the gathered tensors
        update_weights_rdt(group)      # (rank 0) inference workers pull their
                                       # slices from rank 0's cache over NIXL
        free_group(group)              # rank 0 drops the cached tensors
    finish_weight_update

This mirrors the reference driver in the vllm-rdt-weight-sync fork
(examples/rl/rlhf_sharded_rdt_fsdp_ep.py), adapted to run inside SkyRL's
training workers (every rank runs ``send_chunks``; only rank 0 talks to the
inference client and serves NIXL pulls).

All engine-specific wiring lives here, behind the generic ``WeightTransferStrategy``
/ ``WeightTransferSender`` hooks (``populate_init_info`` / ``initialize_receivers``
/ ``bind_trainer_worker``), so ``Worker.init_weight_sync_state`` stays backend-agnostic.
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

import ray
import torch

from skyrl.backends.skyrl_train.weight_sync.base import WeightChunk
from skyrl.backends.skyrl_train.weight_sync.transfer_strategy import (
    WeightSyncInitInfo,
    WeightTransferReceiver,
    WeightTransferSender,
    WeightTransferStrategy,
)
from skyrl.train.utils.utils import str_to_torch_dtype

if TYPE_CHECKING:
    from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import (
        InferenceEngineClient,
    )
    from skyrl.train.config.config import InferenceEngineConfig

logger = logging.getLogger(__name__)

# Name the trainer rank-0 actor is registered under and that the vLLM engine
# resolves via ray.get_actor. Must match the name used when the master training
# actor is created (see PPORayActorGroup in workers/worker.py).
RDT_TRAINER_ACTOR_NAME = "skyrl_rdt_trainer"
RDT_PRODUCE_METHOD_NAME = "rdt_produce_weights_batched"
RDT_WARMUP_METHOD_NAME = "rdt_warmup"


def layerwise_groups(names: List[str]) -> List[List[str]]:
    """Partition a flat parameter-name list into layer-aligned groups.

    Ported verbatim from the reference driver. Names with prefix
    ``model.layers.<N>.`` are grouped by ``<N>``; everything before the first
    such name is a single "pre" group (embeddings etc.), everything after is a
    single "post" group (final norm, lm_head). Each group is the unit of
    gather/cache backpressure.
    """
    pre: List[str] = []
    layers: Dict[int, List[str]] = {}
    post: List[str] = []
    seen_layer = False
    for n in names:
        if n.startswith("model.layers."):
            seen_layer = True
            idx = int(n[len("model.layers.") :].split(".", 1)[0])
            layers.setdefault(idx, []).append(n)
        elif not seen_layer:
            pre.append(n)
        else:
            post.append(n)

    groups: List[List[str]] = []
    if pre:
        groups.append(pre)
    for i in sorted(layers):
        groups.append(layers[i])
    if post:
        groups.append(post)
    return groups


@dataclass
class ShardedRdtInitInfo(WeightSyncInitInfo):
    """Initialization info for the sharded_rdt (NIXL pull) backend.

    Carries everything the vLLM ``ShardedRDTWeightTransferEngine`` needs to bake
    its replay plan and resolve the trainer actor. ``names``/``dtype_names``/
    ``shapes`` are filled in by ``populate_init_info`` from the trainer's weight
    extractor before init is sent.
    """

    trainer_actor_name: str
    trainer_actor_namespace: Optional[str]
    model_dtype_str: str = "bfloat16"
    produce_method_name: str = RDT_PRODUCE_METHOD_NAME
    warmup_method_name: Optional[str] = RDT_WARMUP_METHOD_NAME
    names: List[str] = field(default_factory=list)
    dtype_names: List[str] = field(default_factory=list)
    shapes: List[List[int]] = field(default_factory=list)

    @staticmethod
    def strategy_type() -> type:
        return ShardedRdtTransferStrategy

    def to_api_payload(self) -> Dict[str, Any]:
        """Return the dict for the vLLM ``ShardedRDTWeightTransferInitInfo``.

        Sent (identically to every server) via the client's
        ``init_weight_transfer_engine_rdt`` collective_rpc. Note this drops the
        SkyRL-only fields (``override_existing_receiver``, ``model_dtype_str``);
        the vLLM engine's init info has neither (it keys dtype per-param via
        ``dtype_names``).
        """
        return {
            "trainer_actor_name": self.trainer_actor_name,
            "trainer_actor_namespace": self.trainer_actor_namespace,
            "produce_method_name": self.produce_method_name,
            "names": self.names,
            "dtype_names": self.dtype_names,
            "shapes": self.shapes,
            "warmup_method_name": self.warmup_method_name,
        }


class ShardedRdtWeightTransferSender(WeightTransferSender):
    """Drives the pull-based per-layer-group weight sync.

    Runs on every training rank. Only rank 0 talks to the inference client and
    serves NIXL pulls; all ranks participate in the ``gather_layer`` collectives.
    The owning training worker is injected via ``bind_trainer_worker`` (it
    exposes ``gather_layer``/``free_group`` and the NIXL producer method).
    """

    def __init__(
        self,
        init_info: ShardedRdtInitInfo,
        inference_client: "InferenceEngineClient",
    ) -> None:
        self._init_info = init_info
        self._inference_client = inference_client
        # Set by bind_trainer_worker (called from init_weight_sync_state).
        self._worker: Any = None

    def bind_trainer_worker(self, worker: Any) -> None:
        """Keep a reference to the trainer worker that backs gather/free + pulls."""
        self._worker = worker

    async def send_chunks(
        self,
        chunks: Iterable[WeightChunk],
        weight_metadata: Optional[Dict[str, list]] = None,
    ) -> None:
        """Drive the layer-by-layer gather + pull loop.

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

        groups = layerwise_groups(list(weight_metadata["names"]))
        dtype = str_to_torch_dtype(self._init_info.model_dtype_str)
        is_rank0 = torch.distributed.is_initialized() and torch.distributed.get_rank() == 0

        if is_rank0:
            await self._inference_client.start_weight_update(is_checkpoint_format=True)

        # Per-group loop. The gather (full_tensor) is collective, so every rank
        # must reach each group's gather together — that collective IS the
        # backpressure between groups. Only rank 0 caches the gathered tensors,
        # serves the NIXL pull (on the actor threadpool while this coroutine
        # awaits the HTTP update), and frees the group after the pull drains.
        for group_names in groups:
            self._worker.gather_layer(group_names, dtype)
            if is_rank0:
                await self._inference_client.update_weights_rdt({"names": group_names})
                self._worker.free_group(group_names)

        if is_rank0:
            await self._inference_client.finish_weight_update()

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def teardown(self) -> None:
        self._worker = None


class ShardedRdtTransferStrategy(WeightTransferStrategy):
    """Factory for the sharded_rdt (NIXL pull) weight transfer strategy."""

    @staticmethod
    def create_init_info(
        ie_cfg: "InferenceEngineConfig", inference_world_size: Optional[int] = None
    ) -> ShardedRdtInitInfo:
        """Build the RDT init info.

        ``inference_world_size`` is unused (workers pull from a named actor; no
        NCCL world). ``names``/``dtype_names``/``shapes`` are filled in later by
        ``populate_init_info`` from the trainer's weight extractor. The actor
        namespace is the current Ray namespace so the vLLM engine's
        ``ray.get_actor`` resolves the same actor the training side names.
        """
        try:
            namespace = ray.get_runtime_context().namespace or None
        except Exception:  # noqa: BLE001
            namespace = None
        return ShardedRdtInitInfo(
            override_existing_receiver=ie_cfg.override_existing_update_group == "enable",
            trainer_actor_name=RDT_TRAINER_ACTOR_NAME,
            trainer_actor_namespace=namespace,
            model_dtype_str=ie_cfg.model_dtype,
        )

    @staticmethod
    def populate_init_info(init_info: WeightSyncInitInfo, *, weight_extractor: Any) -> None:
        """Fill ``names``/``dtype_names``/``shapes`` from the trainer's live weights.

        The vLLM engine bakes its replay plan over the full parameter-name list
        at init time, so it must be conveyed up front (before
        ``initialize_receivers``). The metadata is read without materializing
        any tensors (state_dict shapes only).
        """
        if weight_extractor is None:
            raise RuntimeError("sharded_rdt requires a weight extractor on the trainer worker.")
        assert isinstance(init_info, ShardedRdtInitInfo)
        dtype = str_to_torch_dtype(init_info.model_dtype_str)
        metadata = weight_extractor.get_weight_metadata(dtype)
        init_info.names = metadata["names"]
        init_info.dtype_names = metadata["dtype_names"]
        init_info.shapes = metadata["shapes"]

    @staticmethod
    def initialize_receivers(init_info: WeightSyncInitInfo, inference_client: "InferenceEngineClient"):
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
        inference_client: "InferenceEngineClient",
    ) -> ShardedRdtWeightTransferSender:
        """Create the sender on every training rank (no process group needed)."""
        return ShardedRdtWeightTransferSender(
            init_info=init_info,
            inference_client=inference_client,
        )

    @staticmethod
    def create_receiver(init_info: ShardedRdtInitInfo) -> WeightTransferReceiver:
        """RDT has no SkyRL-side receiver — the receiver is the vLLM engine.

        The new inference path drives the vLLM ``ShardedRDTWeightTransferEngine``
        directly via the worker extension; SkyRL's ``WeightTransferReceiver``
        (legacy path) is never constructed for this backend.
        """
        raise NotImplementedError(
            "sharded_rdt has no SkyRL-side receiver; the receiver is vLLM's "
            "ShardedRDTWeightTransferEngine, driven via NewInferenceWorkerWrap."
        )


# Mirror of the worker-side LazyRDTTensor allowlist (the ops the producer is
# willing to replay on a live parameter before slicing/cloning).
_RDT_ALLOWED_OPS = frozenset(
    {
        "narrow",
        "view",
        "reshape",
        "__getitem__",
        "unsqueeze",
        "squeeze",
        "transpose",
        "t",
        "permute",
        "flatten",
        "contiguous",
        "chunk",
    }
)


class RdtProducerMixin:
    """Trainer-side NIXL producer for the ``sharded_rdt`` weight-sync backend.

    This is the trainer half of the sharded_rdt strategy. It lives in the
    weight-sync layer but is *mixed into* the training worker actor, because its
    methods must be actor methods: the vLLM inference workers resolve the rank-0
    actor by name and call ``rdt_produce_weights_batched`` /
    ``rdt_warmup`` over Ray's ``tensor_transport="nixl"``, and ``gather_layer``
    runs the FSDP ``full_tensor()`` collective over the worker's live params.
    Mirrors ``FSDPTrainWorker`` in the vllm-rdt-weight-sync reference example
    (examples/rl/rlhf_sharded_rdt_fsdp_ep.py).

    Host-actor requirements (the worker that mixes this in must provide):
      * ``self.model`` — the (FSDP-wrapped) model; ``state_dict()`` yields the
        per-rank shards (DTensors) keyed by the same names ``get_weight_metadata``
        returns, modulo the extractor's ``weight_prefix``.
      * ``self.weight_extractor`` — exposes ``weight_prefix`` and a
        ``_gather_tensor(param)`` that all-gathers a (possibly sharded) param to
        its full tensor (FSDPWeightExtractor satisfies this).
      * a ``torch.distributed`` process group (the FSDP group).

    Only the rank-0 actor — created named + with ``enable_tensor_transport=True``
    — actually caches gathered tensors and serves pulls; every rank participates
    in the ``gather_layer`` collective. State is lazily initialized so the mixin
    composes with the worker's own ``__init__`` (it has none of its own).
    """

    def _rdt_state(self):
        # Lazy init (the mixin has no __init__ of its own).
        if not hasattr(self, "_rdt_cache"):
            self._rdt_cache: dict[str, torch.Tensor] = {}
            self._rdt_cache_cond = threading.Condition()
            self._rdt_gather_error: Optional[BaseException] = None
        return self._rdt_cache

    def _rdt_is_rank0(self) -> bool:
        return torch.distributed.is_initialized() and torch.distributed.get_rank() == 0

    def gather_layer(self, names: list, dtype: torch.dtype) -> None:
        """Collectively all-gather one layer-aligned group of params.

        Every FSDP rank must call this with the SAME ``names`` in the SAME ORDER
        (``full_tensor()`` is a collective). Rank 0 caches each gathered tensor
        (cast to the inference ``dtype``) under its (possibly prefixed) name;
        other ranks discard. Names are the prefixed metadata names; the
        state_dict lookup strips the weight extractor's prefix. ``dtype`` is
        supplied by the sender (from the init info), so the producer keeps no
        dtype state of its own.
        """
        self._rdt_state()
        prefix = getattr(self.weight_extractor, "weight_prefix", "") or ""
        sd = self.model.state_dict()
        try:
            for name in names:
                raw = name[len(prefix) :] if prefix and name.startswith(prefix) else name
                param = sd[raw]
                full = self.weight_extractor._gather_tensor(param).to(dtype).detach().contiguous()
                if self._rdt_is_rank0():
                    with self._rdt_cache_cond:
                        self._rdt_cache[name] = full
                        self._rdt_cache_cond.notify_all()
                else:
                    del full
        except BaseException as e:
            with self._rdt_cache_cond:
                self._rdt_gather_error = e
                self._rdt_cache_cond.notify_all()
            raise

    def free_group(self, names: list) -> None:
        """Drop a layer group's gathered tensors from the rank-0 cache.

        Called only after the group's ``update_weights_rdt`` has drained (every
        inference worker has pulled its slices), so the tensors are safe to free.
        """
        if not self._rdt_is_rank0():
            return
        self._rdt_state()
        with self._rdt_cache_cond:
            for name in names:
                self._rdt_cache.pop(name, None)

    @ray.method(tensor_transport="nixl")
    def rdt_warmup(self):
        """Return a 1-element tensor over NIXL to prime the connection."""
        return torch.zeros(1, device=torch.cuda.current_device())

    @ray.method(tensor_transport="nixl")
    def rdt_produce_weights_batched(self, specs):
        """Serve a batched slice request from a vLLM worker (rank 0 only).

        Waits until every requested name is in the cache (the sender gathers the
        owning group before firing the ``update_weights_rdt`` that triggers this
        pull), replays each op chain on the cached full tensor, clones the slice
        to a contiguous buffer for NIXL, and returns the list. Ported verbatim
        from the reference example's ``rdt_produce_weights_batched``.
        """
        assert self._rdt_is_rank0()
        self._rdt_state()
        needed = sorted({name for name, _ in specs})
        with self._rdt_cache_cond:
            while not all(n in self._rdt_cache for n in needed):
                if self._rdt_gather_error is not None:
                    raise RuntimeError(f"gather loop errored before producing {needed}: {self._rdt_gather_error!r}")
                self._rdt_cache_cond.wait()

        out: list = []
        for name, chain in specs:
            tensor = self._rdt_cache[name]
            for op_name, args, kwargs_items in chain:
                if op_name not in _RDT_ALLOWED_OPS:
                    raise ValueError(
                        f"Spec for {name!r} requested disallowed op {op_name!r}; "
                        f"allowed: {sorted(_RDT_ALLOWED_OPS)}"
                    )
                kwargs = dict(kwargs_items)
                tensor = getattr(tensor, op_name)(*args, **kwargs)
            out.append(tensor.clone(memory_format=torch.contiguous_format))
        torch.accelerator.synchronize()
        return out
