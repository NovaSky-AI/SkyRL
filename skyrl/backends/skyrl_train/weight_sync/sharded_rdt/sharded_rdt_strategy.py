"""``sharded_rdt`` as a :class:`WeightTransferStrategy`.

A thin adapter over the vendored vLLM trainer-send stack — a ``WeightSource`` feeding
a ``ShardedRDTTrainerWeightTransferEngine`` that the inference workers pull from,
driven by :class:`RdtWeightSyncSender` (``rdt_send.py``). Presenting it through the
shared interface keeps the workers to one sender attribute and one send call.

Two places RDT does not fit the push backends' shape, declared as capabilities so the
workers hold no backend conditional:

* ``trainer_init`` opens the inference side itself, through the blocking
  ``SyncRdtControlPlaneClient``, since the bake needs the source metadata and must run
  under ``set_current_vllm_config``. ``sender_initializes_receivers`` is True, and
  worker rank 0 must not also call ``init_weight_update_communicator``.
* The consumers pull, so there is no chunk stream: this sender overrides
  :meth:`WeightTransferSender.send` and never materializes chunks or metadata.
  ``get_weight_metadata`` on the Megatron extractor is a whole-model
  ``export_hf_weights`` pass, the gather the RDT weight source avoids.

REMOVAL: when SkyRL's pinned vLLM ships trainer-send for NCCL/IPC too, those backends
collapse into this shape and the ``WeightTransferStrategy`` layer goes away. This
adapter is what disappears then; ``rdt_send.py`` stays.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterable, Optional

from skyrl.backends.skyrl_train.weight_sync.base import WeightChunk
from skyrl.backends.skyrl_train.weight_sync.transfer_strategy import (
    WeightSyncInitInfo,
    WeightTransferSender,
    WeightTransferStrategy,
)

if TYPE_CHECKING:
    import torch

    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
        RemoteInferenceClient,
    )
    from skyrl.train.config import InferenceEngineConfig


@dataclass
class ShardedRdtInitInfo(WeightSyncInitInfo):
    """Config-derived args for the sharded-RDT sender.

    Deliberately does NOT implement ``for_servers`` / ``to_api_payload``: those
    exist for the driver-side ``init_weight_update_communicator`` fan-out, which
    this backend does not use (see ``sender_initializes_receivers``). The
    inference-side payload is built by the trainer engine instead, from the
    weight source's metadata, and shipped over ``/collective_rpc``.
    """

    model_dtype: str
    """Inference dtype, as a ``torch.dtype`` string. The weight source casts to it."""

    inference_world_size: int
    """Total inference workers across the fleet — the consumer count the
    producers' free barrier counts against."""

    trainer_actor_namespace: Optional[str]
    """Ray namespace the per-rank producer sidecars are registered in, so the
    consumers can resolve them by name."""


class ShardedRdtWeightTransferSender(WeightTransferSender):
    """Presents :class:`RdtWeightSyncSender` as a ``WeightTransferSender``."""

    # The producer sidecar shares every gathered group with this rank over CUDA
    # IPC on every run, not only under colocation, and expandable-segment (VMM)
    # memory makes that export/rebuild 5-10x slower per storage.
    force_disable_expandable_segments = True

    # Publish buffers freed during a sync stay in this process's allocator cache and
    # are reused by the next training step; returning them to CUDA costs 0.25-0.53s
    # per rank at 235B and buys nothing. The worker still empties under colocate_all,
    # where an inference engine wants the physical memory.
    empty_cache_after_send = False

    def __init__(self, sender: Any) -> None:
        self._sender = sender

    async def send(
        self,
        weight_extractor: Any,
        dtype: "torch.dtype",
        **kwargs,
    ) -> None:
        """Run one pull-based sync. Every rank must call it: the gather is a
        collective.

        ``dtype`` is ignored — the weight source was built with the inference
        dtype at init and does the cast itself. The push backends' kwargs
        (``reset_prefix_cache``) are ignored too: this sender does not handle the
        prefix cache, so the worker resets it (``handles_prefix_cache_reset``
        stays False).
        """
        del dtype, kwargs
        await self._sender.send(weight_extractor)

    async def send_chunks(
        self,
        chunks: Iterable[WeightChunk],
        weight_metadata: Optional[Dict[str, list]] = None,
        **kwargs,
    ) -> None:
        """Not applicable: the consumers pull, so there is no chunk stream to
        push. :meth:`send` is this backend's entry point."""
        raise NotImplementedError(
            "sharded_rdt does not push a chunk stream: the inference workers pull "
            "the slices they consume. Call send(weight_extractor, dtype) instead."
        )

    def teardown(self) -> None:
        self._sender.teardown()


class ShardedRdtTransferStrategy(WeightTransferStrategy):
    """Factory for the sharded-RDT (NIXL pull) sender."""

    # trainer_init drives the inference-side init through the engine's own
    # control-plane client; the worker must not also push init_info.
    sender_initializes_receivers = True

    @staticmethod
    def create_init_info(
        ie_cfg: "InferenceEngineConfig",
        inference_world_size: Optional[int] = None,
        base_model_path: Optional[str] = None,
    ) -> ShardedRdtInitInfo:
        """Collect the config-derived args. Runs on every training rank.

        Raises:
            ValueError: the inference world size is missing. It is the consumer
                count the whole ownership arithmetic is sized from, and a wrong
                value silently mis-maps consumers onto slices, so there is no
                safe default.
        """
        del base_model_path  # weights come off the live model, never a checkpoint
        if not inference_world_size:
            raise ValueError(
                f"sharded_rdt requires the inference world size (consumer count); got {inference_world_size!r}."
            )
        return ShardedRdtInitInfo(
            override_existing_receiver=not ie_cfg.run_engines_locally,
            model_dtype=ie_cfg.model_dtype,
            inference_world_size=int(inference_world_size),
            trainer_actor_namespace=_ray_namespace(),
        )

    @staticmethod
    def create_sender(
        init_info: WeightSyncInitInfo,
        inference_client: "RemoteInferenceClient",
        weight_extractor: Any = None,
    ) -> ShardedRdtWeightTransferSender:
        """Build the sender AND rendezvous, on every rank.

        The rendezvous (spawn each rank's producer sidecar, then the sender rank's
        bake) happens here rather than on the first send because deferring it
        deadlocks: rank 0 would block in the inference-side init RPC while the
        other ranks spin in gather collectives, and the sidecars — which share
        those ranks' GPUs — could then never finish NIXL agent creation, because
        libfabric's CUDA probe blocks behind the spinning kernels. Every rank is
        inside ``init_weight_sync_state`` here, so that window is empty.

        Blocking is expected: the worker calls this off the event loop, and
        ``init_weight_sync_state`` already ends in a barrier.

        Raises:
            ValueError: ``init_info`` is not a :class:`ShardedRdtInitInfo`.
            RuntimeError: no ``weight_extractor``. The rendezvous needs the model
                to build its weight source, so this is a wiring error, not a
                config one.
        """
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt import rdt_vllm_register
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.rdt_send import (
            RdtWeightSyncSender,
        )

        if not isinstance(init_info, ShardedRdtInitInfo):
            raise ValueError(f"sharded_rdt requires a ShardedRdtInitInfo, got {type(init_info).__name__}.")
        if weight_extractor is None:
            raise RuntimeError(
                "sharded_rdt weight sync requires the worker's weight_extractor, which "
                "must be built before init_weight_sync_state runs."
            )

        # Registers the consumer engine in vLLM's factory under "sharded_rdt".
        # Needed on the driver and every vLLM worker; harmless (idempotent) here.
        rdt_vllm_register.ensure_registered()

        sender = RdtWeightSyncSender(
            inference_client,
            init_info.model_dtype,
            init_info.inference_world_size,
            init_info.trainer_actor_namespace,
        )
        sender.initialize(weight_extractor)
        return ShardedRdtWeightTransferSender(sender)

    @staticmethod
    def get_vllm_transfer_engine() -> type:
        """The receive-side engine, as registered in vLLM's factory.

        Unlike the push strategies' mapping this one is live: the consumers really
        do construct this class, via ``WeightTransferEngineFactory`` under the
        name ``rdt_vllm_register`` registers it as.
        """
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.sharded_rdt_engine import (
            ShardedRDTWeightTransferEngine,
        )

        return ShardedRDTWeightTransferEngine


def _ray_namespace() -> Optional[str]:
    """This process's Ray namespace, or None outside a Ray runtime.

    The producer sidecars are named actors, so the consumers need the namespace
    they were created in to resolve them.
    """
    try:
        import ray

        return ray.get_runtime_context().namespace or None
    except Exception:  # noqa: BLE001 - no Ray runtime: the caller falls back to the default namespace
        return None
