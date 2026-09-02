"""Build the trainer-side weight-transfer engine for a backend.

:func:`build_trainer_engine` is the whole trainer-side weight-sync surface SkyRL
owns: pick the init info, build the control-plane client, and hand both to
``WeightTransferTrainerFactory.trainer_init``, which rendezvouses and returns an
engine whose ``send_weights()`` owns the round trip.

===============  ==================================================
``nccl``         vLLM's ``NCCLTrainerWeightTransferEngine``
``ipc``          vLLM's ``IPCTrainerWeightTransferEngine``
``delta``        ``weight_sync/delta_trainer.py``
``sharded_rdt``  ``weight_sync/sharded_rdt/sharded_rdt_trainer.py``
===============  ==================================================

The trainer- and worker-side factories keep separate registries, so the trainer
engines use vLLM's ``nccl`` / ``ipc`` keys even though the receive side registers
under ``skyrl_nccl`` / ``skyrl_ipc`` (see ``skyrl_engines.py``).
"""

from __future__ import annotations

import logging
import math
import socket
from typing import TYPE_CHECKING, Any, Callable, Optional

from vllm.distributed.weight_transfer.packed_tensor import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES,
)

from skyrl.backends.skyrl_train.weight_sync.control_plane import (
    SkyrlWeightSyncClient,
    nccl_init_payloads,
    rdt_init_payloads,
)

if TYPE_CHECKING:
    import torch

    from skyrl.train.config.config import InferenceEngineConfig

logger = logging.getLogger(__name__)


def build_trainer_engine(
    *,
    ie_cfg: "InferenceEngineConfig",
    colocate_all: bool,
    rank: int,
    inference_world_size: int,
    source_factory: Callable[["torch.dtype", str], Any],
    server_urls: list,
    data_parallel_size: int,
    base_model_path: Optional[str] = None,
) -> Any:
    """Resolve the backend, build this rank's source, rendezvous, and return the engine.

    Called on **every** trainer rank, off the event loop: rank 0 drives the
    inference-side handshake while the others return without touching the wire.

    The backend is resolved here, from the same two config values the driver uses
    to configure the inference servers (``get_transfer_strategy``), so the two
    sides cannot pick different engines.

    Args:
        ie_cfg: inference engine config. Supplies the backend and the inference
            dtype; ``delta`` also reads its ``delta_weight_sync`` block.
        colocate_all: ``trainer.placement.colocate_all``.
        rank: this trainer process's rank. Rank 0 is the sender.
        inference_world_size: total inference workers, from
            ``client.get_world_size()``.
        source_factory: ``(dtype, backend) -> WeightSource``. A callback because
            the source reads the live model, which only the caller has, and it
            cannot be built until the backend is known -- sharded RDT needs an
            ownership-aware subclass.
        server_urls: every inference server, in deployment-major order.
        data_parallel_size: DP replicas per deployment.
        base_model_path: policy model path. Required by ``delta``, which
            publishes against that checkpoint.
    """
    from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory

    from skyrl.backends.skyrl_train.weight_sync import get_transfer_strategy
    from skyrl.train.utils.utils import str_to_torch_dtype

    backend = get_transfer_strategy(ie_cfg.weight_sync_backend, colocate_all)
    source = source_factory(str_to_torch_dtype(ie_cfg.model_dtype), backend)

    init_info, init_payload_fn = _build_init_info(
        backend=backend,
        ie_cfg=ie_cfg,
        rank=rank,
        inference_world_size=inference_world_size,
        source=source,
        server_urls=server_urls,
        data_parallel_size=data_parallel_size,
        base_model_path=base_model_path,
    )
    client = SkyrlWeightSyncClient(
        server_urls,
        data_parallel_size,
        init_payload_fn=init_payload_fn,
    )
    if backend == "sharded_rdt":
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt import rdt_send

        rdt_send.log_source_choice(source)

    engine = WeightTransferTrainerFactory.trainer_init(init_info, client=client, source=source)

    if backend == "sharded_rdt":
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt import rdt_send

        rdt_send.freeze_trainer_heap()
    return engine


def _packed_buffer_size_bytes(source: Any) -> int:
    """Packed-buffer size that fits the model's largest single parameter.

    The packed producers stream through a fixed reusable buffer, and a parameter
    too large for one raises on the IPC path and over-allocates on NCCL. vLLM's
    1 GiB default is smaller than a large-vocab embedding matrix (Qwen3-235B's is
    151936 x 4096 in bf16 = 1.24 GiB), so size it from the source.

    ``metadata()`` is a collective on a Megatron source, so this runs on every
    rank -- it is called before the sender split, which keeps them in lockstep --
    and the source caches it.
    """
    meta = source.metadata()
    if not meta:
        return DEFAULT_PACKED_BUFFER_SIZE_BYTES
    largest = max(math.prod(m.shape) * m.dtype.itemsize for m in meta)
    return max(DEFAULT_PACKED_BUFFER_SIZE_BYTES, largest)


def _build_init_info(
    *,
    backend: str,
    ie_cfg: "InferenceEngineConfig",
    rank: int,
    inference_world_size: int,
    source: Any,
    server_urls: list,
    data_parallel_size: int,
    base_model_path: Optional[str],
):
    """Return ``(init_info, init_payload_fn)`` for an already-resolved backend.

    ``init_payload_fn`` expands the engine's single worker-side init dict to one
    payload per server; only NCCL and sharded RDT need it (see ``control_plane``).
    """
    _register_skyrl_trainer_engines()

    if backend == "nccl":
        import ray
        from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerInitInfo

        # Only rank 0 opens the endpoint, so only its address/port reaches a
        # worker; the other ranks build and discard theirs.
        master_address = ray._private.services.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            master_port = sock.getsockname()[1]
        return (
            NCCLTrainerInitInfo(
                master_address=master_address,
                master_port=master_port,
                # Every inference worker plus the single trainer sender (rank 0).
                world_size=inference_world_size + 1,
                # Broadcast out of a fixed reusable buffer instead of one NCCL
                # call per parameter. The engine propagates this to the worker at
                # the handshake, so the two sides cannot disagree.
                packed=True,
                packed_buffer_size_bytes=_packed_buffer_size_bytes(source),
                rank=rank,
            ),
            nccl_init_payloads,
        )

    if backend == "ipc":
        from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerInitInfo

        return (
            # packed=True overrides the vLLM default: the unpacked path holds a
            # strong ref to a contiguous copy of EVERY parameter until past
            # `finish_weight_update` (so the consumer's IPC views stay valid),
            # i.e. the whole model resident on the trainer. Packed streams
            # through one reusable buffer.
            IPCTrainerInitInfo(
                packed=True,
                packed_buffer_size_bytes=_packed_buffer_size_bytes(source),
                rank=rank,
            ),
            None,
        )

    if backend == "delta":
        from skyrl.backends.skyrl_train.weight_sync.delta_checkpoint import (
            SUPPORTED_CHECKPOINT_LOAD_FORMATS,
        )
        from skyrl.backends.skyrl_train.weight_sync.delta_trainer import (
            DeltaTrainerInitInfo,
        )

        if base_model_path is None:
            raise ValueError("Delta weight sync requires base_model_path")
        delta_cfg = ie_cfg.delta_weight_sync
        if delta_cfg is None or not delta_cfg.sync_dir:
            raise ValueError("Delta weight sync requires generator.inference_engine.delta_weight_sync.sync_dir")
        if delta_cfg.checkpoint_load_format not in SUPPORTED_CHECKPOINT_LOAD_FORMATS:
            raise ValueError(
                "Delta checkpoint_load_format must be one of "
                f"{sorted(SUPPORTED_CHECKPOINT_LOAD_FORMATS)}, got {delta_cfg.checkpoint_load_format!r}"
            )
        # local_checkpoint_dir and publish_staging_dir are resolved by
        # DeltaWeightSyncConfig.__post_init__, so they are already concrete here.
        return (
            DeltaTrainerInitInfo(
                base_model_path=base_model_path,
                sync_dir=delta_cfg.sync_dir,
                local_checkpoint_dir=delta_cfg.local_checkpoint_dir,
                publish_staging_dir=delta_cfg.publish_staging_dir,
                max_file_size_in_gb=delta_cfg.max_file_size_in_gb,
                cloud_download_workers=delta_cfg.cloud_download_workers,
                publish_num_workers=delta_cfg.publish_num_workers,
                checkpoint_load_format=delta_cfg.checkpoint_load_format,
                multi_thread_safetensors_max_workers=delta_cfg.multi_thread_safetensors_max_workers,
                rank=rank,
            ),
            None,
        )

    if backend == "sharded_rdt":
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt import rdt_send

        return (
            rdt_send.build_rdt_trainer_init_info(
                rank=rank,
                inference_world_size=inference_world_size,
                server_urls=list(server_urls),
                data_parallel_size=data_parallel_size,
            ),
            rdt_init_payloads,
        )

    raise ValueError(f"Unknown weight sync backend {backend!r}.")


_TRAINER_ENGINES_REGISTERED = False


def _register_skyrl_trainer_engines() -> None:
    """Register SkyRL's trainer engines (``delta``, ``sharded_rdt``) once."""
    global _TRAINER_ENGINES_REGISTERED
    if _TRAINER_ENGINES_REGISTERED:
        return
    from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory

    if "delta" not in WeightTransferTrainerFactory._registry:
        WeightTransferTrainerFactory.register_engine(
            "delta",
            "skyrl.backends.skyrl_train.weight_sync.delta_trainer",
            "DeltaTrainerWeightTransferEngine",
        )
    if "sharded_rdt" not in WeightTransferTrainerFactory._registry:
        WeightTransferTrainerFactory.register_engine(
            "sharded_rdt",
            "skyrl.backends.skyrl_train.weight_sync.sharded_rdt.sharded_rdt_trainer",
            "ShardedRDTTrainerWeightTransferEngine",
        )
    _TRAINER_ENGINES_REGISTERED = True


def engine_capability(engine: Any, name: str, default: Any) -> Any:
    """Read a SkyRL capability flag off a trainer engine.

    A ``getattr`` probe rather than a declared attribute: two of the four engines
    are vLLM's own classes and cannot carry SkyRL attributes, so an engine that
    declares nothing must get the default.

    Flags: ``skyrl_handles_prefix_cache_reset``,
    ``skyrl_force_disable_expandable_segments``,
    ``skyrl_empty_cache_after_send``.
    """
    return getattr(engine, f"skyrl_{name}", default)


def maybe_set_reset_prefix_cache(engine: Any, reset: bool) -> None:
    """Tell an engine whether to reset the prefix cache this round, if it cares.

    ``send_weights()`` takes no arguments, so a per-round flag has to ride the
    engine. Only delta implements the setter.
    """
    setter = getattr(engine, "skyrl_set_reset_prefix_cache", None)
    if setter is not None:
        setter(reset)


def teardown_engine(engine: Any) -> None:
    """Shut an engine down and close its control-plane client."""
    if engine is None:
        return
    try:
        engine.shutdown()
    finally:
        client = getattr(engine, "client", None)
        close = getattr(client, "close", None)
        if close is not None:
            close()
