"""Build the trainer-side weight-transfer engine for a backend.

:func:`build_trainer_engine` is the whole trainer-side weight-sync surface SkyRL
owns: pick the init info, build the control-plane client, and hand both to
``WeightTransferTrainerFactory.trainer_init``, which rendezvouses and returns an
engine whose ``send_weights()`` owns the round trip.

===============  ==================================================
``nccl``         vLLM's ``NCCLTrainerWeightTransferEngine``
``ipc``          vLLM's ``IPCTrainerWeightTransferEngine``
``delta``        ``weight_sync/delta/trainer.py``
``sharded_rdt``  ``weight_sync/sharded_rdt/sharded_rdt_trainer.py``
===============  ==================================================

The trainer- and worker-side factories keep separate registries, so the trainer
engines use vLLM's ``nccl`` / ``ipc`` keys even though the receive side registers
under ``skyrl_nccl`` / ``skyrl_ipc`` (see ``weight_receivers.py``).
"""

from __future__ import annotations

import logging
import socket
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Optional

from vllm.distributed.weight_transfer.base import WeightSource
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


SKYRL_NCCL_TRAINER_BACKEND = "skyrl_nccl"
SKYRL_IPC_TRAINER_BACKEND = "skyrl_ipc"


class SkyrlTrainerCapabilities:
    """The three things the worker's memory bracket needs from a trainer engine.

    Declared, not probed. Every SkyRL trainer engine inherits or overrides these,
    so ``Worker._sync_weights_to_inference_engines`` reads plain attributes and a
    typo in a name is an ``AttributeError`` rather than a silent default.

    * ``skyrl_handles_prefix_cache_reset`` -- the engine resets the prefix cache
      itself, at the right point in its own pause/update sequence, and the worker
      must not fire a second concurrent reset. Checkpoint-delta sets this.
    * ``skyrl_force_disable_expandable_segments`` -- the engine shares CUDA
      memory over IPC on every run, not only under colocation, so expandable
      (VMM) segments must be off around the send. Sharded RDT sets this.
    * ``skyrl_empty_cache_after_send`` -- False when the send buffers are reused
      by the next step, where scrubbing them back to CUDA is pure cost.
    """

    skyrl_handles_prefix_cache_reset: bool = False
    skyrl_force_disable_expandable_segments: bool = False
    skyrl_empty_cache_after_send: bool = True


_TRAINER_ENGINE_CACHE: dict[str, tuple[type, type]] = {}


def _build_skyrl_nccl_trainer() -> "tuple[type, type]":
    """vLLM's NCCL trainer engine plus the SkyRL capability declarations.

    A new backend key rather than a shadow of ``nccl``: ``register_engine``
    refuses a duplicate name, and the factory dispatches on the init info's
    ``backend`` ClassVar, so the subclassed init info carries the new key.
    """
    from vllm.distributed.weight_transfer.nccl_engine import (
        NCCLTrainerInitInfo,
        NCCLTrainerWeightTransferEngine,
    )

    @dataclass
    class SkyrlNCCLTrainerInitInfo(NCCLTrainerInitInfo):
        backend: ClassVar[str] = SKYRL_NCCL_TRAINER_BACKEND

    class SkyrlNCCLTrainerWeightTransferEngine(SkyrlTrainerCapabilities, NCCLTrainerWeightTransferEngine):
        init_info_cls = SkyrlNCCLTrainerInitInfo

    return SkyrlNCCLTrainerInitInfo, SkyrlNCCLTrainerWeightTransferEngine


def _build_skyrl_ipc_trainer() -> "tuple[type, type]":
    """vLLM's IPC trainer engine plus the SkyRL capability declarations."""
    from vllm.distributed.weight_transfer.ipc_engine import (
        IPCTrainerInitInfo,
        IPCTrainerWeightTransferEngine,
    )

    @dataclass
    class SkyrlIPCTrainerInitInfo(IPCTrainerInitInfo):
        backend: ClassVar[str] = SKYRL_IPC_TRAINER_BACKEND

    class SkyrlIPCTrainerWeightTransferEngine(SkyrlTrainerCapabilities, IPCTrainerWeightTransferEngine):
        init_info_cls = SkyrlIPCTrainerInitInfo

    return SkyrlIPCTrainerInitInfo, SkyrlIPCTrainerWeightTransferEngine


def get_skyrl_nccl_trainer() -> "tuple[type, type]":
    """``(init_info_cls, engine_cls)`` for ``skyrl_nccl``. Built lazily and cached."""
    if SKYRL_NCCL_TRAINER_BACKEND not in _TRAINER_ENGINE_CACHE:
        _TRAINER_ENGINE_CACHE[SKYRL_NCCL_TRAINER_BACKEND] = _build_skyrl_nccl_trainer()
    return _TRAINER_ENGINE_CACHE[SKYRL_NCCL_TRAINER_BACKEND]


def get_skyrl_ipc_trainer() -> "tuple[type, type]":
    """``(init_info_cls, engine_cls)`` for ``skyrl_ipc``. Built lazily and cached."""
    if SKYRL_IPC_TRAINER_BACKEND not in _TRAINER_ENGINE_CACHE:
        _TRAINER_ENGINE_CACHE[SKYRL_IPC_TRAINER_BACKEND] = _build_skyrl_ipc_trainer()
    return _TRAINER_ENGINE_CACHE[SKYRL_IPC_TRAINER_BACKEND]


def build_trainer_engine(
    *,
    ie_cfg: "InferenceEngineConfig",
    colocate_all: bool,
    rank: int,
    inference_world_size: int,
    source_factory: Callable[["torch.dtype", str], WeightSource],
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
    dtype = str_to_torch_dtype(ie_cfg.model_dtype)
    source = source_factory(dtype, backend)

    init_info, init_payload_fn = _build_sender_init_info(
        backend=backend,
        ie_cfg=ie_cfg,
        rank=rank,
        inference_world_size=inference_world_size,
        dtype=dtype,
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


def _packed_buffer_size_bytes(base_model_path: Optional[str], threshold_in_gb: float, dtype: "torch.dtype") -> int:
    """Packed-buffer size for both push backends: the configured size, floored to
    fit the model's largest single parameter.

    ``generator.inference_engine.weight_transfer_threshold_cuda_ipc_GB`` is the
    configured size. It applies to NCCL as well as IPC.

    The floor is not optional. The packed producers stream through one fixed
    reusable buffer, and a parameter larger than it raises on the IPC path and
    over-allocates on NCCL -- so a large-vocab embedding (Qwen3-235B's is
    151936 x 4096 in bf16 = 1.24 GiB) outgrows the 1 GiB default on its own.

    The floor comes from the checkpoint's safetensors headers rather than the
    live model, so it costs no GPU memory, no collective, and no residency: the
    policy may be offloaded when this runs. A checkpoint whose shapes cannot be
    read contributes no floor and the configured size stands.
    """
    from skyrl.backends.skyrl_train.weight_sync.checkpoint_shapes import max_param_numel

    configured = int(threshold_in_gb * 1024**3) if threshold_in_gb and threshold_in_gb > 0 else 0
    largest = max_param_numel(base_model_path) * dtype.itemsize if base_model_path else 0
    return max(DEFAULT_PACKED_BUFFER_SIZE_BYTES, configured, largest)


def _build_sender_init_info(
    *,
    backend: str,
    ie_cfg: "InferenceEngineConfig",
    rank: int,
    inference_world_size: int,
    dtype: "torch.dtype",
    server_urls: list,
    data_parallel_size: int,
    base_model_path: Optional[str],
):
    """Return ``(init_info, init_payload_fn)`` for an already-resolved backend.

    ``init_payload_fn`` expands the engine's single worker-side init dict to one
    payload per server; only NCCL and sharded RDT need it (see ``control_plane``).
    """
    from skyrl.backends.skyrl_train.weight_sync.register import register_trainer_engines

    register_trainer_engines()

    if backend == "nccl":
        import ray

        nccl_init_info_cls, _ = get_skyrl_nccl_trainer()

        # Only rank 0 opens the endpoint, so only its address/port reaches a
        # worker; the other ranks build and discard theirs.
        master_address = ray._private.services.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            master_port = sock.getsockname()[1]
        return (
            nccl_init_info_cls(
                master_address=master_address,
                master_port=master_port,
                # Every inference worker plus the single trainer sender (rank 0).
                world_size=inference_world_size + 1,
                # Broadcast out of a fixed reusable buffer instead of one NCCL
                # call per parameter. The engine propagates this to the worker at
                # the handshake, so the two sides cannot disagree.
                packed=True,
                packed_buffer_size_bytes=_packed_buffer_size_bytes(
                    base_model_path, ie_cfg.weight_transfer_threshold_cuda_ipc_GB, dtype
                ),
                rank=rank,
            ),
            nccl_init_payloads,
        )

    if backend == "ipc":
        ipc_init_info_cls, _ = get_skyrl_ipc_trainer()

        return (
            # packed=True overrides the vLLM default: the unpacked path holds a
            # strong ref to a contiguous copy of EVERY parameter until past
            # `finish_weight_update` (so the consumer's IPC views stay valid),
            # i.e. the whole model resident on the trainer. Packed streams
            # through one reusable buffer.
            ipc_init_info_cls(
                packed=True,
                packed_buffer_size_bytes=_packed_buffer_size_bytes(
                    base_model_path, ie_cfg.weight_transfer_threshold_cuda_ipc_GB, dtype
                ),
                rank=rank,
            ),
            None,
        )

    if backend == "delta":
        from skyrl.backends.skyrl_train.weight_sync.delta.checkpoint import (
            SUPPORTED_CHECKPOINT_LOAD_FORMATS,
        )
        from skyrl.backends.skyrl_train.weight_sync.delta.trainer import (
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
