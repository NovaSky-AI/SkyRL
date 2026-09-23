"""Registers SkyRL's weight-transfer engines into vLLM's two factories.

vLLM keeps separate registries for the two directions, and they are populated by
different processes:

* ``WeightTransferEngineFactory`` — the **receive** side. Must be populated in
  every vLLM worker process (``Worker.load_model`` builds the engine through the
  factory) and on the **driver**, which validates ``WeightTransferConfig.backend``
  against the registry while building the servers' CLI args.
* ``WeightTransferTrainerFactory`` — the **send** side. Populated on each trainer
  rank, before ``trainer_init`` dispatches on ``init_info.backend``.

Both live here so the two call sites cannot drift: a backend that
``get_vllm_receive_backend`` can select but nobody registered fails only once a
real engine is constructed, inside a worker.

**Registration does not import the engines.** ``delta`` and ``sharded_rdt`` are
registered by module path and class name as strings, which vLLM imports lazily
when a worker constructs the backend — so this module stays cheap and stays
importable without the vLLM wheel. ``skyrl_nccl`` / ``skyrl_ipc`` are the
exception: they are built dynamically as subclasses of vLLM's engines, so there
is no importable module attribute to name and the class itself is passed.

vLLM 0.29 registers the ``sharded_rdt`` receive engine natively. SkyRL uses it
directly, while adapting its native trainer engine with the three explicit
worker-memory capability declarations described in ``weight_senders``.
"""

import logging

logger = logging.getLogger(__name__)

DELTA_BACKEND = "delta"
RDT_BACKEND = "sharded_rdt"

_DELTA_ENGINE_MODULE = "skyrl.backends.skyrl_train.weight_sync.delta.engine"
_DELTA_TRAINER_MODULE = "skyrl.backends.skyrl_train.weight_sync.delta.trainer"
_RDT_ENGINE_MODULE = "skyrl.backends.skyrl_train.weight_sync.sharded_rdt.sharded_rdt_engine"
_RDT_TRAINER_MODULE = "skyrl.backends.skyrl_train.weight_sync.sharded_rdt.sharded_rdt_trainer"

_RECEIVE_REGISTERED = False
_TRAINER_REGISTERED = False


def register_receive_engines() -> None:
    """Register every receive-side engine SkyRL adds (idempotent).

    Call from every vLLM worker process (``new_inference_worker_wrap``, which
    vLLM imports before model init) and from the driver
    (``inference_servers/utils.build_vllm_cli_args``).

    No-op when vLLM is not importable — it is a Linux-only optional dependency
    and half the CPU suite runs without the wheel.
    """
    global _RECEIVE_REGISTERED
    if _RECEIVE_REGISTERED:
        return
    try:
        from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory
    except ImportError:
        logger.debug("vLLM not importable; skipping receive-engine registration.")
        return

    from skyrl.backends.skyrl_train.weight_sync.weight_receivers import (
        SKYRL_IPC_BACKEND,
        SKYRL_NCCL_BACKEND,
        get_skyrl_ipc_engine,
        get_skyrl_nccl_engine,
    )

    # Direct-class registration: these subclass vLLM's engines and are built on
    # demand, so there is no module attribute to name. vLLM is importable here.
    for name, build in ((SKYRL_NCCL_BACKEND, get_skyrl_nccl_engine), (SKYRL_IPC_BACKEND, get_skyrl_ipc_engine)):
        if name not in WeightTransferEngineFactory._registry:
            WeightTransferEngineFactory.register_engine(name, build())

    register_delta_weight_transfer_engine()
    register_rdt_weight_transfer_engine()

    _RECEIVE_REGISTERED = True
    logger.debug("Registered receive-side weight transfer engines.")


def register_delta_weight_transfer_engine() -> None:
    """Register the checkpoint-delta receive engine under ``delta`` (idempotent)."""
    _register_receive_by_path(DELTA_BACKEND, _DELTA_ENGINE_MODULE, "DeltaWeightTransferEngine")


def register_rdt_weight_transfer_engine() -> None:
    """Register the sharded-RDT receive engine under ``sharded_rdt`` (idempotent).

    REMOVAL: drops out once SkyRL's pinned vLLM registers this engine natively.
    """
    _register_receive_by_path(RDT_BACKEND, _RDT_ENGINE_MODULE, "ShardedRDTWeightTransferEngine")


def _register_receive_by_path(name: str, module: str, class_name: str) -> None:
    """Register one receive engine by module path, without importing it.

    No-op when vLLM is not importable, so the per-engine helpers above are safe to
    call from anywhere.
    """
    try:
        from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory
    except ImportError:
        logger.debug("vLLM not importable; skipping %r registration.", name)
        return
    if name not in WeightTransferEngineFactory._registry:
        WeightTransferEngineFactory.register_engine(name, module, class_name)


def register_trainer_engines() -> None:
    """Register SkyRL's trainer-side engines (idempotent).

    Called from ``weight_senders._build_init_info`` on every trainer rank, before
    ``WeightTransferTrainerFactory.trainer_init`` dispatches. All four backends
    are SkyRL's: ``skyrl_nccl`` / ``skyrl_ipc`` subclass vLLM's engines to declare
    the capability attributes the worker's memory bracket reads.
    """
    global _TRAINER_REGISTERED
    if _TRAINER_REGISTERED:
        return
    from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory

    from skyrl.backends.skyrl_train.weight_sync.weight_senders import (
        SKYRL_IPC_TRAINER_BACKEND,
        SKYRL_NCCL_TRAINER_BACKEND,
        get_skyrl_ipc_trainer,
        get_skyrl_nccl_trainer,
        get_skyrl_rdt_trainer,
    )

    # Direct-class registration: like their receive-side counterparts these are
    # built on demand as subclasses of vLLM's engines, so there is no module
    # attribute to name.
    for name, build in (
        (SKYRL_NCCL_TRAINER_BACKEND, get_skyrl_nccl_trainer),
        (SKYRL_IPC_TRAINER_BACKEND, get_skyrl_ipc_trainer),
    ):
        if name not in WeightTransferTrainerFactory._registry:
            WeightTransferTrainerFactory.register_engine(name, build()[1])

    for name, module, cls in ((DELTA_BACKEND, _DELTA_TRAINER_MODULE, "DeltaTrainerWeightTransferEngine"),):
        if name not in WeightTransferTrainerFactory._registry:
            WeightTransferTrainerFactory.register_engine(name, module, cls)

    # vLLM 0.29 owns this backend's wire transport. Its sender intentionally
    # knows nothing about SkyRL's worker-memory bracket, so replace only the
    # lazy loader with a capability-declaring subclass. The native init-info and
    # implementation stay unchanged; a missing/changed native class therefore
    # fails immediately when a trainer is constructed.
    if RDT_BACKEND in WeightTransferTrainerFactory._registry:
        WeightTransferTrainerFactory._registry[RDT_BACKEND] = get_skyrl_rdt_trainer
    else:
        WeightTransferTrainerFactory.register_engine(
            RDT_BACKEND, _RDT_TRAINER_MODULE, "ShardedRDTTrainerWeightTransferEngine"
        )

    _TRAINER_REGISTERED = True
    logger.debug("Registered trainer-side weight transfer engines.")
