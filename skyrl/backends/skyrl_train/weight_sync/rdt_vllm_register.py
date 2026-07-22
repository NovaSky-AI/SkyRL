"""Runtime registration of the SkyRL ``sharded_rdt`` weight-transfer engine into
the pinned ``vllm==0.23.0`` wheel.

Importing this module (idempotently) registers ``ShardedRDTWeightTransferEngine``
in vLLM's ``WeightTransferEngineFactory`` under the name ``"sharded_rdt"`` so
``GPUWorker.load_model`` builds it (via ``create_engine``) when
``weight_transfer_config.backend == "sharded_rdt"``.

vLLM 0.23.0 already types ``WeightTransferConfig.backend`` as
``Literal["nccl", "ipc"] | str`` (accepts arbitrary strings), so — unlike the
0.20.2 stopgap — NO Literal relaxation is needed; only the factory registration.

This must run on the **driver** (which builds the ``WeightTransferConfig`` in
``build_vllm_cli_args``) AND on **every vLLM worker** process (which constructs
the engine via the factory). It is imported from:
  * ``new_inference_worker_wrap.py`` — loaded by vLLM via ``--worker-extension-cls``
    on every worker before model init.
  * ``inference_servers/utils.py`` and ``weight_sync/__init__.py`` — driver side.

``vllm`` is a Linux-only optional dependency, so vLLM imports are inside the
function and ImportError is swallowed on platforms without vLLM.

REMOVAL: delete this module + its imports once SkyRL's pinned vLLM registers the
``sharded_rdt`` engine in ``WeightTransferEngineFactory`` natively.
"""

import logging

logger = logging.getLogger(__name__)

RDT_BACKEND = "sharded_rdt"

_DONE = False


def _register_engine() -> None:
    from vllm.distributed.weight_transfer import WeightTransferEngineFactory

    # Lazy-loading registration: only imports the engine module when a worker
    # actually constructs the backend (keeps this import cheap on the driver).
    if RDT_BACKEND in WeightTransferEngineFactory._registry:
        return
    WeightTransferEngineFactory.register_engine(
        RDT_BACKEND,
        "skyrl.backends.skyrl_train.weight_sync.sharded_rdt_engine",
        "ShardedRDTWeightTransferEngine",
    )
    logger.info("Registered %r weight transfer engine.", RDT_BACKEND)


def ensure_registered() -> None:
    """Idempotently register the engine. Safe to call often.

    No-op if vLLM is not importable (non-Linux platforms / CPU-only test envs
    without the vLLM wheel)."""
    global _DONE
    if _DONE:
        return
    try:
        _register_engine()
        _DONE = True
    except ImportError:
        # vLLM not available (e.g. non-Linux). The sharded_rdt backend is a
        # Linux/GPU feature; nothing to register here.
        logger.debug("vLLM not importable; skipping sharded_rdt registration.")


# Run on import.
ensure_registered()
