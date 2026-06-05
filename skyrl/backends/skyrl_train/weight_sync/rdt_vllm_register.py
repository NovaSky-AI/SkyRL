"""Runtime registration of the SkyRL ``sharded_rdt`` weight-transfer engine into
the **pinned, unmodified** ``vllm==0.20.2`` wheel.

Importing this module (idempotently):

  1. Relaxes ``vllm.config.weight_transfer.WeightTransferConfig.backend`` from a
     pydantic ``Literal["nccl", "ipc"]`` to ``str`` so ``backend="sharded_rdt"``
     validates. The fork made this change in-tree; here we must do it at runtime
     because the wheel is not editable.
  2. Registers ``ShardedRDTWeightTransferEngine`` in vLLM's
     ``WeightTransferEngineFactory`` under the name ``"sharded_rdt"`` so
     ``GPUWorker.__init__`` builds it when ``weight_transfer_config.backend ==
     "sharded_rdt"``.

This must run on the **driver** (which builds the ``WeightTransferConfig`` in
``build_vllm_cli_args``) AND on **every vLLM worker** process (which constructs
the engine via the factory). It is imported from:
  * ``new_inference_worker_wrap.py`` — loaded by vLLM via ``--worker-extension-cls``
    on every worker before model init.
  * ``inference_servers/utils.py`` and ``weight_sync/__init__.py`` — driver side.

``vllm`` is a Linux-only optional dependency, so all vLLM imports are inside the
function and failures are swallowed on platforms without vLLM (matching the
lazy-import convention in ``new_inference_worker_wrap.py``).

REMOVAL: this entire module is a stopgap for the pinned, unmodified vLLM wheel.
Delete it once SkyRL's pinned vLLM ships BOTH of the fork's changes upstream:
  1. ``WeightTransferConfig.backend`` accepts arbitrary strings (the fork changed
     ``Literal["nccl", "ipc"]`` -> ``str``), so no Literal relaxation is needed; and
  2. the ``sharded_rdt`` engine is registered in ``WeightTransferEngineFactory``
     (or otherwise discoverable), so no runtime registration is needed.
At that point ``WeightTransferConfig(backend="sharded_rdt")`` works natively;
drop this file and its imports from ``new_inference_worker_wrap.py``,
``inference_servers/utils.py``, and ``weight_sync/__init__.py``.
"""

import logging

logger = logging.getLogger(__name__)

RDT_BACKEND = "sharded_rdt"

_DONE = False


def _relax_backend_literal() -> bool:
    """Make ``WeightTransferConfig.backend`` accept arbitrary strings.

    vLLM 0.20.2 types it as ``Literal["nccl", "ipc"]`` on a pydantic dataclass,
    so ``WeightTransferConfig(backend="sharded_rdt")`` raises at construction.
    The wheel isn't editable, and swapping the field annotation + rebuilding the
    pydantic schema does NOT stick (the compiled core schema caches the Literal),
    so we wrap ``__init__``: construct with a valid literal, then overwrite
    ``backend`` after validation. Pydantic dataclasses don't validate on
    attribute assignment, and the override survives pickling to Ray workers —
    verified end-to-end by the sharded_rdt GPU weight-sync test. Returns True on
    success.
    """
    import vllm.config
    import vllm.config.weight_transfer as wt_mod

    cls = wt_mod.WeightTransferConfig

    # Already accepts arbitrary backends? (a future vLLM that relaxes the
    # Literal, or a prior import in this process) -> nothing to patch.
    try:
        cls(backend=RDT_BACKEND)
        return True
    except Exception:
        pass

    try:
        _orig_init = cls.__init__

        def _patched_init(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            override = None
            if "backend" in kwargs and kwargs["backend"] not in ("nccl", "ipc"):
                override = kwargs["backend"]
                kwargs = dict(kwargs)
                kwargs["backend"] = "nccl"
            elif args and args[0] not in ("nccl", "ipc"):
                override = args[0]
                args = ("nccl",) + tuple(args[1:])
            _orig_init(self, *args, **kwargs)
            if override is not None:
                object.__setattr__(self, "backend", override)

        cls.__init__ = _patched_init  # type: ignore[method-assign]
        vllm.config.WeightTransferConfig = cls
        cls(backend=RDT_BACKEND)
        logger.info("Relaxed WeightTransferConfig.backend to accept %r via __init__ wrap.", RDT_BACKEND)
        return True
    except Exception as e:  # noqa: BLE001
        logger.error("Could not relax WeightTransferConfig.backend: %s", e)
        return False


def _register_engine() -> None:
    from vllm.distributed.weight_transfer import WeightTransferEngineFactory

    from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_engine import (
        ShardedRDTWeightTransferEngine,
    )

    if RDT_BACKEND in WeightTransferEngineFactory._registry:
        return
    WeightTransferEngineFactory.register_engine(RDT_BACKEND, ShardedRDTWeightTransferEngine)
    logger.info("Registered %r weight transfer engine.", RDT_BACKEND)


def ensure_registered() -> None:
    """Idempotently relax the config + register the engine. Safe to call often.

    No-op if vLLM is not importable (non-Linux platforms / CPU-only test envs
    without the vLLM wheel)."""
    global _DONE
    if _DONE:
        return
    try:
        ok = _relax_backend_literal()
        if not ok:
            raise RuntimeError(
                "Failed to relax WeightTransferConfig.backend to accept "
                f"{RDT_BACKEND!r}; the sharded_rdt backend cannot be configured."
            )
        _register_engine()
        _DONE = True
    except ImportError:
        # vLLM not available (e.g. non-Linux). The sharded_rdt backend is a
        # Linux/GPU feature; nothing to register here.
        logger.debug("vLLM not importable; skipping sharded_rdt registration.")


# Run on import.
ensure_registered()
