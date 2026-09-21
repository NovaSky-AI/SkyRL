"""Record this process's ``GPUModelRunner`` so a weight-transfer engine can reach it.

A ``WeightTransferEngine`` is constructed with ``(config, vllm_config, device,
model)`` -- the main model only. SkyRL's receive path also needs the
speculative-decoding drafter (``model_runner.drafter.model``), a separate module
that ``load_weights`` on the main model never touches, so an MTP model would keep
drafting with pre-sync weights (see ``inference_servers/spec_decode_utils``).

There is no route from an engine to its worker: the engine is created inside
``Worker.load_model``, and the worker-extension class is appended to
``Worker.__bases__`` *after* ``Worker`` (``v1/worker/worker_base.py``), so it can
neither override ``load_model`` nor hand the engine anything. So wrap
``GPUModelRunner.load_model`` and record the runner in a process-global weakref:
one runner per worker process, recorded before any weight sync can run, read
lazily at sync time.

REMOVAL: delete this once vLLM gives ``WeightTransferEngine`` a post-load hook
(or reaches the drafter itself). Nothing else depends on it -- the engines treat a
missing runner as "no drafter".
"""

import weakref
from typing import Any, Optional

_PATCHED = False
_CURRENT_MODEL_RUNNER: Optional["weakref.ReferenceType[Any]"] = None


def apply_model_runner_registry_patch() -> None:
    """Install the recorder on ``GPUModelRunner.load_model`` (idempotent)."""
    global _PATCHED
    if _PATCHED:
        return

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original = GPUModelRunner.load_model

    def load_model(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        global _CURRENT_MODEL_RUNNER
        _CURRENT_MODEL_RUNNER = weakref.ref(self)
        return result

    load_model.__wrapped__ = original
    GPUModelRunner.load_model = load_model
    _PATCHED = True


def current_model_runner() -> Optional[Any]:
    """This process's ``GPUModelRunner``, or None if it was never recorded.

    None means the patch did not run -- not a vLLM worker process, or a vLLM
    version whose runner no longer goes through ``GPUModelRunner.load_model``.
    Callers must treat it as "no drafter available", not as an error.
    """
    if _CURRENT_MODEL_RUNNER is None:
        return None
    return _CURRENT_MODEL_RUNNER()
