"""Record this process's ``GPUModelRunner`` for receive-side sync hooks.

vLLM constructs ``WeightTransferEngine`` instances with the model, not the
worker or runner. SkyRL's receive-side finish hooks need the runner to repair
state that lives outside plain model weights, such as serialized-FP8 KV scale
mirrors. Record the process-local runner after ``GPUModelRunner.load_model`` so
engine hooks can find it later in the same worker process.
"""

from __future__ import annotations

import weakref
from typing import Any

_PATCHED = False
_CURRENT_MODEL_RUNNER: weakref.ReferenceType[Any] | None = None


def apply_model_runner_registry_patch() -> None:
    """Install the process-local runner recorder on ``GPUModelRunner.load_model``."""
    global _PATCHED
    if _PATCHED:
        return

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original = GPUModelRunner.load_model

    def load_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = original(self, *args, **kwargs)
        global _CURRENT_MODEL_RUNNER
        _CURRENT_MODEL_RUNNER = weakref.ref(self)
        return result

    load_model.__wrapped__ = original
    GPUModelRunner.load_model = load_model
    _PATCHED = True


def current_model_runner() -> Any | None:
    """Return this worker process's ``GPUModelRunner``, if it has loaded."""
    if _CURRENT_MODEL_RUNNER is None:
        return None
    return _CURRENT_MODEL_RUNNER()
