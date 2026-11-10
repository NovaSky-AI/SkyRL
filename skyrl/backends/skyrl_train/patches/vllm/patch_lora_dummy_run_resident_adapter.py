"""Data-parallel serving with LoRA: idle DP ranks run their dummy steps on the resident adapter.

Under vLLM data parallelism every DP rank steps in lockstep, and a rank with no scheduled requests runs
``Worker.execute_dummy_batch`` -> ``model_runner._dummy_run`` each step while any other rank has work. With
LoRA, ``_dummy_run`` wraps the forward in ``maybe_dummy_run_with_lora(..., remove_lora=True,
num_active_loras=max_loras)``, built for warmup and CUDA-graph capture: every call creates ``max_loras``
all-zero "warmup" adapters (``add_dummy_lora``: ``create_dummy_lora`` / ``pack_moe`` / ``optimize``), activates
them, runs, then ``remove_all_adapters()``. At serving time that costs, per idle-rank step:

* time: for a model with LoRA on every MoE expert (DeepSeek-V4.1: 256 experts x 43 layers) building the dummy
  adapter takes seconds. Every lockstepped step waits for it, so while any rank is idle the busy ranks decode at
  ~1-3 tok/s (rollout tails, and the drain of a ``wait``-mode pause for a weight sync: one rank's last request
  then outlives SKYRL_INFERENCE_CONTROL_PLANE_TIMEOUT_S and the run dies on the /pause);
* correctness under expert parallelism: the idle rank's MoE layer still runs its local experts on the other
  ranks' tokens (and with SKYRL_VLLM_MOE_LORA_EP_GLOBAL_META applies LoRA to them), but its slots hold the
  all-zero warmup adapters, so those tokens get no adapter delta on this rank's experts;
* the real adapter is removed from the idle rank after every dummy step and reloaded on its next request.

This patch flags runtime dummy steps (``Worker.execute_dummy_batch``) and, when the worker already has adapters
registered, maps the dummy tokens onto them instead: no dummy adapters are created and nothing is removed, the
LoRA kernels see the same number of active adapters, and every rank's experts apply the real adapter. Warmup,
profiling and CUDA-graph capture (no flag, or no adapter registered yet) keep vLLM's behavior.

Opt-in: ``SKYRL_VLLM_LORA_DUMMY_RESIDENT_ADAPTER=1`` (applied in every vLLM worker by ``new_inference_worker_wrap``).
"""

import contextlib
import functools
import os

import numpy as np
from loguru import logger

ENV = "SKYRL_VLLM_LORA_DUMMY_RESIDENT_ADAPTER"
_FLAG = "_skyrl_runtime_dummy_step"
_APPLIED = False


def _resident_lora_ids(runner) -> list:
    manager = getattr(runner, "lora_manager", None)
    if manager is None:
        return []
    try:
        return sorted(i for i in manager.list_adapters() if i > 0)
    except Exception:  # noqa: BLE001 - fall back to vLLM's path on any manager we don't understand
        return []


def apply_lora_dummy_run_resident_adapter_patch() -> bool:
    """Patch ``LoRAModelRunnerMixin.maybe_dummy_run_with_lora`` and ``Worker.execute_dummy_batch``. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return True
    if os.environ.get(ENV, "0").lower() not in ("1", "true"):
        return False
    try:
        from vllm.lora.request import LoRARequest
        from vllm.v1.worker.gpu_worker import Worker
        from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
    except ImportError:
        return False
    if not hasattr(LoRAModelRunnerMixin, "maybe_dummy_run_with_lora") or not hasattr(Worker, "execute_dummy_batch"):
        logger.warning(f"{ENV}=1 but vLLM has no maybe_dummy_run_with_lora / execute_dummy_batch; not patching")
        return False

    orig_dummy_run_with_lora = LoRAModelRunnerMixin.maybe_dummy_run_with_lora

    @contextlib.contextmanager
    def maybe_dummy_run_with_lora(
        self,
        lora_config,
        num_scheduled_tokens,
        num_sampled_tokens=None,
        remove_lora=True,
        num_active_loras=0,
        *args,
        **kwargs,
    ):
        ids = _resident_lora_ids(self) if lora_config is not None and getattr(self, _FLAG, False) else []
        if not ids:
            with orig_dummy_run_with_lora(
                self,
                lora_config,
                num_scheduled_tokens,
                num_sampled_tokens,
                remove_lora,
                num_active_loras,
                *args,
                **kwargs,
            ):
                yield
            return
        num_scheduled_tokens = np.asarray(num_scheduled_tokens, dtype=np.int32)
        if num_sampled_tokens is None:
            num_sampled_tokens = np.ones_like(num_scheduled_tokens, dtype=np.int32)
        num_reqs = len(num_scheduled_tokens)
        # Same number of distinct adapters vLLM's dummy path would activate (capped by what is resident).
        use = ids[: max(1, min(num_active_loras, lora_config.max_loras, len(ids)))] if num_active_loras > 0 else []
        if use:
            prompt_mapping = np.asarray(use, dtype=np.int32)[np.arange(num_reqs) % len(use)]
        else:
            prompt_mapping = np.zeros(num_reqs, dtype=np.int32)
        requests = {
            LoRARequest(lora_name=f"skyrl_resident_{i}", lora_int_id=i, lora_path="/not/a/real/path") for i in use
        }
        mapping_args = list(args[:1]) + ([kwargs["mapping_type"]] if "mapping_type" in kwargs else [])
        self._set_active_loras(
            tuple(np.repeat(prompt_mapping, num_sampled_tokens)),
            tuple(np.repeat(prompt_mapping, num_scheduled_tokens)),
            requests,
            *mapping_args,
        )
        yield

    orig_execute_dummy_batch = Worker.execute_dummy_batch

    @functools.wraps(orig_execute_dummy_batch)
    def execute_dummy_batch(self, *args, **kwargs):
        runner = self.model_runner
        setattr(runner, _FLAG, True)
        try:
            return orig_execute_dummy_batch(self, *args, **kwargs)
        finally:
            setattr(runner, _FLAG, False)

    LoRAModelRunnerMixin.maybe_dummy_run_with_lora = maybe_dummy_run_with_lora
    Worker.execute_dummy_batch = execute_dummy_batch
    _APPLIED = True
    logger.info(f"{ENV}: idle DP ranks' dummy steps reuse the resident LoRA adapter")
    return True
