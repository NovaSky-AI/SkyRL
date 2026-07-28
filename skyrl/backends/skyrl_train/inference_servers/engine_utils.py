import os
from typing import Any, Dict, Optional, Union

from omegaconf import DictConfig, ListConfig

from skyrl.train.config import SamplingParams


def _alloc_conf_with_expandable_segments() -> str:
    """Return a ``PYTORCH_CUDA_ALLOC_CONF`` value with ``expandable_segments:True`` enabled.

    Appended to any value already set in the launching environment rather than overwriting
    it, so other allocator settings (e.g. ``max_split_size_mb`` or a custom backend) are
    preserved. If the user already set ``expandable_segments`` explicitly, their value is
    left untouched (which also avoids a duplicate-key parse error).
    """
    existing = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").strip()
    if not existing:
        return "expandable_segments:True"
    if "expandable_segments" in existing:
        return existing
    return f"{existing},expandable_segments:True"


def build_engine_runtime_env(
    use_expandable_segments: bool = False,
    extra_env_vars: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    """Build the Ray ``runtime_env`` for inference-engine actors.

    Env vars are set here (rather than inside the actor) because they must be present
    before the worker process initializes CUDA -- this is also why the vLLM worker child
    tasks, which inherit the actor's ``runtime_env``, pick them up. Add future engine-side
    env vars by extending this function or by passing ``extra_env_vars``.

    Args:
        use_expandable_segments: Enable PyTorch's ``expandable_segments`` allocator via
            ``PYTORCH_CUDA_ALLOC_CONF`` (appended to any existing value).
        extra_env_vars: Additional env vars to set on the engine actors. Takes precedence
            over the keys this function sets if they collide.

    Returns ``None`` when there is nothing to set, so callers can pass the result straight
    through as ``runtime_env``.
    """
    env_vars: Dict[str, str] = {}
    if use_expandable_segments:
        env_vars["PYTORCH_CUDA_ALLOC_CONF"] = _alloc_conf_with_expandable_segments()
    if extra_env_vars:
        env_vars.update(extra_env_vars)
    if not env_vars:
        return None
    return {"env_vars": env_vars}


def get_vllm_sampling_params(sampling_params: Union[SamplingParams, DictConfig]) -> Dict[str, Any]:
    stop_val = sampling_params.stop
    vllm_sampling_params = {
        "min_tokens": 1,
        "skip_special_tokens": True,
        "include_stop_str_in_output": True,
        "max_tokens": sampling_params.max_generate_length,
        "temperature": sampling_params.temperature,
        "top_p": sampling_params.top_p,
        "top_k": sampling_params.top_k,
        "min_p": sampling_params.min_p,
        "logprobs": sampling_params.logprobs,
        "stop": list(stop_val) if stop_val is not None else None,
    }
    if isinstance(sampling_params, DictConfig):
        exclude_keys = ["max_generate_length"]
        for key, value in sampling_params.items():
            if key not in vllm_sampling_params and key not in exclude_keys:
                # Convert OmegaConf ListConfig to regular list if needed
                if isinstance(value, ListConfig):
                    value = list(value)
                vllm_sampling_params[key] = value
    else:
        if sampling_params.additional_kwargs is not None:
            for key, value in sampling_params.additional_kwargs.items():
                if key not in vllm_sampling_params:
                    vllm_sampling_params[key] = value
    return vllm_sampling_params


def get_sampling_params_for_backend(backend: str, sampling_params: Union[SamplingParams, DictConfig]) -> Dict[str, Any]:
    if backend == "vllm":
        return get_vllm_sampling_params(sampling_params)
    else:
        raise ValueError(f"Unsupported generation backend: {backend}")
