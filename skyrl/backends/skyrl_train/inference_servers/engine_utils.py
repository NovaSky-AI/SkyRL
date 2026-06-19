import hashlib
import os
import random
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


def hash_with_sha256(x: Union[int, str]) -> int:
    return int.from_bytes(hashlib.sha256(str(x).encode()).digest(), "big")


def route_prompts_to_engines(
    num_prompts: int, num_inference_engines: int, session_ids: Optional[Union[list[int], list[str]]]
) -> dict[int, list[int]]:
    """
    Given the number of prompts, number of inference engines, and the session_id, return a mapping
    from engine index to the list of prompt IDs the engine will process.

    Args:
    - num_prompts: int - The number of prompts.
    - num_inference_engines: int - The number of inference engines.
    - session_ids: Optional[Union[list[int], list[str]]] - The session IDs.

    Required:
    - num_prompts > 0
    - num_inference_engines > 0
    - session_ids is a list of integers or strings if provided
    - len(session_ids) == num_prompts if provided

    Returns:
    - dict[int, list[int]] - A mapping from engine index to the list of prompt IDs the engine will process.
    """
    # 0. Validation
    assert num_prompts > 0, "Number of prompts must be greater than 0"
    assert num_inference_engines > 0, "Number of inference engines must be greater than 0"
    if session_ids is not None:
        assert isinstance(session_ids, list) and all(
            isinstance(sid, (int, str)) for sid in session_ids
        ), "Session ID must be a list of integers or strings"
        assert len(session_ids) == num_prompts, "Session ID must have the same length as the number of prompts"

    # 1. session_id not provided, with a single prompt: route to a random engine for a naive load balancing.
    if session_ids is None and num_prompts == 1:
        engine_idx = random.randint(0, num_inference_engines - 1)
        return {engine_idx: [0]}

    # 2. session_id not provided, with a batched prompt: split evenly across engines.
    engine_idx_to_prompt_ids: dict[int, list[int]] = {}
    if session_ids is None:
        dp_item_size = (num_prompts + num_inference_engines - 1) // num_inference_engines
        for dp_rank in range(num_inference_engines):
            start_idx = dp_rank * dp_item_size
            end_idx = min((dp_rank + 1) * dp_item_size, num_prompts)
            prompt_ids = list(range(start_idx, end_idx))
            if len(prompt_ids) > 0:
                engine_idx_to_prompt_ids[dp_rank] = prompt_ids
        return engine_idx_to_prompt_ids

    # 3. session_id provided, we route by session_id
    for i, cur_sid in enumerate(session_ids):
        engine_idx = hash_with_sha256(str(cur_sid)) % num_inference_engines
        engine_idx_to_prompt_ids.setdefault(engine_idx, []).append(i)
    return engine_idx_to_prompt_ids
