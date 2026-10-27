import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional, Union

from omegaconf import DictConfig, ListConfig, OmegaConf

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


# Keys that are consumed by the mapping below (or dropped on purpose) and must not be
# re-emitted verbatim as ``generate`` kwargs by the passthrough loop.
_NEURON_CONSUMED_KEYS = {
    "max_generate_length",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "stop",
    "logprobs",
    "additional_kwargs",
}


def _sampling_params_to_dict(sampling_params: Union[SamplingParams, DictConfig, Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize the three accepted sampling-params shapes to a plain ``dict``.

    Generators pass sampling params as a plain ``dict`` (``dataclasses.asdict`` of
    ``SamplingParams``); config-driven paths pass a ``DictConfig``; and callers may
    pass the typed ``SamplingParams`` directly. Reading keys through a single dict
    view keeps the mapping logic free of shape-specific attribute access.
    """
    if isinstance(sampling_params, dict) and not isinstance(sampling_params, DictConfig):
        return sampling_params
    if isinstance(sampling_params, DictConfig):
        return OmegaConf.to_container(sampling_params, resolve=True)
    if is_dataclass(sampling_params):
        return asdict(sampling_params)
    raise TypeError(f"Unsupported sampling_params type: {type(sampling_params)!r}")


def get_neuron_sampling_params(sampling_params: Union[SamplingParams, DictConfig, Dict[str, Any]]) -> Dict[str, Any]:
    """Map SkyRL ``SamplingParams`` to transformers ``generate`` kwargs.

    The native (TorchNeuron) engine calls ``model.generate(**kwargs)`` directly, so
    the keys are transformers ``generate`` arguments rather than vLLM fields.

    ``logprobs`` is deliberately dropped here: the native engine always computes
    prompt/response logprobs via ``output_logits``, so the vLLM ``logprobs`` key has
    no ``generate`` equivalent.

    ``do_sample`` is derived from ``temperature`` to match vLLM semantics: transformers
    falls back to greedy decoding at ``temperature == 1.0`` (the common RL default),
    which would silently disable the sampling RL exploration depends on.
    """
    params = _sampling_params_to_dict(sampling_params)

    stop_val = params.get("stop")
    generate_kwargs: Dict[str, Any] = {
        "max_new_tokens": params.get("max_generate_length"),
        "temperature": params.get("temperature"),
        "top_p": params.get("top_p"),
        "repetition_penalty": params.get("repetition_penalty"),
        "stop_strings": list(stop_val) if stop_val is not None else None,
    }

    temperature = params.get("temperature")
    if temperature is not None:
        generate_kwargs["do_sample"] = temperature > 0

    top_k = params.get("top_k")
    if top_k is not None and top_k > 0:
        generate_kwargs["top_k"] = top_k
    min_p = params.get("min_p")
    if min_p is not None and min_p > 0:
        generate_kwargs["min_p"] = min_p

    additional_kwargs = params.get("additional_kwargs")
    if additional_kwargs:
        for key, value in additional_kwargs.items():
            generate_kwargs.setdefault(key, value)

    for key, value in params.items():
        if key not in _NEURON_CONSUMED_KEYS and key not in generate_kwargs:
            if isinstance(value, ListConfig):
                value = list(value)
            generate_kwargs[key] = value

    return generate_kwargs


def get_sampling_params_for_backend(backend: str, sampling_params: Union[SamplingParams, DictConfig]) -> Dict[str, Any]:
    if backend == "vllm":
        return get_vllm_sampling_params(sampling_params)
    elif backend == "neuron":
        return get_neuron_sampling_params(sampling_params)
    else:
        raise ValueError(f"Unsupported generation backend: {backend}")
