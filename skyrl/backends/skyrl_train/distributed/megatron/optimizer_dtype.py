"""Torch-only coercion for Megatron optimizer dtype kwargs."""

from typing import Any, Dict, Set

import torch

# Megatron short names plus common YAML spellings. TE stores FP8 optimizer state
# as uint8, matching Megatron-LM's dtype map.
_DTYPE_NAME_TO_TORCH: Dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "float32": torch.float32,
    "float": torch.float32,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "fp8": torch.uint8,
    "float8": torch.uint8,
    "uint8": torch.uint8,
}

# Only TE FusedAdam-backed fields get field-specific checks. ``main_grads_dtype``
# is not forwarded at the pinned megatron-core rev, so it is coerced only and
# left to ``OptimizerConfig.__post_init__``.
_LEGAL_FIELD_DTYPES: Dict[str, Set[torch.dtype]] = {
    "main_params_dtype": {torch.float32, torch.float16},
    "exp_avg_dtype": {torch.float32, torch.bfloat16, torch.float16, torch.uint8},
    "exp_avg_sq_dtype": {torch.float32, torch.bfloat16, torch.float16, torch.uint8},
}


def coerce_optimizer_dtype_kwargs(optimizer_config_kwargs: Dict[str, Any] | None) -> Dict[str, Any]:
    """Return kwargs with recognized ``*_dtype`` strings converted to ``torch.dtype``."""
    if optimizer_config_kwargs is None:
        return {}

    coerced: Dict[str, Any] = {}
    for key, value in optimizer_config_kwargs.items():
        if not key.endswith("_dtype"):
            coerced[key] = value
            continue

        if isinstance(value, torch.dtype):
            dtype = value
        elif isinstance(value, str):
            name = value.strip().lower()
            if name not in _DTYPE_NAME_TO_TORCH:
                raise ValueError(
                    f"Unrecognized dtype name {value!r} for optimizer kwarg {key!r}. "
                    f"Expected one of {sorted(_DTYPE_NAME_TO_TORCH)} or a torch.dtype."
                )
            dtype = _DTYPE_NAME_TO_TORCH[name]
        else:
            # Let Megatron validate non-string, non-dtype values.
            coerced[key] = value
            continue

        legal = _LEGAL_FIELD_DTYPES.get(key)
        if legal is not None and dtype not in legal:
            legal_names = sorted({n for n, d in _DTYPE_NAME_TO_TORCH.items() if d in legal})
            raise ValueError(f"Illegal dtype {dtype} for optimizer kwarg {key!r}; legal values are {legal_names}.")
        coerced[key] = dtype
    return coerced
