"""Pure-Python (torch-only) coercion of Megatron optimizer ``*_dtype`` kwargs.

This is intentionally kept free of any ``megatron.core`` import so the coercion
logic can be unit-tested on the cheap CPU CI lane (which installs torch but not
megatron-core). ``optimizer.py`` imports the public helper from here.
"""

from typing import Any, Dict, Set

import torch

# Canonical dtype-name -> torch.dtype mapping, mirroring Megatron-LM's own
# ``dtype_map`` in ``megatron/training/arguments.py`` (the short forms 'fp32',
# 'bf16', 'fp16', 'fp8' are the ones Megatron itself maps). The extra aliases
# ('bfloat16', 'float16'/'half', 'float32'/'float', 'float8'/'uint8') accept the
# spellings a user might reasonably write in YAML. ``fp8`` maps to ``torch.uint8``
# because TransformerEngine represents FP8 optimizer state as uint8.
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

# Per-field legal dtypes, enforced before the kwargs reach FusedAdam (which would
# otherwise raise a cryptic error deep inside TransformerEngine). Only fields that
# are actually forwarded to TE FusedAdam — and whose accepted set is verifiable in
# the TE source — are listed here. ``main_params_dtype`` (a.k.a. master weights)
# maps to FusedAdam's ``master_weight_dtype``, which only supports fp32/fp16;
# ``exp_avg_dtype`` / ``exp_avg_sq_dtype`` additionally allow bf16/fp8.
# ``main_grads_dtype`` is deliberately NOT listed: at the pinned megatron-core rev
# it is not forwarded to FusedAdam (see megatron/core/optimizer/__init__.py, which
# only passes exp_avg_dtype, exp_avg_sq_dtype, and main_params_dtype as
# master_weight_dtype), so there is no TE-backed legal set to enforce — it is still
# coerced str->dtype here and left for ``OptimizerConfig.__post_init__`` to validate
# (mirroring how ``params_dtype`` is handled: coerced, not field-validated).
# Fields not listed here accept any value in ``_DTYPE_NAME_TO_TORCH``.
_LEGAL_FIELD_DTYPES: Dict[str, Set[torch.dtype]] = {
    "main_params_dtype": {torch.float32, torch.float16},
    "exp_avg_dtype": {torch.float32, torch.bfloat16, torch.float16, torch.uint8},
    "exp_avg_sq_dtype": {torch.float32, torch.bfloat16, torch.float16, torch.uint8},
}


def coerce_optimizer_dtype_kwargs(optimizer_config_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce ``*_dtype`` string values in Megatron optimizer kwargs to ``torch.dtype``.

    Megatron's precision-aware ``OptimizerConfig`` types ``exp_avg_dtype`` /
    ``exp_avg_sq_dtype`` / ``main_params_dtype`` (and friends) as real ``torch.dtype``,
    but SkyRL forwards ``optimizer_config_kwargs`` verbatim from YAML/Hydra, which
    delivers plain strings (e.g. ``"bf16"``). This converts any ``*_dtype`` key whose
    value is a dtype-name string into the corresponding ``torch.dtype`` using Megatron-LM's
    canonical mapping, validates the result against the per-field legal set, and leaves
    everything else (non-``*_dtype`` keys, values already ``torch.dtype``) untouched.

    Returns a new dict; the input is not mutated.

    Raises:
        ValueError: if a ``*_dtype`` value is an unrecognized dtype name, or if a coerced
            dtype is illegal for that specific field (e.g. bf16/fp8 for ``main_params_dtype``).
    """
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
            # Not a dtype-name string or torch.dtype (e.g. None); pass through so
            # Megatron's own validation surfaces any problem.
            coerced[key] = value
            continue

        legal = _LEGAL_FIELD_DTYPES.get(key)
        if legal is not None and dtype not in legal:
            legal_names = sorted({n for n, d in _DTYPE_NAME_TO_TORCH.items() if d in legal})
            raise ValueError(f"Illegal dtype {dtype} for optimizer kwarg {key!r}; legal values are {legal_names}.")
        coerced[key] = dtype
    return coerced
