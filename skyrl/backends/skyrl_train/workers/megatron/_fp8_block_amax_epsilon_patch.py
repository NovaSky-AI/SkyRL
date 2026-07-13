"""Apply SkyRL's optional amax floor to TE blockwise FP8 recipes."""

from __future__ import annotations

import os
from dataclasses import replace
from math import isfinite

from loguru import logger

_ENV_VAR = "NVTE_FP8_BLOCK_AMAX_EPSILON"
_PATCH_FLAG = "_skyrl_amax_epsilon_patched"
_QPARAM_FIELDS = ("fp8_quant_fwd_inp", "fp8_quant_fwd_weight", "fp8_quant_bwd_grad")
_MISSING = object()


def _configured_amax_epsilon() -> float | None:
    raw_value = os.getenv(_ENV_VAR)
    if raw_value is None or not raw_value.strip():
        return None
    try:
        epsilon = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{_ENV_VAR} must be a float, got {raw_value!r}") from exc
    if not isfinite(epsilon) or epsilon < 0:
        raise ValueError(f"{_ENV_VAR} must be finite and non-negative, got {epsilon}")
    return epsilon or None


def apply_fp8_block_amax_epsilon_patch() -> None:
    """Idempotently set ``amax_epsilon`` on TE 2.11 blockwise QParams.

    The environment variable is optional. Once explicitly enabled, an
    unsupported TE API or failed verification is fatal because continuing would
    silently use a different training quantizer than the serialized rollout path.
    """

    epsilon = _configured_amax_epsilon()
    if epsilon is None:
        return

    try:
        from transformer_engine.common.recipe import Float8BlockScaling, Format
    except Exception as exc:
        raise RuntimeError(
            f"{_ENV_VAR} is set, but Transformer Engine's Float8BlockScaling recipe is unavailable"
        ) from exc

    if getattr(Float8BlockScaling, _PATCH_FLAG, None) == epsilon:
        return

    original_qparams = {}
    original_flag = getattr(Float8BlockScaling, _PATCH_FLAG, _MISSING)
    try:
        original_qparams = {name: getattr(Float8BlockScaling, name) for name in _QPARAM_FIELDS}
        patched_qparams = {name: replace(qparams, amax_epsilon=epsilon) for name, qparams in original_qparams.items()}
        for name, qparams in patched_qparams.items():
            setattr(Float8BlockScaling, name, qparams)
        setattr(Float8BlockScaling, _PATCH_FLAG, epsilon)

        probe = Float8BlockScaling(fp8_format=Format.E4M3)
        observed = {name: getattr(probe, name).amax_epsilon for name in _QPARAM_FIELDS}
        if any(value != epsilon for value in observed.values()):
            raise RuntimeError(f"fresh Float8BlockScaling instance reported {observed}")
    except Exception as exc:
        for name, qparams in original_qparams.items():
            setattr(Float8BlockScaling, name, qparams)
        if original_flag is _MISSING:
            try:
                delattr(Float8BlockScaling, _PATCH_FLAG)
            except AttributeError:
                pass
        else:
            setattr(Float8BlockScaling, _PATCH_FLAG, original_flag)
        raise RuntimeError(f"Failed to apply {_ENV_VAR}={epsilon} to Transformer Engine") from exc

    logger.info("Applied {}={} to TE Float8BlockScaling blockwise quantizers", _ENV_VAR, epsilon)
