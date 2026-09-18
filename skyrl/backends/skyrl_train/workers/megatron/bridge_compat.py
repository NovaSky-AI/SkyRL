"""Compatibility helpers for Megatron-Bridge / megatron-core API drift.

SkyRL's Megatron worker targets current AutoBridge APIs. Users may install a
different Bridge or core revision (``MCORE_REV`` / ``BRIDGE_REV``). These helpers
prefer the current APIs and fall back to older names when present.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

logger = logging.getLogger(__name__)


def megatron_core_version() -> str:
    try:
        import megatron.core as mc

        return str(getattr(mc, "__version__", "unknown"))
    except Exception as exc:  # pragma: no cover - import-time environment
        return f"unavailable ({exc})"


def inspect_bridge_capabilities() -> dict[str, Any]:
    """Report which Bridge/core entry points exist in the active environment."""
    caps: dict[str, Any] = {
        "megatron_core_version": megatron_core_version(),
        "auto_bridge": False,
        "from_hf_pretrained": False,
        "from_hf": False,
        "from_pretrained": False,
        "canonical_lora": False,
        "lora": False,
        "initialize_model_parallel_params": [],
    }
    try:
        from megatron.bridge import AutoBridge

        caps["auto_bridge"] = True
        caps["from_hf_pretrained"] = hasattr(AutoBridge, "from_hf_pretrained")
        caps["from_hf"] = hasattr(AutoBridge, "from_hf")
        caps["from_pretrained"] = hasattr(AutoBridge, "from_pretrained")
    except Exception as exc:
        caps["auto_bridge_error"] = str(exc)

    try:
        from megatron.bridge.peft.canonical_lora import CanonicalLoRA  # noqa: F401

        caps["canonical_lora"] = True
    except Exception:
        caps["canonical_lora"] = False

    try:
        from megatron.bridge.peft.lora import LoRA  # noqa: F401

        caps["lora"] = True
    except Exception:
        caps["lora"] = False

    try:
        from megatron.core import parallel_state as mpu

        params = inspect.signature(mpu.initialize_model_parallel).parameters
        caps["initialize_model_parallel_params"] = list(params)
    except Exception as exc:
        caps["initialize_model_parallel_error"] = str(exc)

    return caps


def _filter_kwargs(func: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def load_auto_bridge(source: str, **kwargs: Any):
    """Load an AutoBridge, accepting current and older constructor names."""
    from megatron.bridge import AutoBridge

    errors: list[str] = []
    for name in ("from_hf_pretrained", "from_hf", "from_pretrained"):
        factory = getattr(AutoBridge, name, None)
        if factory is None:
            continue
        call_kwargs = _filter_kwargs(factory, kwargs)
        try:
            logger.info("Loading Megatron AutoBridge via %s (source=%s)", name, source)
            return factory(source, **call_kwargs)
        except TypeError as exc:
            errors.append(f"{name}{call_kwargs}: {exc}")
            try:
                return factory(source)
            except TypeError as inner:
                errors.append(f"{name}(): {inner}")
                continue
    caps = inspect_bridge_capabilities()
    raise RuntimeError(
        "This Megatron-Bridge build does not expose a usable HF loader "
        f"(tried from_hf_pretrained/from_hf/from_pretrained). errors={errors} caps={caps}"
    )


def import_lora_classes() -> tuple[Any, Any]:
    """Return (CanonicalLoRA, LoRA), using LoRA as a CanonicalLoRA fallback."""
    lora_cls = None
    canonical_cls = None
    try:
        from megatron.bridge.peft.lora import LoRA as _LoRA

        lora_cls = _LoRA
    except Exception:
        lora_cls = None
    try:
        from megatron.bridge.peft.canonical_lora import CanonicalLoRA as _Canonical

        canonical_cls = _Canonical
    except Exception:
        canonical_cls = lora_cls
    return canonical_cls, lora_cls


def initialize_model_parallel_compat(**kwargs: Any) -> None:
    """Call mpu.initialize_model_parallel, dropping kwargs this core build rejects."""
    from megatron.core import parallel_state as mpu

    mpu.initialize_model_parallel(**_filter_kwargs(mpu.initialize_model_parallel, kwargs))
