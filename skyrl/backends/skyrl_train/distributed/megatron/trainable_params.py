"""Trainable-parameter selection for Megatron vision-language models.

``megatron_config.freeze_language_model`` / ``freeze_vision_model`` /
``freeze_vision_projection`` are written onto the Megatron-Bridge model
provider, whose ``provide()`` calls the model's own ``freeze()``: every Bridge
VLM provider (Qwen2.5/3/3.5-VL, GLM-4.5V, Kimi K2.5-VL, Gemma 3/4-VL,
Nemotron-VL, Ernie 4.5-VL) carries the three fields and maps them onto its own
module names, so nothing here needs per-family parameter names.

Kept free of Megatron imports so the helpers are unit-testable on CPU.
"""

from collections import OrderedDict
from typing import List, Optional, Union

import torch.nn as nn

VLM_FREEZE_FIELDS = ("freeze_language_model", "freeze_vision_model", "freeze_vision_projection")

_WRAPPER_PREFIXES = ("module.",)


def vlm_freeze_settings(megatron_config) -> dict:
    """The three VLM freeze flags from a ``MegatronConfig``, by field name."""
    return {name: bool(getattr(megatron_config, name, False)) for name in VLM_FREEZE_FIELDS}


def apply_vlm_freeze_to_provider(provider, megatron_config) -> Optional[str]:
    """Write the three VLM freeze flags from ``megatron_config`` onto a Megatron-Bridge model provider.

    All three are always written, set or not, so ``megatron_config`` is the only
    source of truth: Bridge providers carry their own defaults (Qwen3-VL MoE
    defaults ``freeze_language_model`` and ``freeze_vision_model`` to True), which
    would otherwise apply silently whenever no flag is set. A provider without the
    fields (a text-only model) is left alone, unless a flag is set, which raises
    ``ValueError`` instead of silently training everything.

    Returns a one-line summary for the log, or None when the provider has no freeze fields.
    """
    settings = vlm_freeze_settings(megatron_config)
    missing = [name for name in VLM_FREEZE_FIELDS if not hasattr(provider, name)]
    if missing:
        if not any(settings.values()):
            return None
        raise ValueError(
            f"megatron_config.{'/'.join(n for n, v in settings.items() if v)} is set, but the Megatron-Bridge "
            f"provider {type(provider).__name__} has no {missing} field(s): these flags apply to vision-language "
            "models whose Bridge provider supports freezing (e.g. Qwen-VL, GLM-4.5V, Kimi K2.5-VL, Gemma-VL)."
        )
    overridden = {name: getattr(provider, name) for name, value in settings.items() if getattr(provider, name) != value}
    for name, value in settings.items():
        setattr(provider, name, value)
    summary = ", ".join(f"{name}={value}" for name, value in settings.items())
    if overridden:
        summary += " (provider defaults overridden: " + ", ".join(f"{n}={v}" for n, v in overridden.items()) + ")"
    return summary


def _as_list(model: Union[nn.Module, List[nn.Module]]) -> List[nn.Module]:
    return list(model) if isinstance(model, (list, tuple)) else [model]


def _canonical_name(name: str) -> str:
    """Parameter name with DDP / Float16Module wrapper prefixes stripped."""
    changed = True
    while changed:
        changed = False
        for prefix in _WRAPPER_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                changed = True
    return name


def summarize_trainable_parameters(model: Union[nn.Module, List[nn.Module]], depth: int = 2) -> List[str]:
    """Human-readable trainable / total parameter counts, overall and per module prefix.

    Prefixes are the first ``depth`` components of each parameter name after
    wrapper prefixes are stripped (e.g. ``language_model.decoder``,
    ``vision_model.merger``), in first-seen order. Counts are local to the
    rank (its tensor/pipeline shard).
    """
    groups: "OrderedDict[str, list[int]]" = OrderedDict()
    trainable = total = 0
    for chunk in _as_list(model):
        for name, param in chunk.named_parameters():
            n = param.numel()
            prefix = ".".join(_canonical_name(name).split(".")[:depth])
            counts = groups.setdefault(prefix, [0, 0])
            counts[1] += n
            total += n
            if param.requires_grad:
                counts[0] += n
                trainable += n
    pct = 100.0 * trainable / total if total else 0.0
    lines = [f"trainable parameters: {_fmt(trainable)} of {_fmt(total)} ({pct:.2f}%)"]
    for prefix, (t, n) in groups.items():
        state = "trainable" if t == n else ("frozen" if t == 0 else "partly trainable")
        lines.append(f"  {prefix}: {_fmt(t)} / {_fmt(n)} {state}")
    return lines


def _fmt(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)
