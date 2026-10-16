"""Trainable-parameter selection for Megatron vision-language models.

``megatron_config.freeze_language_model`` / ``freeze_vision_model`` /
``freeze_vision_projection`` are forwarded to the Megatron-Bridge model
provider, whose ``provide()`` calls the model's own ``freeze()``: every Bridge
VLM provider (Qwen2.5/3/3.5-VL, GLM-4.5V, Kimi K2.5-VL, Gemma 3/4-VL,
Nemotron-VL, Ernie 4.5-VL) carries the three fields and maps them onto its own
module names, so nothing here needs per-family parameter names.

With LoRA, Megatron-Bridge's PEFT freezes every base parameter before adding
adapters. :func:`make_keep_trainable_hooks` records which base parameters the
freeze flags left trainable before the LoRA pre-wrap hook runs and re-enables
them after it, so "adapters plus a fully trained projector" trains both.

Kept free of Megatron imports so the helpers are unit-testable on CPU.
"""

from collections import OrderedDict
from typing import Callable, Iterable, List, Optional, Union

import torch.nn as nn

VLM_FREEZE_FIELDS = ("freeze_language_model", "freeze_vision_model", "freeze_vision_projection")

_WRAPPER_PREFIXES = ("module.",)


def vlm_freeze_settings(megatron_config) -> dict:
    """The three VLM freeze flags from a ``MegatronConfig``, by field name."""
    return {name: bool(getattr(megatron_config, name, False)) for name in VLM_FREEZE_FIELDS}


def any_vlm_freeze(megatron_config) -> bool:
    return any(vlm_freeze_settings(megatron_config).values())


def apply_vlm_freeze_to_provider(provider, megatron_config) -> bool:
    """Set the VLM freeze flags on a Megatron-Bridge model provider.

    No-op (returns False) when none of the flags is set. Otherwise raises
    ``ValueError`` if the provider does not declare the fields (a text-only
    model, or a Bridge VLM provider without freeze support), so an unsupported
    model fails at build time instead of silently training everything.
    """
    settings = vlm_freeze_settings(megatron_config)
    if not any(settings.values()):
        return False
    missing = [name for name in VLM_FREEZE_FIELDS if not hasattr(provider, name)]
    if missing:
        raise ValueError(
            f"megatron_config.{'/'.join(n for n, v in settings.items() if v)} is set, but the Megatron-Bridge "
            f"provider {type(provider).__name__} has no {missing} field(s): these flags apply to vision-language "
            "models whose Bridge provider supports freezing (e.g. Qwen-VL, GLM-4.5V, Kimi K2.5-VL, Gemma-VL)."
        )
    for name, value in settings.items():
        setattr(provider, name, value)
    return True


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


def make_keep_trainable_hooks(
    on_restore: Optional[Callable[[List[nn.Module]], None]] = None,
) -> tuple[Callable, Callable]:
    """Pre-wrap hooks that keep the base parameters left trainable by the freeze flags.

    Register ``record`` before the LoRA pre-wrap hook and ``restore`` after it.
    ``record`` stores the names of the parameters that require grad (per model
    chunk, after the provider's ``freeze()``); ``restore`` sets
    ``requires_grad=True`` on them again once PEFT has frozen the base, then
    calls ``on_restore`` (e.g. to refresh PEFT's ``params_to_save``). Adapter
    parameters PEFT added are left as they are.
    """
    recorded: List[set] = []

    def record(model):
        recorded.clear()
        for chunk in _as_list(model):
            recorded.append({_canonical_name(n) for n, p in chunk.named_parameters() if p.requires_grad})
        return model

    def restore(model):
        chunks = _as_list(model)
        if len(chunks) != len(recorded):
            raise RuntimeError(
                f"keep-trainable hooks saw {len(recorded)} model chunk(s) before LoRA and {len(chunks)} after"
            )
        for chunk, names in zip(chunks, recorded):
            for name, param in chunk.named_parameters():
                if _canonical_name(name) in names:
                    param.requires_grad = True
        if on_restore is not None:
            on_restore(model)
        return model

    return record, restore


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


def lora_checkpoint_state_dict(named_parameters: Iterable) -> "OrderedDict[str, object]":
    """Tensors a LoRA checkpoint must hold: adapters plus any trainable base parameter.

    A base parameter is trainable under LoRA only when the VLM freeze flags kept
    it (e.g. a fully trained vision projection next to LM adapters); it is
    saved under its own name so the ``strict=False`` load restores it.
    """
    state = OrderedDict()
    for name, param in named_parameters:
        if ".adapter" in name.lower() or param.requires_grad:
            state[name] = param.data
    return state


def _fmt(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)
