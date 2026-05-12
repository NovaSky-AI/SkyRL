"""Translate Tinker LoRA options into SkyRL-Train LoRA target modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from skyrl.tinker import types

FSDP_ATTN_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "query_key_value",
    "attn.c_attn",
    "attn.c_proj",
)
FSDP_MLP_TARGET_MODULES = (
    "gate_proj",
    "up_proj",
    "down_proj",
    "fc1",
    "fc2",
    "c_fc",
    "mlp.c_proj",
)
FSDP_UNEMBED_TARGET_MODULES = (
    "lm_head",
    "embed_out",
    "output_projection",
)

MEGATRON_LORA_ATTN_TARGET_MODULES = ("linear_qkv", "linear_proj")
MEGATRON_LORA_MLP_TARGET_MODULES = ("linear_fc1", "linear_fc2")
MEGATRON_CANONICAL_LORA_ATTN_TARGET_MODULES = ("linear_q", "linear_k", "linear_v", "linear_proj")
MEGATRON_CANONICAL_LORA_MLP_TARGET_MODULES = ("linear_fc1_up", "linear_fc1_gate", "linear_fc2")
MEGATRON_UNEMBED_TARGET_MODULES = ("output_layer",)


@dataclass(frozen=True)
class ResolvedSkyRLTrainLoraConfig:
    target_modules: str | list[str]
    exclude_modules: list[str] | None = None


def normalize_lora_targets(target_modules: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(target_modules, str):
        return (target_modules,)
    return tuple(target_modules)


def _dedupe_targets(target_modules: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(target_modules))


def _validate_train_targets(lora_config: types.LoraConfig) -> None:
    if lora_config.rank > 0 and not (lora_config.train_attn or lora_config.train_mlp or lora_config.train_unembed):
        raise ValueError("At least one of train_attn, train_mlp, or train_unembed must be true for LoRA rank > 0")


def resolve_skyrl_train_lora_config(
    lora_config: types.LoraConfig,
    strategy: str,
    lora_type: str = "lora",
    pipeline_parallel_size: int = 1,
) -> ResolvedSkyRLTrainLoraConfig:
    """Resolve Tinker LoRA train flags to the target module surface SkyRL-Train expects."""

    _validate_train_targets(lora_config)
    if lora_config.rank <= 0:
        return ResolvedSkyRLTrainLoraConfig(target_modules="all-linear")
    if lora_config.train_attn and lora_config.train_mlp and not lora_config.train_unembed:
        return ResolvedSkyRLTrainLoraConfig(target_modules="all-linear")

    if strategy in ("fsdp", "fsdp2"):
        target_modules: list[str] = []
        if lora_config.train_attn:
            target_modules.extend(FSDP_ATTN_TARGET_MODULES)
        if lora_config.train_mlp:
            target_modules.extend(FSDP_MLP_TARGET_MODULES)
        if lora_config.train_unembed:
            target_modules.extend(FSDP_UNEMBED_TARGET_MODULES)
        return ResolvedSkyRLTrainLoraConfig(target_modules=_dedupe_targets(target_modules))

    if strategy == "megatron":
        if lora_config.train_unembed and pipeline_parallel_size > 1:
            raise ValueError(
                "train_unembed=True is not supported for the Megatron SkyRL-Train backend when "
                "pipeline_model_parallel_size > 1 because output_layer only exists on the final pipeline stage"
            )
        if lora_type == "canonical_lora":
            attn_targets = MEGATRON_CANONICAL_LORA_ATTN_TARGET_MODULES
            mlp_targets = MEGATRON_CANONICAL_LORA_MLP_TARGET_MODULES
        elif lora_type == "lora":
            attn_targets = MEGATRON_LORA_ATTN_TARGET_MODULES
            mlp_targets = MEGATRON_LORA_MLP_TARGET_MODULES
        else:
            raise ValueError(f"Unsupported Megatron LoRA type: {lora_type!r}")

        target_modules = []
        if lora_config.train_attn:
            target_modules.extend(attn_targets)
        if lora_config.train_mlp:
            target_modules.extend(mlp_targets)
        if lora_config.train_unembed:
            target_modules.extend(MEGATRON_UNEMBED_TARGET_MODULES)
        return ResolvedSkyRLTrainLoraConfig(target_modules=_dedupe_targets(target_modules))

    raise ValueError(f"Unsupported SkyRL-Train strategy for Tinker LoRA config: {strategy!r}")


def skyrl_train_lora_signature(
    lora_config: types.LoraConfig,
    strategy: str,
    lora_type: str = "lora",
    pipeline_parallel_size: int = 1,
) -> tuple:
    resolved = resolve_skyrl_train_lora_config(
        lora_config,
        strategy=strategy,
        lora_type=lora_type,
        pipeline_parallel_size=pipeline_parallel_size,
    )
    return (
        int(lora_config.rank),
        int(lora_config.alpha),
        normalize_lora_targets(resolved.target_modules),
        tuple(resolved.exclude_modules or ()),
        lora_type if strategy == "megatron" else strategy,
    )
