"""Utilities for training with persistent Transformer Engine FP8 parameters."""

from collections.abc import Mapping
from typing import Any

import torch

from skyrl.backends.skyrl_train.distributed.megatron.packing_utils import is_fp8_enabled


def is_fp8_param_enabled(transformer_config_kwargs: Mapping[str, Any]) -> bool:
    """Read persistent-FP8 enablement from SkyRL's dictionary config."""
    return is_fp8_enabled(transformer_config_kwargs.get("fp8_param", False))


def _copy_model_shards_to_main_params(
    megatron_optimizer: Any,
    state_dict: Mapping[str, Any],
) -> int:
    """Reload every MCore master shard after an HF import.

    ``DistributedOptimizer._copy_model_params_to_main_params`` has the same
    state-dict-aware behavior for normal optimizers.  Its
    ``HybridDeviceOptimizer`` branch returns before updating distributed FP32
    masters, so CPU-offloaded runs need the equivalent copy here.

    ``state_dict`` contains Megatron-Bridge's exact converted checkpoint tensors.
    That avoids seeding FP32 masters from dequantized FP8 compute weights and also
    reloads ordinary BF16 parameters such as embeddings and normalization weights.
    """
    model_groups = getattr(megatron_optimizer, "model_float16_groups", None)
    main_groups = getattr(megatron_optimizer, "shard_fp32_from_float16_groups", None)
    model_fp32_groups = getattr(megatron_optimizer, "model_fp32_groups", None)
    main_fp32_groups = getattr(megatron_optimizer, "shard_fp32_groups", None)
    get_range = getattr(megatron_optimizer, "_get_model_param_range_map", None)
    if model_groups is None or main_groups is None or not callable(get_range):
        raise TypeError(
            "Persistent FP8 parameter training with CPU optimizer offload requires "
            "Megatron DistributedOptimizer shard metadata."
        )

    build_state_dict_map = getattr(megatron_optimizer, "_build_model_param_to_state_dict_param_map", None)
    if not callable(build_state_dict_map):
        raise TypeError(
            "Persistent FP8 parameter training requires Megatron's "
            "_build_model_param_to_state_dict_param_map() to reload exact checkpoint masters."
        )
    state_dict_params = build_state_dict_map(state_dict)

    group_pairs = [(model_groups, main_groups)]
    if model_fp32_groups is not None and main_fp32_groups is not None:
        group_pairs.append((model_fp32_groups, main_fp32_groups))

    copied = 0
    for grouped_model_params, grouped_main_params in group_pairs:
        for model_group, main_group in zip(grouped_model_params, grouped_main_params):
            for model_param, main_param in zip(model_group, main_group):
                if main_param is None:
                    continue

                param_range = get_range(model_param)["param"]
                if param_range.size != main_param.numel():
                    raise RuntimeError(
                        "Persistent FP8 master-shard range does not match its FP32 master tensor: "
                        f"{param_range.size=} {main_param.numel()=}."
                    )

                source_param = state_dict_params[model_param]
                source = source_param.detach().reshape(-1)[param_range.start : param_range.end]
                main_param.data.copy_(source.to(device=main_param.device, dtype=main_param.dtype))
                copied += 1
    return copied


def _sync_hybrid_device_optimizer_masters(hybrid_optimizer: Any) -> int:
    """Synchronize HybridDeviceOptimizer's CPU/secondary master copies.

    CPU-offloaded parameters are cloned before SkyRL imports the HF checkpoint.
    MCore's public reload helper leaves those clones unchanged when the exposed
    distributed shards are already FP32, so the first AdamW step otherwise
    writes pre-import values back into the persistent FP8 model.
    """
    copied = 0
    for cpu_param, gpu_param in getattr(hybrid_optimizer, "cpu_copys_map_gpu_param", {}).items():
        cpu_param.data.copy_(
            gpu_param.detach().to(device=cpu_param.device, dtype=cpu_param.dtype),
            non_blocking=False,
        )
        copied += 1

    for param, fp32_param in getattr(hybrid_optimizer, "param_to_fp32_param", {}).items():
        fp32_param.data.copy_(
            param.detach().to(device=fp32_param.device, dtype=fp32_param.dtype),
            non_blocking=False,
        )
        copied += 1
    return copied


def _uses_hybrid_device_optimizer(megatron_optimizer: Any) -> bool:
    """Identify MCore's CPU-offload optimizer without importing it eagerly."""
    optimizer = getattr(megatron_optimizer, "optimizer", None)
    return hasattr(optimizer, "cpu_copys_map_gpu_param") and hasattr(optimizer, "param_to_fp32_param")


def initialize_fp8_param_optimizer_masters(
    optimizer: Any,
    *,
    fp8_param: bool,
    fp8_param_gather: bool,
    state_dict: Mapping[str, Any] | None = None,
) -> int:
    """Seed FP32 optimizer masters from the exact imported checkpoint state.

    Transformer Engine creates persistent FP8 GEMM weights during model
    construction. Megatron-Bridge intentionally skips random initialization
    before importing Hugging Face weights.  ``state_dict`` is the bridge's
    per-rank, unquantized conversion result and is therefore the only exact
    source for optimizer masters after that import.  CPU-offloaded
    HybridDeviceOptimizer keeps an additional CPU copy of every master, so
    both the MCore shard and that copy must be seeded.
    """
    if not fp8_param:
        return 0
    if not fp8_param_gather:
        raise ValueError(
            "Persistent FP8 parameters require ddp_config.fp8_param_gather=true "
            "so updated FP32 master weights are requantized into FP8 compute weights."
        )
    if state_dict is None:
        raise ValueError(
            "Persistent FP8 optimizer masters require Megatron-Bridge's exact unquantized checkpoint state."
        )

    optimizers = getattr(optimizer, "chained_optimizers", None)
    if optimizers is None:
        optimizers = [optimizer]

    initialized = 0
    with torch.no_grad():
        for megatron_optimizer in optimizers:
            if _uses_hybrid_device_optimizer(megatron_optimizer):
                copied = _copy_model_shards_to_main_params(megatron_optimizer, state_dict)
                if copied == 0:
                    raise RuntimeError(
                        "Persistent FP8 parameter training did not find any model shards "
                        "to seed into CPU-offloaded optimizer masters."
                    )
                _sync_hybrid_device_optimizer_masters(megatron_optimizer.optimizer)
            else:
                reload_main_params = getattr(megatron_optimizer, "_copy_model_params_to_main_params", None)
                if not callable(reload_main_params):
                    raise TypeError(
                        "Persistent FP8 parameter training requires a Megatron optimizer "
                        "with _copy_model_params_to_main_params()."
                    )
                reload_main_params(state_dict=state_dict)
            initialized += 1
    return initialized
