"""Serialized weight strategy registration and dispatch."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import (
    LEGACY_BATCHED_MOE_PREFIX,
    SERIALIZED_WEIGHT_PREFIX,
    SerializedWeightStrategy,
)
from .model_specs import MODEL_QUANTIZATION_SPECS

StrategyFactory = Callable[[], SerializedWeightStrategy]

_SERIALIZED_WEIGHT_STRATEGIES: dict[str, StrategyFactory] = {}
_RECEIVER_STRATEGIES: dict[str, SerializedWeightStrategy] = {}


def register_serialized_weight_strategy(mode: str, factory: StrategyFactory) -> None:
    if mode in _SERIALIZED_WEIGHT_STRATEGIES:
        raise ValueError(f"Duplicate serialized weight strategy: {mode}")
    factory_mode = getattr(factory, "mode", mode)
    if factory_mode != mode:
        raise ValueError(f"Strategy factory for {mode!r} declares mode={factory_mode!r}")
    _SERIALIZED_WEIGHT_STRATEGIES[mode] = factory


def get_serialized_weight_strategy(mode: str) -> SerializedWeightStrategy:
    try:
        factory = _SERIALIZED_WEIGHT_STRATEGIES[mode]
    except KeyError as exc:
        supported = sorted(_SERIALIZED_WEIGHT_STRATEGIES)
        raise ValueError(f"Unsupported serialized weight mode {mode!r}; supported: {supported}") from exc
    return factory()


def serialized_weight_modes() -> tuple[str, ...]:
    return tuple(_SERIALIZED_WEIGHT_STRATEGIES)


def should_use_serialized_weights(mode: str | None) -> bool:
    return mode in _SERIALIZED_WEIGHT_STRATEGIES


def get_serialized_weight_sync_mode(inference_config: Any) -> str | None:
    mode = getattr(inference_config, "serialized_weight_sync_mode", None)
    legacy_mode = getattr(inference_config, "fp8_weight_sync_mode", None)
    if mode is not None and legacy_mode is not None:
        raise ValueError("Set only one of serialized_weight_sync_mode and fp8_weight_sync_mode")
    return mode if mode is not None else legacy_mode


def parse_serialized_wire_name(wire_name: str) -> tuple[str | None, str] | None:
    if wire_name.startswith(LEGACY_BATCHED_MOE_PREFIX):
        return None, wire_name.removeprefix(LEGACY_BATCHED_MOE_PREFIX)
    if not wire_name.startswith(SERIALIZED_WEIGHT_PREFIX):
        return None
    payload = wire_name.removeprefix(SERIALIZED_WEIGHT_PREFIX)
    mode, separator, checkpoint_name = payload.partition(":")
    if not separator or not mode or not checkpoint_name:
        raise ValueError(f"Invalid serialized weight wire name {wire_name!r}")
    return mode, checkpoint_name


def resolve_serialized_receiver_target(mode: str | None, checkpoint_name: str) -> tuple[str, str] | None:
    modes = (mode,) if mode is not None else serialized_weight_modes()
    matches = {
        target
        for candidate_mode in modes
        if (
            target := _get_receiver_strategy(candidate_mode).receiver_target(
                checkpoint_name,
                MODEL_QUANTIZATION_SPECS,
            )
        )
        is not None
    }
    if len(matches) > 1:
        raise ValueError(f"Ambiguous serialized weight receiver target for {checkpoint_name!r}")
    return next(iter(matches), None)


def _get_receiver_strategy(mode: str) -> SerializedWeightStrategy:
    strategy = _RECEIVER_STRATEGIES.get(mode)
    if strategy is None:
        strategy = get_serialized_weight_strategy(mode)
        _RECEIVER_STRATEGIES[mode] = strategy
    return strategy


def strategy_uses_block_scale_runtime_contract(mode: str | None) -> bool:
    if mode is None:
        return False
    return get_serialized_weight_strategy(mode).uses_block_scale_runtime_contract


def _register_builtin_strategies() -> None:
    from .blockwise_fp8 import BlockwiseFp8Strategy
    from .mxfp8 import Mxfp8Strategy

    register_serialized_weight_strategy(BlockwiseFp8Strategy.mode, BlockwiseFp8Strategy)
    register_serialized_weight_strategy(Mxfp8Strategy.mode, Mxfp8Strategy)


_register_builtin_strategies()
