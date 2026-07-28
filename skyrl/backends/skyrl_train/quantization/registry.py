"""Built-in serialized weight strategies."""

from __future__ import annotations

from typing import Literal, TypeAlias

from .base import SerializedWeightStrategy
from .blockwise_fp8 import SERIALIZED_BLOCKWISE_FP8, BlockwiseFp8Strategy
from .mxfp8 import SERIALIZED_MXFP8, Mxfp8Strategy

SerializedWeightMode: TypeAlias = Literal["serialized_blockwise", "serialized_mxfp8"]

SERIALIZED_WEIGHT_STRATEGIES: dict[SerializedWeightMode, type[SerializedWeightStrategy]] = {
    SERIALIZED_BLOCKWISE_FP8: BlockwiseFp8Strategy,
    SERIALIZED_MXFP8: Mxfp8Strategy,
}


def get_serialized_weight_strategy(mode: str) -> SerializedWeightStrategy:
    """Construct the serialized weight strategy registered for a mode."""

    try:
        strategy_cls = SERIALIZED_WEIGHT_STRATEGIES[mode]
    except KeyError as exc:
        supported = sorted(SERIALIZED_WEIGHT_STRATEGIES)
        raise ValueError(f"Unsupported serialized weight mode {mode!r}; supported: {supported}") from exc
    return strategy_cls()
