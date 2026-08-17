"""Built-in serialized quantization strategies."""

from __future__ import annotations

from functools import cache

from .base import QuantizationStrategy
from .blockwise_fp8 import BLOCKWISE_FP8, BlockwiseFp8Strategy
from .mxfp8 import MXFP8, Mxfp8ExpertStrategy

SERIALIZED_WEIGHT_STRATEGIES: dict[str, type[QuantizationStrategy]] = {
    BLOCKWISE_FP8: BlockwiseFp8Strategy,
    MXFP8: Mxfp8ExpertStrategy,
}


@cache
def get_serialized_weight_strategy(mode: str) -> QuantizationStrategy:
    """Construct the strategy registered for a serialized weight mode."""

    try:
        strategy_cls = SERIALIZED_WEIGHT_STRATEGIES[mode]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported serialized weight mode {mode!r}; supported: {sorted(SERIALIZED_WEIGHT_STRATEGIES)}"
        ) from exc
    return strategy_cls()
