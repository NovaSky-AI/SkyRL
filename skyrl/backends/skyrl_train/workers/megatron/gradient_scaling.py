import math
from typing import Any


def scale_gradients(model_chunks: list[Any], scale: float) -> None:
    """Scale reduced DDP buffers once, before clipping and updating Adam moments."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("gradient_scale must be finite and positive")
    if scale != 1.0:
        for chunk in model_chunks:
            for buffer in list(chunk.buffers) + list(chunk.expert_parallel_buffers):
                buffer.grad_data.mul_(scale)
