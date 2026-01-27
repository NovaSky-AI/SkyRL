"""
Cutile-based kernel implementations for SkyRL-tx.

This package provides optimized CUDA kernels using NVIDIA's cuTile (cuda-tile on PyPI)
for LoRA expert parallelism computation.
"""

import os

# Feature flag for cutile LoRA
USE_CUTILE_LORA = os.environ.get("TX_USE_CUTILE_LORA", "0") == "1"

# Try to import cutile implementation
if USE_CUTILE_LORA:
    try:
        from .cutile_lora import cutile_ragged_dot

        CUTILE_AVAILABLE = True
    except ImportError as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"Cutile not available, falling back to ragged_dot: {e}")
        CUTILE_AVAILABLE = False
        cutile_ragged_dot = None
else:
    CUTILE_AVAILABLE = False
    cutile_ragged_dot = None

__all__ = [
    "USE_CUTILE_LORA",
    "CUTILE_AVAILABLE",
    "cutile_ragged_dot",
]
