"""
Configuration for cutile kernel parameters.

Tile sizes and other kernel configuration options.
"""

from dataclasses import dataclass
import os


@dataclass
class CutileConfig:
    """Configuration for cutile LoRA kernels."""

    # Tile sizes (must be powers of 2 for efficient computation)
    tile_m: int = 128  # M dimension (rows/tokens)
    tile_n: int = 128  # N dimension (columns/output features)
    tile_k: int = 64  # K dimension (inner/reduction)

    # Block swizzling for cache locality
    group_size_m: int = 8

    @classmethod
    def from_env(cls):
        """Create config from environment variables."""
        return cls(
            tile_m=int(os.environ.get("TX_CUTILE_TILE_M", 128)),
            tile_n=int(os.environ.get("TX_CUTILE_TILE_N", 128)),
            tile_k=int(os.environ.get("TX_CUTILE_TILE_K", 64)),
            group_size_m=int(os.environ.get("TX_CUTILE_GROUP_SIZE_M", 8)),
        )


# Global config instance
config = CutileConfig.from_env()
