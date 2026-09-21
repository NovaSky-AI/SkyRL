"""Compressed byte helpers for disk-based checkpoint weight sync."""

from __future__ import annotations

from typing import Optional

import torch
import zstandard as zstd


def compress_bytes(data: bytes, level: int = 3) -> bytes:
    return zstd.ZstdCompressor(level=level).compress(data)


def decompress_bytes(data: bytes, expected_size: Optional[int] = None) -> bytes:
    result = zstd.ZstdDecompressor().decompress(data, max_output_size=expected_size or 0)
    if expected_size is not None and len(result) != expected_size:
        raise ValueError(f"Expected decompressed size {expected_size}, got {len(result)}")
    return result


def bytes_to_uint8_tensor(data: bytes) -> torch.Tensor:
    # bytearray gives torch owned mutable storage without an intermediate Python list.
    return torch.frombuffer(bytearray(data), dtype=torch.uint8)


def uint8_tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    if tensor.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 tensor, got {tensor.dtype}")
    return tensor.detach().cpu().contiguous().numpy().tobytes()
