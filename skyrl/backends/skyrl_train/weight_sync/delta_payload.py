"""Compressed delta payload helpers for disk-based checkpoint weight sync."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import zstandard as zstd


def dtype_to_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def dtype_from_name(name: str) -> torch.dtype:
    normalized = name.removeprefix("torch.")
    if not hasattr(torch, normalized):
        raise ValueError(f"Unknown torch dtype: {name}")
    dtype = getattr(torch, normalized)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Expected torch dtype name, got: {name}")
    return dtype


def tensor_to_bytes_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return a CPU uint8 tensor that preserves the input tensor's exact bytes."""
    return tensor.detach().cpu().contiguous().view(torch.uint8).reshape(-1)


def bytes_tensor_to_tensor(data: torch.Tensor, dtype: torch.dtype, shape: list[int]) -> torch.Tensor:
    """Reinterpret a flat uint8 CPU tensor as ``dtype`` and ``shape``."""
    if data.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 byte tensor, got {data.dtype}")
    element_size = torch.empty((), dtype=dtype).element_size()
    expected_bytes = element_size
    for dim in shape:
        expected_bytes *= dim
    if data.numel() != expected_bytes:
        raise ValueError(f"Expected {expected_bytes} bytes for {dtype} tensor with shape {shape}, got {data.numel()}")
    return data.contiguous().view(dtype).reshape(shape)


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


@dataclass
class DeltaPayload:
    """A sparse/dense delta payload with exact zstd compression metadata.

    ``values`` stores either raw tensor bytes or compressed bytes. ``positions``
    is retained for compatibility with the existing sparse payload abstraction;
    disk checkpoint deltas use dense XOR bytes and leave it as ``None``.
    """

    values: torch.Tensor
    positions: Optional[torch.Tensor] = None
    is_compressed: bool = False
    values_dtype: Optional[str] = None
    values_shape: Optional[list[int]] = None
    positions_dtype: Optional[str] = None
    positions_shape: Optional[list[int]] = None
    values_uncompressed_num_bytes: Optional[int] = None
    positions_uncompressed_num_bytes: Optional[int] = None

    @property
    def is_seed(self) -> bool:
        return self.positions is None

    def compress(self) -> "DeltaPayload":
        """Compress values and positions losslessly with zstd."""
        if self.is_compressed:
            return self

        values_bytes = uint8_tensor_to_bytes(tensor_to_bytes_tensor(self.values))
        compressed_values = compress_bytes(values_bytes)

        compressed_positions = None
        positions_dtype = None
        positions_shape = None
        positions_num_bytes = None
        if self.positions is not None:
            positions_bytes = uint8_tensor_to_bytes(tensor_to_bytes_tensor(self.positions))
            compressed_positions = bytes_to_uint8_tensor(compress_bytes(positions_bytes))
            positions_dtype = dtype_to_name(self.positions.dtype)
            positions_shape = list(self.positions.shape)
            positions_num_bytes = len(positions_bytes)

        return DeltaPayload(
            values=bytes_to_uint8_tensor(compressed_values),
            positions=compressed_positions,
            is_compressed=True,
            values_dtype=dtype_to_name(self.values.dtype),
            values_shape=list(self.values.shape),
            positions_dtype=positions_dtype,
            positions_shape=positions_shape,
            values_uncompressed_num_bytes=len(values_bytes),
            positions_uncompressed_num_bytes=positions_num_bytes,
        )

    def recover(self) -> "DeltaPayload":
        """Recover the uncompressed payload exactly."""
        if not self.is_compressed:
            return self
        if self.values_dtype is None or self.values_shape is None or self.values_uncompressed_num_bytes is None:
            raise ValueError("Compressed payload is missing values metadata")

        values_bytes = decompress_bytes(uint8_tensor_to_bytes(self.values), self.values_uncompressed_num_bytes)
        values = bytes_tensor_to_tensor(
            bytes_to_uint8_tensor(values_bytes),
            dtype_from_name(self.values_dtype),
            self.values_shape,
        ).clone()

        positions = None
        if self.positions is not None:
            if (
                self.positions_dtype is None
                or self.positions_shape is None
                or self.positions_uncompressed_num_bytes is None
            ):
                raise ValueError("Compressed payload is missing positions metadata")
            positions_bytes = decompress_bytes(
                uint8_tensor_to_bytes(self.positions), self.positions_uncompressed_num_bytes
            )
            positions = bytes_tensor_to_tensor(
                bytes_to_uint8_tensor(positions_bytes),
                dtype_from_name(self.positions_dtype),
                self.positions_shape,
            ).clone()

        return DeltaPayload(values=values, positions=positions, is_compressed=False)
