"""Compact routed-expert HTTP payloads."""

import math
from typing import Any

import numpy as np
import pybase64

from skyrl.utils.routed_experts import (
    ROUTED_EXPERT_DTYPES,
    RoutedExpertIndices,
    compact_routed_expert_indices,
)

_DTYPES = {dtype.name: dtype for dtype in ROUTED_EXPERT_DTYPES}


def pack_routed_experts(routed_experts: RoutedExpertIndices) -> dict[str, Any]:
    compact = compact_routed_expert_indices(routed_experts)
    return {
        "data": pybase64.b64encode(memoryview(compact)).decode("ascii"),
        "shape": list(compact.shape),
        "dtype": compact.dtype.name,
    }


def decode_packed_routed_experts(payload: dict[str, Any]) -> RoutedExpertIndices:
    if not isinstance(payload, dict):
        raise TypeError("packed routed expert indices must be an object")
    try:
        dtype = _DTYPES[payload["dtype"]]
        shape = tuple(payload["shape"])
        data = pybase64.b64decode_as_bytearray(payload["data"], validate=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid packed routed_experts payload") from exc
    if len(shape) != 3 or any(type(dim) is not int or dim < 0 for dim in shape):
        raise ValueError(f"invalid packed routed_experts shape: {shape}")
    expected_size = math.prod(shape) * dtype.itemsize
    if len(data) != expected_size:
        raise ValueError(f"packed routed_experts has {len(data)} bytes, expected {expected_size}")
    decoded = np.frombuffer(data, dtype=dtype).reshape(shape)
    compact = compact_routed_expert_indices(decoded)
    if compact.dtype != dtype:
        raise ValueError(f"packed routed_experts uses non-canonical dtype {dtype.name}; expected {compact.dtype.name}")
    return compact
