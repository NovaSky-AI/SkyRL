"""Packed wire format for dense sampler-support vocab IDs."""

from typing import Literal, TypedDict

import numpy as np
import pybase64


class PackedSampleSupportSet(TypedDict):
    data: str
    shape: list[int]
    dtype: Literal["int32"]


_DTYPE = np.dtype("int32")


def _validated_dense_support(values: np.ndarray) -> np.ndarray:
    if not isinstance(values, np.ndarray):
        raise TypeError("sample support must be a NumPy array")
    if values.ndim != 2:
        raise ValueError(f"sample support must have shape [tokens, top_k], got {values.shape!r}")
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("sample support must contain integer vocab IDs")
    if values.size:
        minimum = int(values.min())
        maximum = int(values.max())
        if minimum < -1 or maximum > np.iinfo(np.int32).max:
            raise ValueError("sample support IDs must be -1 padding or non-negative int32 vocab IDs")
        if np.any((values[:, :-1] == -1) & (values[:, 1:] >= 0)):
            raise ValueError("sample support padding must be trailing -1 values")
    return np.ascontiguousarray(values, dtype=_DTYPE)


def encode_sample_support_set(values: np.ndarray) -> PackedSampleSupportSet:
    dense = _validated_dense_support(values)
    return {
        "data": pybase64.b64encode(memoryview(dense)).decode("ascii"),
        "shape": list(dense.shape),
        "dtype": "int32",
    }


def decode_sample_support_set(payload: PackedSampleSupportSet) -> np.ndarray:
    if not isinstance(payload, dict):
        raise TypeError("packed sample support must be a dictionary")
    try:
        data_b64 = payload["data"]
        raw_shape = payload["shape"]
        dtype_name = payload["dtype"]
    except KeyError as exc:
        raise ValueError("packed sample support is missing data/shape/dtype") from exc
    if dtype_name != "int32":
        raise ValueError("packed sample support dtype must be int32")
    if not isinstance(data_b64, str):
        raise ValueError("packed sample support data must be a base64 string")
    if not isinstance(raw_shape, list) or len(raw_shape) != 2:
        raise ValueError("packed sample support shape must be [tokens, top_k]")
    if any(not isinstance(dim, int) or isinstance(dim, bool) for dim in raw_shape):
        raise ValueError("packed sample support shape must contain integers")
    if any(dim < 0 for dim in raw_shape):
        raise ValueError("packed sample support shape must be non-negative")
    try:
        data = pybase64.b64decode_as_bytearray(data_b64, validate=True) if data_b64 else bytearray()
    except ValueError as exc:
        raise ValueError("packed sample support data must be valid base64") from exc
    expected_nbytes = raw_shape[0] * raw_shape[1] * _DTYPE.itemsize
    if len(data) != expected_nbytes:
        raise ValueError(
            f"packed sample support byte-size mismatch: data={len(data)} bytes, expected={expected_nbytes}"
        )
    dense = np.frombuffer(data, dtype=_DTYPE).reshape(raw_shape)
    return _validated_dense_support(dense)
