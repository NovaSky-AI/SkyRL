import base64

import numpy as np
import pytest

from skyrl.backends.skyrl_train.inference_servers.routed_experts_wire import (
    decode_packed_routed_experts,
    pack_routed_experts,
)
from skyrl.utils.routed_experts import compact_routed_expert_indices


@pytest.mark.parametrize(
    "routes,expected_dtype",
    [
        (np.arange(12).reshape(3, 2, 2), "uint8"),
        (np.array([[[2**8 - 1]]]), "uint8"),
        (np.array([[[0, 2**8]]]), "int16"),
        (np.array([[[0, 2**15 - 1]]]), "int16"),
        (np.array([[[0, 2**15]]]), "int32"),
        (np.array([[[0, 2**31 - 1]]], dtype=np.int64), "int32"),
        (np.empty((0, 2, 2), dtype=np.int64), "uint8"),
        (np.arange(24).reshape(6, 2, 2)[::2], "uint8"),
    ],
)
def test_packed_routed_experts_round_trip(routes, expected_dtype):
    payload = pack_routed_experts(routes)
    decoded = decode_packed_routed_experts(payload)

    assert payload["dtype"] == expected_dtype
    assert decoded.dtype.name == expected_dtype
    assert decoded.flags.c_contiguous
    assert np.array_equal(decoded, routes)


def test_packed_routed_experts_uses_raw_base64():
    assert pack_routed_experts(np.array([[[1, 2, 3]]]))["data"] == "AQID"


@pytest.mark.parametrize(
    "routes",
    [np.array([1, 2]), np.array([[[-1]]]), np.array([[[2**31]]], dtype=np.uint64)],
)
def test_pack_rejects_invalid_routes(routes):
    with pytest.raises(ValueError):
        pack_routed_experts(routes)


def test_pack_rejects_nested_lists():
    with pytest.raises(TypeError, match="NumPy array"):
        pack_routed_experts([[[1, 2]]])


def test_compaction_makes_read_only_arrays_writable():
    routes = np.arange(12, dtype=np.uint8).reshape(3, 2, 2)
    routes.flags.writeable = False

    compact = compact_routed_expert_indices(routes)

    assert compact.dtype == np.uint8
    assert compact.flags.c_contiguous
    assert compact.flags.writeable


def test_decode_rejects_incorrect_byte_count():
    with pytest.raises(ValueError, match="bytes"):
        decode_packed_routed_experts({"data": "AQ==", "shape": [2, 1, 1], "dtype": "uint8"})


@pytest.mark.parametrize(
    "payload",
    [
        {"data": "AQ==", "shape": [1, 1, 1], "dtype": "uint16"},
        {"data": "!", "shape": [1, 1, 1], "dtype": "uint8"},
        {"data": "AQ==", "shape": [True, 1, 1], "dtype": "uint8"},
    ],
)
def test_decode_rejects_malformed_payloads(payload):
    with pytest.raises(ValueError):
        decode_packed_routed_experts(payload)


def test_decode_rejects_noncanonical_dtype():
    routes = np.array([[[300]]], dtype=np.int32)
    payload = {
        "data": base64.b64encode(routes.tobytes()).decode("ascii"),
        "shape": [1, 1, 1],
        "dtype": "int32",
    }

    with pytest.raises(ValueError, match="non-canonical dtype"):
        decode_packed_routed_experts(payload)
