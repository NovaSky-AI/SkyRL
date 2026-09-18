"""Tests for the /skyrl/v1/generate payload contract."""

import base64
import math
from dataclasses import dataclass

import numpy as np
import orjson
import pytest
import torch

from skyrl.backends.skyrl_train.inference_servers.generate_wire import (
    CLAMPED_LOGPROB,
    build_logprobs_content,
    decode_packed_full_logprobs,
    decode_packed_routed_experts,
    pack_full_logprobs,
    pack_routed_experts,
)


@dataclass
class _Logprob:
    logprob: float


@pytest.mark.parametrize(
    "entry",
    [
        {7: _Logprob(float("-inf"))},
        {7: _Logprob(float("inf"))},
        {7: _Logprob(float("nan"))},
        None,
        {},
        {99: _Logprob(-0.5)},  # present, but not for the sampled token
    ],
)
def test_bad_logprob_is_clamped(entry):
    assert build_logprobs_content([7], [entry]) == ([{"logprob": CLAMPED_LOGPROB}], 1)


def test_finite_logprobs_pass_through_and_count_only_bad_tokens():
    token_ids = [10, 11, 12, 13]
    resp = [{10: _Logprob(-0.25)}, {11: _Logprob(float("-inf"))}, None, {13: _Logprob(-12.3456789)}]
    content, num_clamped = build_logprobs_content(token_ids, resp)
    # Length must match token_ids: callers assert len(logprobs) == len(response_ids).
    assert [e["logprob"] for e in content] == [-0.25, CLAMPED_LOGPROB, CLAMPED_LOGPROB, -12.3456789]
    assert num_clamped == 2


def test_clamped_payload_round_trips_through_orjson():
    # orjson emits `null` for non-finite and then rejects it on the way back in,
    # so a non-finite logprob must never reach the wire.
    assert orjson.dumps({"logprob": float("-inf")}) == b'{"logprob":null}'
    content, _ = build_logprobs_content([7], [{7: _Logprob(float("-inf"))}])
    assert math.isfinite(orjson.loads(orjson.dumps(content))[0]["logprob"])


def test_empty_logprobs_input():
    assert build_logprobs_content([], []) == ([], 0)


def test_null_logprob_entry_is_clamped_not_raised():
    # An entry present but None must take the floor rather than raise AttributeError.
    assert build_logprobs_content([7], [{7: None}]) == ([{"logprob": CLAMPED_LOGPROB}], 1)


def test_packed_full_logprobs_round_trip_float32():
    rows = [
        {token_id: _Logprob(-0.25 * (token_id + 1)) for token_id in range(3)},
        {token_id: _Logprob(-1.5 - token_id) for token_id in range(3)},
    ]

    payload = pack_full_logprobs([1, 2], rows, vocab_size=3)
    decoded = decode_packed_full_logprobs(payload)

    assert payload["dtype"] == "float32"
    assert decoded.dtype == np.dtype("<f4")
    assert decoded.flags.c_contiguous
    np.testing.assert_array_equal(
        decoded,
        np.array([[-0.25, -0.5, -0.75], [-1.5, -2.5, -3.5]], dtype=np.float32),
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"data": "", "shape": [0, 3]},
        {"data": "", "shape": [0, 3], "dtype": "float64"},
        {"data": "!", "shape": [1, 1], "dtype": "float32"},
        {"data": "AAAAAAAA", "shape": [1, 1], "dtype": "float32"},
        {"data": "AAAAAA==", "shape": [True, 1], "dtype": "float32"},
        {"data": "AAAAAA==", "shape": [1, 0], "dtype": "float32"},
    ],
)
def test_decode_rejects_malformed_full_logprobs(payload):
    with pytest.raises(ValueError):
        decode_packed_full_logprobs(payload)


def test_pack_rejects_incomplete_or_nonfinite_full_logprobs():
    with pytest.raises(ValueError, match="every vocabulary token"):
        pack_full_logprobs([1], [{0: _Logprob(-1.0), 1: _Logprob(-2.0)}], vocab_size=3)

    row = {0: _Logprob(-1.0), 1: _Logprob(float("nan")), 2: _Logprob(-3.0)}
    with pytest.raises(ValueError, match="finite"):
        pack_full_logprobs([1], [row], vocab_size=3)


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
    # The coercion in pack_routed_experts must not turn the old nested-list
    # format into a valid payload.
    with pytest.raises(TypeError, match="NumPy array"):
        pack_routed_experts([[[1, 2]]])


def test_pack_accepts_torch_tensors():
    routes = torch.arange(12, dtype=torch.int64).reshape(3, 2, 2)

    decoded = decode_packed_routed_experts(pack_routed_experts(routes))

    assert decoded.dtype == np.uint8
    assert np.array_equal(decoded, routes.numpy())


def test_pack_moves_device_tensors_to_host():
    """np.asarray raises on a CUDA tensor, so packing must detach/cpu/numpy first.

    Simulated rather than GPU-gated: the coercion is duck-typed, so the call
    sequence is identical to the real CUDA path.
    """
    calls = []

    class _DeviceTensor:
        def __init__(self, array):
            self._array = array

        def detach(self):
            calls.append("detach")
            return self

        def cpu(self):
            calls.append("cpu")
            return self

        def numpy(self):
            calls.append("numpy")
            return self._array

    routes = np.arange(12, dtype=np.int64).reshape(3, 2, 2)
    decoded = decode_packed_routed_experts(pack_routed_experts(_DeviceTensor(routes)))

    assert calls == ["detach", "cpu", "numpy"]
    assert np.array_equal(decoded, routes)


@pytest.mark.parametrize("shape", [[1, 1, 1], [np.int64(1), np.int32(1), 1]])
def test_decode_accepts_numpy_integer_dims(shape):
    assert decode_packed_routed_experts({"data": "AQ==", "shape": shape, "dtype": "uint8"}).shape == (1, 1, 1)


def test_decode_rejects_incorrect_byte_count():
    with pytest.raises(ValueError, match="bytes"):
        decode_packed_routed_experts({"data": "AQ==", "shape": [2, 1, 1], "dtype": "uint8"})


@pytest.mark.parametrize(
    "payload",
    [
        {"data": "AQ==", "shape": [1, 1, 1], "dtype": "uint16"},
        {"data": "!", "shape": [1, 1, 1], "dtype": "uint8"},
        # bool is a subclass of int, so widening the dim check must not admit it.
        {"data": "AQ==", "shape": [True, 1, 1], "dtype": "uint8"},
        {"data": "AQ==", "shape": [np.bool_(True), 1, 1], "dtype": "uint8"},
        {"data": "AQ==", "shape": [1.0, 1, 1], "dtype": "uint8"},
        {"data": "AQ==", "shape": [-1, 1, 1], "dtype": "uint8"},
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
