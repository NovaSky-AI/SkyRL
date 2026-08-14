"""Tests for the /skyrl/v1/generate payload contract."""

import base64
import json
import math
from dataclasses import dataclass

import numpy as np
import orjson
import pytest
import torch

from skyrl.backends.skyrl_train.inference_servers.generate_wire import (
    CLAMPED_LOGPROB,
    PACKED_SIDE_CHANNEL_FIELDS,
    PackedArrayKey,
    PackedField,
    build_logprobs_content,
    decode_packed_routed_experts,
    load_packed_body,
    pack_ndarray,
    pack_routed_experts,
    unpack_ndarray,
)

_FLOAT32 = frozenset({np.dtype(np.float32)})
_INT16 = frozenset({np.dtype(np.int16)})


def _support_envelope(support: np.ndarray) -> dict:
    return pack_ndarray(support, allowed_dtypes=_FLOAT32)


def _body(**choice_fields) -> dict:
    return {"choices": [{"token_ids": [1, 2, 3], "finish_reason": "stop", **choice_fields}]}


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


@pytest.mark.parametrize(
    "arr,allowed_dtypes,extra",
    [
        (np.arange(6, dtype=np.float32).reshape(2, 3), _FLOAT32, None),
        (np.arange(6, dtype=np.float32).reshape(2, 3), _FLOAT32, {"prompt_start": 4, "labels": ["a", "b"]}),
        (np.arange(12, dtype=np.int16).reshape(3, 2, 2), _INT16, {"prompt_start": 0}),
        (np.empty((0, 3), dtype=np.float32), _FLOAT32, None),
    ],
)
def test_ndarray_round_trip_with_sidecar_fields(arr, allowed_dtypes, extra):
    payload = pack_ndarray(arr, allowed_dtypes=allowed_dtypes, extra=extra)
    decoded, sidecar = unpack_ndarray(payload, allowed_dtypes=allowed_dtypes, ndim=arr.ndim)

    assert np.array_equal(decoded, arr)
    assert decoded.dtype == arr.dtype
    assert decoded.flags.c_contiguous
    assert sidecar == (extra or {})


def test_packed_envelope_leads_with_data():
    # load_packed_body finds a blob by byte prefix, so `data` must stay first
    # even when sidecar fields are present.
    payload = pack_ndarray(np.zeros((2, 2), np.float32), allowed_dtypes=_FLOAT32, extra={"prompt_start": 1})

    assert list(payload) == [PackedArrayKey.DATA, PackedArrayKey.SHAPE, PackedArrayKey.DTYPE, "prompt_start"]
    assert orjson.dumps(payload).startswith(b'{"data":"')


def test_pack_routed_experts_is_byte_identical_to_the_hand_built_envelope():
    routes = np.arange(12).reshape(3, 2, 2)

    assert orjson.dumps(pack_routed_experts(routes)) == b'{"data":"AAECAwQFBgcICQoL","shape":[3,2,2],"dtype":"uint8"}'


@pytest.mark.parametrize(
    "wrap", [str, lambda data: memoryview(data.encode("ascii")), lambda data: data.encode("ascii")]
)
def test_unpack_accepts_str_and_buffers(wrap):
    support = np.arange(6, dtype=np.float32).reshape(2, 3)
    payload = dict(_support_envelope(support))
    payload[PackedArrayKey.DATA.value] = wrap(payload[PackedArrayKey.DATA.value])

    decoded, _ = unpack_ndarray(payload, allowed_dtypes=_FLOAT32, ndim=2)
    assert np.array_equal(decoded, support)


def test_pack_rejects_disallowed_dtype():
    with pytest.raises(ValueError, match="dtype"):
        pack_ndarray(np.zeros((2, 2), np.float64), allowed_dtypes=_FLOAT32)


def test_pack_rejects_sidecar_collision_with_envelope_keys():
    with pytest.raises(ValueError, match="collide"):
        pack_ndarray(np.zeros((2, 2), np.float32), allowed_dtypes=_FLOAT32, extra={"dtype": "float64"})


def test_unpack_rejects_disallowed_dtype():
    payload = pack_ndarray(np.zeros((2, 2), np.float32), allowed_dtypes=_FLOAT32)

    with pytest.raises(ValueError, match="dtype"):
        unpack_ndarray(payload, allowed_dtypes=_INT16, ndim=2)


@pytest.mark.parametrize("ndim", [1, 3])
def test_unpack_rejects_wrong_ndim(ndim):
    payload = pack_ndarray(np.zeros((2, 2), np.float32), allowed_dtypes=_FLOAT32)

    with pytest.raises(ValueError, match="dimensions"):
        unpack_ndarray(payload, allowed_dtypes=_FLOAT32, ndim=ndim)


def test_unpack_rejects_byte_count_mismatched_with_declared_shape():
    payload = pack_ndarray(np.zeros((2, 3), np.float32), allowed_dtypes=_FLOAT32)
    payload[PackedArrayKey.SHAPE.value] = [2, 4]

    with pytest.raises(ValueError, match="24 bytes, expected 32"):
        unpack_ndarray(payload, allowed_dtypes=_FLOAT32, ndim=2)


def test_registry_seeds_both_side_channels():
    # A later branch adds the rollout_sample_support producer; the splice must
    # already know the name so that branch stays a pure addition.
    assert PACKED_SIDE_CHANNEL_FIELDS == ("routed_experts", "rollout_sample_support")


def test_load_packed_body_splices_both_blobs_in_one_body():
    routes = np.arange(12).reshape(3, 2, 2)
    support = np.arange(6, dtype=np.float32).reshape(2, 3)
    raw = orjson.dumps(
        _body(
            logprobs={"content": [{"logprob": -0.5}]},
            routed_experts=pack_routed_experts(routes),
            rollout_sample_support=_support_envelope(support),
        )
    )

    choice = load_packed_body(raw)["choices"][0]

    # Both blobs are handed over as memoryviews into `raw`, never as Python strs.
    assert all(
        isinstance(choice[field][PackedArrayKey.DATA], memoryview)
        for field in (PackedField.ROUTED_EXPERTS, PackedField.ROLLOUT_SAMPLE_SUPPORT)
    )
    assert np.array_equal(decode_packed_routed_experts(choice[PackedField.ROUTED_EXPERTS]), routes)
    decoded_support, _ = unpack_ndarray(choice[PackedField.ROLLOUT_SAMPLE_SUPPORT], allowed_dtypes=_FLOAT32, ndim=2)
    assert np.array_equal(decoded_support, support)
    assert choice["logprobs"] == {"content": [{"logprob": -0.5}]}


def test_load_packed_body_keeps_sidecar_fields():
    support = np.arange(4, dtype=np.float32).reshape(2, 2)
    envelope = pack_ndarray(support, allowed_dtypes=_FLOAT32, extra={"prompt_start": 7})
    raw = orjson.dumps(_body(rollout_sample_support=envelope))

    decoded, sidecar = unpack_ndarray(
        load_packed_body(raw)["choices"][0][PackedField.ROLLOUT_SAMPLE_SUPPORT],
        allowed_dtypes=_FLOAT32,
        ndim=2,
    )

    assert np.array_equal(decoded, support)
    assert sidecar == {"prompt_start": 7}


@pytest.mark.parametrize("value", [None, "absent"])
def test_load_packed_body_passes_through_absent_and_null_fields(value):
    fields = {} if value == "absent" else {PackedField.ROUTED_EXPERTS.value: None}
    body = _body(logprobs=None, **fields)

    assert load_packed_body(orjson.dumps(body)) == body


def test_load_packed_body_rejects_reordered_envelope_keys():
    routes = np.arange(12).reshape(3, 2, 2)
    envelope = pack_routed_experts(routes)
    reordered = {key: envelope[key] for key in reversed(list(envelope))}

    with pytest.raises(ValueError, match="layout drifted"):
        load_packed_body(orjson.dumps(_body(routed_experts=reordered)))


def test_load_packed_body_rejects_a_reserialized_body():
    # stdlib json spaces its separators; the blob would silently land in a
    # ~121 MiB Python str instead, which is exactly the cost this avoids.
    raw = json.dumps(_body(routed_experts=pack_routed_experts(np.arange(12).reshape(3, 2, 2)))).encode()

    with pytest.raises(ValueError, match="layout drifted"):
        load_packed_body(raw)


def test_load_packed_body_is_not_spoofable_from_a_string_value():
    routes = np.arange(12).reshape(3, 2, 2)
    spoof = '"routed_experts":{"data":"AAAA","shape":[1,1,1],"dtype":"uint8"}'
    raw = orjson.dumps(_body(note=spoof, routed_experts=pack_routed_experts(routes)))

    choice = load_packed_body(raw)["choices"][0]

    assert choice["note"] == spoof
    assert np.array_equal(decode_packed_routed_experts(choice[PackedField.ROUTED_EXPERTS]), routes)


def test_load_packed_body_leaves_a_spoofed_prefix_alone_when_no_field_is_present():
    body = _body(note='"routed_experts":{"data":"AAAA"')

    assert load_packed_body(orjson.dumps(body)) == body


def test_load_packed_body_ignores_unregistered_packed_fields():
    envelope = pack_ndarray(np.zeros((2, 2), np.float32), allowed_dtypes=_FLOAT32)
    body = _body(some_other_array=envelope)

    assert load_packed_body(orjson.dumps(body)) == body


def test_load_packed_body_honours_a_narrowed_field_registry():
    routes = np.arange(12).reshape(3, 2, 2)
    raw = orjson.dumps(_body(routed_experts=pack_routed_experts(routes)))

    body = load_packed_body(raw, fields=(PackedField.ROLLOUT_SAMPLE_SUPPORT,))

    assert isinstance(body["choices"][0][PackedField.ROUTED_EXPERTS][PackedArrayKey.DATA], str)


def test_load_packed_body_splices_one_blob_per_choice():
    first, second = np.arange(12).reshape(3, 2, 2), np.arange(12, 24).reshape(3, 2, 2)
    raw = orjson.dumps({"choices": [{"routed_experts": pack_routed_experts(routes)} for routes in (first, second)]})

    choices = load_packed_body(raw)["choices"]

    # Blobs are matched to envelopes in document order.
    assert np.array_equal(decode_packed_routed_experts(choices[0][PackedField.ROUTED_EXPERTS]), first)
    assert np.array_equal(decode_packed_routed_experts(choices[1][PackedField.ROUTED_EXPERTS]), second)


def test_load_packed_body_rejects_an_unterminated_blob():
    raw = orjson.dumps(_body(routed_experts=pack_routed_experts(np.arange(12).reshape(3, 2, 2))))
    truncated = raw[: raw.index(b'"shape"') - 2]

    with pytest.raises(ValueError, match="unterminated"):
        load_packed_body(truncated)
