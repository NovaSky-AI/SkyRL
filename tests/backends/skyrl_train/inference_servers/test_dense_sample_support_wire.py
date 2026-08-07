import numpy as np
import orjson
import pytest

from skyrl.backends.skyrl_train.inference_servers.sample_support_set_wire import (
    decode_sample_support_set,
    encode_sample_support_set,
)


def test_packed_round_trip_preserves_dense_support():
    support = np.array([[7, 152064, -1, -1], [9, 10, 11, -1], [12, -1, -1, -1]], dtype=np.int32)

    packed = encode_sample_support_set(support)
    restored = decode_sample_support_set(orjson.loads(orjson.dumps(packed)))

    np.testing.assert_array_equal(restored, support)
    assert restored.dtype == np.int32
    assert restored.flags.c_contiguous


def test_packed_round_trip_preserves_empty_token_dimension():
    support = np.empty((0, 8), dtype=np.int32)

    packed = encode_sample_support_set(support)

    assert packed["shape"] == [0, 8]
    np.testing.assert_array_equal(decode_sample_support_set(packed), support)


def test_packed_wire_rejects_legacy_list_payload():
    with pytest.raises(TypeError, match="dictionary"):
        decode_sample_support_set([[7, 8]])


@pytest.mark.parametrize(
    ("support", "message"),
    [
        (np.array([[1.5, 2.5]]), "integer"),
        (np.array([[1, -1, 2]]), "trailing"),
        (np.array([[-2, 1]]), "-1 padding"),
        (np.array([[2**31, -1]], dtype=np.int64), "int32"),
    ],
)
def test_wire_rejects_invalid_dense_support(support, message):
    with pytest.raises(ValueError, match=message):
        encode_sample_support_set(support)


@pytest.mark.parametrize("shape", [[1], [1, 2, 3], [1.5, 2], ["1", 2], [True, 2], [-1, 2]])
def test_packed_wire_rejects_invalid_shape(shape):
    with pytest.raises(ValueError, match="shape"):
        decode_sample_support_set({"data": "", "shape": shape, "dtype": "int32"})


def test_packed_wire_rejects_bad_base64_and_size():
    with pytest.raises(ValueError, match="valid base64"):
        decode_sample_support_set({"data": "!", "shape": [1, 1], "dtype": "int32"})
    with pytest.raises(ValueError, match="byte-size mismatch"):
        decode_sample_support_set({"data": "", "shape": [1, 1], "dtype": "int32"})
