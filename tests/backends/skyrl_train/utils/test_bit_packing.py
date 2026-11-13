import numpy as np
import pytest

from skyrl.backends.skyrl_train.utils.bit_packing import pack_ids, unpack_ids


@pytest.mark.parametrize("bits", [1, 17, 18, 31])
@pytest.mark.parametrize("count", [0, 1, 3, 63, 64, 65, 1001])
def test_ids_round_trip_in_order(bits, count):
    ids = np.random.default_rng(count).integers(0, 1 << bits, count).astype(np.int32)

    words = pack_ids(ids, bits)
    words.setflags(write=False)  # as a stored payload's view arrives

    assert words.nbytes == 8 * -(-count * bits // 64)
    np.testing.assert_array_equal(unpack_ids(words, count, bits), ids)


@pytest.mark.parametrize("bits", [17, 18])
def test_extreme_ids_survive(bits):
    ids = np.array([0, (1 << bits) - 1, 1, (1 << bits) - 2, 131_071], dtype=np.int32)

    np.testing.assert_array_equal(unpack_ids(pack_ids(ids, bits), ids.size, bits), ids)


@pytest.mark.parametrize("bad", [-1, 1 << 17])
def test_an_id_the_width_cannot_hold_is_refused(bad):
    with pytest.raises(ValueError, match="17 bits"):
        pack_ids(np.array([5, bad], dtype=np.int32), 17)


@pytest.mark.parametrize("bits", [0, 32])
def test_widths_outside_int32_ids_are_refused(bits):
    with pytest.raises(ValueError, match="bits"):
        pack_ids(np.zeros(1, dtype=np.int32), bits)


def test_too_few_words_are_refused():
    with pytest.raises(ValueError, match="cannot hold"):
        unpack_ids(np.zeros(1, dtype=np.uint64), 4, 17)
