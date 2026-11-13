"""Fixed-width bit packing of non-negative ids: each id below ``2**bits`` takes ``bits`` bits.

Id ``i`` occupies bits ``[i * bits, (i + 1) * bits)`` of a little-endian stream of uint64 words, so
ids keep their order and one may straddle two words.
"""

import numba
import numpy as np

# Ids are int32 here, so 31 bits hold any of them.
MAX_BITS = 31


@numba.njit
def _pack(ids, bits, words):
    for i in range(ids.size):
        word, offset = (i * bits) >> 6, (i * bits) & 63
        value = np.uint64(ids[i])
        words[word] |= value << np.uint64(offset)
        if offset + bits > 64:
            words[word + 1] |= value >> np.uint64(64 - offset)


@numba.njit
def _unpack(words, bits, ids):
    mask = (np.uint64(1) << np.uint64(bits)) - np.uint64(1)
    for i in range(ids.size):
        word, offset = (i * bits) >> 6, (i * bits) & 63
        value = words[word] >> np.uint64(offset)
        if offset + bits > 64:
            value |= words[word + 1] << np.uint64(64 - offset)
        ids[i] = value & mask


def _check_bits(bits: int) -> None:
    if not 1 <= bits <= MAX_BITS:
        raise ValueError(f"ids pack at 1 to {MAX_BITS} bits, got {bits}")


def pack_ids(ids: np.ndarray, bits: int) -> np.ndarray:
    """``ids``, each in ``[0, 2**bits)``, packed in order into ``ceil(len(ids) * bits / 64)`` uint64 words."""
    _check_bits(bits)
    ids = np.ascontiguousarray(ids)
    if ids.ndim != 1:
        raise ValueError(f"expected a flat id array, got shape {ids.shape}")
    if ids.size and (ids.min() < 0 or ids.max() >= 1 << bits):
        raise ValueError(f"ids span [{ids.min()}, {ids.max()}], which {bits} bits cannot hold")
    words = np.zeros(-(-ids.size * bits // 64), dtype=np.uint64)
    _pack(ids, bits, words)
    return words


def unpack_ids(words: np.ndarray, count: int, bits: int) -> np.ndarray:
    """The first ``count`` ids ``pack_ids(ids, bits)`` packed into ``words``, as int32."""
    _check_bits(bits)
    if words.dtype != np.uint64 or words.size * 64 < count * bits:
        raise ValueError(f"{words.size} {words.dtype} words cannot hold {count} {bits}-bit ids")
    ids = np.empty(count, dtype=np.int32)
    _unpack(words, bits, ids)
    return ids
