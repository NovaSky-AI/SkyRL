"""Payload contract for the ``/skyrl/v1/generate`` endpoint.

``VLLMServerActor`` writes these payloads and ``RemoteInferenceClient`` reads
them; nothing else depends on the encoding. Both sides serialize with orjson,
which rejects non-finite floats and has no notion of NumPy arrays, so the
helpers here exist to get sampled logprobs and NumPy side channels across that
boundary intact.

Side-channel arrays use ``{data: <base64>, shape: [...], dtype: <name>}``
envelopes. Keeping ``data`` first lets ``load_packed_body`` decode it from the
raw response without materializing a large Python ``str``.
"""

import math
from collections import deque
from enum import StrEnum
from typing import Any, Collection, Iterable, Mapping, Optional, Tuple

import numpy as np
import orjson
import pybase64

from skyrl.backends.skyrl_train.utils.routed_experts import (
    ROUTED_EXPERT_DTYPES,
    RoutedExpertIndices,
    compact_routed_expert_indices,
)

# Matches the floor vLLM applies at its own serving boundaries.
CLAMPED_LOGPROB = -9999.0


class PackedArrayKey(StrEnum):
    """Envelope keys, with ``DATA`` first for ``load_packed_body``."""

    DATA = "data"
    SHAPE = "shape"
    DTYPE = "dtype"


class PackedField(StrEnum):
    """Response-body fields whose value is a packed-array envelope."""

    ROUTED_EXPERTS = "routed_experts"
    ROLLOUT_SAMPLE_SUPPORT = "rollout_sample_support"


PACKED_SIDE_CHANNEL_FIELDS: tuple[str, ...] = tuple(PackedField)

_ENVELOPE_KEYS = frozenset(PackedArrayKey)

_ROUTED_EXPERTS_NDIM = 3

_QUOTE = b'"'

# Base64 cannot contain this scan anchor.
_PACKED_DATA_ANCHOR = f':{{"{PackedArrayKey.DATA}":"'.encode()


def build_logprobs_content(
    token_ids: Iterable[int],
    resp_logprobs: Iterable[Optional[Mapping[int, Any]]],
) -> Tuple[list[dict[str, float]], int]:
    """Build ``logprobs.content``, flooring missing and non-finite logprobs.

    vLLM reports a non-finite logprob for a token it just sampled every few
    thousand rollouts, and omits the entry entirely for others. ``isfinite``
    also catches NaN, which vLLM's own ``max(logprob, -9999.0)`` floor misses
    because ``max`` returns its first argument on a False comparison.

    Under ``off_policy_correction.tis_ratio_type="sequence"`` a clamped token
    pins its whole trajectory at the importance-sampling cap; under ``"token"``
    the effect stays bounded to that token.

    Returns the content list and how many entries were clamped.
    """
    content: list[dict[str, float]] = []
    num_clamped = 0
    for tid, lp_dict in zip(token_ids, resp_logprobs):
        # .get over `tid in lp_dict`: an entry present but None would otherwise
        # raise AttributeError instead of taking the floor below.
        entry = lp_dict.get(tid) if lp_dict else None
        logprob = entry.logprob if entry is not None else None
        if logprob is None or not math.isfinite(logprob):
            num_clamped += 1
            logprob = CLAMPED_LOGPROB
        content.append({"logprob": logprob})
    return content, num_clamped


def _to_host_array(routed_experts: Any) -> Any:
    """Bring a framework tensor into host memory, leaving anything else alone.

    vLLM hands back a torch tensor that may still live on a CUDA device, where
    ``np.asarray`` raises instead of transferring. Duck-typed so this module
    stays framework-agnostic, and deliberately not a blanket ``np.asarray``:
    objects without these methods (notably the nested lists this endpoint used
    to send) fall through to ``compact_routed_expert_indices`` and are rejected.
    """
    for method in ("detach", "cpu", "numpy"):
        op = getattr(routed_experts, method, None)
        if callable(op):
            routed_experts = op()
    return routed_experts


def pack_ndarray(
    arr: np.ndarray,
    *,
    allowed_dtypes: Collection[np.dtype],
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Encode ``arr`` as a base64 envelope carrying ``extra`` as sidecar fields."""
    if not isinstance(arr, np.ndarray):
        raise TypeError("packed array must be a NumPy array")
    if arr.dtype not in allowed_dtypes:
        allowed = sorted(dtype.name for dtype in allowed_dtypes)
        raise ValueError(f"packed array {PackedArrayKey.DTYPE} {arr.dtype.name!r} is not one of {allowed}")
    if extra is not None:
        collisions = sorted(set(extra) & _ENVELOPE_KEYS)
        if collisions:
            raise ValueError(f"sidecar fields collide with envelope keys: {collisions}")

    contiguous = np.ascontiguousarray(arr)
    # `.value` keys: orjson rejects str subclasses as dict keys.
    payload = {
        PackedArrayKey.DATA.value: pybase64.b64encode(memoryview(contiguous)).decode("ascii"),
        PackedArrayKey.SHAPE.value: list(contiguous.shape),
        PackedArrayKey.DTYPE.value: contiguous.dtype.name,
    }
    if extra is not None:
        payload.update(extra)
    return payload


def unpack_ndarray(
    payload: Mapping[str, Any],
    *,
    allowed_dtypes: Collection[np.dtype],
    ndim: int,
) -> Tuple[np.ndarray, dict[str, Any]]:
    """Decode an envelope whose base64 ``data`` may be a string or buffer."""
    if not isinstance(payload, Mapping):
        raise TypeError("packed array payload must be an object")
    try:
        dtype_name = payload[PackedArrayKey.DTYPE]
        shape = tuple(payload[PackedArrayKey.SHAPE])
        data = pybase64.b64decode_as_bytearray(payload[PackedArrayKey.DATA], validate=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid packed array envelope: {exc}") from exc

    dtypes = {dtype.name: dtype for dtype in allowed_dtypes}
    if not isinstance(dtype_name, str) or dtype_name not in dtypes:
        raise ValueError(f"packed array {PackedArrayKey.DTYPE} {dtype_name!r} is not one of {sorted(dtypes)}")
    dtype = dtypes[dtype_name]
    # Reject bool, an int subclass; accept np.integer for in-process callers.
    if len(shape) != ndim or any(
        not isinstance(dim, (int, np.integer)) or isinstance(dim, bool) or dim < 0 for dim in shape
    ):
        raise ValueError(f"packed array {PackedArrayKey.SHAPE} {shape} is not {ndim} non-negative dimensions")
    expected_size = math.prod(shape) * dtype.itemsize
    if len(data) != expected_size:
        raise ValueError(
            f"packed array {PackedArrayKey.DATA} has {len(data)} bytes, "
            f"expected {expected_size} for {dtype_name}{list(shape)}"
        )

    array = np.frombuffer(data, dtype=dtype).reshape(shape)
    sidecar = {key: value for key, value in payload.items() if key not in _ENVELOPE_KEYS}
    return array, sidecar


def pack_routed_experts(routed_experts: RoutedExpertIndices) -> dict[str, Any]:
    compact = compact_routed_expert_indices(_to_host_array(routed_experts))
    return pack_ndarray(compact, allowed_dtypes=ROUTED_EXPERT_DTYPES)


def decode_packed_routed_experts(payload: dict[str, Any]) -> RoutedExpertIndices:
    decoded, _ = unpack_ndarray(payload, allowed_dtypes=ROUTED_EXPERT_DTYPES, ndim=_ROUTED_EXPERTS_NDIM)
    compact = compact_routed_expert_indices(decoded)
    if compact.dtype != decoded.dtype:
        raise ValueError(
            f"packed routed_experts uses non-canonical dtype {decoded.dtype.name}; expected {compact.dtype.name}"
        )
    return compact


def _data_prefix(field: str) -> bytes:
    """The bytes an orjson-serialized packed ``field`` opens with."""
    return f'"{field}"'.encode() + _PACKED_DATA_ANCHOR


def load_packed_body(raw: bytes, *, fields: tuple[str, ...] = PACKED_SIDE_CHANNEL_FIELDS) -> dict[str, Any]:
    """Parse a response after replacing registered base64 blobs with views.

    Null fields pass through. An envelope layout the scan cannot splice raises
    instead of falling back to materializing the base64 as a Python string.
    """
    prefixes = {field: _data_prefix(field) for field in fields}
    blobs: dict[str, deque[memoryview]] = {field: deque() for field in fields}
    view = memoryview(raw)
    pieces: list[memoryview] = []
    copied = 0
    scan = 0
    while (anchor := raw.find(_PACKED_DATA_ANCHOR, scan)) >= 0:
        field = _match_packed_field(raw, anchor, prefixes)
        if field is None:
            scan = anchor + len(_PACKED_DATA_ANCHOR)
            continue
        start = anchor + len(_PACKED_DATA_ANCHOR)
        end = raw.find(_QUOTE, start)
        if end < 0:
            raise ValueError(f"unterminated base64 {PackedArrayKey.DATA} for {field} in the response body")
        pieces.append(view[copied:start])
        blobs[field].append(view[start:end])
        copied = scan = end

    if pieces:
        pieces.append(view[copied:])
        body = orjson.loads(b"".join(pieces))
    else:
        body = orjson.loads(raw)
    _restore_packed_data(body, blobs)

    unplaced = {field: len(queue) for field, queue in blobs.items() if queue}
    if unplaced:
        raise ValueError(f"spliced packed blobs found no envelope in the response body: {unplaced}")
    return body


def _match_packed_field(raw: bytes, anchor: int, prefixes: Mapping[str, bytes]) -> Optional[str]:
    """Name the registered field whose prefix ends at ``anchor``, if any."""
    for field, prefix in prefixes.items():
        begin = anchor + len(_PACKED_DATA_ANCHOR) - len(prefix)
        if begin >= 0 and raw.startswith(prefix, begin):
            return field
    return None


def _restore_packed_data(node: Any, blobs: Mapping[str, deque[memoryview]]) -> None:
    """Put each blob back on its envelope's ``data`` key, in document order."""
    if isinstance(node, dict):
        for key, value in node.items():
            queue = blobs.get(key)
            if queue is not None and isinstance(value, dict) and PackedArrayKey.DATA in value:
                if not queue:
                    raise ValueError(f"packed {key} survived the scan unspliced; the response-body layout drifted")
                value[PackedArrayKey.DATA.value] = queue.popleft()
            elif isinstance(value, (dict, list)):
                _restore_packed_data(value, blobs)
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, (dict, list)):
                _restore_packed_data(item, blobs)
