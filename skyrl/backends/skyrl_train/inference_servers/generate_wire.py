"""Payload contract for the ``/skyrl/v1/generate`` endpoint.

``VLLMServerActor`` writes these payloads and ``RemoteInferenceClient`` reads
them; nothing else depends on the encoding. Both sides serialize with orjson,
which rejects non-finite floats and has no notion of NumPy arrays, so the
helpers here exist to get sampled logprobs and routed-expert IDs across that
boundary intact.
"""

import math
from typing import Any, Iterable, Mapping, Optional, Tuple

import numpy as np
import pybase64

from skyrl.backends.skyrl_train.utils.routed_experts import (
    ROUTED_EXPERT_DTYPES,
    RoutedExpertIndices,
    compact_routed_expert_indices,
)

# Matches the floor vLLM applies at its own serving boundaries.
CLAMPED_LOGPROB = -9999.0

_DTYPES = {dtype.name: dtype for dtype in ROUTED_EXPERT_DTYPES}


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


def pack_routed_experts(routed_experts: RoutedExpertIndices) -> dict[str, Any]:
    compact = compact_routed_expert_indices(_to_host_array(routed_experts))
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
    # bool is a subclass of int, so it needs an explicit rejection; np.integer is
    # accepted for in-process callers, since orjson only ever yields plain ints.
    if len(shape) != 3 or any(
        not isinstance(dim, (int, np.integer)) or isinstance(dim, bool) or dim < 0 for dim in shape
    ):
        raise ValueError(f"invalid packed routed_experts shape: {shape}")
    expected_size = math.prod(shape) * dtype.itemsize
    if len(data) != expected_size:
        raise ValueError(f"packed routed_experts has {len(data)} bytes, expected {expected_size}")
    decoded = np.frombuffer(data, dtype=dtype).reshape(shape)
    compact = compact_routed_expert_indices(decoded)
    if compact.dtype != dtype:
        raise ValueError(f"packed routed_experts uses non-canonical dtype {dtype.name}; expected {compact.dtype.name}")
    return compact


def clamp_sampled_logprobs(sampled: np.ndarray) -> Tuple[list[dict[str, float]], int]:
    """Build ``logprobs.content`` from a flat array of sampled-token logprobs.

    The array form of :func:`build_logprobs_content`, for vLLM's flat-logprobs
    rows. A non-finite entry here reaches the client as JSON ``null`` (orjson
    emits ``null`` rather than raising), which then fails tensor conversion in
    preprocessing, so it has to be floored before serialization.

    NaN needs the same treatment as ``-inf`` and is easier to miss: callers that
    screen support columns with ``isneginf`` do not catch it.
    """
    finite = np.isfinite(sampled)
    num_clamped = int((~finite).sum())
    values = np.where(finite, sampled, CLAMPED_LOGPROB).tolist()
    return [{"logprob": value} for value in values], num_clamped
