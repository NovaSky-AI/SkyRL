"""Decode vLLM ``/v1/completions`` bodies straight into numpy arrays.

A long-output result is almost entirely two numeric arrays (token ids and
logprobs). Decoding it the ordinary way materializes one Python object per
element and then converts the lists to numpy, which for a 262k-token result
(2.8MB of JSON) costs ~38ms and dominates the API server's CPU. pysimdjson
exposes homogeneous numeric arrays as raw buffers, so the same body decodes in
~6ms with no per-token Python objects. orjson is the fallback when pysimdjson
is unavailable or an array is not purely numeric (a null logprob, say).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import orjson

from skyrl.utils.log import logger

try:
    import simdjson
except ImportError:  # pragma: no cover - exercised via the forced-fallback tests
    simdjson = None


@dataclass
class DecodedChoice:
    finish_reason: str | None
    tokens: np.ndarray  # int32
    logprobs: np.ndarray  # float32, same length as tokens
    # vLLM's raw prompt_logprobs structure, kept only when the request asked for it.
    prompt_logprobs: list[Any] | None = None


def _floats_from_list(values: list[Any], expected_len: int) -> np.ndarray:
    """Float32 array from a decoded list, filling absent values with zeros.

    vLLM occasionally returns None for logprobs under load; zero-fill so RL
    advantage computation doesn't see a ragged shape.
    """
    if not values:
        if expected_len:
            logger.warning("No logprobs returned from vLLM — filling with zeros")
        return np.zeros(expected_len, dtype=np.float32)
    if any(value is None for value in values):
        logger.warning("vLLM returned null logprobs — filling those positions with zeros")
        values = [0.0 if value is None else value for value in values]
    return np.asarray(values, dtype=np.float32)


class CompletionDecoder:
    """Decodes completion bodies; one instance per forwarding client."""

    def __init__(self) -> None:
        self._parser = simdjson.Parser() if simdjson is not None else None

    @property
    def backend(self) -> str:
        return "simdjson" if self._parser is not None else "orjson"

    def decode(self, body: bytes, *, want_prompt_logprobs: bool = False) -> list[DecodedChoice]:
        """Return one :class:`DecodedChoice` per ``choices`` entry.

        Raises ``ValueError`` for a body that is not JSON.
        """
        if self._parser is not None:
            return self._decode_simdjson(body, want_prompt_logprobs)
        return self._decode_orjson(body, want_prompt_logprobs)

    # -- pysimdjson --------------------------------------------------------

    def _decode_simdjson(self, body: bytes, want_prompt_logprobs: bool) -> list[DecodedChoice]:
        # The parser owns one document at a time, so everything is copied out
        # into numpy / Python objects before this method returns.
        doc = self._parser.parse(body)
        choices = []
        for choice in doc.get("choices") or ():
            tokens = _simd_int32(choice.get("token_ids"))
            logprobs_obj = choice.get("logprobs")
            raw_logprobs = logprobs_obj.get("token_logprobs") if logprobs_obj is not None else None
            logprobs = _simd_float32(raw_logprobs, expected_len=len(tokens))
            prompt_logprobs = None
            if want_prompt_logprobs:
                raw = choice.get("prompt_logprobs")
                prompt_logprobs = raw.as_list() if raw is not None else None
            choices.append(
                DecodedChoice(
                    finish_reason=choice.get("finish_reason"),
                    tokens=tokens,
                    logprobs=logprobs,
                    prompt_logprobs=prompt_logprobs,
                )
            )
        return choices

    # -- orjson fallback ---------------------------------------------------

    @staticmethod
    def _decode_orjson(body: bytes, want_prompt_logprobs: bool) -> list[DecodedChoice]:
        result = orjson.loads(body)
        choices = []
        for choice in result.get("choices") or ():
            tokens = np.asarray(choice.get("token_ids") or (), dtype=np.int32)
            raw_logprobs = (choice.get("logprobs") or {}).get("token_logprobs") or []
            choices.append(
                DecodedChoice(
                    finish_reason=choice.get("finish_reason"),
                    tokens=tokens,
                    logprobs=_floats_from_list(raw_logprobs, expected_len=len(tokens)),
                    prompt_logprobs=choice.get("prompt_logprobs") if want_prompt_logprobs else None,
                )
            )
        return choices


def _simd_int32(array) -> np.ndarray:
    if array is None:
        return np.zeros(0, dtype=np.int32)
    try:
        return np.frombuffer(array.as_buffer(of_type="i"), dtype=np.int64).astype(np.int32)
    except TypeError:
        # Not a homogeneous integer array; take the slow path for this one.
        return np.asarray(array.as_list(), dtype=np.int32)


def _simd_float32(array, *, expected_len: int) -> np.ndarray:
    if array is None:
        return _floats_from_list([], expected_len)
    try:
        values = np.frombuffer(array.as_buffer(of_type="d"), dtype=np.float64).astype(np.float32)
    except TypeError:
        # Nulls or mixed ints/floats: fall back to a list for this array only.
        return _floats_from_list(array.as_list(), expected_len)
    if len(values) == 0 and expected_len:
        # vLLM returned an empty logprob list for a non-empty sequence.
        return _floats_from_list([], expected_len)
    return values
