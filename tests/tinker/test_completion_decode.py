"""vLLM completion bodies decode to the same arrays with pysimdjson and with the orjson fallback."""

import numpy as np
import orjson
import pytest

from skyrl.tinker.extra import completion_decode
from skyrl.tinker.extra.completion_decode import CompletionDecoder


def _body(choices: list[dict]) -> bytes:
    return orjson.dumps({"id": "cmpl", "object": "text_completion", "choices": choices, "usage": {}})


def _decoders() -> list[CompletionDecoder]:
    fast = CompletionDecoder()
    slow = CompletionDecoder()
    slow._parser = None
    return [fast, slow]


@pytest.mark.parametrize("decoder", _decoders(), ids=lambda d: d.backend)
def test_decodes_tokens_and_logprobs_to_typed_arrays(decoder):
    choices = decoder.decode(
        _body(
            [
                {
                    "token_ids": [5, 6, 70000],
                    "logprobs": {"token_logprobs": [-0.5, -1.25, -3.0]},
                    "finish_reason": "stop",
                },
                {"token_ids": [1], "logprobs": {"token_logprobs": [0]}, "finish_reason": "length"},
            ]
        )
    )

    assert [c.finish_reason for c in choices] == ["stop", "length"]
    assert choices[0].tokens.dtype == np.int32 and choices[0].logprobs.dtype == np.float32
    assert choices[0].tokens.tolist() == [5, 6, 70000]
    assert choices[0].logprobs.tolist() == [-0.5, -1.25, -3.0]
    # An integer-valued logprob still comes out as float32.
    assert choices[1].logprobs.tolist() == [0.0]
    assert choices[0].prompt_logprobs is None


@pytest.mark.parametrize("decoder", _decoders(), ids=lambda d: d.backend)
def test_missing_or_null_logprobs_are_zero_filled(decoder):
    choices = decoder.decode(
        _body(
            [
                {"token_ids": [1, 2, 3], "logprobs": None, "finish_reason": "length"},
                {"token_ids": [1, 2], "logprobs": {"token_logprobs": []}, "finish_reason": "length"},
                {"token_ids": [1, 2, 3], "logprobs": {"token_logprobs": [None, -0.5, -1.0]}, "finish_reason": "stop"},
                {"token_ids": [], "logprobs": {"token_logprobs": []}, "finish_reason": "stop"},
            ]
        )
    )

    assert choices[0].logprobs.tolist() == [0.0, 0.0, 0.0]
    assert choices[1].logprobs.tolist() == [0.0, 0.0]
    assert choices[2].logprobs.tolist() == [0.0, -0.5, -1.0]
    assert choices[3].tokens.tolist() == [] and choices[3].logprobs.tolist() == []
    assert all(c.logprobs.dtype == np.float32 for c in choices)


@pytest.mark.parametrize("decoder", _decoders(), ids=lambda d: d.backend)
def test_prompt_logprobs_kept_only_when_requested(decoder):
    raw = [None, {"5": {"logprob": -0.1, "rank": 1, "decoded_token": "a"}}]
    body = _body(
        [{"token_ids": [1], "logprobs": {"token_logprobs": [-1.0]}, "finish_reason": "stop", "prompt_logprobs": raw}]
    )

    assert decoder.decode(body)[0].prompt_logprobs is None
    assert decoder.decode(body, want_prompt_logprobs=True)[0].prompt_logprobs == raw


@pytest.mark.parametrize("decoder", _decoders(), ids=lambda d: d.backend)
def test_non_json_body_raises_value_error(decoder):
    with pytest.raises(ValueError):
        decoder.decode(b"<html>502 Bad Gateway</html>")


@pytest.mark.skipif(completion_decode.simdjson is None, reason="pysimdjson not installed")
def test_fast_and_fallback_agree_on_large_arrays():
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 150_000, size=200_000).tolist()
    logprobs = (-rng.random(200_000) * 20).tolist()
    body = _body([{"token_ids": tokens, "logprobs": {"token_logprobs": logprobs}, "finish_reason": "length"}])
    fast, slow = _decoders()

    a, b = fast.decode(body)[0], slow.decode(body)[0]

    assert np.array_equal(a.tokens, b.tokens)
    assert np.array_equal(a.logprobs, b.logprobs)
    assert fast.backend == "simdjson" and slow.backend == "orjson"


def test_decoder_can_be_reused_across_bodies():
    decoder = CompletionDecoder()
    first = decoder.decode(
        _body([{"token_ids": [1, 2], "logprobs": {"token_logprobs": [-1.0, -2.0]}, "finish_reason": "stop"}])
    )
    second = decoder.decode(
        _body([{"token_ids": [9], "logprobs": {"token_logprobs": [-9.0]}, "finish_reason": "length"}])
    )
    # Arrays from the first body survive the second parse (they were copied out).
    assert first[0].tokens.tolist() == [1, 2] and first[0].logprobs.tolist() == [-1.0, -2.0]
    assert second[0].tokens.tolist() == [9]
