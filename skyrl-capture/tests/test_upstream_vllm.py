"""The `vllm` upstream: vLLM's own token-in/token-out wire.

`vllm serve` mounts `/inference/v1/generate` itself — `api_server.py` calls
`register_scale_out_api_routers` for any generate-capable model — so a target is
`{"type": "vllm", "url": "http://engine:8000"}` with nothing in between: no
wrapper script, no translating proxy.

Each test pins one of the five ways it differs from capture's own `tokens`
shape. Four of the five fail loudly. The fifth, the session header, does not:
the server still answers without it and only prefix-cache locality is lost,
which is most of the reason to name a trajectory at all.
"""

from __future__ import annotations

import pytest

from skyrl_capture.tito import upstream as tito
from skyrl_capture.tito.types import TokenUpstreamError


@pytest.fixture
def wire():
    return tito.get("vllm")


@pytest.fixture
def batched():
    """capture's own shape, for contrast."""
    return tito.get("tokens")


def request_for(engine, **overrides):
    kwargs = {
        "prompt_token_ids": [1, 2, 3],
        "sampling_params": {"max_tokens": 16, "logprobs": True},
        "model": "policy",
        "session_id": "0_1",
    }
    kwargs.update(overrides)
    return engine.request(**kwargs)


def reply(token_ids=(7, 8), logprobs=(-0.1, -0.2), finish="stop"):
    return {
        "choices": [
            {
                "token_ids": list(token_ids),
                "finish_reason": finish,
                "logprobs": {"content": [{"logprob": value} for value in logprobs]},
            }
        ]
    }


# -- the request --------------------------------------------------------------


def test_the_request_is_singular_not_a_batch_of_one(wire, batched):
    body, _ = request_for(wire)
    assert body["token_ids"] == [1, 2, 3]
    assert "prompt_token_ids" not in body

    # The contrast is the point: capture's own wire sends a batch.
    other, _ = request_for(batched)
    assert other["prompt_token_ids"] == [[1, 2, 3]]


def test_session_affinity_moves_into_the_header(wire, batched):
    """`SESSION_ID_HEADER = "X-Session-ID"` in vLLM's serving code.

    The quiet one: without it the server still answers, and only prefix-cache
    locality is lost.
    """
    body, headers = request_for(wire)
    assert headers == {"X-Session-ID": "0_1"}
    assert "session_id" not in body and "session_ids" not in body

    other, other_headers = request_for(batched)
    assert other_headers == {} and other["session_id"] == "0_1"


def test_the_generate_path_is_appended_to_a_bare_server_url(wire):
    assert wire.url("http://engine:8000") == "http://engine:8000/inference/v1/generate"
    assert wire.url("http://engine:8000/") == "http://engine:8000/inference/v1/generate"
    already = "http://engine:8000/inference/v1/generate"
    assert wire.url(already) == already


def test_capture_s_own_wire_does_not_rewrite_the_url(batched):
    assert batched.url("http://engine:9000/generate") == "http://engine:9000/generate"


def test_a_boolean_logprobs_flag_becomes_a_count(wire):
    """`logprobs` changes type across this boundary: a boolean in the OpenAI
    shape capture speaks, a count of top logprobs in `SamplingParams`, where 0
    means the sampled token only."""
    body, _ = request_for(wire, sampling_params={"logprobs": True})
    assert body["sampling_params"]["logprobs"] == 0


def test_logprobs_are_requested_even_when_the_caller_did_not(wire):
    """capture refuses a turn without them, so never ask for none."""
    for params in ({"max_tokens": 4}, {"logprobs": False}):
        body, _ = request_for(wire, sampling_params=params)
        assert body["sampling_params"]["logprobs"] == 0


def test_sampling_params_the_engine_would_reject_are_dropped(wire):
    """The field is vLLM's own `SamplingParams`, which rejects unknown keys."""
    body, _ = request_for(
        wire,
        sampling_params={"max_tokens": 8, "temperature": 0.7, "user": "someone", "stream": True},
    )
    params = body["sampling_params"]
    assert params["max_tokens"] == 8 and params["temperature"] == 0.7
    assert "user" not in params and "stream" not in params


def test_the_model_rides_at_the_top_level(wire):
    body, _ = request_for(wire)
    assert body["model"] == "policy"


def test_a_sampling_field_the_caller_did_not_send_is_not_invented(wire):
    """Per-inference settings come from the inference request and nowhere else.

    `cache_salt` is the case that made this rule: it is keyed on a weight
    version that moves every training step, so it can be neither startup
    configuration nor a trajectory setting. It rides in the request that wants
    it, which means the wire adds nothing of its own here.
    """
    body, _ = request_for(wire)
    assert "cache_salt" not in body


# -- the response -------------------------------------------------------------


def test_a_choice_is_read_back_into_what_a_turn_needs(wire):
    parsed = wire.response(reply())
    assert parsed["completion_ids"] == [7, 8]
    assert parsed["completion_logprobs"] == [-0.1, -0.2]
    assert parsed["stop_reason"] == "stop"


def test_a_missing_finish_reason_defaults_rather_than_failing(wire):
    assert wire.response(reply(finish=None))["stop_reason"] == "stop"


def test_a_response_without_logprobs_is_refused(wire):
    body = reply()
    del body["choices"][0]["logprobs"]
    with pytest.raises(TokenUpstreamError, match="selected-token logprobs"):
        wire.response(body)


def test_logprobs_that_do_not_cover_the_completion_are_refused(wire):
    with pytest.raises(TokenUpstreamError, match="different lengths"):
        wire.response(reply(token_ids=(1, 2, 3), logprobs=(-0.1, -0.2)))


def test_a_real_batch_is_refused(wire):
    body = reply()
    body["choices"].append(body["choices"][0])
    with pytest.raises(TokenUpstreamError, match="exactly one choice"):
        wire.response(body)


def test_routed_experts_are_omitted_not_guessed_at(wire):
    """vLLM returns a base64 `.npy` covering `num_tokens - 1` rows, because the
    last sampled token has not been forwarded yet. capture refuses a turn whose
    routed experts cover only part of the sequence, so a partial decode would
    be worse than none."""
    body = reply()
    body["choices"][0]["routed_experts"] = "<base64 npy>"
    assert wire.response(body)["routed_experts"] is None


# -- the kind itself ----------------------------------------------------------


def test_it_is_an_engine_wire_and_not_something_a_client_speaks(wire):
    """Which registry it is in is the whole of what it is.

    A token engine has no client-facing route, no environment variables and no
    error envelope, because nothing points a provider SDK at it -- capture
    calls it. Asking the text registry for it is an error rather than a
    half-working adapter.
    """
    from skyrl_capture import upstream as text

    assert tito.is_registered("vllm") and not text.is_registered("vllm")
    assert not hasattr(wire, "client_environment")


def test_a_fork_inherits_the_wire_and_overrides_only_the_path(wire):
    """What a trainer serving the same shape elsewhere has to write.

    SkyRL forks this endpoint onto `/skyrl/v1/generate`; the payloads are
    identical, so the subclass is a name and a path.
    """

    class Forked(tito.VLLMTitoProtocol):
        name = "forked"
        generate_path = "/forked/v1/generate"

    forked = Forked()
    assert forked.url("http://engine:8000") == "http://engine:8000/forked/v1/generate"
    same = {"prompt_token_ids": [1], "sampling_params": {"logprobs": True},
            "model": None, "session_id": "s"}
    assert forked.request(**same) == wire.request(**same)


# -- fields capture has no opinion about ---------------------------------------
def test_a_request_field_capture_does_not_know_is_carried_not_dropped():
    """An engine behind capture accepts parameters capture has never heard of.

    `cache_salt` is the case that made this rule -- it keys an engine's prefix
    cache on the policy weights, so it moves every training step and can be
    neither startup configuration nor a trajectory setting. But the rule is
    deliberately not about `cache_salt`, and capture does not name it. Naming
    the fields of every upstream would make this file the place their
    vocabularies accumulate; dropping them would put those upstreams out of
    reach through capture entirely. So they are carried, unread, and the
    protocol for that upstream decides what any of them mean.
    """
    import orjson

    from skyrl_capture.tito.protocol import parse_chat_request

    request = parse_chat_request(
        orjson.dumps(
            {
                "model": "policy",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "temperature": 0.7,
                "cache_salt": "weights-42",
                "some_future_engine_flag": True,
            }
        ),
        registered_model=None,
    )
    # What capture reads itself stays out of the extras.
    assert request.sampling == {"temperature": 0.7}
    assert "model" not in request.extras
    assert "messages" not in request.extras
    assert "stream" not in request.extras
    # What it does not read is offered onward, whatever it is.
    assert request.extras == {"cache_salt": "weights-42", "some_future_engine_flag": True}


def test_a_field_vllm_does_not_accept_never_reaches_its_sampling_params(wire):
    """Carrying is not applying.

    `SamplingParams` rejects a request holding a key it does not know, so a
    protocol has to drop what it does not recognise. vLLM's own wire filters
    against the set it accepts, which is what makes carrying the rest safe.
    """
    body, _ = request_for(
        wire,
        sampling_params={"temperature": 0.7, "cache_salt": "weights-42", "nonsense": 1},
    )
    assert "cache_salt" not in body["sampling_params"]
    assert "nonsense" not in body["sampling_params"]
    assert "cache_salt" not in body
    assert body["sampling_params"]["temperature"] == 0.7
