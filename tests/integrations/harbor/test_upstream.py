"""The `skyrl` token wire: what this router expects, versus what capture sends.

The translation used to run as a proxy process beside the trainer, owned by
neither project. It is an upstream kind now, registered into capture from
`examples/train_integrations/harbor/icap/upstream.py`, so a target is
`{"type": "skyrl", "url": "http://router:8000"}` and nothing sits in between.

Each test pins one of the five differences. They run with capture stubbed --
see conftest -- so they belong on the CPU pipeline.
"""

from __future__ import annotations

import pytest


def request_for(wire, **overrides):
    kwargs = {
        "prompt_token_ids": [1, 2, 3],
        "sampling_params": {"max_tokens": 16, "logprobs": True},
        "model": "policy",
        "session_id": "0_1",
        "cache_salt": None,
    }
    kwargs.update(overrides)
    return wire.token_request(**kwargs)


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


# -- registration -------------------------------------------------------------


def test_importing_the_module_registers_the_wire(registered):
    """capture resolves a target by name, so the name is the contract."""
    assert registered.name == "skyrl"
    assert registered.mode == "tokens"
    assert registered.requires_tokenizer is True


# -- the request --------------------------------------------------------------


def test_the_request_is_singular_not_a_batch_of_one(registered):
    body, _ = request_for(registered)
    assert body["token_ids"] == [1, 2, 3]
    assert "prompt_token_ids" not in body


def test_session_affinity_moves_into_the_router_header(registered):
    """The quiet one: without it the router still answers and only prefix-cache
    locality is lost, which is most of the reason to name a trajectory."""
    body, headers = request_for(registered)
    assert headers == {"X-Session-ID": "0_1"}
    assert "session_id" not in body and "session_ids" not in body


def test_the_generate_path_is_appended_to_a_bare_router_url(registered):
    assert registered.token_url("http://router:8000") == "http://router:8000/skyrl/v1/generate"
    assert registered.token_url("http://router:8000/") == "http://router:8000/skyrl/v1/generate"
    already = "http://router:8000/skyrl/v1/generate"
    assert registered.token_url(already) == already


def test_a_boolean_logprobs_flag_becomes_this_engines_count(registered):
    """`logprobs` changes type across the boundary: a boolean in the OpenAI
    shape capture speaks, a count of top logprobs here, where 0 means the
    sampled token only."""
    body, _ = request_for(registered, sampling_params={"logprobs": True})
    assert body["sampling_params"]["logprobs"] == 0


def test_logprobs_are_requested_even_when_the_caller_did_not(registered):
    """Capture refuses a turn without them, so never ask for none."""
    for params in ({"max_tokens": 4}, {"logprobs": False}):
        body, _ = request_for(registered, sampling_params=params)
        assert body["sampling_params"]["logprobs"] == 0


def test_sampling_params_this_engine_would_reject_are_dropped(registered):
    body, _ = request_for(
        registered,
        sampling_params={"max_tokens": 8, "temperature": 0.7, "user": "someone", "stream": True},
    )
    params = body["sampling_params"]
    assert params["max_tokens"] == 8 and params["temperature"] == 0.7
    assert "user" not in params and "stream" not in params


def test_model_and_cache_salt_ride_at_the_top_level(registered):
    body, _ = request_for(registered, cache_salt="weights-7")
    assert body["cache_salt"] == "weights-7" and body["model"] == "policy"


# -- the response -------------------------------------------------------------


def test_a_choice_is_read_back_into_what_a_turn_needs(registered):
    parsed = registered.token_response(reply())
    assert parsed["completion_ids"] == [7, 8]
    assert parsed["completion_logprobs"] == [-0.1, -0.2]
    assert parsed["stop_reason"] == "stop"


def test_a_missing_finish_reason_defaults_rather_than_failing(registered):
    assert registered.token_response(reply(finish=None))["stop_reason"] == "stop"


def test_a_response_without_logprobs_is_refused(registered, wire_error):
    body = reply()
    del body["choices"][0]["logprobs"]
    with pytest.raises(wire_error, match="selected-token logprobs"):
        registered.token_response(body)


def test_logprobs_that_do_not_cover_the_completion_are_refused(registered, wire_error):
    with pytest.raises(wire_error, match="different lengths"):
        registered.token_response(reply(token_ids=(1, 2, 3), logprobs=(-0.1, -0.2)))


def test_a_real_batch_is_refused(registered, wire_error):
    body = reply()
    body["choices"].append(body["choices"][0])
    with pytest.raises(wire_error, match="exactly one choice"):
        registered.token_response(body)


def test_routed_experts_are_omitted_not_guessed_at(registered):
    """They are packed under this project's own codec, and capture refuses a
    turn whose routed experts cover only part of the sequence -- so a partial
    decode would be worse than none."""
    body = reply()
    body["choices"][0]["routed_experts"] = {"data": "<b64>", "shape": [2, 4], "dtype": "int32"}
    assert registered.token_response(body)["routed_experts"] is None
