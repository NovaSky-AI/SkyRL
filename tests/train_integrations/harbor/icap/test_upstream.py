"""The `skyrl` upstream: our token wire, spoken as this router expects.

These replace `work_icap/tests/test_shim.py`. The translation used to live in a
proxy process that neither project owned, started in a thread beside the
trainer; it is now an upstream kind registered into capture from here, so a
target is `{"type": "skyrl", "url": "http://router:8000"}` and nothing sits in
between.

It lives in this tree rather than capture's because the wire is ours: capture
speaks protocols, not engines. Each test pins one of the five differences named
in `SkyRLTokensServer`.
"""

from __future__ import annotations

import pytest

from inference_capture.tokens.types import TokenUpstreamError
from inference_capture.upstream import registry

# Importing is registering: this is the module under test.
from examples.train_integrations.harbor.icap import upstream as _register  # noqa: F401


@pytest.fixture
def skyrl():
    return registry.get("skyrl")


@pytest.fixture
def default():
    return registry.get("tokens")


def _request(server, **overrides):
    kwargs = {
        "prompt_token_ids": [1, 2, 3],
        "sampling_params": {"max_tokens": 16, "logprobs": True},
        "model": "policy",
        "session_id": "0_1",
        "cache_salt": None,
    }
    kwargs.update(overrides)
    return server.token_request(**kwargs)


def _reply(token_ids=(7, 8), logprobs=(-0.1, -0.2), finish="stop"):
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


def test_the_request_is_singular_not_a_batch_of_one(skyrl, default):
    body, _ = _request(skyrl)
    assert body["token_ids"] == [1, 2, 3]
    assert "prompt_token_ids" not in body

    # The contrast is the point: the default wire sends a batch.
    batched, _ = _request(default)
    assert batched["prompt_token_ids"] == [[1, 2, 3]]


def test_session_affinity_moves_to_the_router_header(skyrl, default):
    """The quiet failure: get this wrong and the router still answers.

    Only prefix-cache locality is lost, which is most of the reason to name a
    trajectory in the first place.
    """
    body, headers = _request(skyrl)
    assert headers == {"X-Session-ID": "0_1"}
    assert "session_id" not in body and "session_ids" not in body

    batched, batched_headers = _request(default)
    assert batched_headers == {}
    assert batched["session_id"] == "0_1"


def test_the_generate_path_is_appended_to_a_bare_router_url(skyrl):
    assert skyrl.token_url("http://router:8000") == "http://router:8000/skyrl/v1/generate"
    assert skyrl.token_url("http://router:8000/") == "http://router:8000/skyrl/v1/generate"
    # Already-complete URLs are left alone, so an explicit target still works.
    full = "http://router:8000/skyrl/v1/generate"
    assert skyrl.token_url(full) == full


def test_the_default_wire_does_not_rewrite_the_url(default):
    assert default.token_url("http://engine:9000/generate") == "http://engine:9000/generate"


def test_a_boolean_logprobs_flag_becomes_vllms_count(skyrl):
    """`logprobs` changes type across this boundary.

    A boolean in the OpenAI shape capture speaks; a count of top logprobs in
    vLLM's, where 0 means "the sampled token only".
    """
    body, _ = _request(skyrl, sampling_params={"logprobs": True})
    assert body["sampling_params"]["logprobs"] == 0


def test_logprobs_are_requested_even_when_the_caller_did_not(skyrl):
    """Capture rejects a turn without them, so never ask for none."""
    body, _ = _request(skyrl, sampling_params={"max_tokens": 4})
    assert body["sampling_params"]["logprobs"] == 0

    body, _ = _request(skyrl, sampling_params={"logprobs": False})
    assert body["sampling_params"]["logprobs"] == 0


def test_sampling_params_vllm_would_reject_are_dropped(skyrl):
    body, _ = _request(
        skyrl,
        sampling_params={"max_tokens": 8, "temperature": 0.7, "user": "someone", "stream": True},
    )
    params = body["sampling_params"]
    assert params["max_tokens"] == 8 and params["temperature"] == 0.7
    assert "user" not in params and "stream" not in params


def test_cache_salt_and_model_ride_at_the_top_level(skyrl):
    body, _ = _request(skyrl, cache_salt="weights-7")
    assert body["cache_salt"] == "weights-7"
    assert body["model"] == "policy"


# -- the response -------------------------------------------------------------


def test_a_choice_is_read_back_into_what_a_turn_needs(skyrl):
    parsed = skyrl.token_response(_reply())
    assert parsed["completion_ids"] == [7, 8]
    assert parsed["completion_logprobs"] == [-0.1, -0.2]
    assert parsed["stop_reason"] == "stop"


def test_a_missing_finish_reason_defaults_rather_than_failing(skyrl):
    assert skyrl.token_response(_reply(finish=None))["stop_reason"] == "stop"


def test_a_response_without_logprobs_is_refused(skyrl):
    reply = _reply()
    del reply["choices"][0]["logprobs"]
    with pytest.raises(TokenUpstreamError, match="selected-token logprobs"):
        skyrl.token_response(reply)


def test_logprobs_that_do_not_cover_the_completion_are_refused(skyrl):
    with pytest.raises(TokenUpstreamError, match="different lengths"):
        skyrl.token_response(_reply(token_ids=(1, 2, 3), logprobs=(-0.1, -0.2)))


def test_a_real_batch_is_refused(skyrl):
    reply = _reply()
    reply["choices"].append(reply["choices"][0])
    with pytest.raises(TokenUpstreamError, match="exactly one choice"):
        skyrl.token_response(reply)


def test_routed_experts_are_omitted_not_guessed_at(skyrl):
    """SkyRL packs them under its own codec; capture does not own that format.

    Capture refuses a turn whose routed experts cover only part of the
    sequence, so a partial decode would be worse than none.
    """
    reply = _reply()
    reply["choices"][0]["routed_experts"] = {"data": "<base64>", "shape": [2, 4], "dtype": "int32"}
    assert skyrl.token_response(reply)["routed_experts"] is None


# -- the kind itself ----------------------------------------------------------


def test_it_is_a_tokens_target_in_every_other_respect(skyrl, default):
    assert skyrl.mode == "tokens" == default.mode
    assert skyrl.requires_tokenizer is True
    assert skyrl.client_suffix == default.client_suffix
    assert skyrl.describe()["name"] == "skyrl"
