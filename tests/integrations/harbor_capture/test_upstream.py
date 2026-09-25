"""SkyRL's fork of vLLM's token-in/token-out endpoint.

capture ships the wire as `vllm`, because the payloads are vLLM's own and
`vllm serve` mounts `/inference/v1/generate` itself. Three things are ours: the
path, `cache_salt`, and routed experts. Those are what this file pins.

The wire itself is capture's and is tested there, in
`tests/test_upstream_vllm.py`. The stub in `conftest.py` returns sentinels from
`request`/`response`, so a test here can tell what the fork *added* from what
it reimplemented -- and a fork that grows a second copy of the wire fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from skyrl.backends.skyrl_train.inference_servers.generate_wire import (
    pack_routed_experts,
)


def call(registered, **overrides):
    kwargs = {
        "prompt_token_ids": [1, 2, 3],
        "sampling_params": {"temperature": 0.7},
        "model": "policy",
        "session_id": "tr_01ABC",
    }
    kwargs.update(overrides)
    return registered.request(**kwargs)


def reply(routed=None, token_ids=(7, 8, 9)):
    choice = {"token_ids": list(token_ids), "finish_reason": "stop"}
    if routed is not None:
        choice["routed_experts"] = routed
    return {"choices": [choice]}


# -- the name and the path -----------------------------------------------------
def test_importing_the_module_registers_the_fork(registered):
    """capture resolves an upstream by name, so the name is the contract."""
    assert registered.name == "skyrl"


def test_it_serves_the_same_shape_on_skyrls_path(registered):
    assert registered.generate_path == "/skyrl/v1/generate"
    assert registered.url("http://router:8000") == "http://router:8000/skyrl/v1/generate"
    assert registered.url("http://router:8000/") == "http://router:8000/skyrl/v1/generate"
    already = "http://router:8000/skyrl/v1/generate"
    assert registered.url(already) == already


def test_the_wire_itself_is_inherited_not_reimplemented(registered, inherited):
    """If this starts failing, the fork has grown a second copy of the wire --
    which is what putting it in capture was meant to prevent."""
    body, headers = call(registered)
    expected_body, expected_headers = inherited
    assert body == expected_body
    assert headers == expected_headers


# -- cache_salt ----------------------------------------------------------------
def test_cache_salt_is_lifted_to_the_top_level(registered):
    """capture carries it without knowing it -- it is just one of the request
    fields capture has no opinion about. Deciding it means "prefix-cache key"
    is this protocol's job, and vLLM reads it from the request body."""
    body, _ = call(registered, sampling_params={"temperature": 0.7, "cache_salt": "weights-42"})
    assert body["cache_salt"] == "weights-42"


def test_cache_salt_is_taken_out_before_the_base_class_sees_it(registered):
    """`SamplingParams` rejects a request carrying a key it does not know, so
    the lift has to happen before the inherited wire filters what is left."""
    call(registered, sampling_params={"temperature": 0.7, "cache_salt": "weights-42"})
    handed_on = registered.seen["sampling_params"]
    assert "cache_salt" not in handed_on
    assert handed_on["temperature"] == 0.7


def test_no_cache_salt_is_sent_when_the_caller_did_not_ask(registered):
    """Absent, not null: the field is optional at the endpoint."""
    body, _ = call(registered)
    assert "cache_salt" not in body


# -- routed experts ------------------------------------------------------------
def test_routed_experts_decode_and_align_with_the_completion(registered):
    """One row per sampled token, each row layers x experts. This is the whole
    reason the fork exists rather than using vLLM's own path."""
    routes = np.arange(3 * 2 * 4, dtype=np.int64).reshape(3, 2, 4)
    result = registered.response(reply(routed=pack_routed_experts(routes)))
    assert result["routed_experts"] == routes.tolist()
    assert len(result["routed_experts"]) == len(result["completion_ids"])


def test_no_routed_experts_is_a_supported_state(registered):
    """Not requested, or an engine built without them. Absent is fine; wrong
    is not."""
    assert registered.response(reply())["routed_experts"] is None


def test_routed_experts_covering_the_wrong_tokens_fail_closed(registered, upstream_error):
    """capture cannot align a partial mask, and padding one would be a lie
    about which expert served which token."""
    routes = np.arange(2 * 2 * 4, dtype=np.int64).reshape(2, 2, 4)
    with pytest.raises(upstream_error, match="routed experts cover 2 tokens"):
        registered.response(reply(routed=pack_routed_experts(routes)))


@pytest.mark.parametrize(
    "routed",
    [
        [[1, 2]],
        "packed",
        17,
        {"shape": [1, 1, 1]},
        {"data": "!!", "shape": [1], "dtype": "uint8"},
    ],
)
def test_a_payload_that_is_not_the_packed_object_fails_closed(registered, upstream_error, routed):
    """The endpoint used to send nested lists. Anything but the packed object
    is a version skew, and guessing at it would mistrain silently."""
    with pytest.raises(upstream_error):
        registered.response(reply(routed=routed))
