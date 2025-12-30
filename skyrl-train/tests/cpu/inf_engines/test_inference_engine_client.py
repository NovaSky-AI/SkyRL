"""
Test for `skyrl-train/skyrl_train/inference_engines/inference_engine_client.py` functinoalities
that can be mocked. Also tests for `skyrl-train/skyrl_train/inference_engines/utils.py`.

Run with:
uv run --isolated --extra dev pytest tests/cpu/inf_engines/test_inference_engine_client.py
"""

from http import HTTPStatus
from unittest.mock import patch

from transformers import AutoTokenizer
from skyrl_train.inference_engines.utils import (
    postprocess_completion_request,
    route_prompts_to_engines,
    hash_with_sha256,
)
from skyrl_train.inference_engines.inference_engine_client_http_endpoint import (
    ErrorResponse,
)
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.base import InferenceEngineInput, InferenceEngineOutput
from omegaconf import OmegaConf
import asyncio
import pytest
import random
from copy import deepcopy

# -------------------------------------------
# tests for postprocess_completion_request
# --------------------------------------------


def test_postprocess_single_string_no_session_id():
    prompt = "hello world"
    traj, processed = postprocess_completion_request(prompt, None)
    assert traj is None
    assert isinstance(processed, list)
    assert processed == [prompt]


def test_postprocess_single_string_scalar_session_id():
    prompt = "hello world"
    traj, processed = postprocess_completion_request(prompt, 123)
    assert traj == [123]
    assert processed == [prompt]


def test_postprocess_single_string_list_session_id_singleton():
    prompt = "hello world"
    traj, processed = postprocess_completion_request(prompt, ["abc"])  # accepts str ids
    assert traj == ["abc"]
    assert processed == [prompt]


def test_postprocess_single_string_list_session_id_wrong_len():
    prompt = "hello world"
    traj, processed = postprocess_completion_request(prompt, [1, 2])
    assert isinstance(traj, ErrorResponse)
    assert processed == [prompt]
    assert traj.error.code == HTTPStatus.BAD_REQUEST.value


def test_postprocess_single_token_ids_no_session_id():
    prompt = [1, 2, 3]
    traj, processed = postprocess_completion_request(prompt, None)
    assert traj is None
    assert processed == [prompt]


def test_postprocess_single_token_ids_scalar_session_id():
    prompt = [1, 2, 3]
    traj, processed = postprocess_completion_request(prompt, 7)
    assert traj == [7]
    assert processed == [prompt]


def test_postprocess_single_token_ids_list_session_id_singleton():
    prompt = [1, 2, 3]
    traj, processed = postprocess_completion_request(prompt, [8])
    assert traj == [8]
    assert processed == [prompt]


def test_postprocess_single_token_ids_list_session_id_wrong_len():
    prompt = [1, 2, 3]
    traj, processed = postprocess_completion_request(prompt, [8, 9])
    assert isinstance(traj, ErrorResponse)
    assert processed == [prompt]
    assert traj.error.code == HTTPStatus.BAD_REQUEST.value


def test_postprocess_batched_token_ids_no_session_id():
    prompt = [[1, 2], [3, 4, 5]]
    traj, processed = postprocess_completion_request(prompt, None)
    assert traj is None
    assert processed is prompt  # unchanged shape


def test_postprocess_batched_token_ids_with_matching_session_ids():
    prompt = [[1, 2], [3, 4, 5]]
    traj, processed = postprocess_completion_request(prompt, ["a", "b"])  # accepts str ids too
    assert traj == ["a", "b"]
    assert processed is prompt


def test_postprocess_batched_token_ids_with_wrong_session_ids_length():
    prompt = [[1, 2], [3, 4, 5]]
    traj, processed = postprocess_completion_request(prompt, [1])
    assert isinstance(traj, ErrorResponse)
    assert processed is prompt
    assert traj.error.code == HTTPStatus.BAD_REQUEST.value


def test_postprocess_batched_strings_no_session_id():
    prompt = ["p0", "p1"]
    traj, processed = postprocess_completion_request(prompt, None)
    assert traj is None
    assert processed is prompt


def test_postprocess_batched_strings_with_matching_session_ids():
    prompt = ["p0", "p1", "p2"]
    traj, processed = postprocess_completion_request(prompt, [10, 11, 12])
    assert traj == [10, 11, 12]
    assert processed is prompt


def test_postprocess_batched_strings_with_wrong_session_ids_length():
    prompt = ["p0", "p1", "p2"]
    traj, processed = postprocess_completion_request(prompt, [10, 11])
    assert isinstance(traj, ErrorResponse)
    assert processed is prompt
    assert traj.error.code == HTTPStatus.BAD_REQUEST.value


def test_postprocess_batched_strings_with_wrong_session_ids_length_2():
    prompt = ["p0", "p1", "p2"]
    traj, processed = postprocess_completion_request(prompt, 10)
    assert isinstance(traj, ErrorResponse)
    assert processed is prompt
    assert traj.error.code == HTTPStatus.BAD_REQUEST.value


# -------------------------------------------
# tests for InferenceEngineClient.completion
# --------------------------------------------


@pytest.mark.parametrize("num_prompts", [1, 50, 100])
@pytest.mark.parametrize("with_session_id", [True, False])
@pytest.mark.parametrize("num_engines", [1, 3, 4, 8, 16])
def test_completion_batched_routing_and_order_preservation(num_prompts, with_session_id, num_engines):
    """
    In InferenceEngineClient.completion, when the request is batched, we distribute the batch
    and route to engines. If session_id is provided, we map to the corresponding engine; if unprovided,
    we split it evenly. While the routing is done by `route_prompts_to_engines`, the aggregation is done
    by the client. We expect the aggregated results returned to the user in the original order, and
    this test checks exactly that.

    Related test: `test_route_prompts_to_engines_xxx` functions test the specific routing logic,
    while this will call `route_prompts_to_engines` and check the end-to-end behavior.
    """

    class MockEngine:
        async def completion(self, request_payload):
            """
            Given input [i, j, k, ...], return output [f"{i}{i}", f"{j}{j}", f"{k}{k}", ...] with
            indices 0, 1, 2, 3, ...
            """
            body = request_payload["json"]
            my_prompts = body["prompt"]
            # Return per-sub-batch indices 0..len-1; client is expected to remap to global order
            choices = []
            for i, p in enumerate(my_prompts):
                choices.append(
                    {
                        "index": i,
                        "text": f"{p}{p}",
                        "finish_reason": "stop",
                    }
                )
            num_prompt_tokens = sum(len(p) for p in my_prompts)
            num_completion_tokens = num_prompt_tokens * 2  # since we doubled the prompts
            return {
                "id": "cmpl-mock",
                "object": "text_completion",
                "model": body.get("model", "dummy-model"),
                "choices": choices,
                "usage": {
                    "prompt_tokens": num_prompt_tokens,
                    "total_tokens": num_prompt_tokens + num_completion_tokens,
                    "completion_tokens": num_completion_tokens,
                    "prompt_tokens_details": {
                        "cached_tokens": num_prompt_tokens,
                    },
                },
            }

    # Create a minimal config to avoid spinning up HTTP endpoint
    cfg = OmegaConf.create(
        {
            "trainer": {
                "policy": {"model": {"path": "dummy-model"}},
            },
            "generator": {
                "backend": "vllm",
                "enable_http_endpoint": False,
                "http_endpoint_host": "127.0.0.1",
                "http_endpoint_port": 0,
            },
        }
    )

    engines = [MockEngine() for _ in range(num_engines)]
    tokenizer = object()  # not used by completion()
    client = InferenceEngineClient(engines=engines, tokenizer=tokenizer, full_config=cfg)

    prompts = [str(i) for i in range(num_prompts)]
    if with_session_id:
        session_ids = [random.randint(1, 100) for _ in range(num_prompts)]
    else:
        session_ids = None
    request_payload = {
        "json": {
            "model": "dummy-model",
            "prompt": prompts,
            "session_id": session_ids,
            "max_tokens": 32,
        },
        "headers": {"Content-Type": "application/json"},
    }

    resp = asyncio.run(client.completion(request_payload))

    assert resp.get("object") != "error"
    assert "choices" in resp and len(resp["choices"]) == len(prompts)
    # Ensure outputs align with inputs and indices are global order 0..n-1
    expected_texts = [f"{i}{i}" for i in range(num_prompts)]
    for i, choice in enumerate(resp["choices"]):
        assert choice["index"] == i
        assert choice["text"] == expected_texts[i]

    # also check usage aggregation here
    global_num_prompt_tokens = sum(len(p) for p in prompts)
    global_num_completion_tokens = global_num_prompt_tokens * 2  # since we doubled the prompts
    assert resp["usage"] == {
        "prompt_tokens": global_num_prompt_tokens,
        "total_tokens": global_num_prompt_tokens + global_num_completion_tokens,
        "completion_tokens": global_num_completion_tokens,
        "prompt_tokens_details": {
            "cached_tokens": global_num_prompt_tokens,
        },
    }


# -------------------------------------------
# tests for InferenceEngineClient.generate
# --------------------------------------------


@pytest.mark.parametrize("num_prompts", [1, 50, 100])
@pytest.mark.parametrize("with_session_id", [True, False])
@pytest.mark.parametrize("num_engines", [1, 3, 4, 8, 16])
def test_generate_batched_routing_and_order_preservation(num_prompts, with_session_id, num_engines):
    """
    See the `test_completion_batched_routing_and_order_preservation` test for more details.
    Essentially `InferenceEngineClient.generate` does the same routing and aggregation as
    `InferenceEngineClient.completion`.
    """

    class MockEngine:
        async def generate(self, input_batch):
            # input_batch["prompt_token_ids"] is a local sub-batch list of token id lists
            prompt_token_ids = input_batch["prompt_token_ids"]
            responses = []
            response_ids = []
            stop_reasons = []
            for ids in prompt_token_ids:
                # construct a deterministic text and token output based on first id
                base = ids[0]
                responses.append(f"{base}{base}")
                response_ids.append([base, base])
                stop_reasons.append("stop")
            return {
                "responses": responses,
                "response_ids": response_ids,
                "stop_reasons": stop_reasons,
            }

    # Minimal config, do not spin up HTTP endpoint
    cfg = OmegaConf.create(
        {
            "trainer": {
                "policy": {"model": {"path": "dummy-model"}},
            },
            "generator": {
                "backend": "vllm",
                "enable_http_endpoint": False,
                "http_endpoint_host": "127.0.0.1",
                "http_endpoint_port": 0,
            },
        }
    )

    engines = [MockEngine() for _ in range(num_engines)]
    tokenizer = object()  # not used when prompt_token_ids are provided
    client = InferenceEngineClient(engines=engines, tokenizer=tokenizer, full_config=cfg)

    # Build token id prompts [[0], [1], ..., [n-1]]
    prompt_token_ids = [[i] for i in range(num_prompts)]
    if with_session_id:
        session_ids = [random.randint(1, 100) for _ in range(num_prompts)]
    else:
        session_ids = None

    input_batch = {
        "prompts": None,
        "prompt_token_ids": prompt_token_ids,
        "sampling_params": None,
        "session_ids": session_ids,
    }

    out = asyncio.run(client.generate(input_batch))

    # Validate reconstruction and ordering
    assert len(out["responses"]) == num_prompts
    assert len(out["response_ids"]) == num_prompts
    assert len(out["stop_reasons"]) == num_prompts
    expected_texts = [f"{i}{i}" for i in range(num_prompts)]
    for i in range(num_prompts):
        assert out["responses"][i] == expected_texts[i]
        assert out["response_ids"][i] == [i, i]
        assert out["stop_reasons"][i] == "stop"


# -----------------------------
# Test for route_prompts_to_engines function that routes prompts to inference engines
# in inference engine client.
# -------------------------------


def test_route_prompts_to_engines_single_prompt_no_trajectory_random_engine():
    # Force deterministic random routing to engine index 1
    with patch("random.randint", return_value=1):
        mapping = route_prompts_to_engines(num_prompts=1, num_inference_engines=4, session_ids=None)
    assert mapping == {1: [0]}


def test_route_prompts_to_engines_batched_even_split_exact_multiple():
    # 4 prompts, 2 engines => [0,1] and [2,3]
    num_prompts = 4
    num_engines = 2
    mapping = route_prompts_to_engines(num_prompts=num_prompts, num_inference_engines=num_engines, session_ids=None)
    assert mapping == {0: [0, 1], 1: [2, 3]}


def test_route_prompts_to_engines_batched_uneven_split():
    # 5 prompts, 2 engines => ceil(5/2)=3 => [0,1,2] and [3,4]
    mapping = route_prompts_to_engines(num_prompts=5, num_inference_engines=2, session_ids=None)
    assert mapping == {0: [0, 1, 2], 1: [3, 4]}

    # 5 prompts, 3 engines => ceil(5/3)=2 => [0,1] and [2,3] and [4]
    mapping = route_prompts_to_engines(num_prompts=5, num_inference_engines=3, session_ids=None)
    assert mapping == {0: [0, 1], 1: [2, 3], 2: [4]}

    # 5 prompts, 4 engines => ceil(5/4)=2 => [0,1] and [2,3] and [4]
    mapping = route_prompts_to_engines(num_prompts=5, num_inference_engines=4, session_ids=None)
    assert mapping == {0: [0, 1], 1: [2, 3], 2: [4]}

    # 129 prompts, 4 engines => ceil(129/4)=33 => [0,1,2,...,32] and [33,34,35,...,65] and [66,67,68,...,99] and [100,101,102,...,128]
    mapping = route_prompts_to_engines(num_prompts=129, num_inference_engines=4, session_ids=None)
    assert mapping == {0: list(range(33)), 1: list(range(33, 66)), 2: list(range(66, 99)), 3: list(range(99, 129))}


def test_route_prompts_to_engines_batched_more_engines_than_prompts():
    # 2 prompts, 4 engines => size=1 => {0:[0], 1:[1]}
    mapping = route_prompts_to_engines(num_prompts=2, num_inference_engines=4, session_ids=None)
    assert mapping == {0: [0], 1: [1]}


def test_route_prompts_to_engines_with_session_ids_grouping_and_partition():
    num_engines = 4
    # Ensure same session IDs route to the same engine index
    sids = ["A", "A", "B", "C", "B"]
    # hash A ends in 45, B ends in 44, C ends in 69, with % 4 they become 1, 0, 1
    engine_idx = [hash_with_sha256(sid) % num_engines for sid in sids]  # what we do in route_prompts_to_engines
    assert engine_idx == [1, 1, 0, 1, 0]
    mapping = route_prompts_to_engines(num_prompts=5, num_inference_engines=num_engines, session_ids=sids)

    assert mapping == {1: [0, 1, 3], 0: [2, 4]}


def test_route_prompts_to_engines_validation_errors():
    # num_prompts must be > 0
    with pytest.raises(AssertionError):
        route_prompts_to_engines(num_prompts=0, num_inference_engines=1, session_ids=None)

    # num_inference_engines must be > 0
    with pytest.raises(AssertionError):
        route_prompts_to_engines(num_prompts=1, num_inference_engines=0, session_ids=None)

    # session_ids length must match
    with pytest.raises(AssertionError):
        route_prompts_to_engines(num_prompts=2, num_inference_engines=1, session_ids=["x"])  # len 1 != 2

    # session_ids type checking
    with pytest.raises(AssertionError):
        route_prompts_to_engines(num_prompts=2, num_inference_engines=1, session_ids=[1, 2.0])  # float invalid

    # No error
    route_prompts_to_engines(num_prompts=2, num_inference_engines=1, session_ids=[1, 2])
    route_prompts_to_engines(num_prompts=2, num_inference_engines=1, session_ids=None)
    route_prompts_to_engines(num_prompts=1, num_inference_engines=1, session_ids=None)


# -------------------------------------------
# tests for InferenceEngineClient.chat_completion retry logic
# --------------------------------------------


def _make_min_cfg():
    return OmegaConf.create(
        {
            "trainer": {
                "policy": {"model": {"path": "dummy-model"}},
            },
            "generator": {
                "backend": "vllm",
                "enable_http_endpoint": False,
                "http_endpoint_host": "127.0.0.1",
                "http_endpoint_port": 0,
            },
        }
    )


@pytest.mark.asyncio
async def test_chat_completion_retry_accumulates_and_sends_continuations():
    """
    First response aborts with tokens; second aborts with 0 tokens (ignored);
    third finishes. Assert:
    - Continuation requests append accumulated assistant content with correct role
    - continue_final_message/add_generation_prompt flags are set
    - remaining max_tokens decreases by accumulated completion tokens
    - Final response accumulates content, logprobs, token_ids and recomputes usage correctly
    - Each retry request is what we expect the engine to receive
    """

    class MockEngine:
        def __init__(self):
            self.calls = []  # capture full request payloads {"json":..., "headers":...}
            # Pre-programmed partial responses
            self.responses = [
                # 1) abort with 1 token "A"
                {
                    "id": "cmpl-1",
                    "object": "chat.completion",
                    "model": "dummy-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "A"},
                            "finish_reason": "abort",
                            "logprobs": {
                                "content": [
                                    {
                                        "token": "token_id:11",
                                        "logprob": -0.1,
                                        "bytes": [84, 111],
                                        "top_logprobs": [{"token": "token_id:11", "logprob": -0.1, "bytes": [116]}],
                                    },
                                ]
                            },
                            "token_ids": [11],
                        }
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
                },
                # 2) abort with 0 tokens (should be ignored for accumulation)
                {
                    "id": "cmpl-2",
                    "object": "chat.completion",
                    "model": "dummy-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": ""},
                            "finish_reason": "abort",
                            "logprobs": {"content": []},
                            "token_ids": [],
                        }
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5},
                },
                # 3) finish with 1 token "B"
                {
                    "id": "cmpl-3",
                    "object": "chat.completion",
                    "model": "dummy-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "B"},
                            "finish_reason": "stop",
                            "logprobs": {
                                "content": [
                                    {
                                        "token": "token_id:12",
                                        "logprob": -0.1,
                                        "bytes": [84, 111],
                                        "top_logprobs": [{"token": "token_id:12", "logprob": -0.1, "bytes": [116]}],
                                    },
                                ]
                            },
                            "token_ids": [12],
                        }
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
                },
            ]

        async def chat_completion(self, request_payload):
            self.calls.append(deepcopy(request_payload))
            idx = len(self.calls) - 1
            assert idx < len(self.responses), f"Unexpected extra call {idx}"
            return deepcopy(self.responses[idx])

    engines = [MockEngine()]
    cfg = _make_min_cfg()
    client = InferenceEngineClient(engines=engines, tokenizer=object(), full_config=cfg)

    original = {
        "json": {
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 8,
            # ask for structures that the client can accumulate
            "logprobs": True,
            "top_logprobs": 1,
            "return_tokens_as_token_ids": True,
        },
        "headers": {"Content-Type": "application/json"},
    }

    out = await client.chat_completion(original)

    # Verify engine received 3 calls
    assert len(engines[0].calls) == 3
    first_call = engines[0].calls[0]
    second_call = engines[0].calls[1]
    third_call = engines[0].calls[2]

    # First call should be identical to original json (no continuation flags)
    assert first_call["json"] == original["json"]
    assert first_call["headers"] == original["headers"]
    assert first_call["json"].get("continue_final_message") is None
    assert first_call["json"].get("add_generation_prompt") is None
    assert first_call["json"]["messages"] == [{"role": "user", "content": "Hi"}]
    assert first_call["json"]["max_tokens"] == 8

    # Second/third calls should be continuation requests
    for call in (second_call, third_call):
        assert call["headers"] == original["headers"]
        # Flags
        assert call["json"].get("continue_final_message") is True
        assert call["json"].get("add_generation_prompt") is False
        # Accumulated assistant message appended with content "A"
        assert call["json"]["messages"][-1] == {"role": "assistant", "content": "A"}
        # Original user message preserved
        assert call["json"]["messages"][0] == {"role": "user", "content": "Hi"}
        # Remaining max_tokens reduced by 1 (we already generated one token)
        assert call["json"].get("max_tokens") == 7
        # Other params preserved
        assert call["json"]["model"] == "dummy-model"
        assert call["json"]["logprobs"] is True
        assert call["json"]["top_logprobs"] == 1
        assert call["json"]["return_tokens_as_token_ids"] is True

    # Final response should accumulate content/logprobs/token_ids and usage
    choice = out["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["content"] == "AB"
    assert len(choice["logprobs"]["content"]) == 2
    assert choice["logprobs"]["content"][0]["token"] == "token_id:11"
    assert choice["logprobs"]["content"][1]["token"] == "token_id:12"
    assert choice["token_ids"] == [11, 12]

    # usage: prompt_tokens from base (5), completion_tokens summed (2), total 7
    assert out["usage"]["prompt_tokens"] == 5
    assert out["usage"]["completion_tokens"] == 2
    assert out["usage"]["total_tokens"] == 7


@pytest.mark.asyncio
async def test_chat_completion_retry_resends_original_when_no_tokens_generated_yet():
    """
    First response aborts with 0 tokens, so the next request should resend the original
    payload unchanged. Second response finishes; client returns it directly.
    """

    class MockEngine:
        def __init__(self):
            self.calls = []  # capture full payloads
            self.responses = [
                # 1) abort with 0 tokens
                {
                    "id": "cmpl-a1",
                    "object": "chat.completion",
                    "model": "dummy-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": ""},
                            "finish_reason": "abort",
                            "logprobs": {"content": []},
                            "token_ids": [],
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10},
                },
                # 2) finish with tokens "XYZ" (3 tokens)
                {
                    "id": "cmpl-a2",
                    "object": "chat.completion",
                    "model": "dummy-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "XYZ"},
                            "finish_reason": "stop",
                            "logprobs": {
                                "content": [
                                    {
                                        "token": "token_id:21",
                                        "logprob": -0.1,
                                        "bytes": [84, 111],
                                        "top_logprobs": [{"token": "token_id:21", "logprob": -0.1, "bytes": [116]}],
                                    },
                                    {
                                        "token": "token_id:22",
                                        "logprob": -0.1,
                                        "bytes": [84, 111],
                                        "top_logprobs": [{"token": "token_id:22", "logprob": -0.1, "bytes": [116]}],
                                    },
                                    {
                                        "token": "token_id:23",
                                        "logprob": -0.1,
                                        "bytes": [84, 111],
                                        "top_logprobs": [{"token": "token_id:23", "logprob": -0.1, "bytes": [116]}],
                                    },
                                ]
                            },
                            "token_ids": [21, 22, 23],
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
                },
            ]

        async def chat_completion(self, request_payload):
            self.calls.append(deepcopy(request_payload))
            return deepcopy(self.responses[len(self.calls) - 1])

    engines = [MockEngine()]
    cfg = _make_min_cfg()
    client = InferenceEngineClient(engines=engines, tokenizer=object(), full_config=cfg)

    original = {
        "json": {
            "model": "dummy-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 16,
            "logprobs": True,
            "top_logprobs": 1,
        },
        "headers": {"Content-Type": "application/json"},
    }

    out = await client.chat_completion(original)

    # Two calls should have been made
    assert len(engines[0].calls) == 2
    first_call = engines[0].calls[0]
    second_call = engines[0].calls[1]

    # After 0-token abort, the next call should resend the original unchanged
    assert first_call["json"] == original["json"]
    assert second_call["json"] == original["json"]
    assert first_call["headers"] == original["headers"]
    assert second_call["headers"] == original["headers"]
    # No continuation flags should appear
    assert first_call["json"].get("continue_final_message") is None
    assert second_call["json"].get("continue_final_message") is None

    # Since finish_reason != abort on the second call and base_response was None,
    # client should return the second response directly (no accumulation)
    assert out == engines[0].responses[1]


# -------------------------------------------
# tests for InferenceEngineClient.generate retry logic
# --------------------------------------------


@pytest.mark.parametrize("max_tokens_key", ["max_tokens", "max_completion_tokens"])
@pytest.mark.asyncio
async def test_generate_retry_some_gen_no_gen_finish(max_tokens_key):
    """
    Test that generate() with retry logic properly accumulates tokens and adjusts subsequent requests.

    First response aborts with tokens [21, 22]; second aborts with 0 tokens (ignored);
    third finishes with tokens [23, 24]. Assert:
    - Continuation requests append accumulated tokens to prompt_token_ids
    - remaining max_tokens decreases by accumulated tokens
    - Final response accumulates all tokens and uses last stop_reason
    """

    class MockEngine:
        def __init__(self):
            self.calls = []  # capture InferenceEngineInput calls
            # Pre-programmed responses
            self.responses = [
                # 1) abort with 2 tokens
                InferenceEngineOutput(
                    responses=["something"],  # will be ignored since we decode the final output
                    response_ids=[[21, 22]],
                    stop_reasons=["abort"],
                    response_logprobs=[[-0.1, -0.2]],
                ),
                # 2) abort with 0 tokens (should be ignored)
                InferenceEngineOutput(
                    responses=[""],
                    response_ids=[[]],
                    stop_reasons=["abort"],
                    response_logprobs=None,
                ),
                # 3) finish with 2 tokens
                InferenceEngineOutput(
                    responses=[" something"],  # will be ignored since we decode the final output
                    response_ids=[[23, 24]],
                    stop_reasons=["stop"],
                    response_logprobs=[[-0.3, -0.4]],
                ),
            ]

        async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
            self.calls.append(deepcopy(input_batch))
            idx = len(self.calls) - 1
            assert idx < len(self.responses), f"Unexpected extra call {idx}"
            return deepcopy(self.responses[idx])

    engines = [MockEngine()]
    cfg = _make_min_cfg()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    client = InferenceEngineClient(engines=engines, tokenizer=tokenizer, full_config=cfg)

    # Original request
    prompt_token_ids = [[1, 2, 3, 4, 5]]  # 5 prompt tokens
    sampling_params = {max_tokens_key: 10, "temperature": 0.7}

    input_batch = InferenceEngineInput(
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
    )

    out = await client.generate(input_batch)

    # Verify engine received 3 calls
    assert len(engines[0].calls) == 3
    first_call = engines[0].calls[0]
    second_call = engines[0].calls[1]
    third_call = engines[0].calls[2]

    # First call should have original prompt
    assert first_call["prompt_token_ids"] == [[1, 2, 3, 4, 5]]
    assert first_call["sampling_params"][max_tokens_key] == 10
    assert first_call["sampling_params"]["temperature"] == 0.7

    # Second call should have prompt + first response tokens
    assert second_call["prompt_token_ids"] == [[1, 2, 3, 4, 5, 21, 22]]
    assert second_call["sampling_params"][max_tokens_key] == 8  # 10 - 2 already generated
    assert second_call["sampling_params"]["temperature"] == 0.7

    # Third call should also have prompt + first response tokens (second was ignored)
    assert third_call["prompt_token_ids"] == [[1, 2, 3, 4, 5, 21, 22]]
    assert third_call["sampling_params"][max_tokens_key] == 8  # 10 - 2 already generated
    assert third_call["sampling_params"]["temperature"] == 0.7

    # Final response should accumulate all tokens
    expected_final_response_ids = [21, 22, 23, 24]
    expected_final_text_response = tokenizer.decode(expected_final_response_ids, skip_special_tokens=True)
    assert out["responses"] == [expected_final_text_response]
    assert out["response_ids"] == [expected_final_response_ids]
    assert out["stop_reasons"] == ["stop"]
    assert out["response_logprobs"] == [[-0.1, -0.2, -0.3, -0.4]]


@pytest.mark.asyncio
async def test_generate_retry_direct_return():
    """
    Test that if the first generate() request doesn't abort, it returns directly without retries.
    """

    class MockEngine:
        def __init__(self):
            self.calls = []
            # Single response that completes immediately
            self.response = InferenceEngineOutput(
                responses=["something"],
                response_ids=[[21, 22, 23, 24]],
                stop_reasons=["stop"],
                response_logprobs=[[-0.1, -0.2, -0.3, -0.4]],
            )

        async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
            self.calls.append(deepcopy(input_batch))
            return deepcopy(self.response)

    engines = [MockEngine()]
    cfg = _make_min_cfg()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    client = InferenceEngineClient(engines=engines, tokenizer=tokenizer, full_config=cfg)

    prompt_token_ids = [[1, 2, 3, 4, 5]]
    sampling_params = {"max_tokens": 10}

    input_batch = InferenceEngineInput(
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
    )

    out = await client.generate(input_batch)

    # Verify only one call was made
    assert len(engines[0].calls) == 1

    # Verify response is returned as-is
    expected_final_response_ids = [21, 22, 23, 24]
    # note since we completed in one turn, we return the text response of the first turn returned by
    # the underlying engine instead re-tokenizing like others
    assert out["responses"] == ["something"]
    assert out["response_ids"] == [expected_final_response_ids]
    assert out["stop_reasons"] == ["stop"]
    assert out["response_logprobs"] == [[-0.1, -0.2, -0.3, -0.4]]


@pytest.mark.asyncio
async def test_generate_retry_no_gen_finish():
    """
    First response aborts with 0 tokens; next finishes.
    The second request should resend the original unchanged and the final output equals the second response.
    """
    final_response_ids = [21, 22, 23]

    class MockEngine:
        def __init__(self):
            self.calls = []
            self.responses = [
                # 1) abort with 0 tokens
                InferenceEngineOutput(
                    responses=[""],
                    response_ids=[[]],
                    stop_reasons=["abort"],
                    response_logprobs=[[]],
                ),
                # 2) finish directly
                InferenceEngineOutput(
                    responses=["something"],  # will be ignored since we decode the final output
                    response_ids=[final_response_ids],
                    stop_reasons=["stop"],
                    response_logprobs=[[-0.1, -0.1, -0.1]],
                ),
            ]

        async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
            self.calls.append(deepcopy(input_batch))
            return deepcopy(self.responses[len(self.calls) - 1])

    engines = [MockEngine()]
    cfg = _make_min_cfg()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    client = InferenceEngineClient(engines=engines, tokenizer=tokenizer, full_config=cfg)

    original_prompt_ids = [7, 8, 9]
    input_batch = InferenceEngineInput(
        prompt_token_ids=[original_prompt_ids],
        sampling_params={"max_tokens": 16},
    )

    out = await client.generate(input_batch)

    # Two calls should have been made, identical inputs since first had 0 tokens
    assert len(engines[0].calls) == 2
    first_call, second_call = engines[0].calls
    assert first_call["prompt_token_ids"] == [original_prompt_ids]
    assert second_call["prompt_token_ids"] == [original_prompt_ids]
    assert first_call["sampling_params"]["max_tokens"] == 16
    assert second_call["sampling_params"]["max_tokens"] == 16

    # Since finish_reason != abort on the second call and no accumulation occurred,
    # client should return the second response directly (no aggregation)
    # Besides, since we completed in one turn, we return the text response of the first turn returned by
    # the underlying engine instead re-tokenizing the accumulated tokens
    assert out == engines[0].responses[1]
