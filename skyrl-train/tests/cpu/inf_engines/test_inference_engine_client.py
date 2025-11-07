"""
Test for `skyrl-train/skyrl_train/inference_engines/inference_engine_client.py` functinoalities
that can be mocked. Also tests for `skyrl-train/skyrl_train/inference_engines/utils.py`.

Run with:
uv run --isolated --extra dev pytest tests/cpu/inf_engines/test_inference_engine_client.py
"""

from http import HTTPStatus
from unittest.mock import patch

from skyrl_train.inference_engines.utils import (
    postprocess_completion_request,
    route_prompts_to_engines,
    hash_with_sha256,
)
from skyrl_train.inference_engines.inference_engine_client_http_endpoint import (
    ErrorResponse,
)
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from omegaconf import OmegaConf
import asyncio
import pytest
import random

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
