"""
Test pause and continue generation with inference engine client HTTP endpoint.

uv run --isolated --extra dev --extra vllm pytest tests/gpu/gpu_ci/test_pause_and_continue_generation.py -m "vllm"
"""

import pytest
import asyncio
from tests.gpu.gpu_ci.test_inference_engine_client_http_endpoint import get_test_actor_config
from tests.gpu.utils import init_inference_engines, get_test_prompts
from skyrl_train.inference_engines.base import ConversationType
from transformers import AutoTokenizer
from typing import List
from skyrl_train.inference_engines.inference_engine_client_http_endpoint import (
    serve,
    wait_for_server_ready,
    shutdown_server,
)
import threading
import requests

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TP_SIZE = 2
SERVER_PORT = 8123
SERVER_HOST = "127.0.0.1"


@pytest.mark.vllm
def test_continue_generation_vllm_engine_chat_completion(ray_init_fixture):
    """
    We send 6 requests via `/chat/completions` to two engines concurrently with vLLM `max_num_seqs=2`
    so that in each engine, 2 run and 1 wait. We ignore eos and let model geneate 2048 tokens.
    We pause and then resume generation twice in the middle. We expect each response to
    finish with reason `length` and have exactly `max_tokens` completion tokens.
    """
    server_thread = None
    num_engines = 2
    num_requests = 6
    max_num_seqs = 2
    try:
        # 1. Build engine and start server
        cfg = get_test_actor_config(num_inference_engines=num_engines, model=MODEL)
        cfg.trainer.placement.colocate_all = True
        cfg.generator.weight_sync_backend = "nccl"
        cfg.trainer.strategy = "fsdp2"
        sampling_params = {
            "max_tokens": 2048,
            "stop": None,
            "stop_token_ids": None,
            "ignore_eos": True,
            "stream": False,
            "temperature": 0.0,
            # Ensure logprobs and token ids are returned for accumulation checks
            "logprobs": True,
            "top_logprobs": 1,
            "return_tokens_as_token_ids": True,
        }
        client, _ = init_inference_engines(
            cfg=cfg,
            use_local=True,
            async_engine=cfg.generator.async_engine,
            tp_size=cfg.generator.inference_engine_tensor_parallel_size,
            colocate_all=cfg.trainer.placement.colocate_all,
            backend="vllm",
            model=MODEL,
            num_inference_engines=cfg.generator.num_inference_engines,
            sleep_level=1,
            # We test aborting 2 running requests and 1 waiting requests
            max_num_seqs=max_num_seqs,
        )

        def run_server():
            serve(client, host=SERVER_HOST, port=SERVER_PORT, log_level="warning")

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        wait_for_server_ready(host=SERVER_HOST, port=SERVER_PORT, max_wait_seconds=30)
        base_url = f"http://{SERVER_HOST}:{SERVER_PORT}/v1"

        # 2. Prepare input
        messages: List[ConversationType] = get_test_prompts(MODEL, num_samples=1)[0]

        # 3. Fire 6 concurrent HTTP requests, then pause/resume mid-flight
        results = {}

        def send_request(i: int):
            r = requests.post(
                f"{base_url}/chat/completions",
                json={"model": MODEL, "messages": messages, **sampling_params},
            )
            # Store minimal structured result for assertions
            content_type = r.headers.get("Content-Type", "")
            resp_json = r.json() if content_type.startswith("application/json") else {}
            results[i] = {
                "status_code": r.status_code,
                "text": r.text,
                "response": resp_json,
            }

        threads = [threading.Thread(target=send_request, args=(i,), daemon=True) for i in range(num_requests)]
        for t in threads:
            t.start()

        # Let the requests start and enqueue; with max_num_seqs=2, 2 run and 1 wait
        asyncio.run(asyncio.sleep(1))

        # Pause then resume while requests are in-flight
        asyncio.run(client.pause_generation())
        asyncio.run(client.resume_generation())
        # Run for another two seconds, then pause and resume again
        asyncio.run(asyncio.sleep(2))
        asyncio.run(client.pause_generation())
        asyncio.run(client.resume_generation())

        # Wait for all requests to finish
        for t in threads:
            t.join(timeout=180)

        # Ensure we collected all num_requests results
        assert len(results) == num_requests, f"Expected {num_requests} responses, got {len(results)}"

        # 4. Validate each output: finish_reason is length and completion_tokens == max_tokens
        for i in range(num_requests):
            assert i in results, f"Missing result for index {i}"
            cur = results[i]
            assert cur.get("status_code") == 200, f"Request {i} failed: {cur.get('status_code')} {cur.get('text')}"
            out = cur["response"]
            assert "choices" in out and len(out["choices"]) == 1, f"Invalid choices for request {i}: {out}"
            assert (
                out["choices"][0].get("finish_reason") == "length"
            ), f"Request {i} finish_reason is not 'length': {out['choices'][0].get('finish_reason')}"

            choice = out["choices"][0]
            logprobs = choice["logprobs"]
            token_count_from_logprobs = len(logprobs["content"])
            print(f"Output first 1500 chars: {choice['message']['content'][:1500]}...")

            # Check completion tokens
            assert (
                out["usage"]["completion_tokens"] == sampling_params["max_tokens"]
            ), f"Request {i} expected completion_tokens={sampling_params['max_tokens']}, got {out['usage']['completion_tokens']}"
            assert (
                token_count_from_logprobs == sampling_params["max_tokens"]
            ), f"Request {i} expected {sampling_params['max_tokens']} tokens from logprobs, got {token_count_from_logprobs}"

            # Spot-check structure of each logprob entry: token contains token_id and top_logprobs length matches request
            top_logprobs = sampling_params["top_logprobs"]
            for entry in logprobs["content"]:
                # tokens are token_id:<int> when return_tokens_as_token_ids=True
                parts = str(entry["token"]).split(":")
                assert (
                    len(parts) >= 2 and parts[-1].isdigit()
                ), f"Request {i} token field not token_id:int: {entry['token']}"
                assert (
                    len(entry["top_logprobs"]) == top_logprobs
                ), f"Request {i} expected top_logprobs len {top_logprobs}, got {len(entry['top_logprobs'])}"
            # Check prompt tokens
            prompt_tokens = client.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
            assert (
                len(prompt_tokens) == out["usage"]["prompt_tokens"]
            ), f"Request {i} expected {len(prompt_tokens)} tokens from prompt, got {out['usage']['prompt_tokens']}"
            # TODO(Charlie): after we bump vllm such that it supports returnining tokens, check `choice["token_ids"]`
            # TODO(Charlie): after we add model version to the output, check that as well
    finally:
        shutdown_server(host=SERVER_HOST, port=SERVER_PORT, max_wait_seconds=5)
        if server_thread is not None and server_thread.is_alive():
            server_thread.join(timeout=5)


@pytest.mark.vllm
def test_continue_generation_generate_vllm_engine_generation(ray_init_fixture):
    """
    Identical to `test_continue_generation_vllm_engine_chat_completion` except we use `generate()`
    instead of `/chat/completions`.

    Launch 6 concurrent single-request generate() calls against two engines with vLLM `max_num_seqs=2`
    so that in each engine, 2 run and 1 wait. Ignore EOS and request a long generation (2048 tokens).
    Pause and then resume generation twice mid-flight. Expect each request to finish with reason `length`
    and have exactly `max_tokens` completion tokens (i.e., len(response_ids[0]) == max_tokens and
    len(response_logprobs[0]) == max_tokens).
    """
    num_engines = 2
    num_requests = 6
    max_num_seqs = 2

    # 1. Build engines (no HTTP server needed for generate())
    cfg = get_test_actor_config(num_inference_engines=num_engines, model=MODEL)
    cfg.trainer.placement.colocate_all = True
    cfg.generator.weight_sync_backend = "nccl"
    cfg.trainer.strategy = "fsdp2"
    sampling_params = {
        "max_tokens": 2048,
        "stop": None,
        "stop_token_ids": None,
        "ignore_eos": True,
        "temperature": 0.0,
        # Request token logprobs (vLLM SamplingParams expects an int for how many to return)
        "logprobs": 1,
    }
    client, _ = init_inference_engines(
        cfg=cfg,
        use_local=True,
        async_engine=cfg.generator.async_engine,
        tp_size=cfg.generator.inference_engine_tensor_parallel_size,
        colocate_all=cfg.trainer.placement.colocate_all,
        backend="vllm",
        model=MODEL,
        num_inference_engines=cfg.generator.num_inference_engines,
        sleep_level=1,
        max_num_seqs=max_num_seqs,
    )

    # 2. Prepare a single ConversationType prompt; each generate() call will be single-request
    messages: List[ConversationType] = get_test_prompts(MODEL, num_samples=1)[0]

    # 3. Fire 6 concurrent client.generate() single-request calls, then pause/resume mid-flight
    async def run_requests_then_pause():
        async def one_req(i: int):
            engine_input = {
                "prompts": [messages],  # single request path
                "prompt_token_ids": None,
                "sampling_params": dict(sampling_params),
                "session_ids": [i],
            }
            return await client.generate(engine_input)

        tasks = [asyncio.create_task(one_req(i)) for i in range(num_requests)]
        # Let requests start and enqueue; with max_num_seqs=2, 2 run and 1 wait per engine
        await asyncio.sleep(1)
        # Pause then resume while requests are in-flight
        await client.pause_generation()
        await client.resume_generation()
        # Run for another two seconds, then pause and resume again
        await asyncio.sleep(2)
        await client.pause_generation()
        await client.resume_generation()
        return await asyncio.gather(*tasks)

    outputs = asyncio.run(run_requests_then_pause())

    # 4. Validate each output: stop_reason is "length" and tokens/logprobs == max_tokens
    assert len(outputs) == num_requests, f"Expected {num_requests} outputs, got {len(outputs)}"
    for i, out in enumerate(outputs):
        # InferenceEngineOutput shape checks
        assert "responses" in out and "response_ids" in out and "stop_reasons" in out
        assert len(out["responses"]) == 1 and len(out["response_ids"]) == 1 and len(out["stop_reasons"]) == 1
        assert out["stop_reasons"][0] == "length", f"Request {i} stop_reason is not 'length': {out['stop_reasons'][0]}"
        # Check completion tokens via response_ids
        token_ids = out["response_ids"][0]
        assert (
            len(token_ids) == sampling_params["max_tokens"]
        ), f"Request {i} expected {sampling_params['max_tokens']} tokens, got {len(token_ids)}"
        # Check response_logprobs length
        assert "response_logprobs" in out, f"Request {i} missing response_logprobs"
        assert (
            len(out["response_logprobs"][0]) == sampling_params["max_tokens"]
        ), f"Request {i} expected {sampling_params['max_tokens']} logprobs, got {len(out['response_logprobs'][0])}"
        # Check string output is
        assert out["responses"][0] == client.tokenizer.decode(token_ids, skip_special_tokens=True)
        # Print a preview to aid debugging
        print(f"Output first 1500 chars: {out['responses'][0][:1500]}...")


@pytest.mark.vllm
def test_abort_generation_vllm_engine(ray_init_fixture):
    """
    We send 4 requests that are really long to `InferenceEngineClient.engines[0].chat_completion`
    and then call abort. We set max_num_seqs=2 to test aborting 2 running requests and 2 waiting
    requests. We expect 2 requests to be returned with completion_tokens=0 and 2 with non-zero
    completion_tokens. We also expect the finish_reason to be "abort" for all requests.
    """
    # 1. Build engine
    cfg = get_test_actor_config(num_inference_engines=1, model=MODEL)
    cfg.trainer.placement.colocate_all = True
    cfg.generator.weight_sync_backend = "nccl"
    cfg.trainer.strategy = "fsdp2"
    # We generate 8192 tokens ad ignore eos.
    sampling_params = {
        "max_tokens": 8192,
        "stop": None,
        "stop_token_ids": None,
        "ignore_eos": True,
        "stream": False,
    }
    client, _ = init_inference_engines(
        cfg=cfg,
        use_local=True,
        async_engine=cfg.generator.async_engine,
        tp_size=cfg.generator.inference_engine_tensor_parallel_size,
        colocate_all=cfg.trainer.placement.colocate_all,
        backend="vllm",
        model=MODEL,
        num_inference_engines=cfg.generator.num_inference_engines,
        sleep_level=1,
        # We test aborting 2 running requests and 2 waiting requests
        max_num_seqs=2,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    for api in ["chat_completion", "completion"]:

        # 2. Build 4 chat prompts that have no early stops
        convs: List[ConversationType] = [
            [
                {"role": "system", "content": "You are a token generator that keeps talking endlessly."},
                {"role": "user", "content": "Write a very long rambling response without ending."},
            ]
            for _ in range(4)
        ]

        # 3. Fire 4 concurrent requests directly to engine[0]
        async def run_requests_then_pause():
            async def one_req(i: int):
                if api == "chat_completion":
                    body = {
                        "model": MODEL,
                        "messages": convs[i],
                        **sampling_params,
                    }
                    return await client.engines[0].chat_completion({"json": body, "headers": {}})
                else:
                    # completions: prompt is a string
                    prompt_str = tokenizer.apply_chat_template(convs[i], add_generation_prompt=True, tokenize=False)
                    body = {
                        "model": MODEL,
                        "prompt": prompt_str,
                        **sampling_params,
                    }
                    return await client.engines[0].completion({"json": body, "headers": {}})

            tasks = [asyncio.create_task(one_req(i)) for i in range(4)]
            # Wait to let it run a bit, then pause generation
            await asyncio.sleep(1)
            await client.pause_generation()
            return await asyncio.gather(*tasks)

        outputs = asyncio.run(run_requests_then_pause())

        # 5. Validate outputs: each should be a ChatCompletionResponse; finish_reason is either "abort" or "length"
        num_completion_tokens_is_zero = 0
        for out in outputs:
            assert "choices" in out and len(out["choices"]) == 1
            if out["usage"]["completion_tokens"] == 0:
                num_completion_tokens_is_zero += 1
            assert out["choices"][0].get("finish_reason") == "abort"

        # Two requests should have never got to run because we have max_num_seqs=2, and yet they should
        # be aborted.
        assert (
            num_completion_tokens_is_zero == 2
        ), f"Expected 2 requests with completion_tokens=0, got {num_completion_tokens_is_zero}."

        # Unpause for the next API run
        asyncio.run(client.resume_generation())
