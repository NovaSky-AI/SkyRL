"""
Test pause and continue generation with inference engine client HTTP endpoint.

uv run --isolated --extra dev --extra vllm pytest tests/gpu/gpu_ci/test_pause_and_continue_generation.py -m "vllm"
"""

import pytest
import asyncio
from tests.gpu.gpu_ci.test_inference_engine_client_http_endpoint import get_test_actor_config
from tests.gpu.utils import init_inference_engines
from skyrl_train.inference_engines.base import ConversationType
from transformers import AutoTokenizer
from typing import List

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TP_SIZE = 1
SERVER_PORT = 8123
SERVER_HOST = "127.0.0.1"


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
        client.resume_generation()
