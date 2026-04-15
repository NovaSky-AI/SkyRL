"""
To run:
uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/test_engine_generation.py
"""

import asyncio

import pytest
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl.backends.skyrl_train.inference_engines.utils import (
    get_sampling_params_for_backend,
)
from skyrl.env_vars import _SKYRL_USE_NEW_INFERENCE
from skyrl.train.config import SkyRLTrainConfig
from tests.backends.skyrl_train.gpu.utils import (
    InferenceEngineState,
    are_responses_similar,
    get_test_prompts,
)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MOE_MODEL = "Qwen/Qwen1.5-MoE-A2.7B"


def get_test_actor_config(model: str = MODEL) -> SkyRLTrainConfig:
    """Get base config with test-specific overrides."""
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = model

    cfg.generator.sampling_params.temperature = 0.0
    cfg.generator.sampling_params.top_p = 1
    cfg.generator.sampling_params.top_k = -1
    cfg.generator.sampling_params.max_generate_length = 1024
    cfg.generator.sampling_params.min_p = 0.0
    cfg.generator.sampling_params.logprobs = None

    return cfg


async def run_batch_generation(client, prompts, sampling_params):
    engine_input = InferenceEngineInput(prompts=prompts, sampling_params=sampling_params)
    engine_output = await client.generate(engine_input)
    return engine_output["responses"], engine_output["stop_reasons"]


async def run_single_generation(client, prompts, sampling_params):
    tasks = []
    for prompt in prompts:
        engine_input = InferenceEngineInput(prompts=[prompt], sampling_params=sampling_params)
        task = client.generate(engine_input)
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    responses = []
    finish_reasons = []
    for result in results:
        responses.extend(result["responses"])
        finish_reasons.extend(result["stop_reasons"])

    return responses, finish_reasons


async def run_batch_generation_with_tokens(client, prompt_token_ids, sampling_params):
    engine_input = InferenceEngineInput(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
    engine_output = await client.generate(engine_input)
    return engine_output["responses"], engine_output["stop_reasons"]


async def run_single_generation_with_tokens(client, prompt_token_ids, sampling_params):
    tasks = []
    for tokens in prompt_token_ids:
        engine_input = InferenceEngineInput(prompt_token_ids=[tokens], sampling_params=sampling_params)
        task = client.generate(engine_input)
        tasks.append(task)

    results = await asyncio.gather(*tasks)

    responses = []
    finish_reasons = []
    for result in results:
        responses.extend(result["responses"])
        finish_reasons.extend(result["stop_reasons"])

    return responses, finish_reasons


@pytest.mark.parametrize(
    "tp_size,pp_size,dp_size,model,distributed_executor_backend",
    [
        pytest.param(2, 1, 1, MODEL, "ray"),
        pytest.param(2, 2, 1, MODEL, "ray"),
        pytest.param(2, 1, 2, MOE_MODEL, "ray"),
        pytest.param(2, 1, 2, MOE_MODEL, "mp"),
    ],
    ids=["tp2_pp1_dp1_ray", "tp2_pp2_dp1_ray", "tp2_pp1_dp2_moe_ray", "tp2_pp1_dp2_moe_mp"],
)
@pytest.mark.asyncio
async def test_token_based_generation(
    ray_init_fixture, tp_size: int, pp_size: int, dp_size: int, model: str, distributed_executor_backend: str
):
    """Test generation using prompt_token_ids."""

    cfg = get_test_actor_config(model)

    prompts = get_test_prompts(model, 3)
    tokenizer = AutoTokenizer.from_pretrained(model)
    prompt_token_ids = tokenizer.apply_chat_template(
        prompts, add_generation_prompt=True, tokenize=True, return_dict=True
    )["input_ids"]

    cfg.generator.inference_engine.tensor_parallel_size = tp_size
    cfg.generator.inference_engine.pipeline_parallel_size = pp_size
    cfg.generator.inference_engine.data_parallel_size = dp_size
    cfg.generator.inference_engine.distributed_executor_backend = distributed_executor_backend

    async with InferenceEngineState.create(cfg, sleep_level=1) as engines:
        llm_client = engines.client
        sampling_params = get_sampling_params_for_backend(
            cfg.generator.inference_engine.backend, cfg.generator.sampling_params
        )

        # Test batch generation with tokens
        token_batch_responses, _ = await run_batch_generation_with_tokens(llm_client, prompt_token_ids, sampling_params)
        assert len(token_batch_responses) == len(prompts)

        # Test single generation with tokens
        token_single_responses, _ = await run_single_generation_with_tokens(
            llm_client, prompt_token_ids, sampling_params
        )
        assert len(token_single_responses) == len(prompts)

        # Ensure batched and single generation outputs are (roughly) the same
        for i in range(len(prompts)):
            if not are_responses_similar(token_batch_responses[i], token_single_responses[i], tolerance=0.01):
                print(
                    f"Token batch and single generation responses are not similar, got batch={token_batch_responses[i]} and single={token_single_responses[i]}"
                )


@pytest.mark.skipif(not _SKYRL_USE_NEW_INFERENCE, reason="PD requires new inference pathway")
def test_pd_generation(ray_init_fixture):
    """Test generation with prefill-decode disaggregation (1P1D, 2 GPUs)."""
    cfg = get_test_actor_config(MODEL)

    prompts = get_test_prompts(MODEL, 3)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompt_token_ids = tokenizer.apply_chat_template(
        prompts, add_generation_prompt=True, tokenize=True, return_dict=True
    )["input_ids"]

    with InferenceEngineState.create(
        cfg,
        tp_size=1,
        num_inference_engines=2,
        sleep_level=1,
        enable_pd=True,
        num_prefill=1,
        use_new_inference_servers=True,
        engine_init_kwargs={
            "kv_transfer_config": {
                "kv_connector": "NixlConnector",
            },
        },
    ) as engines:
        llm_client = engines.client
        sampling_params = get_sampling_params_for_backend(
            cfg.generator.inference_engine.backend, cfg.generator.sampling_params
        )

        # Batch generation
        batch_responses, batch_finish_reasons = asyncio.run(
            run_batch_generation_with_tokens(llm_client, prompt_token_ids, sampling_params)
        )
        assert len(batch_responses) == len(prompts)
        assert len(batch_finish_reasons) == len(prompts)

        # Single generation
        single_responses, single_finish_reasons = asyncio.run(
            run_single_generation_with_tokens(llm_client, prompt_token_ids, sampling_params)
        )
        assert len(single_responses) == len(prompts)


@pytest.mark.skipif(not _SKYRL_USE_NEW_INFERENCE, reason="PD requires new inference pathway")
@pytest.mark.parametrize(
    "num_prefill,num_decode,colocate_all",
    [
        pytest.param(1, 1, False, id="1P1D_non_colocated"),
    ],
)
def test_pd_generation_non_colocated(
    ray_init_fixture,
    num_prefill: int,
    num_decode: int,
    colocate_all: bool,
):
    """Test PD generation in non-colocated mode (separate placement groups).

    Exercises the shared-PG creation path in create_inference_servers when
    placement_group=None (non-colocated).
    """
    cfg = get_test_actor_config(MODEL)

    prompts = get_test_prompts(MODEL, 3)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    prompt_token_ids = tokenizer.apply_chat_template(
        prompts, add_generation_prompt=True, tokenize=True, return_dict=True
    )["input_ids"]

    num_engines = num_prefill + num_decode

    with InferenceEngineState.create(
        cfg,
        tp_size=1,
        num_inference_engines=num_engines,
        sleep_level=1,
        enable_pd=True,
        num_prefill=num_prefill,
        colocate_all=colocate_all,
        use_new_inference_servers=True,
        engine_init_kwargs={
            "kv_transfer_config": {
                "kv_connector": "NixlConnector",
            },
        },
    ) as engines:
        llm_client = engines.client
        sampling_params = get_sampling_params_for_backend(
            cfg.generator.inference_engine.backend, cfg.generator.sampling_params
        )

        # Batch generation
        batch_responses, batch_finish_reasons = asyncio.run(
            run_batch_generation_with_tokens(llm_client, prompt_token_ids, sampling_params)
        )
        assert len(batch_responses) == len(prompts)
        assert len(batch_finish_reasons) == len(prompts)

        # Single generation
        single_responses, single_finish_reasons = asyncio.run(
            run_single_generation_with_tokens(llm_client, prompt_token_ids, sampling_params)
        )
        assert len(single_responses) == len(prompts)
