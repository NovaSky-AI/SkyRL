"""
uv run --extra dev --extra vllm --isolated pytest tests/gpu/gpu_ci/test_skyrl_gym_generator.py
"""

import os
import pytest
import ray
from transformers import AutoTokenizer
from skyrl_train.inference_engines.ray_wrapped_inference_engine import create_ray_wrapped_inference_engines
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.generators.skyrl_gym_generator import SkyRLGymGenerator
from skyrl_train.generators.base import GeneratorInput
from tests.gpu.utils import Timer, get_test_generator_input
from omegaconf import DictConfig
from skyrl_train.utils.utils import initialize_ray
from skyrl_gym.envs import register
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from typing import Any, Dict
import hydra
from skyrl_train.entrypoints.main_base import config_dir
from loguru import logger

OBSERVATION_PROMPT = "give me another solution"


def get_test_actor_config() -> DictConfig:
    """Get base config with test-specific overrides."""
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")
        cfg.generator.backend = "vllm"

        return cfg


# Setup for formatting tests
class TestEnv(BaseTextEnv):
    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()
        self.max_turns = 3

    def init(self, prompt):
        return prompt, {}

    def step(self, action: str):
        self.turns += 1
        done = self.turns >= self.max_turns
        return BaseTextEnvStepOutput(
            observations=[{"role": "user", "content": f"{OBSERVATION_PROMPT} {self.turns}"}] if not done else [],
            reward=0,
            done=done,
            metadata={},
        )


register(
    id="test_env",
    entry_point="tests.gpu.gpu_ci.test_skyrl_gym_generator:TestEnv",
)

MODEL_TO_GENERATION_PROMPT = {
    "Qwen/Qwen2.5-1.5B-Instruct": "<|im_start|>assistant\n",
    "unsloth/Llama-3.2-1B-Instruct": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "Qwen/Qwen3-0.6B": "<|im_start|>assistant\n",
}


async def run_generator_end_to_end(
    use_async_engine,
    batched,
    n_samples_per_prompt,
    num_inference_engines,
    tensor_parallel_size,
    model="Qwen/Qwen2.5-1.5B-Instruct",
    max_prompt_length=512,
    max_input_length=2048,
    max_generate_length=1024,
    data_path=os.path.expanduser("~/data/gsm8k/validation.parquet"),
    env_class="gsm8k",
    num_prompts=2,
    max_turns=1,
    use_conversation_multi_turn=True,
    max_env_workers=10,
):
    """
    End to end generator test - requires minimum 2 GPUs
    """
    tokenizer = AutoTokenizer.from_pretrained(model)

    inference_engines = create_ray_wrapped_inference_engines(
        num_inference_engines=num_inference_engines,
        tensor_parallel_size=tensor_parallel_size,
        model_dtype="bfloat16",
        pretrain=model,
        seed=42,
        vllm_v1_disable_multiproc=True,
        enable_prefix_caching=True,
        enforce_eager=True,
        shared_pg=None,
        gpu_memory_utilization=0.8,
        inference_engine_enable_sleep=True,
        async_engine=use_async_engine,
        max_num_batched_tokens=8192,
        max_num_seqs=1024,
        tokenizer=tokenizer,
        sleep_level=1,  # in unit tests that do not explicitly sync weights, we do not discard weights
    )

    # Create a mock generator config
    generator_cfg = DictConfig(
        {
            "sampling_params": {
                "max_generate_length": max_generate_length,
                "logprobs": None,
            },
            "append_eos_token_after_stop_str_in_multi_turn": True,  # for search
            "max_input_length": max_input_length,
            "batched": batched,
            "max_turns": max_turns,
            "zero_reward_on_non_stop": False,
            "use_conversation_multi_turn": use_conversation_multi_turn,
            "apply_overlong_filtering": False,
            "backend": "vllm",
            "enable_http_endpoint": False,
            "http_endpoint_host": "127.0.0.1",
            "http_endpoint_port": 8000,
        }
    )

    env_cfg = DictConfig(
        {
            "search": {
                "log_requests": True,
                "search_url": "http://127.0.0.1:8000/retrieve",
                "topk": 3,
                "timeout": 30,
            },
            "max_env_workers": max_env_workers,
        }
    )

    cfg = get_test_actor_config()
    cfg.trainer.policy.model.path = model
    cfg.generator = generator_cfg
    inference_engine_client = InferenceEngineClient(
        inference_engines,
        tokenizer,
        cfg,
    )

    await inference_engine_client.wake_up()

    generator = SkyRLGymGenerator(
        generator_cfg=generator_cfg,
        skyrl_gym_cfg=env_cfg,
        inference_engine_client=inference_engine_client,
        tokenizer=tokenizer,
        model_name=model,
    )

    input_batch: GeneratorInput = get_test_generator_input(
        model=model,
        num_prompts=num_prompts,
        n_samples_per_prompt=n_samples_per_prompt,
        max_prompt_length=max_prompt_length,
        data_path=data_path,
        env_class=env_class,
    )
    # Attach request-time sampling params into the generator input
    input_batch["sampling_params"] = get_sampling_params_for_backend(
        "vllm",
        DictConfig(
            {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_generate_length": max_generate_length,
                "min_p": 0.0,
                "logprobs": None,
                "stop": ["</search>", "</answer>"] if env_class == "search" else None,
            }
        ),
    )

    with Timer(f"generate_responses_async_engine_{use_async_engine}"):
        generator_output = await generator.generate(input_batch)

    prompts_out = generator_output["prompt_token_ids"]
    outputs = [
        {
            "response": generator_output["response_ids"][i],
            "loss_mask": generator_output["loss_masks"][i],
        }
        for i in range(len(generator_output["response_ids"]))
    ]

    output_keys = [
        "prompt_token_ids",
        "response_ids",
        "rewards",
        "loss_masks",
        "stop_reasons",
        "rollout_metrics",
    ]
    for key in output_keys:
        assert key in generator_output, f"Key {key} not found in generator output"
    assert len(prompts_out) == len(outputs), "Mismatch between prompts and outputs"
    assert isinstance(prompts_out[0], list), "Prompts output should be a list"
    assert isinstance(prompts_out[0][0], int), "Prompts output should be a list of list of token ids"
    assert isinstance(outputs[0]["response"][0], int), "Prompts output should be a list of list of token ids"
    assert len(outputs) == num_prompts * n_samples_per_prompt, "Mismatch between number of outputs and expected outputs"
    for i in range(len(outputs)):
        response_length = len(outputs[i]["response"])
        # TODO (erictang000): make this more precise for multi-turn
        assert response_length <= max_generate_length + max_input_length, f"Output {i} exceeds max length"
        assert response_length == len(outputs[i]["loss_mask"]), f"Output {i} loss mask length mismatch"

    # TODO (tgriggs): Extend this test to compare the outputs to HF generation with temperature 0
    return generator_output


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("use_async_engine", "batched", "n_samples_per_prompt", "num_inference_engines", "tensor_parallel_size"),
    [
        (False, True, 5, 2, 1),  # tests SkyRLGymGenerator.generate_batched for single-turn
        (True, False, 5, 1, 2),  # tests SkyRLGymGenerator.agent_loop for single-turn
        # Add more combinations as needed
    ],
)
async def test_generator_single_turn_gsm8k(
    use_async_engine, batched, n_samples_per_prompt, num_inference_engines, tensor_parallel_size
):
    """
    Test the generator with a single turn of GSM8K
    """
    initialize_ray(get_test_actor_config())
    try:
        await run_generator_end_to_end(
            use_async_engine=use_async_engine,
            batched=batched,
            n_samples_per_prompt=n_samples_per_prompt,
            num_inference_engines=num_inference_engines,
            tensor_parallel_size=tensor_parallel_size,
        )
    finally:
        ray.shutdown()


@pytest.mark.asyncio
async def test_generator_multi_turn_search():
    """
    Test the generator with multiple turns of search
    """
    initialize_ray(get_test_actor_config())
    try:
        await run_generator_end_to_end(
            use_async_engine=True,
            batched=False,
            n_samples_per_prompt=5,
            num_inference_engines=2,
            tensor_parallel_size=2,
            model="Qwen/Qwen2.5-1.5B-Instruct",
            max_prompt_length=2048,
            max_input_length=4096,
            max_generate_length=1000,
            data_path=os.path.expanduser("~/data/searchR1/validation.parquet"),
            env_class="search",
            num_prompts=2,
            max_turns=2,
            use_conversation_multi_turn=False,
            max_env_workers=0,
        )
    finally:
        ray.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name", ["unsloth/Llama-3.2-1B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen3-0.6B"]
)
async def test_generator_formatting_use_conversation_multi_turn(model_name):
    """
    Test generator formatting when using conversation formatting for multi-turn
    """
    initialize_ray(get_test_actor_config())
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        generator_output = await run_generator_end_to_end(
            use_async_engine=True,
            batched=False,
            n_samples_per_prompt=1,
            num_inference_engines=1,
            tensor_parallel_size=1,
            model=model_name,
            max_prompt_length=3000,
            max_input_length=10000,
            max_generate_length=3000,
            env_class="test_env",
            num_prompts=2,
            max_turns=3,
            use_conversation_multi_turn=True,
        )

        for i, resp_ids in enumerate(generator_output["response_ids"]):
            loss_mask = generator_output["loss_masks"][i]
            prompt_token_ids = generator_output["prompt_token_ids"][i]
            stop_reason = generator_output["stop_reasons"][i]
            masked_out_resp_ids = [resp_ids[j] for j in range(len(resp_ids)) if loss_mask[j] == 0]
            masked_in_resp_ids = [resp_ids[j] for j in range(len(resp_ids)) if loss_mask[j] == 1]

            masked_out_resp_str = tokenizer.decode(masked_out_resp_ids)
            masked_in_resp_str = tokenizer.decode(masked_in_resp_ids)

            assert (
                MODEL_TO_GENERATION_PROMPT[model_name] in masked_out_resp_str
                and MODEL_TO_GENERATION_PROMPT[model_name] not in masked_in_resp_str
            ), "generation prompts should be loss masked out"

            # Observations and EOS expectations only strictly apply when the model finished turns
            if stop_reason == "stop":
                assert (
                    f"{OBSERVATION_PROMPT} 1" in masked_out_resp_str
                ), f'"{OBSERVATION_PROMPT} 1" observation should be loss masked out'
                assert (
                    f"{OBSERVATION_PROMPT} 2" in masked_out_resp_str
                ), f'"{OBSERVATION_PROMPT} 2" observation should be loss masked out'
                # count number of eos tokens in masked_in_resp_ids: 1 eos per assistant response (3 turns)
                assert sum(1 for _ in masked_in_resp_ids if _ == tokenizer.eos_token_id) == 3
                # total eos in full response: 2 user eos + 3 assistant eos
                assert sum(1 for _ in resp_ids if _ == tokenizer.eos_token_id) == 5
            else:
                # On length stops, the model may not produce EOS at the end of each assistant turn.
                # Only check that generation prompts are masked out.
                logger.warning(f"Got stop reason {stop_reason}, so we did not fully check the response")
            if model_name == "Qwen/Qwen3-0.6B":
                assert (
                    sum(1 for _ in prompt_token_ids if _ == tokenizer.eos_token_id) == 1
                )  # 1 user eos (no system for Qwen3)
            else:
                assert sum(1 for _ in prompt_token_ids if _ == tokenizer.eos_token_id) == 2  # 1 system eos, 1 user eos
    finally:
        ray.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name", ["unsloth/Llama-3.2-1B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen3-0.6B"]
)
async def test_generator_formatting_no_use_conversation_multi_turn(model_name):
    """
    Test generator formatting when not using conversation formatting for multi-turn
    """
    initialize_ray(get_test_actor_config())
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        generator_output = await run_generator_end_to_end(
            use_async_engine=True,
            batched=False,
            n_samples_per_prompt=1,
            num_inference_engines=1,
            tensor_parallel_size=1,
            model=model_name,
            max_prompt_length=3000,
            max_input_length=10000,
            max_generate_length=3000,
            env_class="test_env",
            num_prompts=2,
            max_turns=3,
            use_conversation_multi_turn=False,
        )

        for i, resp_ids in enumerate(generator_output["response_ids"]):
            loss_mask = generator_output["loss_masks"][i]
            prompt_token_ids = generator_output["prompt_token_ids"][i]
            masked_out_resp_ids = [resp_ids[j] for j in range(len(resp_ids)) if loss_mask[j] == 0]
            masked_in_resp_ids = [resp_ids[j] for j in range(len(resp_ids)) if loss_mask[j] == 1]

            prompt_str = tokenizer.decode(prompt_token_ids)
            resp_str = tokenizer.decode(resp_ids)
            masked_out_resp_str = tokenizer.decode(masked_out_resp_ids)
            masked_in_resp_str = tokenizer.decode(masked_in_resp_ids)

            assert (
                f"{OBSERVATION_PROMPT} 1" in masked_out_resp_str
            ), f'"{OBSERVATION_PROMPT} 1" observation should be loss masked out'
            assert (
                f"{OBSERVATION_PROMPT} 2" in masked_out_resp_str
            ), f'"{OBSERVATION_PROMPT} 2" observation should be loss masked out'
            assert (
                prompt_str.count(MODEL_TO_GENERATION_PROMPT[model_name])
                + resp_str.count(MODEL_TO_GENERATION_PROMPT[model_name])
                == 1
            ), "the single generation prompt should be included in the prompt"
            assert (
                MODEL_TO_GENERATION_PROMPT[model_name] in prompt_str
                and MODEL_TO_GENERATION_PROMPT[model_name] not in masked_in_resp_str
            ), "the single generation prompt should be included in the prompt"

            # count number of eos tokens in masked_in_resp_ids
            assert (
                sum(1 for _ in masked_in_resp_ids if _ == tokenizer.eos_token_id) == 1
            )  # 1 eos for each assistant response
            if model_name == "Qwen/Qwen3-0.6B":
                assert (
                    sum(1 for _ in prompt_token_ids if _ == tokenizer.eos_token_id) == 1
                )  # 1 user eos (no system for Qwen3)
            else:
                assert sum(1 for _ in prompt_token_ids if _ == tokenizer.eos_token_id) == 2  # 1 system eos, 1 user eos
    finally:
        ray.shutdown()
