"""
uv run --extra dev --extra vllm --isolated pytest tests/gpu/test_skyrl_gym_generator.py
"""

import os
import pytest
import ray
from skyrl_train.utils.utils import initialize_ray
from tests.gpu.gpu_ci.test_skyrl_gym_generator import run_generator_end_to_end, get_test_actor_config


# TODO: Make this test lightweight. It currently requires a ~20GB dataset download. Then, transfer the test to gpu_ci.
@pytest.mark.asyncio
async def test_generator_multi_turn_text2sql():
    """
    Test the generator with multiple turns of text2sql
    """
    initialize_ray(get_test_actor_config())
    try:
        await run_generator_end_to_end(
            use_async_engine=True,
            batched=False,
            n_samples_per_prompt=5,
            num_inference_engines=2,
            tensor_parallel_size=2,
            model="Qwen/Qwen2.5-Coder-7B-Instruct",
            max_prompt_length=6000,
            max_input_length=29048,
            max_generate_length=3000,
            data_path=os.path.expanduser("~/data/sql/validation.parquet"),
            env_class="text2sql",
            num_prompts=2,
            max_turns=6,
            use_conversation_multi_turn=False,
        )
    finally:
        ray.shutdown()
