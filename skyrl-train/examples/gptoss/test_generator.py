import pytest
from omegaconf import DictConfig
from unittest.mock import AsyncMock, MagicMock
from skyrl_gym.envs.base_text_env import BaseTextEnvStepOutput
from skyrl_train.inference_engines.remote_inference_engine import create_remote_inference_engines
from gpt_oss_generator_step_wise import GPTOSSGenerator

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

assistant_message = """<|channel|>analysis<|message|>Compute 15% of 220 = 33. So new price = 220+33=253.<|end|><|start|>assistant<|channel|>final<|message|>The price increased by 15% means it is multiplied by \(1 + 0.15 = 1.15\).

\[
220 \times 1.15 = 220 \times \left(\frac{115}{100}\right)
               = 220 \times 1.15
               = 220 + 33
               = 253
\]

The new selling price is **$253**.

####<|return|>"""
MOCK_LLM_OUTPUT_IDS = tokenizer.encode(assistant_message)


def create_remote_engines(remote_inference_engine_urls, model_name_or_path, backend, tokenizer, inference_engine_tensor_parallel_size, inference_engine_data_parallel_size, inference_engine_expert_parallel_size):
    inference_engines = create_remote_inference_engines(
        urls=remote_inference_engine_urls,
        model_name=model_name_or_path,
        engine_backend=backend,
        tokenizer=tokenizer,
        tensor_parallel_size=inference_engine_tensor_parallel_size,
        data_parallel_size=inference_engine_data_parallel_size,
        expert_parallel_size=inference_engine_expert_parallel_size,
    )
    return inference_engines


@pytest.fixture
def mock_env():
    mock_env_instance = MagicMock()
    mock_env_instance.step.side_effect = lambda x: BaseTextEnvStepOutput(
        observations=[{"role": "user", "content": "next"}], reward=1.0, done=True, metadata={}
    )
    mock_env_instance.close.return_value = None
    return mock_env_instance


@pytest.fixture
def mock_generator_cfg():
    cfg = DictConfig(
        {
            "sampling_params": {
                "max_generate_length": 5,
            },
            "logprobs": None,
            "apply_overlong_filtering": False,
            "max_input_length": 512,
            "batched": True,
            "max_turns": 1,
            "chat_template_kwargs": {},
            "chat_template": {"source": "name", "name_or_path": None},
            "use_conversation_multi_turn": True,
        },
    )
    return cfg


@pytest.fixture
def mock_llm():
    """
    This replaces InferenceEngineClient, where `.generate()` always returns MOCK_LLM_OUTPUT_IDS
    for each prompt, with corresponding string output "mocked output".
    """
    mock = MagicMock()

    # Mock the new generate method
    def mock_generate(input_batch):
        num_prompts = len(input_batch["prompts"]) if "prompts" in input_batch else len(input_batch["prompt_token_ids"])
        return {
            "responses": ["mocked output"] * num_prompts,
            "stop_reasons": ["stop"] * num_prompts,
            # say response gets tokenized to 3 tokens
            "response_logprobs": [[0.1] * len(MOCK_LLM_OUTPUT_IDS)] * num_prompts,
            "response_ids": [MOCK_LLM_OUTPUT_IDS.copy()] * num_prompts,
        }

    mock.generate = AsyncMock(side_effect=mock_generate)
    return mock


@pytest.mark.asyncio
async def test_generate_single_turn(mock_generator_cfg, mock_llm):
    env_config = DictConfig({"max_env_workers": 0})

    generator = GPTOSSGenerator(
        generator_cfg=mock_generator_cfg,
        skyrl_gym_cfg=env_config,
        inference_engine_client=mock_llm,
        tokenizer=tokenizer,
        model_name="test_model",
    )
    
    output = await generator.agent_loop(
        prompt=[{"role": "user", "content": "What is 2 + 2?"}],
        env_class="gsm8k",
        env_extras={"reward_spec": {"method": "rule", "ground_truth": "253"}},
        max_tokens=8000,
        max_input_length=512,
    )
    print(output)
