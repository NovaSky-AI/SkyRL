"""
uv run --extra dev --isolated pytest tests/cpu/generators/test_skyrl_gym_generator_compare.py
"""

import pytest
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock
from skyrl_train.generators.skyrl_gym_generator import SkyRLGymGenerator
from skyrl_train.generators.base import GeneratorInput, GeneratorOutput
from skyrl_train.generators.utils import CUSTOM_CHAT_TEMPLATES

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from omegaconf import DictConfig
from transformers import AutoTokenizer
from skyrl_gym.envs import register


class CPUTestEnv(BaseTextEnv):
    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()
        self.max_turns = 3

    def init(self, prompt):
        return prompt, {}

    def step(self, action: str):
        self.turns += 1
        done = self.turns >= self.max_turns
        return BaseTextEnvStepOutput(
            observations=[{"role": "user", "content": f"{self.turns}"}] if not done else [],
            reward=0,
            done=done,
            metadata={},
        )


def _register_test_env_if_needed():
    """Register the test env only if it's not already registered."""
    try:
        register(
            id="cpu_test_env",
            entry_point="tests.cpu.generators.test_skyrl_gym_generator_compare:CPUTestEnv",
        )
    except Exception:
        pass


def _create_mock_llm(tokenizer, response_text="b"):
    """Helper to create mock LLM with consistent behavior."""
    mock_llm = MagicMock()

    def mock_generate(input_batch):
        num_prompts = len(input_batch["prompts"]) if "prompts" in input_batch else len(input_batch["prompt_token_ids"])
        mock_llm_output_text = response_text + tokenizer.eos_token
        return {
            "responses": [response_text] * num_prompts,
            "stop_reasons": ["stop"] * num_prompts,
            "response_logprobs": None,
            "response_ids": [tokenizer.encode(mock_llm_output_text, add_special_tokens=False)] * num_prompts,
        }

    mock_llm.generate = AsyncMock(side_effect=mock_generate)
    return mock_llm


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chat_template_config,expected_template",
    [
        ({"source": "name", "name_or_path": "original_chat_template"}, "original_chat_template"),
        ({"source": "name", "name_or_path": "qwen3_without_thinking"}, "qwen3_without_thinking"),
        ({"source": "name", "name_or_path": None}, None),
        (None, None),
    ],
)
async def test_chat_template_configs(chat_template_config, expected_template):
    """Test different chat template configurations."""
    _register_test_env_if_needed()
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    mock_llm = _create_mock_llm(tokenizer)

    generator_cfg = DictConfig(
        {
            "sampling_params": {"max_generate_length": 200, "logprobs": None},
            "max_input_length": 200,
            "batched": False,
            "max_turns": 1,
            "zero_reward_on_non_stop": False,
            "apply_overlong_filtering": False,
            "use_conversation_multi_turn": True,
            "chat_template": chat_template_config,
            "append_eos_token_after_stop_str_in_multi_turn": True,
        }
    )

    env_cfg = DictConfig({"max_env_workers": 0, "env_class": "cpu_test_env"})

    generator = SkyRLGymGenerator(
        generator_cfg=generator_cfg,
        skyrl_gym_cfg=env_cfg,
        inference_engine_client=mock_llm,
        tokenizer=tokenizer,
        model_name=model_name,
    )

    # Test that the correct chat template is loaded
    if expected_template is None:
        assert generator.custom_chat_template is None
    else:
        assert generator.custom_chat_template == CUSTOM_CHAT_TEMPLATES[expected_template]

    # Test basic generation works
    input_batch: GeneratorInput = {
        "prompts": [[{"role": "user", "content": "Hello"}]],
        "env_extras": [{"answer": "test"}],
        "env_classes": ["cpu_test_env"],
    }
    generator_output: GeneratorOutput = await generator.generate(input_batch)

    assert len(generator_output["prompt_token_ids"]) == 1
    assert len(generator_output["response_ids"]) == 1
    assert len(generator_output["loss_masks"]) == 1


def test_qwen3_original_vs_without_thinking_chat_template():
    """
    Test comparing original Qwen3 chat template with qwen3_without_thinking.
    Explains why we deleted the '\\n' - to match the tokenizer's actual behavior.
    """
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    messages = [
        {"content": "hi", "role": "system"},
        {"content": "hi", "role": "user"},
        {"content": "<think>thinking</think>hi", "role": "assistant"},
        {"content": "hi", "role": "user"},
        {"content": "<think>thinking</think>hi", "role": "assistant"},
        {"content": "hi", "role": "user"},
    ]

    # Apply our custom chat templates
    qwen3_without_thinking_str = tokenizer.apply_chat_template(
        messages, chat_template=CUSTOM_CHAT_TEMPLATES["qwen3_without_thinking"], tokenize=False
    )
    original_template_str = tokenizer.apply_chat_template(
        messages, chat_template=CUSTOM_CHAT_TEMPLATES["original_chat_template"], tokenize=False
    )
    # Apply default chat template
    default_template_str = tokenizer.apply_chat_template(messages, chat_template=None, tokenize=False)

    # The original_chat_template should match the tokenizer exactly
    assert default_template_str == original_template_str
    assert (
        default_template_str == qwen3_without_thinking_str
    ), f"default_template_str: {default_template_str}, qwen3_without_thinking_str: {qwen3_without_thinking_str}"

    # # The qwen3_without_thinking should match original minus trailing newline
    # # We removed '\\n' because our generator follows token-in-token-out behavior
    # if default_template_str.endswith("\n"):
    #     expected_without_thinking = default_template_str[:-1]
    #     assert qwen3_without_thinking_str == expected_without_thinking
    # else:
    #     assert qwen3_without_thinking_str == default_template_str


@pytest.mark.asyncio
async def test_multi_turn_conversation_formatting():
    """Test that multi-turn conversation formatting works correctly with custom templates."""
    _register_test_env_if_needed()
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    mock_llm = _create_mock_llm(tokenizer)

    generator_cfg = DictConfig(
        {
            "sampling_params": {"max_generate_length": 200, "logprobs": None},
            "max_input_length": 200,
            "batched": False,
            "max_turns": 3,
            "zero_reward_on_non_stop": False,
            "apply_overlong_filtering": False,
            "use_conversation_multi_turn": True,
            "chat_template": {"source": "name", "name_or_path": "qwen3_without_thinking"},
            "append_eos_token_after_stop_str_in_multi_turn": True,
        }
    )

    env_cfg = DictConfig({"max_env_workers": 0, "env_class": "cpu_test_env"})

    generator = SkyRLGymGenerator(
        generator_cfg=generator_cfg,
        skyrl_gym_cfg=env_cfg,
        inference_engine_client=mock_llm,
        tokenizer=tokenizer,
        model_name=model_name,
    )

    input_batch: GeneratorInput = {
        "prompts": [[{"role": "user", "content": "a"}]],
        "env_extras": [{"answer": "4"}],
        "env_classes": ["cpu_test_env"],
    }
    generator_output: GeneratorOutput = await generator.generate(input_batch)

    # Check that the conversation was formatted correctly
    expected_chat_history = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "2"},
        {"role": "assistant", "content": "b"},
    ]

    prompt_str = tokenizer.decode(generator_output["prompt_token_ids"][0])
    resp_str = tokenizer.decode(generator_output["response_ids"][0])
    full_str = prompt_str + resp_str

    expected_str = tokenizer.apply_chat_template(
        expected_chat_history, chat_template=generator.custom_chat_template, tokenize=False
    )

    assert full_str == expected_str
