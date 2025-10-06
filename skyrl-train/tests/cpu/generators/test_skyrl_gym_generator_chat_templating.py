"""
uv run --extra dev --isolated pytest tests/cpu/generators/test_skyrl_gym_generator_chat_templating.py
"""

import pytest
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock
from skyrl_train.generators.skyrl_gym_generator import SkyRLGymGenerator
from skyrl_train.generators.base import GeneratorInput, GeneratorOutput

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from omegaconf import DictConfig
from transformers import AutoTokenizer
from skyrl_gym.envs import register
from skyrl_train.generators.utils import get_custom_chat_template


# Setup for formatting tests
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
            entry_point="tests.cpu.generators.test_skyrl_gym_generator_chat_templating:CPUTestEnv",
        )
    except Exception:
        # Environment already registered, ignore
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name", ["Qwen/Qwen2.5-0.5B-Instruct", "unsloth/Llama-3.2-1B-Instruct", "Qwen/Qwen3-0.6B"]
)
async def test_skyrl_gym_generator_chat_templating_exact(model_name):
    _register_test_env_if_needed()  # Register only when needed
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    mock_llm = MagicMock()

    # Mock the new generate method
    def mock_generate(input_batch):
        num_prompts = len(input_batch["prompts"]) if "prompts" in input_batch else len(input_batch["prompt_token_ids"])
        mock_llm_output_text = "b" + tokenizer.eos_token
        return {
            # no tokenizer.eos_token for responses because `skip_special_tokens` is True in sampling params
            "responses": ["b"] * num_prompts,
            "stop_reasons": ["stop"] * num_prompts,
            "response_logprobs": None,
            # add_special_tokens needs to be False, otherwise for instance Llama will always
            # add a `<|begin_of_text|>` before the assistant response.
            "response_ids": [tokenizer.encode(mock_llm_output_text, add_special_tokens=False)] * num_prompts,
        }

    mock_llm.generate = AsyncMock(side_effect=mock_generate)
    # Create a mock generator config
    generator_cfg = DictConfig(
        {
            "sampling_params": {"max_generate_length": 200, "logprobs": None},
            "max_input_length": 200,
            "batched": False,
            "max_turns": 3,
            "zero_reward_on_non_stop": False,
            "apply_overlong_filtering": False,
            "use_conversation_multi_turn": True,
            "append_eos_token_after_stop_str_in_multi_turn": True,
        }
    )
    env_cfg = DictConfig(
        {
            "max_env_workers": 0,
            "env_class": "cpu_test_env",
        }
    )
    generator = SkyRLGymGenerator(
        generator_cfg=generator_cfg,
        skyrl_gym_cfg=env_cfg,
        inference_engine_client=mock_llm,
        tokenizer=tokenizer,
        model_name=model_name,
    )

    prompt = [[{"role": "user", "content": "a"}]]
    extras = [{"answer": "4"}]

    input_batch: GeneratorInput = {
        "prompts": prompt,
        "env_extras": extras,
        "env_classes": [env_cfg.env_class],
    }
    generator_output: GeneratorOutput = await generator.generate(input_batch)

    # assume every actual message is 1 token for loss mask checking
    expected_chat_history = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "1"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "2"},
        {"role": "assistant", "content": "b"},
    ]

    # For Qwen2.5 generator_output_str, we have (note the missing \n after the eos token):
    # <|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n
    # <|im_start|>user\na<|im_end|>\n<|im_start|>assistant\nb<|im_end|>\n
    # <|im_start|>user\n1<|im_end|>\n<|im_start|>assistant\nb<|im_end|>\n
    # <|im_start|>user\n2<|im_end|>\n<|im_start|>assistant\nb<|im_end|>

    # check that the full response is exactly string matching with applying the chat template on history
    prompt_str = tokenizer.decode(generator_output["prompt_token_ids"][0])
    resp_str = tokenizer.decode(generator_output["response_ids"][0])
    custom_chat_template = get_custom_chat_template(model_name)
    if custom_chat_template is not None:
        assert prompt_str + resp_str == tokenizer.apply_chat_template(
            expected_chat_history, chat_template=custom_chat_template, tokenize=False
        )
    else:
        generator_output_str = prompt_str + resp_str
        expected_str = tokenizer.apply_chat_template(expected_chat_history, tokenize=False)
        if "Qwen" in model_name:
            # For Qwen models, there is an `\n` after the eos token. Our generator follows token-in-token-out,
            # so it will not generate anything after the eos token, and hence will not have the `\n`.
            # e.g. `<|assistant|>\Some content<|im_end|>\n` for expected_str, but
            # `<|assistant|>\Some content<|im_end|>` for generator_output_str.
            if expected_str.endswith("\n"):
                expected_str = expected_str[:-1]
        assert generator_output_str == expected_str

    # check loss mask exact matches
    system_prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": ""}] if "Llama" in model_name else [{}], tokenize=True
    )
    empty_user = tokenizer.apply_chat_template([{"role": "user", "content": ""}], tokenize=True)
    empty_user_with_generation_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True
    )
    # TODO (erictang000): consider hard coding the full loss mask for each model to avoid copying logic in code
    generation_prompt_ids = empty_user_with_generation_prompt[len(empty_user) :]  # `<|im_start|>assistant\n`
    empty_user = empty_user[len(system_prompt) :]  # `<|im_start|>user\n<|im_end|>\n`

    # `<|im_start|>assistant\nb<|im_end|>\n`
    expected_assistant_loss_mask = [0] * len(generation_prompt_ids) + [1, 1]  # 1 for single response token, 1 for eos
    expected_assistant_no_generation_prompt_loss_mask = [1, 1]  # 1 for single response token, 1 for eos
    if "Qwen" in model_name:
        expected_assistant_loss_mask += [0]  # extra 0 for \n for qwen templates
        expected_assistant_no_generation_prompt_loss_mask += [0]
    # `<|im_start|>user\n1<|im_end|>\n`
    expected_user_loss_mask = [0] * len(empty_user) + [0]  # extra 0 for single observation token

    if custom_chat_template is not None:
        # For custom_chat_template, the first generation prompt IDs are part of `resp_str`, hence has corresponding mask
        expected_loss_masks = (
            expected_assistant_loss_mask  # <|im_start|>assistant\nb<|im_end|>\n
            + expected_user_loss_mask  # <|im_start|>user\n1<|im_end|>\n
        ) * 2 + expected_assistant_loss_mask  # last <|im_start|>assistant\nb<|im_end|>\n
    else:
        # For non-custom_chat_template, `resp_str` directly starts with what the model generates
        expected_loss_masks = (
            expected_assistant_no_generation_prompt_loss_mask  # b<|im_end|>\n
            + (
                expected_user_loss_mask  # <|im_start|>user\n1
                + expected_assistant_loss_mask  # <|im_start|>assistant\nb<|im_end|>\n
            )
            * 2
        )
        if "Qwen" in model_name:
            expected_loss_masks = expected_loss_masks[:-1]  # remove the extra 0 for \n
    assert len(expected_loss_masks) == len(generator_output["loss_masks"][0])
    assert generator_output["loss_masks"][0] == expected_loss_masks


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name", ["Qwen/Qwen2.5-0.5B-Instruct", "unsloth/Llama-3.2-1B-Instruct", "Qwen/Qwen3-0.6B"]
)
async def test_append_eos_after_stop_multi_turn(model_name):
    """
    Test the behavior of `append_eos_token_after_stop_str_in_multi_turn`, which is applicable
    when `sampling_params.stop` is not `null` and `use_conversation_multi_turn` is `true` in
    the ``agent_loop()`` function.
    It is used in scripts `examples/search/run_search_conversation_format.sh` and
    `examples/text_to_sql/run_skyrl_sql_conversation_format.sh`.
    """
    _register_test_env_if_needed()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    stop_tag = "</solution>"
    mock_text = "b" + stop_tag

    async def make_generator(append_flag: bool):
        mock_llm = MagicMock()

        # The LLM engine will generate and return the stop tag, but no EOS token ID.
        def mock_generate(input_batch):
            num_prompts = (
                len(input_batch["prompts"]) if "prompts" in input_batch else len(input_batch["prompt_token_ids"])
            )
            return {
                "responses": [mock_text] * num_prompts,
                "stop_reasons": ["stop"] * num_prompts,
                "response_logprobs": None,
                "response_ids": [tokenizer.encode(mock_text, add_special_tokens=False)] * num_prompts,
            }

        mock_llm.generate = AsyncMock(side_effect=mock_generate)

        generator_cfg = DictConfig(
            {
                "sampling_params": {"max_generate_length": 200, "logprobs": None, "stop": [stop_tag]},
                "max_input_length": 200,
                "batched": False,
                "max_turns": 3,
                "zero_reward_on_non_stop": False,
                "apply_overlong_filtering": False,
                "use_conversation_multi_turn": True,
                "append_eos_token_after_stop_str_in_multi_turn": append_flag,
            }
        )
        env_cfg = DictConfig({"max_env_workers": 0, "env_class": "cpu_test_env"})
        gen = SkyRLGymGenerator(
            generator_cfg=generator_cfg,
            skyrl_gym_cfg=env_cfg,
            inference_engine_client=mock_llm,
            tokenizer=tokenizer,
            model_name=model_name,
        )
        return gen

    prompt = [[{"role": "user", "content": "a"}]]
    extras = [{"answer": "4"}]
    sp = {"stop": [stop_tag]}

    # Case 1: append flag = True
    generator_true = await make_generator(True)
    out_true: GeneratorOutput = await generator_true.generate(
        {"prompts": prompt, "env_extras": extras, "env_classes": ["cpu_test_env"], "sampling_params": sp}
    )

    # Case 2: append flag = False
    generator_false = await make_generator(False)
    out_false: GeneratorOutput = await generator_false.generate(
        {"prompts": prompt, "env_extras": extras, "env_classes": ["cpu_test_env"], "sampling_params": sp}
    )

    # Common assertions
    assert out_true["stop_reasons"][0] == "stop"
    assert out_false["stop_reasons"][0] == "stop"
    assert len(out_true["response_ids"][0]) == len(out_true["loss_masks"][0])
    assert len(out_false["response_ids"][0]) == len(out_false["loss_masks"][0])

    if "Qwen3" in model_name:
        # Retokenize path is not affected by append_eos_token_after_stop_str_in_multi_turn
        # The chat template does things like '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n'
        # So regardless of append_eos_token_after_stop_str_in_multi_turn, the last tokens are:
        # stop_tag, eos_token_id and \n
        last_token_ids = tokenizer.encode(stop_tag + tokenizer.eos_token + "\n")
        num_last_tokens = len(last_token_ids)
        response_ids_true = out_true["response_ids"][0]
        response_ids_false = out_false["response_ids"][0]
        assert response_ids_true[-num_last_tokens:] == last_token_ids
        assert response_ids_false[-num_last_tokens:] == last_token_ids
        assert response_ids_true == response_ids_false
    else:
        # Non-retokenize path: last token is eos only when append flag is True
        last_token_id_true = out_true["response_ids"][0][-1]
        last_token_id_false = out_false["response_ids"][0][-1]
        assert last_token_id_true == tokenizer.eos_token_id
        assert last_token_id_false == tokenizer.encode(mock_text, add_special_tokens=False)[-1]
