"""
uv run --extra dev --isolated pytest tests/cpu/generators/test_skyrl_gym_generator_chat_templating.py
"""

import pytest
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock
from skyrl_train.generators.skyrl_gym_generator import SkyRLGymGenerator
from skyrl_train.generators.base import GeneratorInput, GeneratorOutput
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from transformers import AutoTokenizer
from skyrl_gym.envs import register
from skyrl_train.generators.utils import get_custom_chat_template
from skyrl_train.config import GeneratorConfig, SamplingParams, ChatTemplateConfig, SkyRLGymConfig
from skyrl_train.generators.utils import CUSTOM_CHAT_TEMPLATES
from pathlib import Path
from tests.cpu.generators.chat_templating_test_constants import (
    QWEN2_5_EXPECTED_STR,
    LLAMA3_2_EXPECTED_STR,
    QWEN3_TITO_EXPECTED_STR,
    QWEN3_WITHOUT_THINKING_EXPECTED_STR,
    get_expected_chat_history,
)


# Setup for formatting tests
class CPUTestEnv(BaseTextEnv):
    def __init__(self, env_config: Any, extras: Dict[str, Any] = {}):
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


def _build_generator(tokenizer, model_name: str, chat_template_config, extra_overrides: Dict[str, Any] | None = None):
    """Helper to create a SkyRLGymGenerator with common config/env settings."""
    # Build chat template config dataclass
    if chat_template_config is None:
        ct_cfg = ChatTemplateConfig()
    elif isinstance(chat_template_config, dict):
        ct_cfg = ChatTemplateConfig(
            source=chat_template_config.get("source", "name"),
            name_or_path=chat_template_config.get("name_or_path"),
        )
    else:
        ct_cfg = chat_template_config

    # Build sampling params, allowing extra_overrides to override
    sp_kwargs = {"max_generate_length": 200, "logprobs": None}
    if extra_overrides and "sampling_params" in extra_overrides:
        sp_kwargs.update(extra_overrides.pop("sampling_params"))
    sampling_params = SamplingParams(**sp_kwargs)

    gen_kwargs = dict(
        sampling_params=sampling_params,
        max_input_length=200,
        batched=False,
        max_turns=3,
        zero_reward_on_non_stop=False,
        apply_overlong_filtering=False,
        use_conversation_multi_turn=True,
        chat_template=ct_cfg,
        append_eos_token_after_stop_str_in_multi_turn=True,
    )
    if extra_overrides:
        gen_kwargs.update(extra_overrides)

    generator_cfg = GeneratorConfig(**gen_kwargs)
    env_cfg = SkyRLGymConfig(max_env_workers=0)
    return SkyRLGymGenerator(
        generator_cfg=generator_cfg,
        skyrl_gym_cfg=env_cfg,
        inference_engine_client=None,  # to be replaced per-test
        tokenizer=tokenizer,
        model_name=model_name,
    )


def _default_prompt_and_extras():
    """Standard single-trajectory prompt and extras used throughout tests."""
    prompt = [[{"role": "user", "content": "a"}]]
    extras = [{"answer": "4"}]
    return prompt, extras


def _make_input_batch(prompt, extras):
    return {"prompts": prompt, "env_extras": extras, "env_classes": ["cpu_test_env"]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name,tokenization_codepath,expected_str",
    [
        ("Qwen/Qwen2.5-0.5B-Instruct", "tito", QWEN2_5_EXPECTED_STR),
        ("unsloth/Llama-3.2-1B-Instruct", "tito", LLAMA3_2_EXPECTED_STR),
        # Qwen3: test all three tokenization paths
        ("Qwen/Qwen3-0.6B", "tito", QWEN3_TITO_EXPECTED_STR),
        ("Qwen/Qwen3-0.6B", "custom_chat_template_from_path", QWEN3_WITHOUT_THINKING_EXPECTED_STR),
        ("Qwen/Qwen3-0.6B", "custom_chat_template_builtin", QWEN3_WITHOUT_THINKING_EXPECTED_STR),
    ],
    ids=[
        "qwen2_5-tito",
        "llama3_2-tito",
        "qwen3-tito",
        "qwen3-custom_chat_template_from_path",
        "qwen3-custom_chat_template_builtin",
    ],
)
async def test_skyrl_gym_generator_chat_templating_exact(model_name, tokenization_codepath, expected_str):
    """
    Tests the behavior of chat templating for various models in multi-turn conversation.

    `tokenization_codepath` being `tito` means token-in-token-out, which is codepath 1 described in
    `skyrl_gym_generator.rst`. For Qwen3, we also test `generator.chat_template` being defined.

    We hardcode the expected string in the constants file, so it is easier to check. But we also double
    check that those expected strings are correct by applying the chat template on the expected chat history.
    """
    # 1. Preparations to mock the generation.
    _register_test_env_if_needed()  # Register only when needed
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    mock_llm = MagicMock()

    # Parameterize mock response: Qwen3 uses thinking tokens, others use simple 'b'
    mock_response_text = "b"
    if "Qwen3" in model_name:
        mock_response_text = "<think>\nmock thinking\n</think>\n\n" + mock_response_text

    def mock_generate(input_batch):
        num_prompts = len(input_batch["prompts"]) if "prompts" in input_batch else len(input_batch["prompt_token_ids"])

        mock_llm_output_text = mock_response_text + tokenizer.eos_token

        return {
            # no tokenizer.eos_token for responses because `skip_special_tokens` is True in sampling params
            "responses": [mock_response_text] * num_prompts,
            "stop_reasons": ["stop"] * num_prompts,
            "response_logprobs": None,
            # add_special_tokens needs to be False, otherwise for instance Llama will always
            # add a `<|begin_of_text|>` before the assistant response.
            "response_ids": [tokenizer.encode(mock_llm_output_text, add_special_tokens=False)] * num_prompts,
        }

    mock_llm.generate = AsyncMock(side_effect=mock_generate)
    chat_template_config = None
    if "Qwen3" in model_name and tokenization_codepath == "custom_chat_template_from_path":
        template_path = Path(__file__).parent / "qwen3_acc_without_thinking.jinja2"
        chat_template_config = ChatTemplateConfig(source="file", name_or_path=str(template_path))
    elif "Qwen3" in model_name and tokenization_codepath == "custom_chat_template_builtin":
        chat_template_config = ChatTemplateConfig(source="name", name_or_path="qwen3_without_thinking")
    else:
        chat_template_config = ChatTemplateConfig(source="name", name_or_path=None)
    # Create a mock generator config
    generator = _build_generator(tokenizer, model_name, chat_template_config)
    generator.inference_engine_client = mock_llm

    prompt, extras = _default_prompt_and_extras()
    input_batch: GeneratorInput = _make_input_batch(prompt, extras)

    generator_output: GeneratorOutput = await generator.generate(input_batch)

    # 2. Double check that the hardcoded expected string is correct by recreating them.
    expected_chat_history = get_expected_chat_history(mock_response_text)
    if "Qwen3" in model_name and tokenization_codepath == "tito":
        keep_thinking_chat_template = get_custom_chat_template(
            ChatTemplateConfig(source="name", name_or_path="qwen3_with_thinking")
        )
        assert expected_str == tokenizer.apply_chat_template(
            expected_chat_history, tokenize=False, chat_template=keep_thinking_chat_template
        )
    else:
        assert expected_str == tokenizer.apply_chat_template(expected_chat_history, tokenize=False)

    # 3. Check that the full response is exactly string matching with applying the chat template on history
    prompt_str = tokenizer.decode(generator_output["prompt_token_ids"][0])
    resp_str = tokenizer.decode(generator_output["response_ids"][0])
    generator_output_str = prompt_str + resp_str
    if tokenization_codepath == "tito" and "Qwen" in model_name:
        # For Qwen models, there is an `\n` after the eos token. Our generator follows token-in-token-out,
        # so it will not generate anything after the eos token, and hence will not have the `\n`.
        # e.g. `<|assistant|>\Some content<|im_end|>\n` for expected_str, but
        # `<|assistant|>\Some content<|im_end|>` for generator_output_str.
        if expected_str.endswith("\n"):
            expected_str = expected_str[:-1]
    assert generator_output_str == expected_str

    # 4. Check loss mask exact matches
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

    # Build expected_loss_masks
    if "Qwen3" in model_name:
        # last [1, 0] -- 1 is for eos, 0 is for `\n`
        # Qwen3 with thinking content
        num_tokens_with_thinking = len(tokenizer.encode(mock_response_text))
        num_tokens_without_thinking = 1
        expected_user_loss_mask = [0] * len(empty_user) + [0]
        # `<|im_start|>assistant\n<think>\nmock thinking\n</think>\n\nb<|im_end|>\n`
        expected_assistant_loss_mask_with_thinking = (
            [0] * len(generation_prompt_ids) + [1] * num_tokens_with_thinking + [1, 0]
        )
        # `<|im_start|>assistant\nb<|im_end|>\n`
        expected_assistant_loss_mask_without_thinking = (
            [0] * len(generation_prompt_ids) + [1] * num_tokens_without_thinking + [1, 0]
        )
        # `<think>\nmock thinking\n</think>\n\nb<|im_end|>\n`
        expected_assistant_no_generation_prompt_loss_mask_with_thinking = [1] * num_tokens_with_thinking + [1, 0]

        if tokenization_codepath == "tito":
            # For non-custom_chat_template, `resp_str` directly starts with what the model generates
            expected_loss_masks = (
                expected_assistant_no_generation_prompt_loss_mask_with_thinking  # <think>\nmock thinking\n</think>\n\nb<|im_end|>\n
                + (
                    expected_user_loss_mask  # <|im_start|>user\n1<|im_end|>\n
                    + expected_assistant_loss_mask_with_thinking  # `<|im_start|>assistant\n<think>\nmock thinking\n</think>\n\nb<|im_end|>\n`
                )
                * 2
            )
            expected_loss_masks = expected_loss_masks[:-1]  # remove the extra 0 for \n
        else:
            # For chat templating, the first generation prompt IDs are part of `resp_str`, hence has corresponding mask
            expected_loss_masks = (
                expected_assistant_loss_mask_without_thinking  # `<|im_start|>assistant\nb<|im_end|>\n`
                + expected_user_loss_mask  # `<|im_start|>user\n1<|im_end|>\n`
            ) * 2 + expected_assistant_loss_mask_with_thinking  # last `<|im_start|>assistant\n<think>\nmock thinking\n</think>\n\nb<|im_end|>\n`
    else:
        # `<|im_start|>assistant\nb<|im_end|>\n`
        expected_assistant_loss_mask = [0] * len(generation_prompt_ids) + [
            1,
            1,
        ]  # 1 for single response token, 1 for eos
        expected_assistant_no_generation_prompt_loss_mask = [1, 1]  # 1 for single response token, 1 for eos
        if "Qwen" in model_name:
            expected_assistant_loss_mask += [0]  # extra 0 for \n for qwen templates
            expected_assistant_no_generation_prompt_loss_mask += [0]
        # `<|im_start|>user\n1<|im_end|>\n`
        expected_user_loss_mask = [0] * len(empty_user) + [0]  # extra 0 for single observation token

        assert tokenization_codepath == "tito"  # we only test custom chat template for Qwen3 models
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


def test_qwen3_original_vs_without_thinking_chat_template():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

    messages = [
        {"content": "hi", "role": "system"},
        {"content": "hi", "role": "user"},
        {"content": "<think>thinking</think>hi", "role": "assistant"},
        {"content": "hi", "role": "user"},
        {"content": "<think>thinking</think>hi", "role": "assistant"},
        {"content": "hi", "role": "user"},
    ]

    # Apply custom chat template
    qwen3_without_thinking_str = tokenizer.apply_chat_template(
        messages, chat_template=CUSTOM_CHAT_TEMPLATES["qwen3_without_thinking"], tokenize=False
    )

    # Apply custom chat template from file
    file_path = Path(__file__).parent / "qwen3_acc_without_thinking.jinja2"
    with open(file_path, "r", encoding="utf-8") as f:
        template_from_file = f.read()

    qwen3_without_thinking_str_from_file = tokenizer.apply_chat_template(
        messages, chat_template=template_from_file, tokenize=False
    )

    # Apply default chat template
    default_template_str = tokenizer.apply_chat_template(messages, chat_template=None, tokenize=False)

    # The original_chat_template should match the tokenizer exactly
    assert default_template_str == qwen3_without_thinking_str
    assert qwen3_without_thinking_str == qwen3_without_thinking_str_from_file


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_name,tokenization_codepath",
    [
        ("Qwen/Qwen2.5-0.5B-Instruct", "tito"),
        ("unsloth/Llama-3.2-1B-Instruct", "tito"),
        ("Qwen/Qwen3-0.6B", "tito"),
        ("Qwen/Qwen3-0.6B", "custom_chat_template_builtin"),
    ],
)
async def test_append_eos_after_stop_multi_turn(model_name, tokenization_codepath):
    """
    Test the behavior of `append_eos_token_after_stop_str_in_multi_turn`, which is applicable
    when `sampling_params.stop` is not `null` and `use_conversation_multi_turn` is `true` in
    the ``agent_loop()`` function.
    It is used in scripts `examples/search/run_search_conversation_format.sh` and
    `examples/text_to_sql/run_skyrl_sql_conversation_format.sh`.
    `tokenization_codepath` being `tito` means token-in-token-out, which is codepath 1 described in
    `skyrl_gym_generator.rst`. For Qwen3, we also test `generator.chat_template` being defined.
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
        chat_template_config = None
        if "Qwen3" in model_name and tokenization_codepath == "custom_chat_template_builtin":
            chat_template_config = ChatTemplateConfig(source="name", name_or_path="qwen3_without_thinking")
        else:
            chat_template_config = ChatTemplateConfig(source="name", name_or_path=None)
        extra_overrides = {
            "sampling_params": {"max_generate_length": 200, "logprobs": None, "stop": [stop_tag]},
            "append_eos_token_after_stop_str_in_multi_turn": append_flag,
        }
        gen = _build_generator(tokenizer, model_name, chat_template_config, extra_overrides)
        gen.inference_engine_client = mock_llm
        return gen

    prompt, extras = _default_prompt_and_extras()
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

    if "Qwen3" in model_name and not tokenization_codepath == "tito":
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
