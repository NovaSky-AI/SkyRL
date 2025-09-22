"""
uv run --extra dev --isolated pytest tests/cpu/generators/test_skyrl_gym_generator.py
"""

import pytest
from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch
from skyrl_train.generators.skyrl_gym_generator import SkyRLGymGenerator
from skyrl_train.generators.base import GeneratorInput, GeneratorOutput, ConversationType
from skyrl_train.generators.utils import concatenate_generator_outputs, get_metrics_from_generator_output
from skyrl_gym.envs.base_text_env import BaseTextEnvStepOutput


# Mock constants, where 4 is the eos token id
MOCK_LLM_OUTPUT_IDS = [1, 10, 12, 4]
MOCK_TOKENIZER_ENCODED_IDS = [1, 2, 3, 4]


# TODO (erictang000): clean up the mocking for tests in this file
@pytest.fixture
def mock_tokenizer():
    """
    A mock tokenizer that encodes any non-empty string to `MOCK_TOKENIZER_ENCODED_IDS`.
    For chat template, if `tokenize=False`, concatenate the content of each message.
    If `tokenize=True`, return `MOCK_TOKENIZER_ENCODED_IDS` for each message.
    """
    tokenizer = MagicMock()

    def mock_apply_chat_template(x, **kwargs):
        if not kwargs.get("tokenize", True):
            return "".join([str(i["content"]) for i in x])
        else:
            # Non-dict return
            if isinstance(x, list) and len(x) > 0 and isinstance(x[0], list):
                # Multiple prompts
                return [MOCK_TOKENIZER_ENCODED_IDS.copy() for _ in x]
            else:
                # Single prompt or conversation
                return MOCK_TOKENIZER_ENCODED_IDS.copy()

    def mock_encode(x, **kwargs):
        if x != "":
            return MOCK_TOKENIZER_ENCODED_IDS.copy()
        else:
            return []

    tokenizer.apply_chat_template.side_effect = mock_apply_chat_template
    tokenizer.decode.side_effect = lambda x: "decoded_output"
    tokenizer.encode.side_effect = mock_encode
    tokenizer.eos_token_id = 4
    tokenizer.eos_token = "<|end_of_turn|>"
    tokenizer.return_value = {"input_ids": MOCK_TOKENIZER_ENCODED_IDS.copy()}  # simulate tokenized response
    return tokenizer


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
    cfg = MagicMock()
    cfg.sampling_params.max_generate_length = 5
    cfg.sampling_params.logprobs = None
    cfg.apply_overlong_filtering = False
    cfg.max_input_length = 512
    cfg.batched = True
    cfg.max_turns = 1
    return cfg


@pytest.fixture
def mock_env_cfg():
    cfg = MagicMock()
    cfg.max_env_workers = 0
    cfg.env_class = "gsm8k"
    return cfg


def validate_generator_input(input_batch: GeneratorInput) -> bool:
    """Validate that input_batch conforms to GeneratorInput TypedDict interface."""
    # Check that input_batch has the required keys
    required_keys = {"prompts", "env_extras"}
    if not all(key in input_batch for key in required_keys):
        return False

    # Validate prompts: List[ConversationType] where ConversationType = List[MessageType]
    prompts = input_batch["prompts"]
    if not isinstance(prompts, list):
        return False

    for conversation in prompts:
        if not isinstance(conversation, list):
            return False
        for message in conversation:
            if not isinstance(message, dict):
                return False
            if not all(isinstance(k, str) and isinstance(v, str) for k, v in message.items()):
                return False

    # Validate env_extras: Optional[List[Dict[str, Any]]]
    env_extras = input_batch["env_extras"]
    if env_extras is not None:
        if not isinstance(env_extras, list):
            return False
        for extra in env_extras:
            if not isinstance(extra, dict):
                return False
            if not all(isinstance(k, str) for k in extra.keys()):
                return False

    return True


def validate_generator_output(output: GeneratorOutput) -> bool:
    """Validate that output conforms to GeneratorOutput TypedDict interface."""
    # Check that output has all required keys
    required_keys = {
        "prompt_token_ids",
        "response_ids",
        "rewards",
        "loss_masks",
        "stop_reasons",
        "rollout_metrics",
        "rollout_logprobs",
    }
    if not all(key in output for key in required_keys):
        return False

    # Validate prompt_token_ids: List[List[int]]
    prompt_token_ids = output["prompt_token_ids"]
    if not isinstance(prompt_token_ids, list):
        return False
    for token_ids in prompt_token_ids:
        if not isinstance(token_ids, list):
            return False
        if not all(isinstance(token, int) for token in token_ids):
            return False

    # Validate response_ids: List[List[int]]
    response_ids = output["response_ids"]
    if not isinstance(response_ids, list):
        return False
    for token_ids in response_ids:
        if not isinstance(token_ids, list):
            return False
        if not all(isinstance(token, int) for token in token_ids):
            return False

    # Validate rewards: List[float] or List[List[float]]
    rewards = output["rewards"]
    if not isinstance(rewards, list):
        return False
    is_list_of_float = all(isinstance(r, (int, float)) for r in rewards)
    is_list_of_list_float = all(isinstance(r, list) and all(isinstance(x, (int, float)) for x in r) for r in rewards)
    if not (is_list_of_float or is_list_of_list_float):
        return False

    # Validate loss_masks: List[List[int]]
    loss_masks = output["loss_masks"]
    if not isinstance(loss_masks, list):
        return False
    for mask in loss_masks:
        if not isinstance(mask, list):
            return False
        if not all(isinstance(val, int) for val in mask):
            return False

    # Validate stop_reasons: Optional[List[str]]
    stop_reasons = output["stop_reasons"]
    if stop_reasons is not None:
        if not isinstance(stop_reasons, list):
            return False
        if not all(isinstance(reason, str) for reason in stop_reasons):
            return False

    # Validate rollout_metrics: Optional[Dict[str, Any]]
    rollout_metrics = output["rollout_metrics"]
    if rollout_metrics is not None:
        if not isinstance(rollout_metrics, dict):
            return False
        if not all(isinstance(k, str) for k in rollout_metrics.keys()):
            return False

    rollout_logprobs = output["rollout_logprobs"]
    if rollout_logprobs is not None:
        if not isinstance(rollout_logprobs, list):
            return False
        for sample_logprobs in rollout_logprobs:
            if not isinstance(sample_logprobs, list):
                return False
            if not all(isinstance(val, (int, float)) for val in sample_logprobs):
                return False
    return True


@pytest.mark.asyncio
@patch("skyrl_gym.make")
@pytest.mark.parametrize("use_conversation_multi_turn", [True, False])
async def test_agent_loop_single_turn(
    mock_make, mock_tokenizer, mock_llm, mock_env, mock_generator_cfg, use_conversation_multi_turn, mock_env_cfg
):
    """
    This test mocks when we call SkyRLGymGenerator.agent_loop() despite being a single-turn generation.
    This is when `batched=False`. Here the environment does nothing.
    """
    mock_generator_cfg.use_conversation_multi_turn = use_conversation_multi_turn
    mock_env.step.side_effect = lambda x: BaseTextEnvStepOutput(observations=[], reward=1.0, done=True, metadata={})
    mock_tokenizer.eos_token_id = 4  # bypass check for eos token id for this test

    mock_make.return_value = mock_env
    mock_env.init.return_value = ([{"role": "user", "content": "Initial input"}], {})

    generator = SkyRLGymGenerator(
        generator_cfg=mock_generator_cfg,
        skyrl_gym_cfg=mock_env_cfg,
        inference_engine_client=mock_llm,
        tokenizer=mock_tokenizer,
        model_name="test_model",
    )
    generator.base_conversation_token_ids = []  # to make sure observation_ids are encoded correctly

    prompt = [{"role": "user", "content": "What is 2 + 2?"}]
    extras = {"answer": "4"}
    output = await generator.agent_loop(prompt, mock_env_cfg.env_class, extras, max_tokens=8, max_input_length=512)

    assert output.response_ids == MOCK_LLM_OUTPUT_IDS
    if isinstance(output.reward, list):
        assert sum(output.reward) == 1.0
    else:
        assert output.reward == 1.0
    assert output.stop_reason == "stop"
    assert output.loss_mask == [1] * len(MOCK_LLM_OUTPUT_IDS)


@pytest.mark.asyncio
@patch("skyrl_gym.make")
async def test_generate_batched(mock_make, mock_tokenizer, mock_llm, mock_env, mock_generator_cfg, mock_env_cfg):
    mock_make.return_value = mock_env
    mock_env.init.return_value = ([{"role": "user", "content": "Initial input"}], {})

    generator = SkyRLGymGenerator(
        generator_cfg=mock_generator_cfg,
        skyrl_gym_cfg=mock_env_cfg,
        inference_engine_client=mock_llm,
        tokenizer=mock_tokenizer,
        model_name="test_model",
    )
    generator.base_conversation_token_ids = []  # to make sure observation_ids are encoded correctly

    prompts = [[{"role": "user", "content": "What is 3 + 5?"}]]
    env_extras = [{"answer": "8"}]

    input_batch: GeneratorInput = {
        "prompts": prompts,
        "env_extras": env_extras,
        "env_classes": [mock_env_cfg.env_class for _ in prompts],  # Mock environment class for each prompt
    }

    generator_output: GeneratorOutput = await generator.generate(input_batch)

    # uses output from llm directly
    assert generator_output["response_ids"][0] == MOCK_LLM_OUTPUT_IDS

    assert generator_output["rewards"][0] == 1.0
    assert generator_output["stop_reasons"][0] == "stop"
    assert generator_output["loss_masks"][0] == [1] * len(MOCK_LLM_OUTPUT_IDS)


def test_generator_output_concatenation():
    # First ensure that the GeneratorOutput fields are what we expect
    expected_fields = [
        "prompt_token_ids",
        "response_ids",
        "rewards",
        "loss_masks",
        "stop_reasons",
        "rollout_metrics",
        "rollout_logprobs",
    ]
    assert set(GeneratorOutput.__annotations__.keys()) == set(expected_fields), (
        "GeneratorOutput fields are not what we expect. "
        "Please update the test and `concatenate_generator_outputs()` to reflect the new fields."
        "It is needed to help Trainer.eval() record the full GeneratorOutput information."
    )

    generator_output_1: GeneratorOutput = {
        "prompt_token_ids": [[1, 2], [3, 4]],
        "response_ids": [[1, 2], [3, 4]],
        "rewards": [1.0, 2.0],
        "loss_masks": [[1, 1], [1, 1]],
        "stop_reasons": ["stop", "stop"],
        "rollout_logprobs": [[0.1, 0.2], [0.3, 0.4]],
    }

    generator_output_2: GeneratorOutput = {
        "prompt_token_ids": [[5, 6, 7], [8]],
        "response_ids": [[5, 6, 7], [8]],
        "rewards": [2.0, 3.0],
        "loss_masks": [[1, 1, 1], [1, 1, 1]],
        "stop_reasons": ["stop", "stop"],
        "rollout_logprobs": [[0.5, 0.6], [0.7, 0.8]],
    }

    generator_outputs = [generator_output_1, generator_output_2]
    concatenated_output = concatenate_generator_outputs(generator_outputs)

    assert concatenated_output["prompt_token_ids"] == [[1, 2], [3, 4], [5, 6, 7], [8]]
    assert concatenated_output["response_ids"] == [[1, 2], [3, 4], [5, 6, 7], [8]]
    assert concatenated_output["rewards"] == [1.0, 2.0, 2.0, 3.0]
    assert concatenated_output["loss_masks"] == [[1, 1], [1, 1], [1, 1, 1], [1, 1, 1]]
    assert concatenated_output["stop_reasons"] == ["stop", "stop", "stop", "stop"]
    assert concatenated_output["rollout_logprobs"] == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]


def test_get_metrics_from_generator_output():
    # Per trajectory rewards, where rewards are List[float]
    generator_output: GeneratorOutput = {
        "prompt_token_ids": [[1, 2], [3, 4]],
        "response_ids": [[1, 2], [3, 4]],
        "rewards": [1.0, 2.0],
        "loss_masks": [[1, 1], [1, 1]],
        "stop_reasons": ["stop", "stop"],
        "rollout_logprobs": None,
    }
    uids = ["a", "b"]
    avg_score, pass_at_n = get_metrics_from_generator_output(generator_output, uids)
    assert avg_score == 1.5
    assert pass_at_n == 1.0

    # Per token rewards, where rewards are List[List[float]], so for pass_at_n we use the last
    # token's reward to signify the trajectory's reward
    generator_output["rewards"] = [[1.0, 0.0], [0.0, 1.0]]
    uids = ["a", "b"]
    avg_score, pass_at_n = get_metrics_from_generator_output(generator_output, uids)
    assert avg_score == 1.0
    assert pass_at_n == 0.5


@pytest.mark.asyncio
@pytest.mark.parametrize("batched", [True, False])
@patch("skyrl_gym.make")
async def test_generate_interface_compliance(
    mock_make, mock_tokenizer, mock_llm, mock_env, mock_generator_cfg, mock_env_cfg, batched
):
    """Test that SkyRLGymGenerator.generate() strictly conforms to the TypedDict interface.

    Tests both batched and non-batched modes to ensure interface compliance.
    """
    mock_make.return_value = mock_env
    # Set the batched mode according to the parameter
    mock_generator_cfg.batched = batched
    mock_env.init.return_value = ([{"role": "user", "content": "Initial input"}], {})

    generator = SkyRLGymGenerator(
        generator_cfg=mock_generator_cfg,
        skyrl_gym_cfg=mock_env_cfg,
        inference_engine_client=mock_llm,
        tokenizer=mock_tokenizer,
        model_name="test_model",
    )
    generator.base_conversation_token_ids = []  # to make sure observation_ids are encoded correctly

    # Create test data based on batched mode
    if batched:
        # For batched mode, test with multiple prompts
        prompts: List[ConversationType] = [
            [{"role": "user", "content": "What is 3 + 5?"}],
            [{"role": "user", "content": "Solve 10 - 7"}],
        ]
        env_extras: List[Dict[str, Any]] = [{"answer": "8"}, {"answer": "3"}]
    else:
        # For non-batched mode, test with single prompt
        prompts: List[ConversationType] = [[{"role": "user", "content": "What is 2 * 3?"}]]
        env_extras: List[Dict[str, Any]] = [{"answer": "6"}]
    env_classes = [mock_env_cfg.env_class for _ in prompts]

    input_batch: GeneratorInput = {
        "prompts": prompts,
        "env_extras": env_extras,
        "env_classes": env_classes,
    }

    # Validate input conforms to interface
    assert validate_generator_input(
        input_batch
    ), f"Input does not conform to GeneratorInput interface (batched={batched})"

    # Call generate method
    generator_output: GeneratorOutput = await generator.generate(input_batch)

    # Validate output conforms to interface
    assert validate_generator_output(
        generator_output
    ), f"Output does not conform to GeneratorOutput interface (batched={batched})"

    # Additional specific type checks
    assert isinstance(generator_output, dict), "Output should be a dictionary"
    assert len(generator_output["response_ids"]) == len(
        prompts
    ), f"Number of responses should match number of prompts (batched={batched})"
    assert len(generator_output["rewards"]) == len(
        prompts
    ), f"Number of rewards should match number of prompts (batched={batched})"
    assert len(generator_output["loss_masks"]) == len(
        prompts
    ), f"Number of loss masks should match number of prompts (batched={batched})"

    # Test with None env_extras to ensure Optional handling works (only test this once)
    if batched:
        input_batch_with_none: GeneratorInput = {
            "prompts": prompts[:1],  # Just one prompt
            "env_extras": None,
        }

        # This should not raise an error even with None env_extras
        assert validate_generator_input(input_batch_with_none), "Input with None env_extras should be valid"


@pytest.mark.asyncio
@pytest.mark.parametrize("turns_to_exceed", [1, 3])  # Test single-turn and multi-turn scenarios
@patch("skyrl_gym.make")
async def test_length_limit_exceeded_during_conversation(
    mock_make, mock_tokenizer, mock_llm, mock_env, mock_generator_cfg, mock_env_cfg, turns_to_exceed
):
    """Test that length limit is enforced during multi-turn conversations.

    Tests both single-turn (turns_to_exceed=1) and multi-turn (turns_to_exceed=3) scenarios
    to verify length accumulation and limit enforcement.
    """
    mock_make.return_value = mock_env
    mock_generator_cfg.batched = False  # Use agent_loop mode
    mock_generator_cfg.max_turns = 5  # Allow multiple turns
    mock_generator_cfg.use_conversation_multi_turn = True
    mock_env.init.return_value = ([{"role": "user", "content": "Initial input"}], {})

    # Configure environment to never set done=True naturally (we want to hit length limit)
    def mock_step_never_done(output):
        return BaseTextEnvStepOutput(
            observations=[{"role": "user", "content": "next"}],
            reward=0.5,
            done=False,
            metadata={},
        )

    # We start with initial prompt len 4 due to mock_apply_chat_template
    # Each turn, observation is 4 tokens due to mock_encode
    mock_env.step.side_effect = mock_step_never_done
    max_input_length = 20  # Low limit to trigger length exceeded

    # Mock the new generate method
    def mock_generate(input_batch):
        num_prompts = len(input_batch["prompts"]) if "prompts" in input_batch else len(input_batch["prompt_token_ids"])
        if turns_to_exceed == 1:
            mock_llm_output_ids = [1] * 20  # Enough to exceed limit immediately (4 + 20 + 4 = 28 > 20)
            assert (
                len(MOCK_TOKENIZER_ENCODED_IDS) + len(mock_llm_output_ids) + len(MOCK_TOKENIZER_ENCODED_IDS)
                > max_input_length
            )
        else:
            assert turns_to_exceed == 3
            mock_llm_output_ids = [1] * 2  # Enough to exceed limit after 3 turns (4 + (2 + 4) * 3 = 22 > 20)
            assert (
                len(MOCK_TOKENIZER_ENCODED_IDS)
                + (len(mock_llm_output_ids) + len(MOCK_TOKENIZER_ENCODED_IDS)) * turns_to_exceed
                > max_input_length
            )
        return {
            "responses": ["mocked output"] * num_prompts,
            "stop_reasons": ["stop"] * num_prompts,
            "response_logprobs": [[0.1] * len(mock_llm_output_ids)] * num_prompts,
            "response_ids": [mock_llm_output_ids.copy()] * num_prompts,
        }

    mock_llm.generate = AsyncMock(side_effect=mock_generate)

    generator = SkyRLGymGenerator(
        generator_cfg=mock_generator_cfg,
        skyrl_gym_cfg=mock_env_cfg,
        inference_engine_client=mock_llm,
        tokenizer=mock_tokenizer,
        model_name="test_model",
    )
    generator.base_conversation_token_ids = []  # to make sure observation_ids are encoded correctly

    prompt = [{"role": "user", "content": "Start conversation"}]
    extras = {"test": "value"}

    output = await generator.agent_loop(prompt, "test_env", extras, max_tokens=100, max_input_length=max_input_length)

    # Verify that length limit was hit
    assert output.stop_reason == "length", f"Expected stop_reason='length', got '{output.stop_reason}'"

    # Verify environment step was called the expected number of times
    expected_calls = turns_to_exceed
    assert (
        mock_env.step.call_count == expected_calls
    ), f"Expected {expected_calls} environment steps, got {mock_env.step.call_count}"

    # Verify response is still properly formatted
    assert isinstance(output.response_ids, list)
    assert isinstance(output.loss_mask, list)
    assert isinstance(output.reward, float) or isinstance(output.reward, list)


@pytest.mark.asyncio
@patch("skyrl_gym.make")
async def test_multi_turn_response_truncation(
    mock_make, mock_tokenizer, mock_llm, mock_env, mock_generator_cfg, mock_env_cfg
):
    """
    Tests that in a multi-turn conversation, if the final tokenized response exceeds the
    calculated maximum length, it is correctly truncated and the stop reason is set to 'length'.
    """
    mock_make.return_value = mock_env
    mock_generator_cfg.max_turns = 3  # Ensure multi-turn logic is triggered
    mock_generator_cfg.batched = False  # Test is for agent_loop
    mock_generator_cfg.use_conversation_multi_turn = True
    mock_env.init.return_value = ([{"role": "user", "content": "Initial input"}], {})

    # Configure environment to run for multiple turns to generate enough tokens for truncation
    step_count = 0

    def mock_step_multi_turn(output):
        nonlocal step_count
        step_count += 1
        done = step_count >= 10  # Allow many turns to exceed length limit
        return BaseTextEnvStepOutput(
            observations=[{"role": "user", "content": "next turn"}], reward=0.5, done=done, metadata={}
        )

    mock_env.step.side_effect = mock_step_multi_turn

    # Define token lengths to control the test
    initial_prompt_len = 13
    max_tokens_from_llm = 20
    max_input_len = 50

    # Expected max response tokens = max_tokens + max_input_length - initial_prompt_length
    expected_max_response_tokens = max_tokens_from_llm + max_input_len - initial_prompt_len  # 20 + 50 - 13 = 57

    def mock_apply_chat_template(messages, **kwargs):
        if kwargs.get("tokenize", True):
            # Return initial prompt tokens
            return [1] * initial_prompt_len
        else:
            # Not used in messages_mode=False
            return "".join([msg.get("content", "") for msg in messages])

    def mock_encode(text, **kwargs):
        # This makes observation_ids to always be 13 tokens
        return [1] * 13

    mock_tokenizer.apply_chat_template.side_effect = mock_apply_chat_template
    mock_tokenizer.encode.side_effect = mock_encode

    # The intitial prompt is 13 tokens due to mock_apply_chat_template
    # Each turn, observation is 13 tokens due to mock_encode and empty system_prompt_ids
    # And the LLM response is 4 tokens due to MOCK_LLM_OUTPUT_IDS
    # So input_ids are 13, 30, 47, 64. And 64 would cause a break in the loop due to exceeding max_input_len.
    # Then with 64, we get the `input_ids[initial_prompt_length:]`, which makes our final
    # response_ids to be 64 - 13 = 51 tokens. So in this case, we are not truncated by expected_max_response_tokens.
    expected_final_response_tokens = 51

    generator = SkyRLGymGenerator(
        generator_cfg=mock_generator_cfg,
        skyrl_gym_cfg=mock_env_cfg,
        inference_engine_client=mock_llm,
        tokenizer=mock_tokenizer,
        model_name="test_model",
    )
    generator.base_conversation_token_ids = []  # to make sure observation_ids are encoded correctly

    prompt = [{"role": "user", "content": "Initial prompt"}]
    extras = {}

    output = await generator.agent_loop(
        prompt, "test_env", extras, max_tokens=max_tokens_from_llm, max_input_length=max_input_len
    )

    # Verify truncation occurred
    assert len(output.response_ids) <= expected_max_response_tokens
    assert (
        len(output.response_ids) == expected_final_response_tokens
    ), f"Expected {expected_final_response_tokens} response tokens, got {len(output.response_ids)}"
    assert (
        len(output.loss_mask) == expected_final_response_tokens
    ), f"Expected {expected_final_response_tokens} loss mask entries, got {len(output.loss_mask)}"

    # Verify stop reason is "length" due to truncation
    assert output.stop_reason == "length", f"Expected stop_reason='length', got '{output.stop_reason}'"


@pytest.mark.asyncio
@patch("skyrl_gym.make")
async def test_postprocessed_action_used(
    mock_make, mock_tokenizer, mock_llm, mock_env, mock_env_cfg, mock_generator_cfg
):
    """
    Tests that if the environment returns a `postprocessed_action`, it is used
    in the chat history instead of the original LLM response.
    """
    mock_make.return_value = mock_env
    mock_generator_cfg.max_turns = 1  # Single turn
    mock_generator_cfg.batched = False
    mock_env.init.return_value = ([{"role": "user", "content": "Initial input"}], {})

    postprocessed_response = "This is a clean response."
    llm_raw_response = "RAW LLM OUTPUT"

    # Environment step returns a postprocessed version of the LLM response
    def mock_step(_):
        return BaseTextEnvStepOutput(
            observations=[{"role": "user", "content": "new input"}],
            reward=1.0,
            done=True,
            metadata={},
            postprocessed_action=postprocessed_response,
        )

    mock_env.step.side_effect = mock_step

    # The LLM will output a raw string, which should be overridden
    mock_llm.generate.return_value = {
        "responses": [llm_raw_response],
        "stop_reasons": ["stop"],
    }

    def mock_apply_chat_template(messages, **kwargs):
        if kwargs.get("tokenize", True):
            return [1] * 5  # Initial prompt tokens
        else:
            return "".join([msg.get("content", "") for msg in messages])

    def mock_encode(text, **kwargs):
        # The key test: postprocessed response should be encoded, not raw LLM output
        if postprocessed_response in str(text):
            return [42] * 10  # Distinctive tokens for postprocessed response
        elif "new input" in str(text):
            return [5] * 2  # Observation tokens
        else:
            return [1] * 3  # Default tokens

    mock_tokenizer.apply_chat_template.side_effect = mock_apply_chat_template
    mock_tokenizer.encode.side_effect = mock_encode

    generator = SkyRLGymGenerator(
        generator_cfg=mock_generator_cfg,
        skyrl_gym_cfg=mock_env_cfg,
        inference_engine_client=mock_llm,
        tokenizer=mock_tokenizer,
        model_name="test_model",
    )
    generator.base_conversation_token_ids = []  # to make sure observation_ids are encoded correctly

    prompt = [{"role": "user", "content": "Initial input"}]
    env_extras = {}

    output = await generator.agent_loop(prompt, "test_env", env_extras, max_tokens=20, max_input_length=50)

    # Check that the postprocessed response tokens (42) are present in response_ids
    # This verifies that postprocessed_action was used instead of raw LLM output
    assert any(
        token == 42 for token in output.response_ids
    ), f"Expected postprocessed response tokens (42) in {output.response_ids}"
    # Make sure raw LLM tokens (99) are NOT present
    assert not any(
        token == 99 for token in output.response_ids
    ), f"Raw LLM output tokens (99) should not be in {output.response_ids}"

    if isinstance(output.reward, list):
        assert sum(output.reward) == 1.0
    else:
        assert output.reward == 1.0
    assert output.stop_reason == "stop"
    assert len(output.response_ids) == len(output.loss_mask)


@pytest.mark.asyncio
@patch("skyrl_gym.make")
async def test_apply_overlong_filtering_non_batched(
    mock_make, mock_tokenizer, mock_llm, mock_env, mock_generator_cfg, mock_env_cfg
):
    """
    Test that apply_overlong_filtering correctly zeroes out loss masks for truncated trajectories
    in non-batched mode (using agent_loop).

    Tests both truncated and non-truncated responses to verify that:
    - Trajectories with responses not ending with eos token have their loss masks zeroed out
    - Trajectories with responses ending with eos token keep their original loss masks
    """
    mock_make.return_value = mock_env
    mock_generator_cfg.apply_overlong_filtering = True  # Enable filtering
    mock_generator_cfg.batched = False
    mock_generator_cfg.max_turns = 1
    mock_generator_cfg.use_conversation_multi_turn = False
    mock_env.init.return_value = ([{"role": "user", "content": "Initial input"}], {})

    # Mock out the environment and inference engine generation.
    mock_env.step.side_effect = lambda x: BaseTextEnvStepOutput(observations=[], reward=1.0, done=True, metadata={})

    def mock_apply_chat_template(messages, **kwargs):
        if kwargs.get("tokenize", True):
            return [1, 2, 3, 4, 5]  # 5 tokens for prompt
        else:
            return "".join([msg.get("content", "") for msg in messages])

    mock_tokenizer.apply_chat_template.side_effect = mock_apply_chat_template
    mock_tokenizer.eos_token_id = 4  # Set EOS token ID

    generator = SkyRLGymGenerator(
        generator_cfg=mock_generator_cfg,
        skyrl_gym_cfg=mock_env_cfg,
        inference_engine_client=mock_llm,
        tokenizer=mock_tokenizer,
        model_name="test_model",
    )
    generator.base_conversation_token_ids = []  # to make sure observation_ids are encoded correctly

    # First test: response that doesn't end with eos token (should be filtered)
    mock_llm.generate = AsyncMock(
        return_value={
            "responses": ["truncated response"],
            "stop_reasons": ["length"],
            "response_ids": [[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]],  # 10 tokens, will be truncated
        }
    )

    input_batch_truncated: GeneratorInput = {
        "prompts": [[{"role": "user", "content": "Test prompt"}]],
        "env_extras": [{"test": "value"}],
        "env_classes": [mock_env_cfg.env_class],
    }

    output_truncated = await generator.generate(input_batch_truncated)

    # Verify truncated response has zeroed loss mask
    assert len(output_truncated["loss_masks"]) == 1
    assert len(output_truncated["loss_masks"][0]) == 5  # Truncated to max_generate_length=5
    assert output_truncated["loss_masks"][0] == [
        0,
        0,
        0,
        0,
        0,
    ], "Loss mask should be all zeros for response not ending with eos token"
    # Note: The long response gets truncated by max_response_tokens, so it doesn't end with eos token

    # Second test: response that ends with eos token (should not be filtered)
    # Reset the environment init to ensure clean state
    mock_env.init.return_value = ([{"role": "user", "content": "Fresh input"}], {})
    mock_llm.generate = AsyncMock(
        return_value={
            "responses": ["truncated response"],
            "stop_reasons": ["length"],
            "response_ids": [[20, 21, 4]],  # 3 tokens, ends with eos token 4
        }
    )

    input_batch_normal: GeneratorInput = {
        "prompts": [[{"role": "user", "content": "Another test prompt"}]],
        "env_extras": [{"test": "value"}],
        "env_classes": [mock_env_cfg.env_class],
    }

    output_normal = await generator.generate(input_batch_normal)

    # Verify normal response keeps original loss mask (all 1s)
    assert len(output_normal["loss_masks"]) == 1
    assert len(output_normal["loss_masks"][0]) == 3  # 3 response tokens (already includes EOS token)
    assert output_normal["loss_masks"][0] == [
        1,
        1,
        1,
    ], "Loss mask should remain as 1s for response ending with eos token"


@pytest.mark.asyncio
@patch("skyrl_gym.make")
async def test_apply_overlong_filtering_batched(
    mock_make,
    mock_tokenizer,
    mock_llm,
    mock_env,
    mock_generator_cfg,
    mock_env_cfg,
):
    """
    Test that apply_overlong_filtering correctly zeroes out loss masks for truncated trajectories
    in batched mode.

    Tests a response that doesn't end with eos token to verify that it gets filtered.
    """
    mock_make.return_value = mock_env
    mock_generator_cfg.apply_overlong_filtering = True  # Enable filtering
    mock_generator_cfg.batched = True
    mock_generator_cfg.max_turns = 1
    mock_env.init.return_value = ([{"role": "user", "content": "Initial input"}], {})

    # Mock out environment and inference engine generation.
    mock_env.step.side_effect = lambda x: BaseTextEnvStepOutput(observations=[], reward=1.0, done=True, metadata={})
    mock_llm.generate = AsyncMock(
        return_value={
            "responses": ["truncated response"],
            "stop_reasons": ["length"],
            "response_ids": [[10, 11, 12, 13]],
        }
    )

    def mock_apply_chat_template(messages, **kwargs):
        if kwargs.get("tokenize", True):
            return [[1, 2, 3, 4, 5] for _ in messages]  # 5 tokens for each prompt
        else:
            return "".join([msg.get("content", "") for msg in messages])

    def mock_encode_or_tokenize(text):
        return [10, 11, 12, 13]  # 4 tokens, doesn't end with eos_token_id=4

    mock_tokenizer.apply_chat_template.side_effect = mock_apply_chat_template
    mock_tokenizer.side_effect = lambda text: {"input_ids": mock_encode_or_tokenize(text)}
    mock_tokenizer.eos_token_id = 4  # Set EOS token ID

    generator = SkyRLGymGenerator(
        generator_cfg=mock_generator_cfg,
        skyrl_gym_cfg=mock_env_cfg,
        inference_engine_client=mock_llm,
        tokenizer=mock_tokenizer,
        model_name="test_model",
    )
    generator.base_conversation_token_ids = []  # to make sure observation_ids are encoded correctly

    # Test batched mode with response that doesn't end with eos token
    prompts = [[{"role": "user", "content": "Test prompt"}]]
    env_extras = [{"test": "value"}]
    env_classes = [mock_env_cfg.env_class]

    input_batch: GeneratorInput = {
        "prompts": prompts,
        "env_extras": env_extras,
        "env_classes": env_classes,
    }

    generator_output = await generator.generate(input_batch)

    # Verify that the loss mask is zeroed out for the response not ending with eos token
    assert len(generator_output["loss_masks"]) == 1
    assert len(generator_output["loss_masks"][0]) == 4  # Should match response length
    assert generator_output["loss_masks"][0] == [
        0,
        0,
        0,
        0,
    ], "Loss mask should be all zeros for response not ending with eos token"


@pytest.mark.asyncio
@patch("skyrl_gym.make")
async def test_agent_loop_token_level_rewards_multi_turn(mock_make, mock_tokenizer, mock_llm, mock_env_cfg):
    """use_conversation_multi_turn=False; verify rewards at assistant turn ends across two steps."""
    # Tokenizer behavior
    mock_tokenizer.eos_token_id = 4

    def apply_chat_template_side_effect(messages, **kwargs):
        # initial prompt tokenization
        if kwargs.get("tokenize", True):
            return [101, 102]
        else:
            return "".join([m.get("content", "") for m in messages])

    def encode_side_effect(text, **kwargs):
        # one token for each observation
        return [77] if text else []

    mock_tokenizer.apply_chat_template.side_effect = apply_chat_template_side_effect
    mock_tokenizer.encode.side_effect = encode_side_effect

    # LLM returns fixed response tokens per step: 3 tokens + eos
    async def llm_generate_side_effect(input_batch):
        num = (
            len(input_batch["prompt_token_ids"]) if "prompt_token_ids" in input_batch else len(input_batch["prompts"])
        )  # noqa: E501
        return {
            "responses": ["aaa"] * num,
            "stop_reasons": ["stop"] * num,
            "response_logprobs": None,
            "response_ids": [[10, 11, 12, mock_tokenizer.eos_token_id] for _ in range(num)],
        }

    mock_llm.generate = AsyncMock(side_effect=llm_generate_side_effect)

    # Two-step env with rewards 0.3 then 1.7
    class TwoStepEnv:
        def __init__(self):
            self.turns = 0

        def init(self, prompt):
            return prompt, {}

        def step(self, action):
            self.turns += 1
            if self.turns == 1:
                return BaseTextEnvStepOutput(
                    observations=[{"role": "user", "content": "obs1"}], reward=0.3, done=False, metadata={}
                )
            else:
                return BaseTextEnvStepOutput(observations=[], reward=1.7, done=True, metadata={})

        def close(self):
            return None

    mock_make.return_value = TwoStepEnv()

    # Generator config
    cfg = MagicMock()
    cfg.sampling_params.max_generate_length = 50
    cfg.sampling_params.logprobs = None
    cfg.apply_overlong_filtering = False
    cfg.max_input_length = 512
    cfg.batched = False
    cfg.max_turns = 10
    cfg.zero_reward_on_non_stop = False
    cfg.use_conversation_multi_turn = False

    generator = SkyRLGymGenerator(
        generator_cfg=cfg,
        skyrl_gym_cfg=mock_env_cfg,
        inference_engine_client=mock_llm,
        tokenizer=mock_tokenizer,
        model_name="test_model",
    )

    # Run agent loop
    prompt = [{"role": "user", "content": "Q?"}]
    extras = {}
    out = await generator.agent_loop(prompt, mock_env_cfg.env_class, extras, max_tokens=50, max_input_length=512)

    # Response ids layout: step1 (3 tokens) + obs (1) + step2 (3) + final eos (1) = 8
    assert len(out.response_ids) == 8
    # Indices: 2 (end of step1 assistant), 6 (end of step2 assistant), 7 (manually appended eos token)
    # Note that the last reward is placed at the 7 instead of at 6 since we manually move
    # it using the flag `appended_eos_token` in skyrl_gym_generator.py
    expected_rewards = [0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 1.7]
    assert isinstance(out.reward, list)
    assert out.reward == expected_rewards
    assert out.stop_reason == "stop"


@pytest.mark.asyncio
@patch("skyrl_gym.make")
async def test_agent_loop_token_level_rewards_multi_turn_conversation_format(
    mock_make, mock_tokenizer, mock_llm, mock_env_cfg
):
    """use_conversation_multi_turn=True; verify rewards placed at ends of assistant segments before observations."""
    mock_tokenizer.eos_token_id = 4

    # Tokenizer: initial prompt -> 2 tokens; observation template -> 2 tokens each call
    def apply_chat_template_side_effect(messages, **kwargs):
        if kwargs.get("tokenize", True):
            # For observations path, generator passes [*base_conversation, *new_obs] with add_generation_prompt=True
            return [201, 202]
        else:
            return "".join([m.get("content", "") for m in messages])

    mock_tokenizer.apply_chat_template.side_effect = apply_chat_template_side_effect

    # LLM outputs include EOS and are kept in multi-turn path
    async def llm_generate_side_effect(input_batch):
        num = (
            len(input_batch["prompt_token_ids"]) if "prompt_token_ids" in input_batch else len(input_batch["prompts"])
        )  # noqa: E501
        return {
            "responses": ["aaa"] * num,
            "stop_reasons": ["stop"] * num,
            "response_logprobs": None,
            "response_ids": [[10, 11, 12, mock_tokenizer.eos_token_id] for _ in range(num)],
        }

    mock_llm.generate = AsyncMock(side_effect=llm_generate_side_effect)

    # Env: two steps with rewards 0.5 then 0.25; first step has an observation, second has none
    class MTEnv:
        def __init__(self):
            self.turns = 0

        def init(self, prompt):
            return prompt, {}

        def step(self, action):
            self.turns += 1
            if self.turns == 1:
                return BaseTextEnvStepOutput(
                    observations=[{"role": "user", "content": "obs1"}], reward=0.5, done=False, metadata={}
                )
            else:
                return BaseTextEnvStepOutput(observations=[], reward=0.25, done=True, metadata={})

        def close(self):
            return None

    mock_make.return_value = MTEnv()

    # Generator config
    cfg = MagicMock()
    cfg.sampling_params.max_generate_length = 50
    cfg.sampling_params.logprobs = None
    cfg.apply_overlong_filtering = False
    cfg.max_input_length = 512
    cfg.batched = False
    cfg.max_turns = 10
    cfg.zero_reward_on_non_stop = False
    cfg.use_conversation_multi_turn = True

    mock_env_cfg.env_class = "mt_env"

    generator = SkyRLGymGenerator(
        generator_cfg=cfg,
        skyrl_gym_cfg=mock_env_cfg,
        inference_engine_client=mock_llm,
        tokenizer=mock_tokenizer,
        model_name="test_model",
    )
    # Ensure base_conversation_token_ids doesn't shift observation slicing in test
    generator.base_conversation_token_ids = []

    prompt = [{"role": "user", "content": "Q?"}]
    extras = {}
    out = await generator.agent_loop(prompt, mock_env_cfg.env_class, extras, max_tokens=50, max_input_length=512)

    # Response ids layout: step1 assistant (4 incl. eos) + obs(2) + step2 assistant(4 incl. eos) = 10
    assert len(out.response_ids) == 10
    # Rewards at indices: 3 (end of step1 assistant), 9 (end of step2 assistant)
    expected = [0.0] * 10
    expected[3] = 0.5
    expected[9] = 0.25
    assert isinstance(out.reward, list)
    assert out.reward == expected
    assert out.stop_reason == "stop"


@pytest.mark.asyncio
@patch("skyrl_gym.make")
async def test_agent_loop_retokenize_returns_float_reward(mock_make, mock_tokenizer, mock_llm, mock_env_cfg):
    """Retokenize mode should return a single float reward (last non-None step reward) because token-level rewards are not yet supported."""
    mock_tokenizer.eos_token_id = 4

    # Tokenizer: initial prompt ids and final retokenized response with masks
    def apply_chat_template_side_effect(messages, **kwargs):
        if kwargs.get("return_dict", False):
            # Final retokenization output
            return {"assistant_masks": [1, 0, 1], "input_ids": [5, 6, 7]}
        if kwargs.get("tokenize", True):
            return [301, 302]
        else:
            return "".join([m.get("content", "") for m in messages])

    mock_tokenizer.apply_chat_template.side_effect = apply_chat_template_side_effect

    # LLM generate in retokenize mode uses prompts; we can return any ids
    async def llm_generate_side_effect(input_batch):
        num = (
            len(input_batch["prompts"]) if "prompts" in input_batch else len(input_batch["prompt_token_ids"])
        )  # noqa: E501
        return {
            "responses": ["bbb"] * num,
            "stop_reasons": ["stop"] * num,
            "response_logprobs": None,
            "response_ids": [[20, 21, 22] for _ in range(num)],
        }

    mock_llm.generate = AsyncMock(side_effect=llm_generate_side_effect)

    # Env with rewards: None then 2.5
    class RetokEnv:
        def __init__(self):
            self.turns = 0

        def init(self, prompt):
            return prompt, {}

        def step(self, action):
            self.turns += 1
            if self.turns == 1:
                return BaseTextEnvStepOutput(
                    observations=[{"role": "user", "content": "o1"}], reward=None, done=False, metadata={}
                )  # noqa: E501
            else:
                return BaseTextEnvStepOutput(observations=[], reward=2.5, done=True, metadata={})

        def close(self):
            return None

    mock_make.return_value = RetokEnv()

    # Generator config enabling retokenize path
    cfg = MagicMock()
    cfg.sampling_params.max_generate_length = 50
    cfg.sampling_params.logprobs = None
    cfg.apply_overlong_filtering = False
    cfg.max_input_length = 512
    cfg.batched = False
    cfg.max_turns = 10
    cfg.zero_reward_on_non_stop = False
    cfg.use_conversation_multi_turn = True

    generator = SkyRLGymGenerator(
        generator_cfg=cfg,
        skyrl_gym_cfg=mock_env_cfg,
        inference_engine_client=mock_llm,
        tokenizer=mock_tokenizer,
        model_name="Qwen/Qwen3-0.6B",  # ensures custom_chat_template is truthy in get_custom_chat_template
    )
    # Force retokenize path regardless of model resolution logic if needed
    generator.custom_chat_template = "<custom>"

    prompt = [{"role": "user", "content": "Q?"}]
    extras = {}
    out = await generator.agent_loop(prompt, mock_env_cfg.env_class, extras, max_tokens=50, max_input_length=512)

    assert isinstance(out.reward, float)
    assert out.reward == 2.5
    assert out.stop_reason == "stop"


@pytest.mark.asyncio
@patch("skyrl_gym.make")
async def test_agent_loop_truncation_drops_out_of_range_rewards(mock_make, mock_tokenizer, mock_llm, mock_env_cfg):
    """Non-retokenize path: ensure rewards whose indices fall beyond truncated response are ignored."""

    # Configure tokenizer: initial prompt -> 2 tokens
    def apply_chat_template_side_effect(messages, **kwargs):
        if kwargs.get("tokenize", True):
            return [101, 102]
        else:
            return "".join([m.get("content", "") for m in messages])

    mock_tokenizer.apply_chat_template.side_effect = apply_chat_template_side_effect
    mock_tokenizer.eos_token_id = 4

    # LLM returns 4 assistant tokens per turn (no eos here; final EOS appended by generator for non-conv-mt)
    async def llm_generate_side_effect(input_batch):
        num = len(input_batch["prompt_token_ids"]) if "prompt_token_ids" in input_batch else len(input_batch["prompts"])
        return {
            "responses": ["step"] * num,
            "stop_reasons": ["stop"] * num,
            "response_logprobs": None,
            "response_ids": [[10, 11, 12, 13] for _ in range(num)],
        }

    mock_llm.generate = AsyncMock(side_effect=llm_generate_side_effect)

    # Env with two steps, rewards on both; no observations to keep math simple
    class TruncEnv:
        def __init__(self):
            self.turns = 0

        def init(self, prompt):
            return prompt, {}

        def step(self, action):
            self.turns += 1
            if self.turns == 1:
                return BaseTextEnvStepOutput(observations=[], reward=1.0, done=False, metadata={})
            else:
                return BaseTextEnvStepOutput(observations=[], reward=2.0, done=True, metadata={})

        def close(self):
            return None

    mock_make.return_value = TruncEnv()

    # Generator config: non-retokenize message mode; max_turns=1 so max_response_tokens = max_tokens
    cfg = MagicMock()
    cfg.sampling_params.max_generate_length = 5  # enforce truncation
    cfg.sampling_params.logprobs = None
    cfg.apply_overlong_filtering = False
    cfg.max_input_length = 1000  # prevent earlier length break
    cfg.batched = False
    cfg.max_turns = 1
    cfg.zero_reward_on_non_stop = False
    cfg.use_conversation_multi_turn = False

    generator = SkyRLGymGenerator(
        generator_cfg=cfg,
        skyrl_gym_cfg=mock_env_cfg,
        inference_engine_client=mock_llm,
        tokenizer=mock_tokenizer,
        model_name="test_model",
    )

    prompt = [{"role": "user", "content": "Q?"}]
    extras = {}
    out = await generator.agent_loop(prompt, mock_env_cfg.env_class, extras, max_tokens=5, max_input_length=1000)

    # Untruncated response would be: 4 (step1) + 4 (step2) + 1 (final eos) = 9; we expect truncation to 5
    assert len(out.response_ids) == 5
    assert isinstance(out.reward, list)
    assert len(out.reward) == 5
    # Step1 end index relative should be 3 (0-based), step2 end index would be 7 -> out of range after truncation
    assert out.reward[3] == 1.0
    assert sum(out.reward) == 1.0
    assert out.stop_reason == "length"
