"""
CPU tests for the VLM generator's obs-token extraction

uv run --extra dev --isolated pytest tests/train/generators/test_skyrl_vlm_generator.py -v
"""

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from transformers import AutoTokenizer

from skyrl.train.config import (
    ChatTemplateConfig,
    GeneratorConfig,
    SamplingParams,
    SkyRLGymConfig,
)
from skyrl.train.generators.base import GeneratorInput, GeneratorOutput
from skyrl.train.generators.skyrl_vlm_generator import SkyRLVLMGymGenerator
from skyrl_gym.envs import deregister, register
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.registration import registry

MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"


# ---------------------------------------------------------------------------
# Test environment
# ---------------------------------------------------------------------------


class CPUVLMTestEnv(BaseTextEnv):
    """3-turn text env for testing the VLM generator on CPU."""

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
            reward=1.0 if done else 0.0,
            done=done,
            metadata={},
        )


if "cpu_vlm_test_env" not in registry:
    register(
        id="cpu_vlm_test_env",
        entry_point="tests.train.generators.test_skyrl_vlm_generator:CPUVLMTestEnv",
    )


@pytest.fixture(autouse=True, scope="module")
def deregister_test_env():
    yield
    deregister("cpu_vlm_test_env")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_vlm_generator(tokenizer):
    """Build a SkyRLVLMGymGenerator with mock inference client."""
    generator_cfg = GeneratorConfig(
        sampling_params=SamplingParams(max_generate_length=200, logprobs=None),
        max_input_length=4096,
        batched=False,
        max_turns=3,
        zero_reward_on_non_stop=False,
        apply_overlong_filtering=False,
        use_conversation_multi_turn=True,
        chat_template=ChatTemplateConfig(source="name", name_or_path=None),
        step_wise_trajectories=False,
    )
    env_cfg = SkyRLGymConfig(max_env_workers=0)
    mock_client = MagicMock()
    mock_client.model_name = MODEL_NAME
    generator = SkyRLVLMGymGenerator(
        generator_cfg=generator_cfg,
        skyrl_gym_cfg=env_cfg,
        inference_engine_client=mock_client,
        tokenizer=tokenizer,
    )
    return generator


def _make_mock_renderer(tokenizer):
    """Create an AsyncMock for render_chat_completion."""

    async def mock_render(request_payload):
        messages = request_payload["json"]["messages"]
        token_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )
        return {"token_ids": token_ids, "features": None}

    return AsyncMock(side_effect=mock_render)


def _make_mock_llm(tokenizer, response_text: str):
    """Create an AsyncMock for the inference engine's generate method."""

    async def mock_generate(input_batch):
        num_prompts = len(input_batch["prompt_token_ids"])
        text_with_eos = response_text + tokenizer.eos_token
        ids = tokenizer.encode(text_with_eos, add_special_tokens=False)
        return {
            "responses": [response_text] * num_prompts,
            "stop_reasons": ["stop"] * num_prompts,
            "response_logprobs": None,
            "response_ids": [ids] * num_prompts,
        }

    return AsyncMock(side_effect=mock_generate)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("skyrl.train.generators.skyrl_vlm_generator.decode_mm_kwargs")
async def test_vlm_obs_offset(mock_decode):
    """Validate that obs token boundaries are correctly identified."""
    mock_decode.return_value = {"pixel_values": None, "image_grid_thw": None}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    response_text = "b"

    generator = _build_vlm_generator(tokenizer)
    generator.inference_engine_client.render_chat_completion = _make_mock_renderer(tokenizer)
    generator.inference_engine_client.generate = _make_mock_llm(tokenizer, response_text)

    prompt = [[{"role": "user", "content": "a"}]]
    input_batch: GeneratorInput = {
        "prompts": prompt,
        "env_extras": [{"answer": "4"}],
        "env_classes": ["cpu_vlm_test_env"],
    }
    output: GeneratorOutput = await generator.generate(input_batch)

    response_ids = output["response_ids"][0]
    loss_mask = output["loss_masks"][0]
    rewards = output["rewards"][0]

    # ── Basic structural checks ──────────────────────────────────────
    assert len(response_ids) == len(loss_mask), "response_ids and loss_mask must be same length"
    assert len(rewards) == len(response_ids), "rewards and response_ids must be same length"

    # ── Compute expected token counts ────────────────────────────────
    gen_ids_per_turn = tokenizer.encode(response_text + tokenizer.eos_token, add_special_tokens=False)
    num_gen_tokens_per_turn = len(gen_ids_per_turn)

    # 3 turns: turn 1 and 2 produce observations, turn 3 is final (no obs).
    total_gen_tokens = num_gen_tokens_per_turn * 3

    # Count 1s and 0s in loss_mask
    num_ones = sum(loss_mask)
    num_zeros = len(loss_mask) - num_ones

    assert num_ones == total_gen_tokens, f"Expected {total_gen_tokens} generated tokens (loss_mask=1), got {num_ones}"
    # There are 2 observation segments (after turns 1 and 2), each containing
    # the obs user message + generation prompt tokens, masked with 0.
    assert num_zeros > 0, "Expected some observation tokens (loss_mask=0)"

    # ── Verify loss mask pattern: gen(1s) then obs(0s), alternating ──
    # [gen_turn1(1s)] [obs_turn1(0s)] [gen_turn2(1s)] [obs_turn2(0s)] [gen_turn3(1s)]
    pos = 0
    for turn in range(3):
        # gen segment
        gen_segment = loss_mask[pos : pos + num_gen_tokens_per_turn]
        assert all(m == 1 for m in gen_segment), f"Turn {turn+1}: expected all 1s in gen segment"
        pos += num_gen_tokens_per_turn

        if turn < 2:
            # obs segment: find next stretch of 0s
            obs_start = pos
            while pos < len(loss_mask) and loss_mask[pos] == 0:
                pos += 1
            assert pos > obs_start, f"Turn {turn+1}: expected obs tokens (0s) after gen tokens"

    assert pos == len(loss_mask), f"Expected to consume all tokens, but {len(loss_mask) - pos} remain"

    # ── Verify reward placement ──────────────────────────────────────
    # Turns 1,2 have reward 0.0, turn 3 has reward 1.0.
    nonzero_rewards = [(i, r) for i, r in enumerate(rewards) if r != 0.0]
    assert len(nonzero_rewards) == 1, f"Expected exactly 1 nonzero reward, got {nonzero_rewards}"
    _, reward_val = nonzero_rewards[0]
    assert reward_val == 1.0


@pytest.mark.asyncio
@patch("skyrl.train.generators.skyrl_vlm_generator.decode_mm_kwargs")
async def test_vlm_obs_tokens_match_expected(mock_decode):
    """Verify the exact obs tokens extracted match what the chat template
    produces for the observation messages."""
    mock_decode.return_value = {"pixel_values": None, "image_grid_thw": None}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    response_text = "b"

    generator = _build_vlm_generator(tokenizer)
    generator.inference_engine_client.render_chat_completion = _make_mock_renderer(tokenizer)
    generator.inference_engine_client.generate = _make_mock_llm(tokenizer, response_text)

    prompt = [[{"role": "user", "content": "a"}]]
    input_batch: GeneratorInput = {
        "prompts": prompt,
        "env_extras": [{"answer": "4"}],
        "env_classes": ["cpu_vlm_test_env"],
    }
    output: GeneratorOutput = await generator.generate(input_batch)

    response_ids = output["response_ids"][0]
    loss_mask = output["loss_masks"][0]

    # Extract the obs token segments (contiguous 0s in loss_mask)
    obs_segments: list[list[int]] = []
    i = 0
    while i < len(loss_mask):
        if loss_mask[i] == 0:
            seg_start = i
            while i < len(loss_mask) and loss_mask[i] == 0:
                i += 1
            obs_segments.append(response_ids[seg_start:i])
        else:
            i += 1

    assert len(obs_segments) == 2, f"Expected 2 obs segments, got {len(obs_segments)}"

    for seg_idx, obs_text in enumerate(["1", "2"]):
        decoded = tokenizer.decode(obs_segments[seg_idx], skip_special_tokens=True)
        assert (
            obs_text in decoded
        ), f"Obs segment {seg_idx}: expected '{obs_text}' in decoded obs tokens, got '{decoded}'"
