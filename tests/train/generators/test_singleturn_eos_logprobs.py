"""Single-assistant-message EOS bookkeeping at the rollout/training boundary."""

import math
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
import torch

from skyrl.backends.skyrl_train.utils.off_policy_correction_utils import (
    compute_outlier_token_mask,
    compute_tis_ratio,
)
from skyrl.train.config import (
    ChatTemplateConfig,
    GeneratorConfig,
    OffPolicyCorrectionConfig,
    SkyRLGymConfig,
)
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator


class Tokenizer:
    eos_token_id = 4
    eos_token = "<eos>"

    def apply_chat_template(self, messages, **kwargs):
        return [101, 102]

    def encode(self, text, **kwargs):
        return [77] if text else []


async def run_agent_loop(monkeypatch, turns, *, get_logprobs=True, max_input_length=256):
    """Run the real generator, replacing only its tokenizer, environment and LLM."""
    cfg = GeneratorConfig()
    cfg.batched = False
    cfg.use_conversation_multi_turn = False
    cfg.sampling_params.logprobs = 0 if get_logprobs else None
    cfg.chat_template = ChatTemplateConfig(source="name", name_or_path=None)
    cfg.inference_engine.enable_return_routed_experts = any("routes" in turn for turn in turns)
    env_cfg = SkyRLGymConfig()
    env_cfg.max_env_workers = 0
    prompts = []
    turn_index = 0

    class Env:
        def init(self, prompt):
            return prompt, {}

        def step(self, text):
            nonlocal turn_index
            turn = turns[turn_index]
            turn_index += 1
            return {
                "observations": turn.get("observations", []),
                "reward": turn.get("reward", 1.0),
                "done": turn_index == len(turns),
            }

        def get_metrics(self):
            return {}

        def close(self):
            pass

    async def generate(input_batch, model=None):
        prompts.append(input_batch["prompt_token_ids"][0].copy())
        turn = turns[turn_index]
        return {
            "responses": ["answer" if turn["tokens"] else ""],
            "response_ids": [turn["tokens"].copy()],
            "response_logprobs": [turn["logprobs"].copy()] if get_logprobs else None,
            "stop_reasons": [turn.get("stop_reason", "stop")],
            "rollout_expert_indices": [turn["routes"]] if "routes" in turn else None,
        }

    client = SimpleNamespace(generate=generate, finish_session=AsyncMock())
    monkeypatch.setattr("skyrl_gym.make", lambda *args, **kwargs: Env())
    generator = SkyRLGymGenerator(cfg, env_cfg, client, Tokenizer())
    output = await generator.agent_loop(
        [{"role": "user", "content": "Q"}],
        "gsm8k",
        {},
        max_tokens=32,
        max_input_length=max_input_length,
        sampling_params={"max_tokens": 32, "logprobs": cfg.sampling_params.logprobs},
    )
    client.finish_session.assert_awaited_once()
    return output, prompts


@pytest.mark.asyncio
async def test_sampled_final_eos_preserves_on_policy_importance_ratios(monkeypatch):
    sampled_logprobs = [-0.2, math.log(0.01)]
    output, _ = await run_agent_loop(monkeypatch, [{"tokens": [10, 4], "logprobs": sampled_logprobs}])

    assert output.response_ids == [10, 4]
    assert output.loss_mask == [1, 1]
    assert output.reward == [0.0, 1.0]

    # Unchanged trainer/inference policies must have ratio 1, even if EOS is unlikely.
    old_logprobs = torch.tensor([sampled_logprobs])
    rollout_logprobs = torch.tensor([output.rollout_logprobs])
    loss_mask = torch.tensor([output.loss_mask])
    correction = OffPolicyCorrectionConfig(outlier_token_is_threshold_low=0.1)
    ratio, _ = compute_tis_ratio(old_logprobs, rollout_logprobs, loss_mask, "sequence", correction)
    accepted, _ = compute_outlier_token_mask(old_logprobs, rollout_logprobs, loss_mask, correction)
    torch.testing.assert_close(ratio, torch.ones_like(ratio))
    torch.testing.assert_close(accepted, torch.ones_like(accepted))
    assert output.rollout_logprobs == sampled_logprobs


@pytest.mark.asyncio
async def test_synthetic_eos_is_not_an_action_or_reward_target(monkeypatch):
    output, _ = await run_agent_loop(monkeypatch, [{"tokens": [10], "logprobs": [-0.2]}])

    assert output.response_ids == [10, 4]
    assert output.loss_mask == [1, 0]
    assert output.rollout_logprobs == [-0.2, 0.0]
    assert output.reward == [1.0, 0.0]

    correction = OffPolicyCorrectionConfig(outlier_token_is_threshold_low=0.1)
    old_logprobs = torch.tensor([[-0.2, -10.0]])
    rollout_logprobs = torch.tensor([output.rollout_logprobs])
    loss_mask = torch.tensor([output.loss_mask])
    ratio, _ = compute_tis_ratio(old_logprobs, rollout_logprobs, loss_mask, "sequence", correction)
    accepted, _ = compute_outlier_token_mask(old_logprobs, rollout_logprobs, loss_mask, correction)
    torch.testing.assert_close(ratio, torch.ones_like(ratio))
    torch.testing.assert_close(accepted, torch.ones_like(accepted))


@pytest.mark.asyncio
async def test_only_intermediate_eos_is_removed_and_routes_stay_aligned(monkeypatch):
    first_routes = np.arange(3, dtype=np.uint8).reshape(3, 1, 1)
    final_routes = np.arange(5, dtype=np.uint8).reshape(5, 1, 1)
    output, prompts = await run_agent_loop(
        monkeypatch,
        [
            {
                "tokens": [10, 4],
                "logprobs": [-0.2, -3.0],
                "reward": 0.3,
                "observations": [{"role": "user", "content": "observation"}],
                "routes": first_routes,
            },
            {"tokens": [11, 4], "logprobs": [-0.4, -5.0], "reward": 1.7, "routes": final_routes},
        ],
    )

    assert prompts == [[101, 102], [101, 102, 10, 77]]
    assert output.response_ids == [10, 77, 11, 4]
    assert output.loss_mask == [1, 0, 1, 1]
    assert output.rollout_logprobs == [-0.2, 0.0, -0.4, -5.0]
    assert output.reward == [0.3, 0.0, 0.0, 1.7]
    # The sampled final token was not itself evaluated: keep the engine's captured
    # prefix+response-minus-one routes, without fabricating an EOS routing row.
    np.testing.assert_array_equal(output.rollout_expert_indices, final_routes)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tokens,logprobs,stop_reason,expected_tokens,expected_rewards",
    [
        ([4], [-3.0], "stop", [4], [1.0]),
        ([], [], "stop", [], []),
        ([], [], "length", [], []),
        ([10], [-0.2], "length", [10], [1.0]),
    ],
)
async def test_empty_eos_only_and_truncated_outputs(
    monkeypatch, tokens, logprobs, stop_reason, expected_tokens, expected_rewards
):
    output, _ = await run_agent_loop(
        monkeypatch, [{"tokens": tokens, "logprobs": logprobs, "stop_reason": stop_reason}]
    )
    assert output.response_ids == expected_tokens
    assert output.loss_mask == [1] * len(expected_tokens)
    assert output.rollout_logprobs == logprobs
    assert output.reward == expected_rewards
    assert output.stop_reason == stop_reason


@pytest.mark.asyncio
async def test_context_limit_does_not_restore_intermediate_eos(monkeypatch):
    output, prompts = await run_agent_loop(
        monkeypatch,
        [
            {
                "tokens": [10, 4],
                "logprobs": [-0.2, -3.0],
                "observations": [{"role": "user", "content": "observation"}],
            },
            {"tokens": [11, 4], "logprobs": [-0.4, -5.0]},
        ],
        max_input_length=3,
    )
    assert prompts == [[101, 102]]
    assert output.response_ids == [10]
    assert output.loss_mask == [1]
    assert output.rollout_logprobs == [-0.2]
    assert output.reward == [1.0]
    assert output.stop_reason == "length"


@pytest.mark.asyncio
async def test_sampled_eos_without_logprob_capture(monkeypatch):
    output, _ = await run_agent_loop(monkeypatch, [{"tokens": [10, 4]}], get_logprobs=False)
    assert output.response_ids == [10, 4]
    assert output.loss_mask == [1, 1]
    assert output.rollout_logprobs is None
    assert output.reward == [0.0, 1.0]
