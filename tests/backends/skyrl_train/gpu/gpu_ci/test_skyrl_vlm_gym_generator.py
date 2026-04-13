"""
GPU E2E test for SkyRLVLMGymGenerator.

Requires a local vLLM install with multi-modal /inference/v1/generate support.

SKYRL_LOCAL_VLLM=1 uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/test_skyrl_vlm_gym_generator.py -m vllm -v
"""

import base64
import io
import os
from typing import Any, Dict

import pytest
import torch
from loguru import logger
from PIL import Image
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.inference_engines.utils import (
    get_sampling_params_for_backend,
)
from skyrl.train.config import SamplingParams, SkyRLTrainConfig
from skyrl.train.generators.base import GeneratorInput, GeneratorOutput, TrajectoryID
from skyrl.train.generators.skyrl_vlm_generator import SkyRLVLMGymGenerator
from skyrl_gym.envs import deregister, register
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from tests.backends.skyrl_train.gpu.utils import InferenceEngineState

requires_local_vllm = pytest.mark.skipif(
    os.environ.get("SKYRL_LOCAL_VLLM") != "1",
    reason="Requires local vLLM with multi-modal /v1/chat/completions/render support",
)

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
TP_SIZE = 1
QUESTION_TEXT = "What color is this square? Answer with just the color name."


# ---------------------------------------------------------------------------
# Helper: create a solid-color image as a data URI
# ---------------------------------------------------------------------------


def _make_color_image_data_uri(color: tuple[int, int, int], size: int = 64) -> str:
    """Create a solid-color JPEG image and return it as a data URI."""
    img = Image.new("RGB", (size, size), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _make_image_message(color: tuple[int, int, int], text: str) -> dict:
    """Build an OpenAI-format user message with an image and text."""
    return {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": _make_color_image_data_uri(color)}},
            {"type": "text", "text": text},
        ],
    }


# ---------------------------------------------------------------------------
# Test environment: 2-turn color classification
# ---------------------------------------------------------------------------


class ColorSquareEnv(BaseTextEnv):
    """2-turn env that shows colored squares and asks the model to name the color.

    Turn 1 (init): red square image + question
    Turn 2 (step): blue square image + question
    Reward: 1.0 if both answers are correct, else 0.0
    """

    RED = (255, 0, 0)
    BLUE = (0, 0, 255)

    def __init__(self, env_config: Any, extras: Dict[str, Any] = {}):
        super().__init__()
        self.max_turns = 2
        self.answers: list[str] = []

    def init(self, prompt):
        prompt.append(_make_image_message(self.RED, QUESTION_TEXT))
        return prompt, {}

    def step(self, action: str):
        self.turns += 1
        self.answers.append(action)
        done = self.turns >= self.max_turns

        if not done:
            observations = [_make_image_message(self.BLUE, QUESTION_TEXT)]
            reward = 0.0
        else:
            red_correct = "red" in self.answers[0].lower()
            blue_correct = "blue" in self.answers[1].lower()
            reward = 1.0 if (red_correct and blue_correct) else 0.0
            observations = []

        return BaseTextEnvStepOutput(
            observations=observations,
            reward=reward,
            done=done,
            metadata={},
        )

    def get_metrics(self) -> Dict[str, Any]:
        return {"answers": self.answers}


register(
    id="color_square_test_env",
    entry_point="tests.backends.skyrl_train.gpu.gpu_ci.test_skyrl_vlm_gym_generator:ColorSquareEnv",
)


@pytest.fixture(autouse=True, scope="module")
def deregister_color_square_env():
    yield
    deregister("color_square_test_env")


# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------


def get_vlm_test_config(model: str) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = model
    cfg.trainer.critic.model.path = ""
    cfg.trainer.placement.colocate_all = True
    cfg.trainer.placement.policy_num_gpus_per_node = TP_SIZE
    cfg.generator.sampling_params = SamplingParams(
        max_generate_length=256,
        temperature=0.0,
        logprobs=1,
    )
    cfg.generator.max_input_length = 4096
    cfg.generator.batched = False
    cfg.generator.max_turns = 2
    cfg.generator.use_conversation_multi_turn = True
    cfg.generator.step_wise_trajectories = False
    cfg.generator.apply_overlong_filtering = False
    cfg.generator.inference_engine.backend = "vllm"
    cfg.generator.async_engine = True
    cfg.generator.inference_engine.num_engines = 1
    cfg.generator.inference_engine.tensor_parallel_size = TP_SIZE
    cfg.generator.run_engines_locally = True
    cfg.generator.inference_engine.served_model_name = model
    cfg.environment.skyrl_gym.max_env_workers = 0
    return cfg


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@requires_local_vllm
@pytest.mark.vllm
@pytest.mark.asyncio
async def test_vlm_generator_color_classification(ray_init_fixture):
    """End-to-end VLM generator test with a 2-turn color classification env."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    cfg = get_vlm_test_config(MODEL)

    async with InferenceEngineState.create(
        cfg=cfg,
        use_local=True,
        backend="vllm",
        model=MODEL,
        sleep_level=1,
        engine_init_kwargs={
            "max_model_len": 4096,
            "limit_mm_per_prompt": {"image": 2, "video": 0},
            "mm_processor_cache_gb": 0,
        },
        use_new_inference_servers=True,
    ) as engines:
        inference_client = engines.client
        env_cfg = cfg.environment.skyrl_gym
        generator_cfg = cfg.generator

        generator = SkyRLVLMGymGenerator(
            generator_cfg=generator_cfg,
            skyrl_gym_cfg=env_cfg,
            inference_engine_client=inference_client,
            tokenizer=tokenizer,
        )

        num_prompts = 2
        prompts = [
            [{"role": "system", "content": "You are a helpful assistant. Answer concisely."}]
            for _ in range(num_prompts)
        ]

        input_batch: GeneratorInput = {
            "prompts": prompts,
            "env_classes": ["color_square_test_env"] * num_prompts,
            "env_extras": [{} for _ in range(num_prompts)],
            "sampling_params": get_sampling_params_for_backend(
                "vllm",
                SamplingParams(
                    temperature=0.0,
                    max_generate_length=256,
                    logprobs=1,
                ),
            ),
            "trajectory_ids": [TrajectoryID(instance_id=str(i), repetition_id=0) for i in range(num_prompts)],
        }

        generator_output: GeneratorOutput = await generator.generate(input_batch)

        # ── Structural assertions ──────────────────────────────────────
        required_keys = {
            "prompt_token_ids",
            "response_ids",
            "rewards",
            "loss_masks",
            "stop_reasons",
            "rollout_metrics",
            "rollout_logprobs",
        }
        for key in required_keys:
            assert key in generator_output, f"Missing key: {key}"

        assert len(generator_output["response_ids"]) == num_prompts
        assert len(generator_output["prompt_token_ids"]) == num_prompts

        # ── Multimodal tensor assertions ───────────────────────────────
        pixel_values = generator_output.get("pixel_values")
        image_grid_thw = generator_output.get("image_grid_thw")
        assert pixel_values is not None, "Expected pixel_values in output"
        assert image_grid_thw is not None, "Expected image_grid_thw in output"

        for i in range(num_prompts):
            pv = pixel_values[i]
            thw = image_grid_thw[i]
            assert isinstance(pv, torch.Tensor), f"pixel_values[{i}] should be a tensor"
            assert pv.numel() > 0, f"pixel_values[{i}] should be non-empty, shape {pv.shape}"
            assert isinstance(thw, torch.Tensor), f"image_grid_thw[{i}] should be a tensor"
            assert thw.shape[-1] == 3, f"image_grid_thw[{i}] last dim should be 3"
            assert thw.dtype == torch.long, f"image_grid_thw[{i}] should be long dtype"

        # ── Per-trajectory assertions ──────────────────────────────────
        for i in range(num_prompts):
            resp_ids = generator_output["response_ids"][i]
            loss_mask = generator_output["loss_masks"][i]
            prompt_ids = generator_output["prompt_token_ids"][i]
            rewards = generator_output["rewards"][i]

            # Length consistency
            assert len(resp_ids) == len(loss_mask), (
                f"Trajectory {i}: response_ids length ({len(resp_ids)}) != " f"loss_mask length ({len(loss_mask)})"
            )

            # Prompt is non-empty list of ints
            assert len(prompt_ids) > 0
            assert all(isinstance(t, int) for t in prompt_ids)

            # Loss mask has both 0s (obs tokens) and 1s (gen tokens)
            assert 0 in loss_mask, f"Trajectory {i}: loss_mask should have masked-out obs tokens"
            assert 1 in loss_mask, f"Trajectory {i}: loss_mask should have masked-in gen tokens"

            # Decode only the generated (loss_mask=1) tokens
            gen_ids = [resp_ids[j] for j in range(len(resp_ids)) if loss_mask[j] == 1]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).lower()
            logger.info(f"Trajectory {i} generated text: {gen_text}")

            assert "red" in gen_text, f"Trajectory {i}: expected 'red' in generated text, got: {gen_text}"
            assert "blue" in gen_text, f"Trajectory {i}: expected 'blue' in generated text, got: {gen_text}"

            # Rewards are per-token list
            assert isinstance(rewards, list), f"Trajectory {i}: rewards should be a list"
            assert len(rewards) == len(resp_ids), f"Trajectory {i}: rewards length mismatch"

        # ── Logprobs assertions ────────────────────────────────────────
        assert generator_output["rollout_logprobs"] is not None
        for i in range(num_prompts):
            logprobs = generator_output["rollout_logprobs"][i]
            assert len(logprobs) == len(generator_output["response_ids"][i])
