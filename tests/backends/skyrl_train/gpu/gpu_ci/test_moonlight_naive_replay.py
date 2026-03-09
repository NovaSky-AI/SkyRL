"""
Diagnostic Moonlight replay test with minimal config overrides.

Purpose:
- Validate whether large vLLM vs Megatron logprob gaps are caused by custom
  test-time config tweaks.
- Run the same high-level flow as router replay:
  vLLM rollout -> build training batch -> Megatron forward (replay on/off).
"""

import asyncio
import math
import os

import pytest
import ray
import torch
from transformers import AutoTokenizer

from tests.backends.skyrl_train.gpu.utils import (
    InferenceEngineState,
    Timer,
    get_test_generator_input,
    init_worker_with_type,
)
from skyrl.backends.skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch
from skyrl.backends.skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.train.config import SamplingParams, SkyRLTrainConfig
from skyrl.train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl.train.generators.base import GeneratorInput
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator
from skyrl.train.utils.utils import validate_cfg


MOONLIGHT_MODEL = os.environ.get("SKYRL_REPLAY_MODEL", "moonshotai/Moonlight-16B-A3B")
NUM_PROMPTS = 2
N_SAMPLES_PER_PROMPT = 2


def get_naive_moonlight_cfg(model_name: str = MOONLIGHT_MODEL) -> SkyRLTrainConfig:
    """Naive config: minimal overrides, no Moonlight-special transformer kwargs."""
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = model_name
    cfg.trainer.logger = "console"
    cfg.trainer.strategy = "megatron"
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.use_sample_packing = False

    cfg.generator.inference_engine.enable_return_routed_experts = True
    cfg.generator.inference_engine.tensor_parallel_size = 8
    cfg.generator.batched = False
    cfg.generator.max_turns = 1
    cfg.generator.sampling_params = SamplingParams(
        max_generate_length=128,
        logprobs=1,
        temperature=1.0,
    )
    # Must match num_rollout_gpus when colocate_all=True (default)
    cfg.trainer.placement.policy_num_gpus_per_node = 8
    validate_cfg(cfg)
    return cfg


def _masked_mean_std(x: torch.Tensor, mask: torch.Tensor):
    vals = x[mask]
    if vals.numel() == 0:
        return float("nan"), float("nan")
    return vals.mean().item(), vals.std().item()


@pytest.mark.megatron
def test_moonlight_naive_replay_logprobs(ray_init_fixture):
    """
    Diagnostic-only test:
    - Uses naive config (minimal overrides)
    - Prints masked/unmasked replay-vs-no-replay stats
    """
    try:
        cfg = get_naive_moonlight_cfg()
        tokenizer = AutoTokenizer.from_pretrained(MOONLIGHT_MODEL, trust_remote_code=True)

        with InferenceEngineState.create(
            cfg=cfg,
            model=MOONLIGHT_MODEL,
            use_local=True,
            colocate_all=True,
            backend="vllm",
            sleep_level=1,
            gpu_memory_utilization=0.9,
        ) as engines:
            client, pg = engines.client, engines.pg
            asyncio.run(client.wake_up())

            generator = SkyRLGymGenerator(
                generator_cfg=cfg.generator,
                skyrl_gym_cfg=cfg.environment.skyrl_gym,
                inference_engine_client=client,
                tokenizer=tokenizer,
            )

            input_batch: GeneratorInput = get_test_generator_input(
                model=MOONLIGHT_MODEL,
                num_prompts=NUM_PROMPTS,
                n_samples_per_prompt=N_SAMPLES_PER_PROMPT,
                max_prompt_length=512,
                env_class="gsm8k",
            )
            input_batch["sampling_params"] = get_sampling_params_for_backend(
                "vllm",
                SamplingParams(
                    temperature=1.0,
                    top_p=1.0,
                    top_k=-1,
                    max_generate_length=128,
                    min_p=0.0,
                    logprobs=1,
                ),
            )

            with Timer("naive_generate_with_router_replay"):
                generator_output = asyncio.run(generator.generate(input_batch))

            indices = generator_output["rollout_inference_indices"]
            responses = generator_output["response_ids"]
            assert indices is not None
            assert len(indices) == len(responses)
            asyncio.run(client.sleep())

        rewards = generator_output["rewards"]
        if rewards and not isinstance(rewards[0], list):
            rewards = [[r] * len(resp) for r, resp in zip(rewards, responses)]

        (sequences, attention_mask, response_mask, rewards_t, loss_mask_t, logprobs_t, rii_tensor) = (
            convert_prompts_responses_to_batch_tensors(
                tokenizer=tokenizer,
                prompts=generator_output["prompt_token_ids"],
                responses=responses,
                rewards=rewards,
                loss_masks=generator_output["loss_masks"],
                logprobs=generator_output.get("rollout_logprobs"),
                rollout_inference_indices=indices,
            )
        )

        assert rii_tensor is not None
        num_actions = response_mask.shape[1]
        batch_size = sequences.shape[0]
        training_input = TrainingInputBatch(
            {
                "sequences": sequences,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "rewards": rewards_t,
                "loss_mask": loss_mask_t,
                "rollout_logprobs": (
                    logprobs_t
                    if logprobs_t is not None
                    else torch.zeros((batch_size, num_actions), dtype=torch.float32)
                ),
                "rollout_inference_indices": rii_tensor,
                "action_log_probs": torch.zeros((batch_size, num_actions), dtype=torch.float32),
                "base_action_log_probs": torch.zeros((batch_size, num_actions), dtype=torch.float32),
                "advantages": torch.zeros((batch_size, num_actions), dtype=torch.float32),
                "action_mask": response_mask.to(dtype=torch.int64),
            }
        )
        training_input.metadata = {"response_length": num_actions}

        # Keep model-parallel sizes identical to the current Moonlight replay setup.
        cfg.trainer.placement.policy_num_gpus_per_node = 8
        cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 4
        cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
        cfg.trainer.policy.megatron_config.context_parallel_size = 1
        cfg.trainer.policy.megatron_config.expert_model_parallel_size = 8
        cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = 1

        def run_megatron_forward(enable_replay: bool) -> torch.Tensor:
            if cfg.trainer.policy.megatron_config.transformer_config_kwargs is None:
                cfg.trainer.policy.megatron_config.transformer_config_kwargs = {}
            cfg.trainer.policy.megatron_config.transformer_config_kwargs["moe_enable_routing_replay"] = enable_replay

            actor_group = init_worker_with_type(
                "policy",
                shared_pg=pg,
                colocate_all=True,
                num_gpus_per_node=8,
                cfg=cfg,
            )
            refs = actor_group.async_run_ray_method("mesh", "forward", data=training_input)
            results = ray.get(refs)
            outputs = concatenate_outputs_after_mesh_dispatch(actor_group.actor_infos, results)["output"]
            for actor in actor_group._actor_handlers:
                ray.kill(actor)
            return outputs

        r3_logprobs = run_megatron_forward(enable_replay=True)
        no_r3_logprobs = run_megatron_forward(enable_replay=False)

        r3_diff = (logprobs_t - r3_logprobs).abs()
        no_r3_diff = (logprobs_t - no_r3_logprobs).abs()
        valid_mask = response_mask.bool()
        pad_mask = ~valid_mask

        # Unmasked summary.
        print("\n=== Naive Moonlight Logprob Summary (all tokens) ===")
        print(f"vLLM logprobs        mean/std: {logprobs_t.mean().item():.6f}/{logprobs_t.std().item():.6f}")
        print(f"Megatron replay      mean/std: {r3_logprobs.mean().item():.6f}/{r3_logprobs.std().item():.6f}")
        print(f"Megatron no-replay   mean/std: {no_r3_logprobs.mean().item():.6f}/{no_r3_logprobs.std().item():.6f}")
        print(f"Diff replay          mean/std: {r3_diff.mean().item():.6f}/{r3_diff.std().item():.6f}")
        print(f"Diff no-replay       mean/std: {no_r3_diff.mean().item():.6f}/{no_r3_diff.std().item():.6f}")

        # Valid-token-only summary (primary).
        vllm_valid_mean, vllm_valid_std = _masked_mean_std(logprobs_t, valid_mask)
        r3_valid_mean, r3_valid_std = _masked_mean_std(r3_logprobs, valid_mask)
        no_r3_valid_mean, no_r3_valid_std = _masked_mean_std(no_r3_logprobs, valid_mask)
        r3_diff_valid_mean, r3_diff_valid_std = _masked_mean_std(r3_diff, valid_mask)
        no_r3_diff_valid_mean, no_r3_diff_valid_std = _masked_mean_std(no_r3_diff, valid_mask)
        print("\n=== Naive Moonlight Logprob Summary (valid response tokens only) ===")
        print(f"valid token count: {valid_mask.sum().item()}")
        print(f"vLLM logprobs        mean/std: {vllm_valid_mean:.6f}/{vllm_valid_std:.6f}")
        print(f"Megatron replay      mean/std: {r3_valid_mean:.6f}/{r3_valid_std:.6f}")
        print(f"Megatron no-replay   mean/std: {no_r3_valid_mean:.6f}/{no_r3_valid_std:.6f}")
        print(f"Diff replay          mean/std: {r3_diff_valid_mean:.6f}/{r3_diff_valid_std:.6f}")
        print(f"Diff no-replay       mean/std: {no_r3_diff_valid_mean:.6f}/{no_r3_diff_valid_std:.6f}")

        if pad_mask.sum().item() > 0:
            vllm_pad_mean, vllm_pad_std = _masked_mean_std(logprobs_t, pad_mask)
            r3_pad_mean, r3_pad_std = _masked_mean_std(r3_logprobs, pad_mask)
            no_r3_pad_mean, no_r3_pad_std = _masked_mean_std(no_r3_logprobs, pad_mask)
            print("\n=== Naive Moonlight Padding-only Stats ===")
            print(f"padding token count: {pad_mask.sum().item()}")
            print(f"vLLM mean/std: {vllm_pad_mean:.6f}/{vllm_pad_std:.6f}")
            print(f"replay mean/std: {r3_pad_mean:.6f}/{r3_pad_std:.6f}")
            print(f"no-replay mean/std: {no_r3_pad_mean:.6f}/{no_r3_pad_std:.6f}")

        # Diagnostic sanity assertions only.
        assert torch.isfinite(r3_logprobs).all()
        assert torch.isfinite(no_r3_logprobs).all()
        assert math.isfinite(r3_diff_valid_mean)
        assert math.isfinite(no_r3_diff_valid_mean)
    finally:
        ray.shutdown()
