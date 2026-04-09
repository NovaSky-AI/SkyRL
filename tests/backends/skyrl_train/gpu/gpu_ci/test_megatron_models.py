"""
Run with:
uv run --isolated --extra dev --extra megatron -- pytest -s tests/backends/skyrl_train/gpu/gpu_ci/test_megatron_models.py
"""

import pytest
import ray
import torch
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.distributed.dispatch import (
    concatenate_outputs_after_mesh_dispatch,
)
from skyrl.backends.skyrl_train.inference_engines.utils import (
    get_sampling_params_for_backend,
)
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.train.config import SamplingParams, SkyRLTrainConfig
from skyrl.train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl.train.generators.base import GeneratorInput
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator
from skyrl.train.utils.utils import validate_cfg
from tests.backends.skyrl_train.gpu.utils import (
    InferenceEngineState,
    Timer,
    get_test_generator_input,
    init_worker_with_type,
)

NUM_PROMPTS = 10
N_SAMPLES_PER_PROMPT = 4
MAX_GENERATE_LENGTH = 128


def get_test_actor_config(model_name) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = model_name
    cfg.trainer.micro_forward_batch_size_per_gpu = 2
    cfg.trainer.micro_train_batch_size_per_gpu = 2
    cfg.trainer.use_sample_packing = True
    cfg.generator.inference_engine.distributed_executor_backend = "mp"
    # flash attn + mla works without sample packing, logprobs are crazy/wrong
    # but flash-attn correctly throws error with sample packing
    # we should add an assert that if you set use_sample_packing=False flash attn can accidentally be used
    # and that we enable nvte fused attn for moonlight models with use_sample_packing=True
    # need to enable nvte fused attn for router replay tests when using moonlight models with use_sample_packing=True
    cfg.trainer.logger = "console"
    if "Moonlight" in model_name:
        if cfg.trainer.policy.megatron_config.transformer_config_kwargs is None:
            cfg.trainer.policy.megatron_config.transformer_config_kwargs = {}

        cfg.trainer.flash_attn = False
    validate_cfg(cfg)
    return cfg


def build_training_input_from_text_samples(
    tokenizer: AutoTokenizer, prompt_response_pairs: list[tuple[str, str]]
) -> TrainingInputBatch:
    prompts = []
    responses = []
    rewards = []
    loss_masks = []

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    for prompt_text, response_text in prompt_response_pairs:
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        response_ids = tokenizer.encode(response_text, add_special_tokens=False)
        if tokenizer.eos_token_id is not None and (not response_ids or response_ids[-1] != tokenizer.eos_token_id):
            response_ids.append(tokenizer.eos_token_id)

        prompts.append(prompt_ids)
        responses.append(response_ids)
        rewards.append([0.0] * len(response_ids))
        loss_masks.append([1] * len(response_ids))

    sequences, attention_mask, response_mask, rewards_t, loss_mask_t, _, _ = convert_prompts_responses_to_batch_tensors(
        tokenizer=tokenizer,
        prompts=prompts,
        responses=responses,
        rewards=rewards,
        loss_masks=loss_masks,
    )

    num_actions = response_mask.shape[1]
    batch_size = sequences.shape[0]
    training_input = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "rewards": rewards_t,
            "loss_mask": loss_mask_t,
            "rollout_logprobs": torch.zeros((batch_size, num_actions), dtype=torch.float32),
            "action_log_probs": torch.zeros((batch_size, num_actions), dtype=torch.float32),
            "base_action_log_probs": torch.zeros((batch_size, num_actions), dtype=torch.float32),
            "advantages": torch.zeros((batch_size, num_actions), dtype=torch.float32),
            "action_mask": response_mask.to(dtype=torch.int64),
        }
    )
    training_input.metadata = {"response_length": num_actions}
    return training_input


@pytest.mark.asyncio
@pytest.mark.megatron
@pytest.mark.parametrize(
    "tp,pp,cp,ep,etp,model_name",
    [
        pytest.param(2, 1, 2, 2, 1, "yujiepan/qwen3.5-moe-tiny-random", id="qwen3.5-moe"),
    ],
)
async def test_logprobs_matching_roundtrip(ray_init_fixture, tp, pp, cp, ep, etp, model_name):
    """
    Check that logprob diff matches acrosss vllm and megatron.
    """
    try:
        cfg = get_test_actor_config(model_name=model_name)
        cfg.trainer.strategy = "megatron"
        cfg.generator.inference_engine.tensor_parallel_size = 8
        cfg.generator.sampling_params = SamplingParams(
            max_generate_length=MAX_GENERATE_LENGTH,
            logprobs=1,
            temperature=1.0,
        )
        cfg.generator.batched = False
        cfg.generator.max_turns = 1

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        async with InferenceEngineState.create(
            cfg=cfg,
            model=model_name,
            use_local=True,
            colocate_all=True,
            backend="vllm",
            sleep_level=1,
            gpu_memory_utilization=0.9,
        ) as engines:
            client, pg = engines.client, engines.pg
            await client.wake_up()

            generator = SkyRLGymGenerator(
                generator_cfg=cfg.generator,
                skyrl_gym_cfg=cfg.environment.skyrl_gym,
                inference_engine_client=client,
                tokenizer=tokenizer,
            )

            input_batch: GeneratorInput = get_test_generator_input(
                model=model_name,
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
                    max_generate_length=MAX_GENERATE_LENGTH,
                    min_p=0.0,
                    logprobs=1,
                ),
            )

            with Timer("generate_with_vllm"):
                generator_output = await generator.generate(input_batch)

            responses = generator_output["response_ids"]
            await client.sleep()

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
                "rollout_expert_indices": rii_tensor,
                "action_log_probs": torch.zeros((batch_size, num_actions), dtype=torch.float32),
                "base_action_log_probs": torch.zeros((batch_size, num_actions), dtype=torch.float32),
                "advantages": torch.zeros((batch_size, num_actions), dtype=torch.float32),
                "action_mask": response_mask.to(dtype=torch.int64),
            }
        )
        training_input.metadata = {"response_length": num_actions}

        cfg.trainer.placement.policy_num_gpus_per_node = 8
        cfg.trainer.policy.megatron_config.tensor_model_parallel_size = tp
        cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = pp
        cfg.trainer.policy.megatron_config.context_parallel_size = cp
        cfg.trainer.policy.megatron_config.expert_model_parallel_size = ep
        cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = etp
        cfg.trainer.micro_forward_batch_size_per_gpu = 2
        cfg.trainer.micro_train_batch_size_per_gpu = 2

        def run_megatron_forward() -> torch.Tensor:
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

        logprobs_megatron = run_megatron_forward()
        mask = response_mask.bool()

        vllm_valid = logprobs_t[mask]
        logprobs_megatron_valid = logprobs_megatron[mask]

        logprobs_diff = (vllm_valid - logprobs_megatron_valid).abs()
        print(f"vLLM logprobs     - mean: {vllm_valid.mean().item():.6f}, std: {vllm_valid.std().item():.6f}")
        print(
            f"Megatron - mean: {logprobs_megatron_valid.mean().item():.6f}, std: {logprobs_megatron_valid.std().item():.6f}"
        )
        print(f"logprob diff mean: {logprobs_diff.mean().item():.6f}, std: {logprobs_diff.std().item():.6f}")

        assert (
            logprobs_diff.mean().item() < 1e-6
        ), f"Logprob diff should be close to 0, but is {logprobs_diff.mean().item():.6f}"
    finally:
        ray.shutdown()
