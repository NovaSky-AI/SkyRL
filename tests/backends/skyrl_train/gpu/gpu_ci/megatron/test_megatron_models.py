"""
Run with:
uv run --isolated --extra dev --extra megatron -- pytest -s tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_megatron_models.py
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
from tests.backends.skyrl_train.gpu.gpu_ci.conftest import ray_init
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
    is_mla_model = "moonlight" in model_name.lower() or "glm-4" in model_name.lower()
    if is_mla_model:
        if cfg.trainer.policy.megatron_config.transformer_config_kwargs is None:
            cfg.trainer.policy.megatron_config.transformer_config_kwargs = {}

        cfg.trainer.flash_attn = False

        # cuDNN fused attention does not support THD (sample packing) layout on
        # pre-Hopper GPUs (sm < 90), FA2 doesn't support MLA, and FA3 is
        # Hopper-only, so there is no viable TE attention backend for
        # MLA + sample_packing on Ada/Ampere.  Fall back to BSHD.
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 9:
            cfg.trainer.use_sample_packing = False
    if "qwen3.5" in model_name.lower():
        # sample packing not yet supported for GDN
        # https://github.com/NVIDIA/Megatron-LM/pull/2644
        cfg.trainer.use_sample_packing = False
    validate_cfg(cfg)
    return cfg


def _extra_env_vars_for_model(model_name: str) -> dict[str, str] | None:
    # MLA models need cuDNN fused attention (the conftest globally sets
    # NVTE_FUSED_ATTN=0; re-enable it here so the fused backend is available).
    if "moonlight" in model_name.lower() or "glm-4" in model_name.lower():
        return {"NVTE_FUSED_ATTN": "1"}
    return None


async def generate_with_vllm(generator, client, model_name, tokenizer, return_training_input=False):
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
            temperature=0.0,
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

    rewards = generator_output["rewards"]
    if rewards and not isinstance(rewards[0], list):
        rewards = [[r] * len(resp) for r, resp in zip(rewards, responses)]

    (sequences, attention_mask, response_mask, rewards_t, loss_mask_t, logprobs_t, _) = (
        convert_prompts_responses_to_batch_tensors(
            tokenizer=tokenizer,
            prompts=generator_output["prompt_token_ids"],
            responses=responses,
            rewards=rewards,
            loss_masks=generator_output["loss_masks"],
            logprobs=generator_output.get("rollout_logprobs"),
        )
    )
    if return_training_input:
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
                "rollout_expert_indices": None,
                "action_log_probs": torch.zeros((batch_size, num_actions), dtype=torch.float32),
                "base_action_log_probs": torch.zeros((batch_size, num_actions), dtype=torch.float32),
                "advantages": torch.zeros((batch_size, num_actions), dtype=torch.float32),
                "action_mask": response_mask.to(dtype=torch.int64),
            }
        )
        training_input.metadata = {"response_length": num_actions}
        return (response_mask, logprobs_t), training_input
    else:
        return (response_mask, logprobs_t)


async def construct_training_input_from_generator_output(generator_output, tokenizer):
    return convert_prompts_responses_to_batch_tensors(
        tokenizer=tokenizer,
        prompts=generator_output["prompt_token_ids"],
        responses=generator_output["response_ids"],
        rewards=generator_output["rewards"],
        loss_masks=generator_output["loss_masks"],
    )


@pytest.mark.asyncio
@pytest.mark.megatron_models
@pytest.mark.parametrize(
    "tp,pp,cp,ep,etp,inference_tp,num_gpus,model_name,vllm_threshold,megatron_threshold",
    [
        pytest.param(2, 1, 1, 2, 1, 2, 4, "eatang/qwen3-moe-tiny-random", 1e-1, 2e-1, id="qwen3-moe_tp2_ep2"),
        pytest.param(1, 2, 2, 1, None, 2, 4, "eatang/qwen3-moe-tiny-random", 1e-1, 2e-1, id="qwen3-moe_pp2_cp2"),
        pytest.param(2, 1, 1, 2, 1, 2, 4, "eatang/glm-4.7-flash-tiny-random", 1e-1, 2e-2, id="glm-4.7-flash_tp2_ep2"),
        pytest.param(
            2,
            1,
            1,
            2,
            1,
            4,
            4,
            "eatang/qwen3.5-moe-tiny-random",
            1e-1,
            2e-1,
            id="qwen3.5-moe_tp2_ep2",
            marks=pytest.mark.skip(reason="running into correctness issues for tiny qwen3.5"),
        ),
    ],
)
async def test_logprobs_matching_roundtrip(
    tp, pp, cp, ep, etp, inference_tp, num_gpus, model_name, vllm_threshold, megatron_threshold
):
    """
    Check that logprob diff matches acrosss vllm and megatron.
    """
    with ray_init(extra_env_vars=_extra_env_vars_for_model(model_name)):
        cfg = get_test_actor_config(model_name=model_name)
        cfg.trainer.strategy = "megatron"
        cfg.generator.inference_engine.tensor_parallel_size = inference_tp
        cfg.generator.inference_engine.num_engines = num_gpus // inference_tp
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
            engine_init_kwargs={},
        ) as engines:
            client, pg = engines.client, engines.pg
            await client.wake_up()

            generator = SkyRLGymGenerator(
                generator_cfg=cfg.generator,
                skyrl_gym_cfg=cfg.environment.skyrl_gym,
                inference_engine_client=client,
                tokenizer=tokenizer,
            )

            (response_mask, logprobs_t), training_input = await generate_with_vllm(
                generator, client, model_name, tokenizer, return_training_input=True
            )
            await client.sleep()
            cfg.trainer.placement.policy_num_gpus_per_node = num_gpus
            cfg.trainer.policy.megatron_config.tensor_model_parallel_size = tp
            cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = pp
            cfg.trainer.policy.megatron_config.context_parallel_size = cp
            cfg.trainer.policy.megatron_config.expert_model_parallel_size = ep
            cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = etp
            cfg.trainer.micro_forward_batch_size_per_gpu = 2
            cfg.trainer.micro_train_batch_size_per_gpu = 2

            policy = init_worker_with_type(
                "policy",
                shared_pg=pg,
                colocate_all=True,
                num_gpus_per_node=num_gpus,
                cfg=cfg,
            )
            ray.get(
                policy.async_run_ray_method(
                    "pass_through", "init_weight_sync_state", client, cfg.generator.inference_engine
                )
            )

            refs = policy.async_run_ray_method("mesh", "forward", data=training_input)
            results = ray.get(refs)
            logprobs_megatron = concatenate_outputs_after_mesh_dispatch(policy.actor_infos, results)["output"]

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
                logprobs_diff.mean().item() < megatron_threshold
            ), f"Logprob diff should be less than {megatron_threshold}, but is {logprobs_diff.mean().item():.6f}"

            # sync weights
            policy.offload_to_cpu(offload_optimizer=True, offload_model=False)
            await client.wake_up(tags=["weights"])
            with Timer("sync_weights"):
                ray.get(
                    policy.async_run_ray_method(
                        "pass_through", "broadcast_to_inference_engines", client, cfg.generator.inference_engine
                    )
                )
            await client.wake_up(tags=["kv_cache"])

            (response_mask_2, logprobs_t_2) = await generate_with_vllm(
                generator, client, model_name, tokenizer, return_training_input=False
            )

            logprobs_t_valid = logprobs_t[response_mask.bool()]
            logprobs_t_2_valid = logprobs_t_2[response_mask_2.bool()]

            assert (
                logprobs_t_valid.shape == logprobs_t_2_valid.shape
            ), f"generator output shapes should match before and after sync, got {logprobs_t_valid.shape} and {logprobs_t_2_valid.shape}"

            logprobs_diff = (logprobs_t_valid - logprobs_t_2_valid).abs()
            print(
                f"vLLM logprobs    - mean: {logprobs_t_valid.mean().item():.6f}, std: {logprobs_t_valid.std().item():.6f}"
            )
            print(
                f"vLLM logprobs after sync - mean: {logprobs_t_2_valid.mean().item():.6f}, std: {logprobs_t_2_valid.std().item():.6f}"
            )
            print(f"vLLM logprob diff mean: {logprobs_diff.mean().item():.6f}, std: {logprobs_diff.std().item():.6f}")
            assert (
                logprobs_diff.mean().item() < vllm_threshold
            ), f"Logprob diff should be less than {vllm_threshold}, but is {logprobs_diff.mean().item():.6f}"
