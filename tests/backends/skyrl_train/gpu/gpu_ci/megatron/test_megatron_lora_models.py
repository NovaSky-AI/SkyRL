"""
Run with:
uv run --isolated --extra dev --extra megatron -- pytest -s tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_megatron_lora_models.py

LoRA rows of ``test_logprobs_matching_roundtrip`` (test_megatron_models.py): the
same models, meshes, generation, forward and weight-sync flow, with a LoRA
adapter on the policy. ``merge_lora=False`` rows sync the adapter and serve the
policy under the adapter name; ``merge_lora=True`` rows broadcast merged full
weights and serve the policy as the base model.
"""

import pytest
import ray
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.distributed.dispatch import (
    WorkerOutput,
    loss_fn_outputs_to_tensor,
)
from skyrl.backends.skyrl_train.inference_servers.utils import (
    _uses_lora_weight_sync,
    resolve_policy_model_name,
)
from skyrl.train.config import SamplingParams, SkyRLLoraConfig, SkyRLTrainConfig
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator
from skyrl.train.utils.utils import validate_cfg
from tests.backends.skyrl_train.gpu.gpu_ci.conftest import ray_init
from tests.backends.skyrl_train.gpu.gpu_ci.megatron.test_megatron_models import (
    MAX_GENERATE_LENGTH,
    _engine_overrides_for_model,
    _extra_env_vars_for_model,
    generate_with_vllm,
    get_test_actor_config,
)
from tests.backends.skyrl_train.gpu.utils import (
    InferenceEngineState,
    Timer,
    init_worker_with_type,
)


def get_test_lora_actor_config(model_name: str, merge_lora: bool) -> SkyRLTrainConfig:
    cfg = get_test_actor_config(model_name=model_name)
    cfg.trainer.strategy = "megatron"
    cfg.trainer.policy.model.lora = SkyRLLoraConfig(rank=8, alpha=16, dropout=0.0, target_modules="all-linear")
    if "glm-4" in model_name.lower():
        # MLA attention has no ``linear_qkv``; adapt the output projection and the
        # MLP, as the Kimi K2.5 row in test_megatron_models.py does.
        cfg.trainer.policy.model.lora.target_modules = ["linear_proj", "linear_fc1", "linear_fc2"]
    cfg.trainer.policy.megatron_config.lora_config.merge_lora = merge_lora
    validate_cfg(cfg)
    return cfg


@pytest.mark.asyncio
@pytest.mark.megatron_models
@pytest.mark.parametrize(
    "tp,pp,cp,ep,etp,inference_tp,num_gpus,model_name,vllm_threshold,megatron_threshold,merge_lora",
    [
        pytest.param(
            2, 1, 1, 2, 1, 2, 4, "eatang/qwen3-moe-tiny-random", 1e-1, 2e-1, False, id="qwen3-moe_tp2_ep2_adapter"
        ),
        pytest.param(
            2, 1, 1, 2, 1, 2, 4, "eatang/qwen3-moe-tiny-random", 1e-1, 2e-1, True, id="qwen3-moe_tp2_ep2_merged"
        ),
        pytest.param(
            1, 2, 2, 1, None, 2, 4, "eatang/qwen3-moe-tiny-random", 1e-1, 2e-1, False, id="qwen3-moe_pp2_cp2_adapter"
        ),
        pytest.param(
            2, 1, 1, 1, None, 2, 2, "Qwen/Qwen3.5-0.8B", 1e-1, 5e-2, False, id="qwen3.5-0.8b-dense_tp2_adapter"
        ),
        pytest.param(2, 1, 1, 1, None, 2, 2, "Qwen/Qwen3.5-0.8B", 1e-1, 5e-2, True, id="qwen3.5-0.8b-dense_tp2_merged"),
        # Large MoE rows on 4xH100-80G, same meshes and engine overrides as the
        # bf16 rows in test_megatron_models.py.
        pytest.param(
            4,
            1,
            1,
            4,
            1,
            4,
            4,
            "zai-org/GLM-4.7-Flash",
            3e-1,
            5e-2,
            False,
            id="glm-4.7-flash_h100_tp4_ep4_adapter",
            marks=pytest.mark.h100,
        ),
        pytest.param(
            4,
            1,
            1,
            4,
            1,
            4,
            4,
            "Qwen/Qwen3.5-35B-A3B",
            3e-1,
            5e-2,
            False,
            id="qwen3.5-35b-a3b_h100_tp4_ep4_adapter",
            marks=pytest.mark.h100,
        ),
    ],
)
async def test_lora_logprobs_matching_roundtrip(
    tp, pp, cp, ep, etp, inference_tp, num_gpus, model_name, vllm_threshold, megatron_threshold, merge_lora
):
    """
    Check that logprob diff matches across vllm and megatron with a LoRA adapter on the policy.
    """
    with ray_init(extra_env_vars=_extra_env_vars_for_model(model_name)):
        cfg = get_test_lora_actor_config(model_name=model_name, merge_lora=merge_lora)
        # With merge_lora=False the policy is served under the adapter name, which
        # only exists after a sync -- so sync first.
        lora_sync = _uses_lora_weight_sync(cfg)
        cfg.generator.inference_engine.tensor_parallel_size = inference_tp
        cfg.generator.inference_engine.num_engines = num_gpus // inference_tp
        cfg.generator.sampling_params = SamplingParams(
            max_generate_length=MAX_GENERATE_LENGTH,
            logprobs=1,
            temperature=0.0,
        )
        cfg.generator.batched = False
        cfg.generator.max_turns = 1

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        engine_overrides = _engine_overrides_for_model(model_name)
        async with InferenceEngineState.create(
            cfg=cfg,
            model=model_name,
            use_local=True,
            colocate_all=True,
            backend="vllm",
            sleep_level=2,  # full sleep — this test explicitly syncs weights
            gpu_memory_utilization=engine_overrides["gpu_memory_utilization"],
            engine_init_kwargs=engine_overrides["engine_init_kwargs"],
            max_num_seqs=engine_overrides.get("max_num_seqs"),
        ) as engines:
            client, pg = engines.client, engines.pg

            generator = SkyRLGymGenerator(
                generator_cfg=cfg.generator,
                skyrl_gym_cfg=cfg.environment.skyrl_gym,
                inference_engine_client=client,
                tokenizer=tokenizer,
                # None for merged rows, keeping them on the default model.
                policy_model_name=resolve_policy_model_name(cfg) if lora_sync else None,
            )

            cfg.trainer.placement.policy_num_gpus_per_node = num_gpus
            cfg.trainer.policy.megatron_config.tensor_model_parallel_size = tp
            cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = pp
            cfg.trainer.policy.megatron_config.context_parallel_size = cp
            cfg.trainer.policy.megatron_config.expert_model_parallel_size = ep
            cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = etp
            cfg.trainer.micro_forward_batch_size_per_gpu = 2
            cfg.trainer.micro_train_batch_size_per_gpu = 2

            policy = None
            if lora_sync:
                # Sync before the first rollout, as the trainer does: adapter rows have
                # no adapter on the engines until one is synced. Build the policy with
                # the engines asleep, then run the same offload/wake/broadcast dance as
                # the sync below.
                await client.sleep()
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
                policy.offload_to_cpu(offload_optimizer=True, offload_model=False)
                await client.wake_up(tags=["weights"])
                with Timer("initial_sync_weights"):
                    ray.get(
                        policy.async_run_ray_method(
                            "pass_through", "broadcast_to_inference_engines", client, cfg.generator.inference_engine
                        )
                    )
                policy.offload_to_cpu(offload_optimizer=False, offload_model=True)
                await client.wake_up(tags=["kv_cache"])
            else:
                await client.wake_up()

            (response_mask, logprobs_t, gen_out_1), training_input = await generate_with_vllm(
                generator, client, model_name, tokenizer, return_training_input=True
            )
            await client.sleep()

            if policy is None:
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
            else:
                policy.backload_to_gpu(backload_optimizer=False, backload_model=True)

            refs = policy.async_run_ray_method("mesh", "forward", data=training_input)
            results = ray.get(refs)
            policy_output = WorkerOutput.cat(policy.actor_infos, results)
            logprobs_megatron = loss_fn_outputs_to_tensor(policy_output.loss_fn_outputs, key="logprobs")

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
            policy.offload_to_cpu(offload_optimizer=False, offload_model=True)
            await client.wake_up(tags=["kv_cache"])

            response_mask_2, logprobs_t_2, gen_out_2 = await generate_with_vllm(
                generator, client, model_name, tokenizer, return_training_input=False
            )

            logprobs_t_valid = logprobs_t[response_mask.bool()]
            logprobs_t_2_valid = logprobs_t_2[response_mask_2.bool()]

            # Pre- and post-sync are two independent sampled generations
            # so truncate to the shorter sequence for the magnitude check.
            if logprobs_t_valid.shape[0] != logprobs_t_2_valid.shape[0]:
                min_len = min(logprobs_t_valid.shape[0], logprobs_t_2_valid.shape[0])
                print(
                    f"NOTE: pre/post-sync generation lengths differ "
                    f"({logprobs_t_valid.shape[0]} vs {logprobs_t_2_valid.shape[0]}); "
                    f"truncating to {min_len} for the magnitude check."
                )
                logprobs_t_valid = logprobs_t_valid[:min_len]
                logprobs_t_2_valid = logprobs_t_2_valid[:min_len]

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
