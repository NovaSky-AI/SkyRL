"""
Run with:
uv run --isolated --extra dev --extra megatron -- pytest -s tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_megatron_lora_models.py

LoRA rows of ``test_logprobs_matching_roundtrip`` (test_megatron_models.py): the
same models, meshes, generation, forward and weight-sync flow, with a LoRA
adapter on the policy. ``merge_lora=False`` rows sync the adapter and serve the
policy under the adapter name; ``merge_lora=True`` rows broadcast merged full
weights and serve the policy as the base model.

Each row runs three phases:

1. Zero adapter: vLLM samples greedily, the trainer scores the sampled tokens
   with its freshly initialized adapter (B = 0), and the two logprob sets must
   agree within ``threshold``.
2. Perturbation: every LoRA B tensor on the trainer receives deterministic,
   name-seeded noise (``perturb_lora_b``). The trainer rescored on the same
   tokens must disagree with the still-stale vLLM logprobs by more than
   ``threshold``, so a sampler that misses the update cannot pass phase 3.
3. Publication: the trainer syncs, vLLM samples again with the updated policy,
   and the trainer scores those tokens. The two logprob sets must agree within
   ``threshold``.
"""

import pytest
import ray
import torch
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.distributed.dispatch import (
    WorkerOutput,
    loss_fn_outputs_to_tensor,
)
from skyrl.backends.skyrl_train.inference_servers.utils import (
    _uses_lora_weight_sync,
    resolve_policy_model_name,
)
from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
    MegatronPolicyWorkerBase,
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
    perturb_lora_b,
)

# Scales the 1e-3 noise std that perturb_lora_b adds to every LoRA B tensor.
LORA_B_MULTIPLIER = 30.0


class LoRAPerturbPolicyWorkerBase(MegatronPolicyWorkerBase):
    def perturb_lora_b(self, multiplier: float) -> dict:
        # One model chunk per virtual pipeline stage.
        named_parameters = (
            (f"chunk{index}.{name}", parameter)
            for index, chunk in enumerate(self.actor_module)
            for name, parameter in chunk.named_parameters()
        )
        return perturb_lora_b(named_parameters, multiplier=multiplier)


LoRAPerturbPolicyWorker = ray.remote(num_gpus=1)(LoRAPerturbPolicyWorkerBase)


def get_test_lora_actor_config(model_name: str, merge_lora: bool, lora_sync_path: str) -> SkyRLTrainConfig:
    cfg = get_test_actor_config(model_name=model_name)
    cfg.trainer.strategy = "megatron"
    cfg.trainer.policy.model.lora = SkyRLLoraConfig(
        rank=8, alpha=16, dropout=0.0, target_modules="all-linear", lora_sync_path=lora_sync_path
    )
    if "glm-4" in model_name.lower():
        # MLA attention has no ``linear_qkv``; adapt the output projection and the
        # MLP, as the Kimi K2.5 row in test_megatron_models.py does.
        cfg.trainer.policy.model.lora.target_modules = ["linear_proj", "linear_fc1", "linear_fc2"]
    cfg.trainer.policy.megatron_config.lora_config.merge_lora = merge_lora
    validate_cfg(cfg)
    return cfg


def _trainer_logprobs(policy, training_input) -> torch.Tensor:
    results = ray.get(policy.async_run_ray_method("mesh", "forward", data=training_input))
    output = WorkerOutput.cat(policy.actor_infos, results)
    return loss_fn_outputs_to_tensor(output.loss_fn_outputs, key="logprobs")


def _mean_abs_diff(reference: torch.Tensor, actual: torch.Tensor, mask: torch.Tensor, label: str) -> float:
    diff = (reference[mask] - actual[mask]).abs()
    print(
        f"{label}: mean abs diff {diff.mean().item():.6f}, max {diff.max().item():.6f}, "
        f"std {diff.std().item():.6f} over {diff.numel()} tokens"
    )
    return diff.mean().item()


async def _sync_weights(policy, client, cfg, label: str):
    """Offload the optimizer, publish the policy to the engines, then offload the model."""
    policy.offload_to_cpu(offload_optimizer=True, offload_model=False)
    await client.wake_up(tags=["weights"])
    with Timer(label):
        ray.get(
            policy.async_run_ray_method(
                "pass_through", "broadcast_to_inference_engines", client, cfg.generator.inference_engine
            )
        )
    policy.offload_to_cpu(offload_optimizer=False, offload_model=True)
    await client.wake_up(tags=["kv_cache"])


@pytest.mark.asyncio
@pytest.mark.megatron_models
@pytest.mark.parametrize(
    "tp,pp,cp,ep,etp,inference_tp,num_gpus,model_name,threshold,merge_lora",
    [
        pytest.param(2, 1, 1, 2, 1, 2, 4, "eatang/qwen3-moe-tiny-random", 2e-1, False, id="qwen3-moe_tp2_ep2_adapter"),
        pytest.param(2, 1, 1, 2, 1, 2, 4, "eatang/qwen3-moe-tiny-random", 2e-1, True, id="qwen3-moe_tp2_ep2_merged"),
        pytest.param(
            1, 2, 2, 1, None, 2, 4, "eatang/qwen3-moe-tiny-random", 2e-1, False, id="qwen3-moe_pp2_cp2_adapter"
        ),
        pytest.param(2, 1, 1, 1, None, 2, 2, "Qwen/Qwen3.5-0.8B", 5e-2, False, id="qwen3.5-0.8b-dense_tp2_adapter"),
        pytest.param(2, 1, 1, 1, None, 2, 2, "Qwen/Qwen3.5-0.8B", 5e-2, True, id="qwen3.5-0.8b-dense_tp2_merged"),
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
            5e-2,
            False,
            id="qwen3.5-35b-a3b_h100_tp4_ep4_adapter",
            marks=pytest.mark.h100,
        ),
    ],
)
async def test_lora_logprobs_matching_roundtrip(
    tp, pp, cp, ep, etp, inference_tp, num_gpus, model_name, threshold, merge_lora, tmp_path
):
    """
    Check that logprob diff matches across vllm and megatron with a zero LoRA adapter and
    again after publishing a perturbed one.
    """
    with ray_init(extra_env_vars=_extra_env_vars_for_model(model_name)):
        cfg = get_test_lora_actor_config(
            model_name=model_name, merge_lora=merge_lora, lora_sync_path=str(tmp_path / "adapter")
        )
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

            # Build the policy with the engines asleep. Adapter rows publish the zero
            # adapter before the first rollout, as the trainer does; merged rows leave
            # the engines on the checkpoint weights they loaded.
            await client.sleep()
            policy = init_worker_with_type(
                "policy",
                shared_pg=pg,
                colocate_all=True,
                num_gpus_per_node=num_gpus,
                cfg=cfg,
                worker_cls=LoRAPerturbPolicyWorker,
            )
            ray.get(
                policy.async_run_ray_method(
                    "pass_through", "init_weight_sync_state", client, cfg.generator.inference_engine
                )
            )
            if lora_sync:
                await _sync_weights(policy, client, cfg, "initial_sync_weights")
            else:
                policy.offload_to_cpu(offload_optimizer=True, offload_model=True)
                await client.wake_up()

            # Phase 1: zero adapter.
            (response_mask, logprobs_t, _), training_input = await generate_with_vllm(
                generator, client, model_name, tokenizer, return_training_input=True
            )
            await client.sleep()
            policy.backload_to_gpu(backload_optimizer=False, backload_model=True)

            mask = response_mask.bool()
            logprobs_megatron = _trainer_logprobs(policy, training_input)
            zero_diff = _mean_abs_diff(logprobs_t, logprobs_megatron, mask, "zero adapter: vLLM vs Megatron")
            assert zero_diff < threshold, f"Logprob diff should be less than {threshold}, but is {zero_diff:.6f}"

            # Phase 2: perturb the trainer's adapter; the engines still serve the zero adapter.
            stats = ray.get(policy.async_run_ray_method("pass_through", "perturb_lora_b", LORA_B_MULTIPLIER))[0]
            print(f"perturbed {stats['changed_tensors']} LoRA B tensors ({stats['changed_elements']} elements)")
            logprobs_megatron_perturbed = _trainer_logprobs(policy, training_input)
            _mean_abs_diff(
                logprobs_megatron, logprobs_megatron_perturbed, mask, "perturbation: Megatron before vs after"
            )
            stale_diff = _mean_abs_diff(
                logprobs_t, logprobs_megatron_perturbed, mask, "stale sampler: vLLM vs perturbed Megatron"
            )
            assert stale_diff > threshold, (
                f"Perturbed Megatron differs from the stale sampler by only {stale_diff:.6f}; "
                f"raise LORA_B_MULTIPLIER so a missed sync fails the {threshold} parity check"
            )

            # Phase 3: publish the perturbed adapter and score the new samples.
            await _sync_weights(policy, client, cfg, "sync_weights")
            (response_mask_2, logprobs_t_2, _), training_input_2 = await generate_with_vllm(
                generator, client, model_name, tokenizer, return_training_input=True
            )
            await client.sleep()
            policy.backload_to_gpu(backload_optimizer=False, backload_model=True)

            logprobs_megatron_2 = _trainer_logprobs(policy, training_input_2)
            updated_diff = _mean_abs_diff(
                logprobs_t_2, logprobs_megatron_2, response_mask_2.bool(), "updated adapter: vLLM vs Megatron"
            )
            assert updated_diff < threshold, f"Logprob diff should be less than {threshold}, but is {updated_diff:.6f}"
