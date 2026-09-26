"""E2E tests for shared-outer grouped-expert LoRA on the Megatron backend.

Covers train step + weight sync to vLLM for both sync modes on a tiny MoE
model: merged full-weight sync (merge_lora=True) exercises the bridge's
grouped-export adapter merge, and adapter sync (merge_lora=False) exercises
the shared-outer adapter export plus the vLLM layout conversion.

Run with:
uv run --isolated --extra dev --extra megatron pytest tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_shared_outer_lora.py
"""

import pytest
import ray
import torch
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.inference_servers.engine_utils import (
    get_sampling_params_for_backend,
)
from skyrl.backends.skyrl_train.inference_servers.utils import resolve_policy_model_name
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.train.config import SkyRLLoraConfig, SkyRLTrainConfig
from skyrl.train.utils.utils import validate_cfg
from tests.backends.skyrl_train.gpu.utils import (
    InferenceEngineState,
    get_test_prompts,
    init_worker_with_type,
    run_inference,
)

MOE_TINY_MODEL = "eatang/qwen3-moe-tiny-random"
NUM_GPUS = 4


def get_test_actor_config(merge_lora: bool) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = MOE_TINY_MODEL
    cfg.trainer.strategy = "megatron"
    cfg.trainer.placement.colocate_all = True
    cfg.trainer.placement.policy_num_gpus_per_node = NUM_GPUS
    cfg.trainer.logger = "console"

    cfg.generator.inference_engine.num_engines = 1
    cfg.generator.inference_engine.run_engines_locally = True
    cfg.generator.inference_engine.weight_sync_backend = "nccl"
    cfg.generator.inference_engine.tensor_parallel_size = NUM_GPUS

    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 2
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.expert_model_parallel_size = 2
    cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = 1
    cfg.trainer.policy.megatron_config.lora_config.merge_lora = merge_lora
    cfg.trainer.policy.megatron_config.lora_config.experts_shared_outer_loras = True
    cfg.trainer.policy.model.lora = SkyRLLoraConfig(rank=8, alpha=8)

    # One optimizer step over a small dummy batch.
    cfg.trainer.train_batch_size = NUM_GPUS * 2
    cfg.trainer.policy_mini_batch_size = NUM_GPUS
    cfg.generator.n_samples_per_prompt = 1
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.micro_forward_batch_size_per_gpu = 2

    validate_cfg(cfg)
    return cfg


def get_test_training_batch(batch_size: int) -> TrainingInputBatch:
    tokenizer = AutoTokenizer.from_pretrained(MOE_TINY_MODEL, trust_remote_code=True)

    sentences = [
        "<|im_start|>system\nYou are a helpful assistant.",
        "<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n",
    ] * (batch_size // 2)
    sequences = [tokenizer.encode(sentence) for sentence in sentences]
    num_actions = 8
    max_seq_length = max(len(seq) for seq in sequences) + 4

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    attention_masks = []
    for i, seq in enumerate(sequences):
        pad_after = max_seq_length - len(seq)
        attention_masks.append([1] * len(seq) + [0] * pad_after)
        sequences[i] = seq + [pad_token_id] * pad_after

    sequences = torch.tensor(sequences)
    attention_masks = torch.tensor(attention_masks)
    loss_masks = attention_masks[:, -num_actions:].float()

    data = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_masks,
            "action_log_probs": torch.full((batch_size, num_actions), 0.1),
            "base_action_log_probs": torch.full((batch_size, num_actions), 0.2),
            "rollout_logprobs": torch.full((batch_size, num_actions), 0.11),
            "values": torch.full((batch_size, num_actions), 0.1),
            "returns": torch.full((batch_size, num_actions), 0.1),
            "advantages": torch.full((batch_size, num_actions), 0.5),
            "loss_mask": loss_masks,
            "response_mask": loss_masks,
        }
    )
    data.metadata = {"response_length": num_actions}
    return data


@pytest.mark.parametrize(
    "merge_lora",
    [
        pytest.param(True, id="merged_sync"),
        pytest.param(False, id="adapter_sync"),
    ],
)
@pytest.mark.asyncio
@pytest.mark.megatron
async def test_shared_outer_lora_train_and_sync(ray_init_fixture, merge_lora):
    """Train one step with shared-outer grouped-expert LoRA, sync to vLLM, generate."""
    cfg = get_test_actor_config(merge_lora=merge_lora)

    # vLLM only needs enable_lora when adapters are synced separately; the
    # merged path pushes plain full weights.
    needs_vllm_lora = not merge_lora

    async with InferenceEngineState.create(
        cfg=cfg,
        model=MOE_TINY_MODEL,
        use_local=True,
        tp_size=cfg.generator.inference_engine.tensor_parallel_size,
        colocate_all=True,
        sleep_level=1 if needs_vllm_lora else 2,
        enable_lora=needs_vllm_lora,
    ) as engines:
        client, pg = engines.client, engines.pg
        await client.sleep(level=1)

        policy = init_worker_with_type(
            "policy",
            shared_pg=pg,
            colocate_all=True,
            num_gpus_per_node=NUM_GPUS,
            cfg=cfg,
        )

        # One training step so the adapters carry non-zero weights into the sync.
        batch = get_test_training_batch(cfg.trainer.train_batch_size)
        batch.metadata["global_step"] = 0
        results = ray.get(policy.async_run_ray_method("mesh", "forward_backward", batch))
        ray.get(policy.async_run_ray_method("pass_through", "optim_step"))
        for result in results:
            assert "policy_loss" in result.metrics

        ray.get(
            policy.async_run_ray_method(
                "pass_through", "init_weight_sync_state", client, cfg.generator.inference_engine
            )
        )
        await client.wake_up(tags=["weights"])
        ray.get(
            policy.async_run_ray_method(
                "pass_through", "broadcast_to_inference_engines", client, cfg.generator.inference_engine
            )
        )
        policy.offload_to_cpu()
        await client.wake_up(tags=["kv_cache"])
        await client.reset_prefix_cache()

        sampling_params = get_sampling_params_for_backend(
            cfg.generator.inference_engine.backend, cfg.generator.sampling_params
        )
        outputs = await run_inference(
            client,
            get_test_prompts(MOE_TINY_MODEL, num_samples=4),
            sampling_params,
            model=resolve_policy_model_name(cfg),
        )
        assert len(outputs["responses"]) == 4
        print(f"Example output: {outputs['responses'][0]!r}, {outputs['stop_reasons'][0]}")
