"""
# Run FSDP tests:
uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/test_lora.py -k "fsdp"

# Run Megatron tests:
uv run --isolated --extra dev --extra megatron pytest tests/backends/skyrl_train/gpu/gpu_ci/test_lora.py -k "megatron"

# Only the adapter-only rows (merge_lora=false), disk and in-memory sync:
uv run --isolated --extra dev --extra megatron pytest tests/backends/skyrl_train/gpu/gpu_ci/test_lora.py -k "megatron_adapter"

Multi-LoRA serving tests live separately in
``tests/backends/skyrl_train/gpu/gpu_ci/inference_servers/test_multi_lora_serving.py``
since they exercise the inference-server LoRA control plane, not the
trainer + weight-sync path covered here.
"""

import os

import pytest
import ray

from skyrl.backends.skyrl_train.inference_servers.engine_utils import (
    get_sampling_params_for_backend,
)
from skyrl.backends.skyrl_train.inference_servers.utils import resolve_policy_model_name
from skyrl.train.config import SkyRLLoraConfig, SkyRLTrainConfig
from tests.backends.skyrl_train.gpu.utils import (
    InferenceEngineState,
    get_test_prompts,
    init_worker_with_type,
    run_inference,
)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def get_test_actor_config(
    strategy: str = "fsdp",
    enable_lora: bool = False,
    colocate_all: bool = False,
    weight_sync_backend: str = "nccl",
    tp_size: int = 2,
    merge_lora: bool = True,
    lora_sync_mode: str = "disk",
    lora_sync_path: str | None = None,
) -> SkyRLTrainConfig:
    """Get base config with test-specific overrides."""
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = MODEL
    cfg.trainer.critic.model.path = ""
    cfg.trainer.strategy = strategy
    cfg.trainer.placement.colocate_all = colocate_all
    cfg.trainer.placement.policy_num_gpus_per_node = 2
    cfg.generator.inference_engine.num_engines = 1
    cfg.generator.inference_engine.run_engines_locally = True
    cfg.generator.inference_engine.weight_sync_backend = weight_sync_backend
    cfg.generator.inference_engine.tensor_parallel_size = tp_size
    cfg.generator.inference_engine.gpu_memory_utilization = 0.6

    if strategy == "megatron":
        cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 2
        cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
        cfg.trainer.policy.megatron_config.lora_config.merge_lora = merge_lora

    if enable_lora:
        cfg.trainer.policy.model.lora = SkyRLLoraConfig(
            rank=32,
            alpha=32,
            dropout=0.1,
            target_modules="all-linear",
            sync_mode=lora_sync_mode,
        )
        if lora_sync_path is not None:
            cfg.trainer.policy.model.lora.lora_sync_path = lora_sync_path

    return cfg


@pytest.mark.parametrize(
    ("colocate_all", "weight_sync_backend", "strategy", "tp_size", "merge_lora", "lora_sync_mode"),
    [
        pytest.param(False, "nccl", "fsdp", 2, True, "disk"),
        pytest.param(True, "nccl", "fsdp", 2, True, "disk"),
        pytest.param(False, "nccl", "megatron", 2, True, "disk", marks=pytest.mark.megatron),
        pytest.param(True, "nccl", "megatron", 2, True, "disk", marks=pytest.mark.megatron),
        pytest.param(False, "nccl", "megatron", 2, False, "disk", marks=pytest.mark.megatron),
        pytest.param(True, "nccl", "megatron", 2, False, "disk", marks=pytest.mark.megatron),
        # Adapter-only sync over the transport itself (NCCL broadcast when
        # non-colocated, CUDA IPC when colocated): no PEFT files are written.
        pytest.param(False, "nccl", "megatron", 2, False, "memory", marks=pytest.mark.megatron),
        pytest.param(True, "nccl", "megatron", 2, False, "memory", marks=pytest.mark.megatron),
    ],
    ids=[
        "no_colocate_nccl_fsdp",
        "colocate_nccl_fsdp",
        "no_colocate_nccl_megatron_merged",
        "colocate_nccl_megatron_merged",
        "no_colocate_nccl_megatron_adapter",
        "colocate_nccl_megatron_adapter",
        "no_colocate_nccl_megatron_adapter_memory",
        "colocate_nccl_megatron_adapter_memory",
    ],
)
@pytest.mark.asyncio
async def test_policy_local_engines_e2e(
    ray_init_fixture, tmp_path, colocate_all, weight_sync_backend, strategy, tp_size, merge_lora, lora_sync_mode
):
    """
    Tests initalizing the policy actor group and inference engine, syncing weights, and performing generation.

    ``lora_sync_path`` is a fresh temporary directory so the assertions at the
    end can tell the disk and in-memory adapter syncs apart by what they wrote.
    """
    lora_sync_path = str(tmp_path / "lora_sync")
    cfg = get_test_actor_config(
        strategy=strategy,
        enable_lora=True,
        colocate_all=colocate_all,
        weight_sync_backend=weight_sync_backend,
        tp_size=tp_size,
        merge_lora=merge_lora,
        lora_sync_mode=lora_sync_mode,
        lora_sync_path=lora_sync_path,
    )

    # Only enable LoRA on the vLLM side when adapters are loaded separately.
    # When merge_lora=True the bridge merges LoRA into the full weights, so
    # vLLM receives plain weights and must NOT have enable_lora (which wraps
    # modules and changes named_parameters(), breaking load_weights).
    needs_vllm_lora = not (strategy == "megatron" and merge_lora)

    # If colocate is True, this will load the engine, sleep, and wake up the engine
    async with InferenceEngineState.create(
        cfg=cfg,
        model=MODEL,
        use_local=True,
        tp_size=cfg.generator.inference_engine.tensor_parallel_size,
        colocate_all=cfg.trainer.placement.colocate_all,
        sleep_level=1 if needs_vllm_lora else 2,
        enable_lora=needs_vllm_lora,
    ) as engines:
        client, pg = engines.client, engines.pg

        await client.sleep(level=1)

        policy = init_worker_with_type(
            "policy",
            shared_pg=pg,
            colocate_all=cfg.trainer.placement.colocate_all,
            num_gpus_per_node=cfg.generator.inference_engine.tensor_parallel_size,
            cfg=cfg,
        )
        sampling_params = get_sampling_params_for_backend(
            cfg.generator.inference_engine.backend, cfg.generator.sampling_params
        )
        await client.wake_up(tags=["weights"])

        ray.get(
            policy.async_run_ray_method(
                "pass_through", "init_weight_sync_state", client, cfg.generator.inference_engine
            )
        )
        ray.get(
            policy.async_run_ray_method(
                "pass_through", "broadcast_to_inference_engines", client, cfg.generator.inference_engine
            )
        )
        policy.offload_to_cpu()
        await client.wake_up(tags=["kv_cache"])
        await client.reset_prefix_cache()
        # Use the same resolver production uses so this test actually exercises
        # the LoRA adapter when vLLM has it loaded (FSDP+LoRA, megatron+adapter)
        # and falls back to the base model for megatron+merge_lora.
        outputs = await run_inference(
            client, get_test_prompts(MODEL), sampling_params, model=resolve_policy_model_name(cfg)
        )
        print(f"Example output: {outputs['responses'][0]}, {outputs['stop_reasons'][0]}")

    # The adapter-only paths differ only in how the adapter reaches vLLM, so the
    # generation above passing is the same evidence for both; what tells them
    # apart is the filesystem. The disk sync writes PEFT files that vLLM reads
    # back; the in-memory sync must leave the directory untouched.
    adapter_file = os.path.join(lora_sync_path, "adapter_model.safetensors")
    if needs_vllm_lora and lora_sync_mode == "disk":
        assert os.path.isfile(adapter_file), f"disk LoRA sync did not write {adapter_file}"
    elif needs_vllm_lora and lora_sync_mode == "memory":
        assert not os.path.exists(lora_sync_path), f"in-memory LoRA sync wrote to {lora_sync_path}"
    # megatron + merge_lora syncs merged full weights and never touches lora_sync_path.
