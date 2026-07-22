"""
Run with:
uv run --isolated --extra dev --extra fsdp pytest -s -vvv tests/backends/skyrl_train/gpu/gpu_ci/test_delta_weight_sync_e2e.py -m "not megatron"
uv run --isolated --extra dev --extra megatron pytest -s -vvv tests/backends/skyrl_train/gpu/gpu_ci/test_delta_weight_sync_e2e.py -m megatron
"""

import pytest
import ray
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.inference_servers.engine_utils import (
    get_sampling_params_for_backend,
)
from skyrl.train.config import SkyRLTrainConfig
from tests.backends.skyrl_train.gpu.gpu_ci.delta_weight_sync_utils import (
    init_policy_worker_for_delta,
)
from tests.backends.skyrl_train.gpu.utils import (
    InferenceEngineState,
    get_test_prompts,
    run_inference,
)

MODEL = "Qwen/Qwen3-0.6B"


def _delta_sync_cfg(strategy: str, tmp_path) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = strategy
    cfg.trainer.logger = "console"
    cfg.trainer.policy.model.path = MODEL
    cfg.trainer.policy.inference_only_init = True
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_gpus_per_node = 1
    cfg.trainer.placement.ref_num_gpus_per_node = 1
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.remove_microbatch_padding = False

    cfg.generator.inference_engine.backend = "vllm"
    cfg.generator.inference_engine.run_engines_locally = True
    cfg.generator.inference_engine.distributed_executor_backend = "ray"
    cfg.generator.inference_engine.num_engines = 1
    cfg.generator.inference_engine.tensor_parallel_size = 1
    cfg.generator.inference_engine.weight_sync_backend = "delta"
    cfg.generator.inference_engine.gpu_memory_utilization = 0.55
    cfg.generator.inference_engine.delta_weight_sync.sync_dir = str(tmp_path / f"sync-{strategy}")
    cfg.generator.inference_engine.delta_weight_sync.local_checkpoint_dir = str(tmp_path / f"receiver-{strategy}")
    cfg.generator.inference_engine.delta_weight_sync.checkpoint_load_format = "vllm_multi_thread_safetensors"
    cfg.generator.inference_engine.delta_weight_sync.multi_thread_safetensors_max_workers = 2
    cfg.generator.inference_engine.delta_weight_sync.publish_num_workers = 2

    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.context_parallel_size = 1
    cfg.trainer.policy.megatron_config.expert_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = 1
    return cfg


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "strategy",
    [
        "fsdp",
        pytest.param("megatron", marks=pytest.mark.megatron),
    ],
)
async def test_delta_weight_sync_sparse_update_e2e(ray_init_fixture, tmp_path, strategy):
    cfg = _delta_sync_cfg(strategy, tmp_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)

    async with InferenceEngineState.create(
        model=MODEL,
        cfg=cfg,
        use_local=True,
        tp_size=cfg.generator.inference_engine.tensor_parallel_size,
        colocate_all=False,
        sleep_level=2,
        backend="vllm",
        gpu_memory_utilization=cfg.generator.inference_engine.gpu_memory_utilization,
        num_inference_engines=cfg.generator.inference_engine.num_engines,
        distributed_executor_backend=cfg.generator.inference_engine.distributed_executor_backend,
    ) as engines:
        client = engines.client
        policy = init_policy_worker_for_delta(
            cfg=cfg,
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
        )
        ray.get(
            policy.async_run_ray_method(
                "pass_through",
                "init_weight_sync_state",
                client,
                cfg.generator.inference_engine,
            )
        )

        ray.get(
            policy.async_run_ray_method(
                "pass_through",
                "broadcast_to_inference_engines",
                client,
                cfg.generator.inference_engine,
            )
        )
        perturb = ray.get(
            policy.async_run_ray_method(
                "pass_through",
                "benchmark_apply_sparse_weight_delta",
                sparsity=0.04,
                delta=1.0e-3,
                phase=1,
            )
        )
        assert any(result["updated_elements"] > 0 for result in perturb)

        ray.get(
            policy.async_run_ray_method(
                "pass_through",
                "broadcast_to_inference_engines",
                client,
                cfg.generator.inference_engine,
            )
        )

        sampling_params = get_sampling_params_for_backend(
            cfg.generator.inference_engine.backend,
            cfg.generator.sampling_params,
        )
        outputs = await run_inference(client, get_test_prompts(MODEL), sampling_params, tokenizer=tokenizer)
        assert outputs["responses"]
