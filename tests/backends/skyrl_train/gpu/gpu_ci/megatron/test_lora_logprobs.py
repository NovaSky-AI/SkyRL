"""LoRA trainer/inference parity on the four-H100 CI worker."""

import pytest
import ray

from examples.model_checks.check_logprobs import check_logprobs
from examples.model_checks.megatron_lora import LoRALogprobWorker
from skyrl.backends.skyrl_train.workers.worker import PPORayActorGroup
from skyrl.train.utils.utils import ResolvedPlacementGroup
from skyrl.utils.tok import get_tokenizer
from tests.backends.skyrl_train.gpu.gpu_ci.conftest import ray_init
from tests.backends.skyrl_train.gpu.gpu_ci.megatron.test_megatron_models import (
    _engine_overrides_for_model,
    _extra_env_vars_for_model,
    get_test_actor_config,
)
from tests.backends.skyrl_train.gpu.utils import InferenceEngineState


@pytest.mark.asyncio
@pytest.mark.h100
@pytest.mark.parametrize(
    "model_name",
    [
        "Qwen/Qwen3-0.6B",
        "zai-org/GLM-4.7-Flash",
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    ],
)
async def test_lora_logprobs_matching_roundtrip(model_name, tmp_path):
    with ray_init(extra_env_vars=_extra_env_vars_for_model(model_name)):
        cfg = get_test_actor_config(model_name)
        cfg.trainer.strategy = "megatron"
        cfg.trainer.placement.colocate_all = True
        cfg.trainer.placement.policy_num_gpus_per_node = 4
        cfg.trainer.policy.inference_only_init = True
        parallel = cfg.trainer.policy.megatron_config
        parallel.tensor_model_parallel_size = 4
        parallel.expert_model_parallel_size = 1 if "Qwen3-0.6B" in model_name else 4
        parallel.expert_tensor_parallel_size = 1
        parallel.lora_config.merge_lora = False
        lora = cfg.trainer.policy.model.lora
        lora.rank = 8
        lora.alpha = 16
        lora.target_modules = ["linear_proj", "linear_fc1", "linear_fc2"]
        lora.lora_sync_path = str(tmp_path / "adapter")
        cfg.generator.inference_engine.tensor_parallel_size = 4
        cfg.generator.inference_engine.num_engines = 1
        overrides = _engine_overrides_for_model(model_name)
        async with InferenceEngineState.create(
            cfg=cfg,
            use_local=True,
            colocate_all=True,
            backend="vllm",
            sleep_level=1,
            enable_lora=True,
            gpu_memory_utilization=overrides["gpu_memory_utilization"],
            engine_init_kwargs=overrides["engine_init_kwargs"],
        ) as engines:
            await engines.client.sleep()
            policy = PPORayActorGroup(
                cfg.trainer,
                num_nodes=1,
                num_gpus_per_node=4,
                ray_actor_type=ray.remote(LoRALogprobWorker),
                pg=ResolvedPlacementGroup(engines.pg),
                num_gpus_per_actor=0.2,
                colocate_all=True,
            )
            ray.get(policy.async_init_model(model_name))
            ray.get(
                policy.async_run_ray_method(
                    "pass_through",
                    "init_weight_sync_state",
                    engines.client,
                    cfg.generator.inference_engine,
                )
            )
            await check_logprobs(policy, engines.client, cfg, get_tokenizer(model_name))
