"""
Tests policy actor with remote inference engines (spawns vLLM/SGLang server as subprocess).

# Run only vllm tests (requires vllm extra):
uv run --isolated --extra dev --extra vllm --extra deepspeed pytest tests/gpu/gpu_ci/test_policy_remote_engines_e2e.py -m "vllm"

# Run only sglang tests (requires sglang extra):
uv run --isolated --extra dev --extra sglang --extra deepspeed pytest tests/gpu/gpu_ci/test_policy_remote_engines_e2e.py -m "sglang"
"""

import pytest
import asyncio
import ray
import hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer

from tests.gpu.utils import (
    init_worker_with_type,
    get_test_prompts,
    run_inference,
    init_remote_inference_servers,
)
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.entrypoints.main_base import config_dir

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def get_test_actor_config() -> DictConfig:
    """Get base config with test-specific overrides."""
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")

        # Override specific parameters
        cfg.trainer.policy.model.path = MODEL
        cfg.trainer.critic.model.path = ""
        cfg.trainer.placement.policy_num_gpus_per_node = 2
        cfg.trainer.placement.colocate_all = False
        cfg.generator.async_engine = True
        cfg.generator.num_inference_engines = 1
        cfg.generator.run_engines_locally = False  # Use remote engines
        cfg.trainer.flash_attn = False
        cfg.trainer.use_sample_packing = False

        return cfg


@pytest.mark.parametrize(
    ("weight_sync_backend", "strategy", "backend", "tp_size"),
    [
        pytest.param("nccl", "fsdp", "vllm", 2, marks=pytest.mark.vllm),
        # pytest.param("gloo", "fsdp", "vllm", 2, marks=pytest.mark.vllm),
        # pytest.param("nccl", "deepspeed", "vllm", 2, marks=pytest.mark.vllm),
        # pytest.param("nccl", "fsdp2", "vllm", 2, marks=pytest.mark.vllm),
        # # TODO(Charlie): add TP > 1 tests for sglang when we support it
        # pytest.param("nccl", "deepspeed", "sglang", 1, marks=pytest.mark.sglang),
        # pytest.param("nccl", "fsdp2", "sglang", 1, marks=pytest.mark.sglang),
        # pytest.param("gloo", "fsdp", "sglang", 1, marks=pytest.mark.sglang),
    ],
    ids=[
        "nccl_fsdp_vllm",
        # "gloo_fsdp_vllm",
        # "nccl_deepspeed_vllm",
        # "nccl_fsdp2_vllm",
        # "nccl_deepspeed_sglang",
        # "nccl_fsdp2_sglang",
        # "gloo_fsdp_sglang",
    ],
)
def test_policy_remote_engines_e2e(ray_init_fixture, weight_sync_backend, strategy, backend, tp_size):
    """
    Tests initializing the policy actor group with a remote inference engine server,
    syncing weights, and performing generation.

    This test spawns a vLLM/SGLang server as a subprocess to act as the remote inference engine.
    """
    server_process = None
    try:
        cfg = get_test_actor_config()
        cfg.generator.weight_sync_backend = weight_sync_backend
        cfg.trainer.strategy = strategy
        cfg.generator.backend = backend
        cfg.generator.inference_engine_tensor_parallel_size = tp_size

        tokenizer = AutoTokenizer.from_pretrained(MODEL)

        # Spawn the remote inference server
        client, server_process = init_remote_inference_servers(
            tp_size=tp_size,
            backend=backend,
            tokenizer=tokenizer,
            config=cfg,
            model=MODEL,
        )

        # Initialize policy worker
        policy = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,  # Remote engines don't support colocation
            num_gpus_per_node=tp_size,
            cfg=cfg,
        )
        sampling_params = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
        outputs = asyncio.run(run_inference(client, get_test_prompts(MODEL), sampling_params))
        # Sync weights and run inference
        ray.get(policy.async_run_ray_method("pass_through", "init_weight_sync_state", client))
        asyncio.run(client.reset_prefix_cache())
        ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", client))

        # sampling_params = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
        outputs = asyncio.run(run_inference(client, get_test_prompts(MODEL), sampling_params))

        print(f"Example output: {outputs['responses'][0]}, {outputs['stop_reasons'][0]}")

        # Verify we got valid outputs
        assert len(outputs["responses"]) > 0, "Expected at least one response"
        assert all(isinstance(r, str) for r in outputs["responses"]), "All responses should be strings"

    finally:
        # Clean up the server process
        if server_process is not None:
            server_process.terminate()
            server_process.wait(timeout=10)
        ray.shutdown()


