"""
uv run --isolated --extra dev --extra sglang pytest tests/gpu/test_policy_sglang_e2e.py
"""

import pytest
import asyncio
import ray
import hydra
from omegaconf import DictConfig

from tests.gpu.utils import init_worker_with_type, get_test_prompts
from skyrl_train.inference_engines.remote_inference_engine import create_remote_inference_engines
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from ray.util.placement_group import placement_group
from skyrl_train.inference_engines.sglang.sglang_server import SGLangServer
from sglang.srt.server_args import ServerArgs
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl_train.utils import get_ray_pg_ready_with_timeout, initialize_ray
from skyrl_train.entrypoints.main_base import config_dir

model = "Qwen/Qwen2.5-1.5B-Instruct"
tp_size = 2


def get_test_actor_config() -> DictConfig:
    """Get base config with test-specific overrides."""
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")

        # Override specific parameters for test
        cfg.trainer.policy.model.path = model
        cfg.trainer.critic.model.path = ""
        cfg.trainer.placement.policy_num_gpus_per_node = 2

        cfg.trainer.placement.colocate_all = False
        cfg.trainer.train_batch_size = 128
        cfg.generator.backend = "sglang"
        cfg.generator.async_engine = True
        cfg.generator.num_inference_engines = 1
        cfg.generator.inference_engine_tensor_parallel_size = tp_size
        cfg.generator.run_engines_locally = False

        return cfg


async def run_generation(client, prompts):
    engine_input = InferenceEngineInput(prompts=prompts)
    await client.generate(engine_input)


def init_sglang_engines(use_local, tp_size, colocate_all, sampling_params):
    assert not use_local, "SGLang currently does not support local engines"
    assert not colocate_all, "SGLang currently does not support colocation"

    initialize_ray(DictConfig({"generator": {"backend": "sglang"}}))

    if colocate_all:
        pg = placement_group([{"GPU": 1, "CPU": 1}] * tp_size, strategy="PACK")
        get_ray_pg_ready_with_timeout(pg, timeout=30)
        sleep = True
    else:
        pg, sleep = None, False

    def get_free_port():
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    engine_port = get_free_port()

    sglang_pg = placement_group([{"GPU": tp_size, "CPU": tp_size}], strategy="PACK")
    get_ray_pg_ready_with_timeout(sglang_pg, timeout=30)

    scheduling_strategy = PlacementGroupSchedulingStrategy(
        placement_group=sglang_pg,
        placement_group_capture_child_tasks=True,
        placement_group_bundle_index=0,
    )

    SGLangServerRayActor = ray.remote(SGLangServer)
    server_actor = SGLangServerRayActor.options(
        num_gpus=tp_size,
        num_cpus=tp_size,
        scheduling_strategy=scheduling_strategy,
    ).remote(
        ServerArgs(
            model_path=model,
            tp_size=tp_size,
            dtype="bfloat16",
            mem_fraction_static=0.7,
            enable_memory_saver=True,
            base_gpu_id=0,
            gpu_id_step=1,
            port=engine_port,
        )
    )
    server_actor.run_server.remote()

    # Wait for server to come online
    import requests
    import time

    def wait_for_server(url: str, timeout: int = 60, interval: float = 1.0):
        start_time = time.time()
        while True:
            try:
                response = requests.get(f"http://{url}/health_generate")
                if response.ok:
                    return
            except requests.exceptions.ConnectionError:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Server at {url} did not come online within {timeout} seconds")
                time.sleep(interval)

    wait_for_server(f"localhost:{engine_port}")
    print(f"Server at localhost:{engine_port} is online")

    engines = create_remote_inference_engines(
        urls=[f"localhost:{engine_port}"],
        model_name=model,
        engine_backend="sglang",
        tensor_parallel_size=tp_size,
        sampling_params=sampling_params,
    )
    client = InferenceEngineClient(engines)
    if sleep:
        asyncio.run(client.wake_up())

    return client, pg, server_actor


@pytest.mark.parametrize(
    ("weight_sync_backend"),
    [
        ("nccl"),
        ("gloo"),
    ],
    ids=[
        "nccl",
        "gloo",
    ],
)
def test_policy_sglang_e2e(weight_sync_backend):
    """
    Tests initalizing the policy actor group and InferenceEngines, syncing weights, and performing generation.
    """
    cfg = get_test_actor_config()
    cfg.generator.weight_sync_backend = weight_sync_backend

    llm_client, pg, server_actor = init_sglang_engines(
        use_local=cfg.generator.run_engines_locally,
        tp_size=cfg.generator.inference_engine_tensor_parallel_size,
        colocate_all=cfg.trainer.placement.colocate_all,
        sampling_params=get_sampling_params_for_backend("sglang", cfg.generator.sampling_params),
    )
    policy = init_worker_with_type(
        "policy",
        shared_pg=pg,
        colocate_all=cfg.trainer.placement.colocate_all,
        num_gpus_per_node=cfg.generator.inference_engine_tensor_parallel_size,
        cfg=cfg,
    )
    ray.get(policy.async_run_ray_method("pass_through", "init_weight_sync_state", llm_client))
    asyncio.run(llm_client.reset_prefix_cache())
    ray.get(policy.async_run_ray_method("pass_through", "broadcast_to_inference_engines", llm_client))
    asyncio.run(run_generation(llm_client, get_test_prompts(model)))
