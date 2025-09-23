"""
Run with:
uv run --isolated --extra dev -- pytest tests/gpu/test_multi_node_pg.py
"""

import ray
import pytest
import hydra
from omegaconf import DictConfig
from ray.util.placement_group import placement_group
from skyrl_train.utils.utils import get_ray_pg_ready_with_timeout

from tests.gpu.utils import (
    init_worker_with_type,
)

from skyrl_train.utils.utils import validate_cfg
from skyrl_train.entrypoints.main_base import config_dir


MODEL_NAME = "Qwen/Qwen3-0.6B"


def get_test_actor_config() -> DictConfig:
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")
    cfg.trainer.policy.model.path = MODEL_NAME
    validate_cfg(cfg)

    return cfg


@pytest.fixture
def cfg() -> DictConfig:
    return get_test_actor_config()


def test_multi_node_pg_init(ray_init_fixture, cfg):
    try:
        cfg = get_test_actor_config()
        cfg.trainer.placement.colocate_all = True
        cfg.generator.weight_sync_backend = "nccl"
        cfg.trainer.strategy = "fsdp"
        cfg.trainer.placement.policy_num_nodes = 5
        cfg.trainer.placement.policy_num_gpus_per_node = 4

        pg = placement_group(
            [{"GPU": 1, "CPU": 1}]
            * cfg.trainer.placement.policy_num_gpus_per_node
            * cfg.trainer.placement.policy_num_nodes,
            strategy="PACK",
        )
        get_ray_pg_ready_with_timeout(pg, timeout=60)

        policy = init_worker_with_type(
            "policy",
            shared_pg=pg,
            colocate_all=cfg.trainer.placement.colocate_all,
            num_nodes=cfg.trainer.placement.policy_num_nodes,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )

        # get info from policy workers
        mesh_ranks = [ray.get(actor.get_mesh_rank.remote()) for actor in policy._actor_handlers]
        gpu_ids = [ray.get(actor.get_gpu_id.remote()) for actor in policy._actor_handlers]
        node_ids = [ray.get(actor.get_ray_node_id.remote()) for actor in policy._actor_handlers]

        # use dp rank in mesh rank as proxy for world rank to verify correct layout
        for rank, mesh_rank in enumerate(mesh_ranks):
            assert rank == mesh_rank.dp, f"Mesh rank {mesh_rank} has incorrect dp rank"
            assert (
                rank % cfg.trainer.placement.policy_num_gpus_per_node == gpu_ids[rank]
            ), f"Mesh rank {mesh_rank} has incorrect gpu id"

        # node ids should be in order
        for i in range(len(set(node_ids))):
            j = 0
            node_id = node_ids[i * cfg.trainer.placement.policy_num_gpus_per_node]
            while j < cfg.trainer.placement.policy_num_gpus_per_node:
                assert (
                    node_id == node_ids[i * cfg.trainer.placement.policy_num_gpus_per_node + j]
                ), f"Node id {node_id} has incorrect node id"
                j += 1
    finally:
        ray.shutdown()
