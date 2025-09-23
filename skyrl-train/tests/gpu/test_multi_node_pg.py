"""
Run with:
uv run --isolated --extra dev -- pytest tests/gpu/test_multi_node_pg.py
NOTE: Placement group bundle ordering across nodes only typically has race conditions when using >16 GPUs
so this test is best run with >16 GPUs to actually see that ordering is correct
"""

import ray
import pytest
import hydra
from omegaconf import DictConfig
from ray.util.placement_group import placement_group

from skyrl_train.utils.utils import get_ray_pg_ready_with_timeout
from skyrl_train.workers.fsdp.fsdp_worker import PolicyWorker
from skyrl_train.workers.worker import PPORayActorGroup
from skyrl_train.utils.utils import validate_cfg
from skyrl_train.entrypoints.main_base import config_dir


MODEL_NAME = "Qwen/Qwen3-0.6B"


def get_test_actor_config() -> DictConfig:
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")
    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.generator.weight_sync_backend = "nccl"
    cfg.trainer.strategy = "fsdp2"
    validate_cfg(cfg)

    return cfg


@pytest.fixture
def cfg() -> DictConfig:
    return get_test_actor_config()


def get_pg(placement_group_type, num_gpus_per_node, num_nodes):
    if placement_group_type == "single_gpu_per_bundle":
        pg = placement_group(
            [{"GPU": 1, "CPU": 1}] * num_gpus_per_node * num_nodes,
            strategy="PACK",
        )
        get_ray_pg_ready_with_timeout(pg, timeout=60)
        return pg
    elif placement_group_type == "whole_node_bundle":
        pg = placement_group(
            [{"GPU": num_gpus_per_node, "CPU": num_gpus_per_node}] * num_nodes,
            strategy="PACK",
        )
        get_ray_pg_ready_with_timeout(pg, timeout=60)
        return pg
    elif placement_group_type == "none":
        return None
    else:
        raise ValueError(f"Invalid placement group type: {placement_group_type}")


def test_multi_node_pg_invalid_pg(ray_init_fixture, cfg):
    colocate_all = True
    num_nodes = 2
    num_gpus_per_node = 4
    pg = get_pg("whole_node_bundle", num_gpus_per_node, num_nodes)
    with pytest.raises(
        AssertionError,
        match="if colocate_all is True, the number of bundles in the shared placement group must match the world size",
    ):
        PPORayActorGroup(
            cfg,
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            ray_actor_type=PolicyWorker,
            pg=pg,
            num_gpus_per_actor=0.2,
            colocate_all=colocate_all,
        )


def test_multi_node_pg_errors_no_pg(ray_init_fixture, cfg):
    colocate_all = True
    num_nodes = 2
    num_gpus_per_node = 4
    pg = None
    with pytest.raises(
        AssertionError,
        match="if colocate_all is True, the shared placement group must be provided to PPORayActorGroup",
    ):
        PPORayActorGroup(
            cfg,
            num_nodes=num_nodes,
            num_gpus_per_node=num_gpus_per_node,
            ray_actor_type=PolicyWorker,
            pg=pg,
            num_gpus_per_actor=0.2,
            colocate_all=colocate_all,
        )


@pytest.mark.parametrize(
    ("colocate_all", "placement_group_type"),
    [
        (True, "single_gpu_per_bundle"),
        (False, "single_gpu_per_bundle"),  # this is technically not used in practice, but testing for completeness
        (False, "whole_node_bundle"),
        (False, "none"),
        # (True, "none"), and ("True", "whole_node_bundle") should fail and are tested above in test_multi_node_pg_init_error
    ],
    ids=[
        "colocate_all_single_gpu_per_bundle",
        "not_colocate_all_single_gpu_per_bundle",
        "not_colocate_all_whole_node_bundle",
        "not_colocate_all_none",
    ],
)
def test_multi_node_pg_init(ray_init_fixture, cfg, colocate_all, placement_group_type):
    try:
        cfg.trainer.placement.colocate_all = colocate_all
        cfg.trainer.placement.policy_num_nodes = 2
        cfg.trainer.placement.policy_num_gpus_per_node = 4

        pg = get_pg(
            placement_group_type, cfg.trainer.placement.policy_num_gpus_per_node, cfg.trainer.placement.policy_num_nodes
        )

        policy = PPORayActorGroup(
            cfg,
            num_nodes=cfg.trainer.placement.policy_num_nodes,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            ray_actor_type=PolicyWorker,
            pg=pg,
            num_gpus_per_actor=0.2 if colocate_all else 0.75,
            colocate_all=colocate_all,
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

        num_nodes = len(set(node_ids))
        gpus_per_node = cfg.trainer.placement.policy_num_gpus_per_node
        for i in range(num_nodes):
            node_ids_for_group = node_ids[i * gpus_per_node : (i + 1) * gpus_per_node]
            unique_node_ids = set(node_ids_for_group)
            assert (
                len(unique_node_ids) == 1
            ), f"Node IDs are not consistent for node group {i}. Found: {unique_node_ids}"
    finally:
        ray.shutdown()
