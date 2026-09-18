from unittest.mock import patch

import torch

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import get_ray_init_num_gpus, validate_inference_engine_cfg


def test_ray_init_gpu_count_is_per_node(monkeypatch):
    monkeypatch.delenv("NUM_GPUS", raising=False)
    cfg = SkyRLTrainConfig()
    cfg.trainer.placement.policy_num_gpus_per_node = 4
    cfg.trainer.placement.policy_num_nodes = 3
    cfg.trainer.placement.ref_num_gpus_per_node = 2
    cfg.trainer.placement.ref_num_nodes = 3

    assert get_ray_init_num_gpus(cfg) == 4


def test_ray_init_gpu_count_allows_container_override(monkeypatch):
    monkeypatch.setenv("NUM_GPUS", "2")

    assert get_ray_init_num_gpus(SkyRLTrainConfig()) == 2


def test_rocm_colocate_switches_vllm_executor_to_mp():
    cfg = SkyRLTrainConfig()
    cfg.trainer.placement.colocate_all = True
    cfg.generator.inference_engine.distributed_executor_backend = "ray"

    with patch.object(torch.version, "hip", "6.4.43484", create=True):
        validate_inference_engine_cfg(cfg)

    assert cfg.generator.inference_engine.distributed_executor_backend == "mp"


def test_rocm_colocate_keeps_ray_for_multinode_vllm():
    cfg = SkyRLTrainConfig()
    cfg.trainer.placement.colocate_all = True
    cfg.trainer.placement.policy_num_gpus_per_node = 2
    cfg.trainer.placement.policy_num_nodes = 2
    cfg.generator.inference_engine.tensor_parallel_size = 4
    cfg.generator.inference_engine.distributed_executor_backend = "ray"

    with patch.object(torch.version, "hip", "6.4.43484", create=True):
        validate_inference_engine_cfg(cfg)

    assert cfg.generator.inference_engine.distributed_executor_backend == "ray"
