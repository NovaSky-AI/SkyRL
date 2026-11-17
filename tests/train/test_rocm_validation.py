"""CPU tests for what config validation rejects on AMD ROCm, and that a CUDA build is unaffected.

ROCm is simulated by setting ``torch.version.hip``, the build attribute ``is_rocm`` reads.

uv run --isolated --extra skyrl-train --extra dev pytest tests/train/test_rocm_validation.py
"""

import pytest
import torch

from skyrl.train.utils import utils as train_utils
from skyrl.train.utils.utils import (
    validate_inference_engine_cfg,
    validate_megatron_cfg,
)
from tests.train.util import example_dummy_config

_NOSET_ENV_VARS = (
    "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
)


@pytest.fixture
def rocm(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.15.26333", raising=False)


@pytest.fixture
def not_rocm(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", None, raising=False)


@pytest.fixture
def ray_masks_gpus(monkeypatch):
    """Ray's default: it writes a per-process GPU mask."""
    for name in _NOSET_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _megatron_cfg():
    cfg = example_dummy_config()
    cfg.trainer.policy_mini_batch_size = cfg.trainer.train_batch_size
    cfg.trainer.critic_mini_batch_size = cfg.trainer.train_batch_size
    cfg.trainer.strategy = "megatron"
    return cfg


def _non_colocated_multi_gpu_engine_cfg():
    cfg = _megatron_cfg()
    cfg.trainer.placement.colocate_all = False
    cfg.generator.inference_engine.weight_sync_backend = "nccl"
    cfg.generator.inference_engine.tensor_parallel_size = 2
    return cfg


def test_rocm_rejects_sharded_rdt_weight_sync(rocm):
    cfg = _megatron_cfg()
    cfg.generator.inference_engine.weight_sync_backend = "sharded_rdt"

    with pytest.raises(ValueError, match="Not supported on AMD ROCm: weight_sync_backend='sharded_rdt'"):
        validate_inference_engine_cfg(cfg)


def test_rocm_rejects_fp8_weight_sync(rocm):
    cfg = _megatron_cfg()
    cfg.generator.inference_engine.fp8_weight_sync_mode = "blockwise"

    with pytest.raises(ValueError, match="Not supported on AMD ROCm: FP8"):
        validate_inference_engine_cfg(cfg)


@pytest.mark.parametrize("worker", ["policy", "ref"])
@pytest.mark.parametrize("key, value", [("fp8", "hybrid"), ("fp8_param", True)])
def test_rocm_rejects_megatron_fp8_before_a_recipe_is_resolved(rocm, monkeypatch, worker, key, value):
    def fail(*_args, **_kwargs):
        raise AssertionError("an FP8 recipe was resolved from compute capability on ROCm")

    monkeypatch.setattr(train_utils, "resolve_auto_fp8_recipe", fail)
    cfg = _megatron_cfg()
    getattr(cfg.trainer, worker).megatron_config.transformer_config_kwargs[key] = value

    with pytest.raises(ValueError, match="Not supported on AMD ROCm: FP8"):
        validate_megatron_cfg(cfg)


def test_rocm_rejects_a_per_module_precision_file(rocm):
    """The file selects quantization recipes per module without the global FP8 switches."""
    cfg = _megatron_cfg()
    cfg.trainer.policy.megatron_config.te_precision_config_file = "/path/to/precision.yaml"

    with pytest.raises(ValueError, match="te_precision_config_file"):
        validate_megatron_cfg(cfg)


def test_rocm_rejects_nccl_to_a_non_colocated_multi_gpu_engine_under_ray_masks(rocm, ray_masks_gpus):
    cfg = _non_colocated_multi_gpu_engine_cfg()

    with pytest.raises(ValueError, match="RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1"):
        validate_megatron_cfg(cfg)


def test_rocm_rejects_nccl_to_a_non_colocated_pipeline_parallel_engine_under_ray_masks(rocm, ray_masks_gpus):
    """Pipeline parallelism spreads an engine over several GPUs just as tensor parallelism does."""
    cfg = _non_colocated_multi_gpu_engine_cfg()
    cfg.generator.inference_engine.tensor_parallel_size = 1
    cfg.generator.inference_engine.pipeline_parallel_size = 2

    with pytest.raises(ValueError, match="RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1"):
        validate_megatron_cfg(cfg)


@pytest.mark.parametrize("value", ["1", "true", "True"])
def test_rocm_accepts_that_engine_when_ray_does_not_mask_amd_gpus(rocm, ray_masks_gpus, monkeypatch, value):
    monkeypatch.setenv("RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES", value)

    validate_megatron_cfg(_non_colocated_multi_gpu_engine_cfg())


def test_rocm_still_rejects_that_engine_with_only_the_cuda_noset_variable(rocm, ray_masks_gpus, monkeypatch):
    """On an AMD node Ray honours the HIP variable alone, so the CUDA one leaves the masks in place."""
    monkeypatch.setenv("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")

    with pytest.raises(ValueError, match="RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1"):
        validate_megatron_cfg(_non_colocated_multi_gpu_engine_cfg())


@pytest.mark.parametrize(
    "overrides",
    [
        {"tensor_parallel_size": 1},
        {"weight_sync_backend": "delta"},
    ],
)
def test_rocm_accepts_the_non_colocated_layouts_that_connect(rocm, ray_masks_gpus, overrides):
    cfg = _non_colocated_multi_gpu_engine_cfg()
    for name, value in overrides.items():
        setattr(cfg.generator.inference_engine, name, value)

    validate_megatron_cfg(cfg)


def test_rocm_accepts_a_colocated_multi_gpu_engine_under_ray_masks(rocm, ray_masks_gpus):
    cfg = _non_colocated_multi_gpu_engine_cfg()
    cfg.trainer.placement.colocate_all = True

    validate_megatron_cfg(cfg)


@pytest.mark.parametrize("noset_env_var", [None, *_NOSET_ENV_VARS])
def test_rocm_megatron_rejects_the_mp_executor_whatever_rays_masking(rocm, ray_masks_gpus, monkeypatch, noset_env_var):
    """With ``mp`` SkyRL confines the engine itself, so turning Ray's masking off does not help."""
    if noset_env_var is not None:
        monkeypatch.setenv(noset_env_var, "1")
    cfg = _megatron_cfg()
    cfg.generator.inference_engine.distributed_executor_backend = "mp"

    with pytest.raises(ValueError, match="distributed_executor_backend='mp' is not supported with the Megatron"):
        validate_megatron_cfg(cfg)


def test_rocm_keeps_the_mp_executor_outside_the_megatron_backend(rocm, ray_masks_gpus):
    """The Tinker + FSDP launcher for AMD defaults to ``mp``; only the Megatron path rejects it."""
    cfg = _megatron_cfg()
    cfg.trainer.strategy = "fsdp"
    cfg.generator.inference_engine.distributed_executor_backend = "mp"

    validate_inference_engine_cfg(cfg)


def test_rocm_accepts_the_default_megatron_config(rocm, ray_masks_gpus):
    cfg = _megatron_cfg()

    validate_inference_engine_cfg(cfg)
    validate_megatron_cfg(cfg)


def test_cuda_build_keeps_the_layouts_rocm_rejects(not_rocm, ray_masks_gpus):
    cfg = _non_colocated_multi_gpu_engine_cfg()
    cfg.generator.inference_engine.distributed_executor_backend = "mp"
    validate_megatron_cfg(cfg)

    cfg.generator.inference_engine.weight_sync_backend = "sharded_rdt"
    validate_inference_engine_cfg(cfg)
