"""Validation of the colocated weight-sync options (``_validate_colocated_weight_sync_cfg``)."""

import pytest

from skyrl.train.utils.utils import _validate_colocated_weight_sync_cfg
from tests.train.util import example_dummy_config


def _colocated_megatron_cfg():
    cfg = example_dummy_config()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.placement.colocate_all = True
    cfg.generator.inference_engine.weight_sync_backend = "nccl"
    return cfg


def test_defaults_pass_without_colocation():
    cfg = example_dummy_config()
    cfg.trainer.placement.colocate_all = False
    _validate_colocated_weight_sync_cfg(cfg)


@pytest.mark.parametrize(
    "path, value",
    [
        ("generator.inference_engine.weight_sync_num_ipc_buffers", 2),
        ("generator.inference_engine.weight_sync_layer_aligned_chunks", True),
        ("generator.inference_engine.weight_sync_local_shard", True),
        ("trainer.policy.megatron_config.weight_sync_from_offloaded_params", True),
    ],
)
def test_every_option_passes_colocated_and_requires_colocation(path, value):
    cfg = _colocated_megatron_cfg()
    *parents, leaf = path.split(".")
    target = cfg
    for name in parents:
        target = getattr(target, name)
    setattr(target, leaf, value)
    _validate_colocated_weight_sync_cfg(cfg)

    cfg.trainer.placement.colocate_all = False
    with pytest.raises(ValueError, match="colocate_all"):
        _validate_colocated_weight_sync_cfg(cfg)


def test_rejects_a_pull_backend():
    cfg = _colocated_megatron_cfg()
    cfg.generator.inference_engine.weight_sync_local_shard = True
    cfg.generator.inference_engine.weight_sync_backend = "sharded_rdt"
    with pytest.raises(ValueError, match="push weight-sync backend"):
        _validate_colocated_weight_sync_cfg(cfg)


def test_rejects_zero_ipc_buffers():
    cfg = _colocated_megatron_cfg()
    cfg.generator.inference_engine.weight_sync_num_ipc_buffers = 0
    with pytest.raises(ValueError, match=">= 1"):
        _validate_colocated_weight_sync_cfg(cfg)


@pytest.mark.parametrize("merge_lora", [True, False])
def test_offloaded_params_reject_lora(merge_lora):
    cfg = _colocated_megatron_cfg()
    cfg.trainer.policy.megatron_config.weight_sync_from_offloaded_params = True
    cfg.trainer.policy.model.lora.rank = 32
    cfg.trainer.policy.megatron_config.lora_config.merge_lora = merge_lora
    with pytest.raises(ValueError, match="LoRA"):
        _validate_colocated_weight_sync_cfg(cfg)


def test_offloaded_params_require_megatron():
    cfg = _colocated_megatron_cfg()
    cfg.trainer.strategy = "fsdp"
    cfg.trainer.policy.megatron_config.weight_sync_from_offloaded_params = True
    with pytest.raises(ValueError, match="strategy=megatron"):
        _validate_colocated_weight_sync_cfg(cfg)


def test_offloaded_params_reject_overlap_param_gather():
    cfg = _colocated_megatron_cfg()
    cfg.trainer.policy.megatron_config.weight_sync_from_offloaded_params = True
    cfg.trainer.policy.megatron_config.ddp_config.overlap_param_gather = True
    with pytest.raises(ValueError, match="overlap_param_gather"):
        _validate_colocated_weight_sync_cfg(cfg)


def test_serialized_fp8_rejects_local_shard_and_offloaded_params():
    cfg = _colocated_megatron_cfg()
    cfg.generator.inference_engine.fp8_weight_sync_mode = "blockwise"
    _validate_colocated_weight_sync_cfg(cfg)  # the ring options alone are fine
    cfg.generator.inference_engine.weight_sync_local_shard = True
    with pytest.raises(ValueError, match="serialized FP8"):
        _validate_colocated_weight_sync_cfg(cfg)
