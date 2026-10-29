"""bf16 meta init (sharded-load path) on the NVFP4 checkpoint layout: packed uint8 weights and the
FP32 scale buffers keep their dtype, and the model loads after ``to_empty``."""

import json
import os

import pytest
import torch


@pytest.fixture(scope="module")
def ml4_nvfp4_model_dir(tmp_path_factory):
    from conftest import ML4_TRAIN_PATH

    if not ML4_TRAIN_PATH:
        pytest.skip("ML4_TRAIN_PATH not set")
    from fake_nvfp4 import write_fake_nvfp4_ckpt
    from ml4_train.convert import convert

    root = tmp_path_factory.mktemp("ml4nvfp4")
    ckpt, model_dir = str(root / "ckpt"), str(root / "model")
    write_fake_nvfp4_ckpt(ckpt)
    convert(ckpt, model_dir)
    cfg_path = os.path.join(model_dir, "config.json")
    cfg = json.load(open(cfg_path))
    cfg.update(rollout_tp=1, rollout_moe_tp=1, rollout_moe_activation="nvfp4_per_token")
    json.dump(cfg, open(cfg_path, "w"))
    return model_dir


def test_bf16_meta_init_keeps_packed_and_fp32_tensors(ml4_nvfp4_model_dir):
    from ml4_train.loading import load_sharded
    from ml4_train.lora import LORA_TARGET_REGEX

    from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper

    wrapper = HFModelWrapper(
        ml4_nvfp4_model_dir,
        bf16=True,
        lora_rank=4,
        lora_alpha=8,
        target_modules=LORA_TARGET_REGEX,
        model_config_kwargs={"attn_implementation": "eager"},
        meta_init=True,
        meta_init_adapters=True,
    )
    model = wrapper.model
    # remote code (trust_remote_code) has its own copy of the class
    experts = [m for m in model.modules() if type(m).__name__ == "NVFP4Linear"]
    assert experts
    for m in experts:
        assert m.weight.dtype == torch.uint8 and m.weight_scale.dtype == torch.uint8
        assert m.weight_scale_2.dtype == torch.float32 and m.input_scale.dtype == torch.float32
    frozen_float = [p for n, p in model.named_parameters() if p.is_floating_point() and "lora_" not in n]
    assert frozen_float and all(p.dtype == torch.bfloat16 for p in frozen_float)

    load_sharded(model, ml4_nvfp4_model_dir)
    assert all(m.weight_scale_2.dtype == torch.float32 and not m.weight.is_meta for m in experts)
    assert any(m.weight_scale_2.item() != 1.0 for m in experts)  # values came from the checkpoint
