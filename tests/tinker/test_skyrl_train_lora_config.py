import pytest

from skyrl.backends.skyrl_train_lora import (
    resolve_skyrl_train_lora_config,
    skyrl_train_lora_signature,
)
from skyrl.tinker import types


def test_resolves_fsdp_lora_train_flags_to_target_modules():
    cfg = types.LoraConfig(
        rank=8,
        alpha=32.0,
        seed=123,
        train_attn=True,
        train_mlp=False,
        train_unembed=True,
    )

    resolved = resolve_skyrl_train_lora_config(cfg, strategy="fsdp2")

    assert resolved.target_modules == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "query_key_value",
        "attn.c_attn",
        "attn.c_proj",
        "lm_head",
        "embed_out",
        "output_projection",
    ]


def test_resolves_legacy_attn_mlp_surface_to_all_linear():
    cfg = types.LoraConfig(rank=8, alpha=32.0, seed=123, train_attn=True, train_mlp=True, train_unembed=False)

    resolved = resolve_skyrl_train_lora_config(cfg, strategy="fsdp2")

    assert resolved.target_modules == "all-linear"


def test_resolves_megatron_canonical_lora_train_flags_to_target_modules():
    cfg = types.LoraConfig(
        rank=8,
        alpha=32.0,
        seed=123,
        train_attn=False,
        train_mlp=True,
        train_unembed=True,
    )

    resolved = resolve_skyrl_train_lora_config(cfg, strategy="megatron", lora_type="canonical_lora")

    assert resolved.target_modules == [
        "linear_fc1_up",
        "linear_fc1_gate",
        "linear_fc2",
        "output_layer",
    ]


def test_skyrl_train_lora_signature_includes_trainable_surface():
    cfg_a = types.LoraConfig(rank=8, alpha=32.0, seed=1, train_attn=True, train_mlp=True, train_unembed=False)
    cfg_b = types.LoraConfig(rank=8, alpha=32.0, seed=1, train_attn=True, train_mlp=True, train_unembed=True)

    assert skyrl_train_lora_signature(cfg_a, strategy="fsdp2") != skyrl_train_lora_signature(
        cfg_b, strategy="fsdp2"
    )


def test_megatron_rejects_train_unembed_with_pipeline_parallelism():
    cfg = types.LoraConfig(rank=8, alpha=32.0, seed=123, train_unembed=True)

    with pytest.raises(ValueError, match="pipeline_model_parallel_size"):
        resolve_skyrl_train_lora_config(cfg, strategy="megatron", pipeline_parallel_size=2)
