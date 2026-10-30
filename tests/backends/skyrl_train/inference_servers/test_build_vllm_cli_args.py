"""Tests for build_vllm_cli_args on GPU-less hosts."""

import logging
from argparse import Namespace
from types import SimpleNamespace

import pytest

from skyrl.backends.skyrl_train.inference_servers.utils import (
    _apply_serialized_fp8_weight_sync_defaults,
    build_vllm_cli_args,
    get_pd_cli_args,
    get_pd_p2p_connector_name,
    resolve_policy_model_name,
)
from skyrl.train.config import SkyRLTrainConfig


def test_serialized_fp8_weight_sync_defaults_configure_vllm_checkpoint_fp8(monkeypatch):
    import skyrl.backends.skyrl_train.inference_servers.utils as inference_utils

    monkeypatch.setattr(inference_utils, "_serialized_fp8_engine_exclude_list", lambda *_args: [])
    cfg = SkyRLTrainConfig()
    ie_cfg = cfg.generator.inference_engine
    ie_cfg.fp8_weight_sync_mode = "blockwise"
    engine_kwargs = {"hf_overrides": {"rope_theta": 10000.0}}

    _apply_serialized_fp8_weight_sync_defaults(ie_cfg, engine_kwargs, model_path="qwen35-test")

    assert engine_kwargs["quantization"] == "fp8"
    assert engine_kwargs["load_format"] == "dummy"
    assert engine_kwargs["hf_overrides"]["rope_theta"] == 10000.0
    assert engine_kwargs["hf_overrides"]["quantization_config"] == {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }


@pytest.mark.parametrize(
    "engine_kwargs",
    [
        {"quantization": "awq"},
        {"load_format": "safetensors"},
    ],
)
def test_serialized_fp8_weight_sync_rejects_conflicting_vllm_settings(engine_kwargs, monkeypatch):
    import skyrl.backends.skyrl_train.inference_servers.utils as inference_utils

    monkeypatch.setattr(inference_utils, "_serialized_fp8_engine_exclude_list", lambda *_args: [])
    cfg = SkyRLTrainConfig()
    cfg.generator.inference_engine.fp8_weight_sync_mode = "blockwise"

    with pytest.raises(ValueError, match="FP8 weight sync"):
        _apply_serialized_fp8_weight_sync_defaults(
            cfg.generator.inference_engine,
            engine_kwargs,
            model_path="qwen35-test",
        )


def test_serialized_fp8_weight_sync_rejects_non_mapping_overrides():
    cfg = SkyRLTrainConfig()
    cfg.generator.inference_engine.fp8_weight_sync_mode = "blockwise"

    with pytest.raises(ValueError, match="must be a dict"):
        _apply_serialized_fp8_weight_sync_defaults(
            cfg.generator.inference_engine,
            {"hf_overrides": []},
            model_path="qwen35-test",
        )


@pytest.mark.parametrize("user_quantization_config", [{"weight_block_size": [64, 128]}, {"store_dtype": "mxfp4"}, []])
def test_serialized_fp8_replaces_a_user_quantization_config(user_quantization_config, monkeypatch, caplog):
    """FP8 weight sync owns the engine's quantization config: it must describe exactly what the trainer sends."""
    import skyrl.backends.skyrl_train.inference_servers.utils as inference_utils

    monkeypatch.setattr(inference_utils, "_serialized_fp8_engine_exclude_list", lambda *_args: [])
    cfg = SkyRLTrainConfig()
    cfg.generator.inference_engine.fp8_weight_sync_mode = "blockwise"
    engine_kwargs = {"hf_overrides": {"quantization_config": user_quantization_config}}

    with caplog.at_level(logging.WARNING, logger=inference_utils.__name__):
        _apply_serialized_fp8_weight_sync_defaults(
            cfg.generator.inference_engine, engine_kwargs, model_path="qwen35-test"
        )

    assert engine_kwargs["hf_overrides"]["quantization_config"] == {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }
    assert "Ignoring engine_init_kwargs.hf_overrides.quantization_config" in caplog.text


def test_serialized_fp8_requires_model_path():
    cfg = SkyRLTrainConfig()
    cfg.generator.inference_engine.fp8_weight_sync_mode = "blockwise"

    with pytest.raises(ValueError, match="model path is required"):
        _apply_serialized_fp8_weight_sync_defaults(cfg.generator.inference_engine, {})


def test_serialized_fp8_fails_when_model_config_cannot_be_inspected(monkeypatch):
    import transformers

    cfg = SkyRLTrainConfig()
    cfg.generator.inference_engine.fp8_weight_sync_mode = "blockwise"

    def fail_config_load(*_args, **_kwargs):
        raise OSError("missing config")

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", fail_config_load)
    with pytest.raises(RuntimeError, match="Could not inspect the model config"):
        _apply_serialized_fp8_weight_sync_defaults(
            cfg.generator.inference_engine,
            {},
            model_path="missing-model",
        )


def test_serialized_fp8_rejects_unsupported_model_layout(monkeypatch):
    import transformers

    cfg = SkyRLTrainConfig()
    cfg.generator.inference_engine.fp8_weight_sync_mode = "blockwise"
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(model_type="llama"),
    )

    with pytest.raises(ValueError, match="no registered model spec"):
        _apply_serialized_fp8_weight_sync_defaults(
            cfg.generator.inference_engine,
            {},
            model_path="unsupported-model",
        )


@pytest.mark.vllm
def test_build_vllm_cli_args_succeeds_on_gpu_less_host(monkeypatch):
    import vllm.platforms
    from vllm.platforms.interface import UnspecifiedPlatform

    # Simulate the GPU-less Ray head-node case: vLLM resolves current_platform
    # to UnspecifiedPlatform (device_type == ""), so AsyncEngineArgs.add_cli_args
    # walks VllmConfig defaults, instantiates DeviceConfig() and its
    # __post_init__ raises "Failed to infer device type" during arg parsing.
    # With the fix in build_vllm_cli_args, current_platform.device_type is
    # pinned to "cuda" before add_cli_args runs.
    monkeypatch.setattr(vllm.platforms, "_current_platform", UnspecifiedPlatform())

    cfg = SkyRLTrainConfig()
    cfg.generator.inference_engine.served_model_name = "served-alias"
    cfg.generator.inference_engine.engine_init_kwargs = {
        "hf_overrides": {"rope_parameters": {"rope_type": "linear", "factor": 2.0, "rope_theta": 10000.0}}
    }
    args = build_vllm_cli_args(cfg)

    assert args is not None
    assert args.model == cfg.trainer.policy.model.path
    assert args.served_model_name == ["served-alias"]
    assert args.tensor_parallel_size == cfg.generator.inference_engine.tensor_parallel_size
    assert args.hf_overrides["rope_parameters"] == {"rope_type": "linear", "factor": 2.0, "rope_theta": 10000.0}
    assert vllm.platforms.current_platform.device_type == "cuda"

    # NOTE: the MTP speculative_config wiring test lives in
    # tests/backends/skyrl_train/mtp/test_build_vllm_cli_args_mtp.py


@pytest.mark.vllm
def test_default_logprobs_mode_is_processed_logprobs():
    args = build_vllm_cli_args(SkyRLTrainConfig())

    assert args.logprobs_mode == "processed_logprobs"


@pytest.mark.vllm
def test_sample_support_uses_processed_top_k_logprobs():
    cfg = SkyRLTrainConfig.from_cli_overrides(
        [
            "generator.inference_engine.enable_return_sample_support_set=true",
            "generator.sampling_params.top_k=8",
        ]
    )

    args = build_vllm_cli_args(cfg)

    assert args.max_logprobs == 8
    assert args.logprobs_mode == "processed_logprobs"


def _megatron_lora_cfg(*overrides):
    return SkyRLTrainConfig.from_cli_overrides(
        [
            "trainer.strategy=megatron",
            "trainer.policy.model.lora.rank=64",
            "trainer.policy.megatron_config.lora_config.merge_lora=false",
            *overrides,
        ]
    )


_SCALE = "generator.inference_engine.scale_moe_lora_rank=true"
_NORMALIZE = "trainer.policy.megatron_config.lora_config.normalize_moe_lora=true"


@pytest.mark.vllm
@pytest.mark.parametrize(
    "overrides,topk,expected",
    [
        # 64 // top-8.
        ([_SCALE, _NORMALIZE], 8, 8),
        # Rounded up to the expert tensor parallel size, which defaults to the TP size.
        ([_SCALE, _NORMALIZE, "trainer.policy.megatron_config.tensor_model_parallel_size=16"], 8, 16),
        ([_SCALE, _NORMALIZE, "trainer.policy.megatron_config.expert_tensor_parallel_size=64"], 8, None),
        # No top-k in the HF config: the experts keep max_lora_rank.
        ([_SCALE, _NORMALIZE], None, None),
        # Off by default.
        ([_NORMALIZE], 8, None),
    ],
)
def test_scale_moe_lora_rank_is_passed_through_additional_config(monkeypatch, overrides, topk, expected):
    import skyrl.backends.skyrl_train.inference_servers.utils as inference_utils

    monkeypatch.setattr(inference_utils, "_moe_router_topk", lambda _model_path: topk)
    args = build_vllm_cli_args(_megatron_lora_cfg(*overrides))

    assert args.enable_lora and args.max_lora_rank == 64
    assert (args.additional_config or {}).get("max_expert_lora_rank") == expected


@pytest.mark.vllm
def test_scale_moe_lora_rank_merges_with_engine_additional_config(monkeypatch):
    import skyrl.backends.skyrl_train.inference_servers.utils as inference_utils

    monkeypatch.setattr(inference_utils, "_moe_router_topk", lambda _model_path: 8)
    cfg = _megatron_lora_cfg(_SCALE, _NORMALIZE)
    cfg.generator.inference_engine.engine_init_kwargs = {"additional_config": {"other": 1}}
    args = build_vllm_cli_args(cfg)

    assert args.additional_config == {"other": 1, "max_expert_lora_rank": 8}


@pytest.mark.parametrize(
    "hf_config,expected",
    [
        ({"num_experts_per_tok": 8}, 8),
        ({"text_config": {"num_experts_per_tok": 4}}, 4),
        ({"moe_topk": [6]}, 6),
        ({"hidden_size": 16}, None),
    ],
)
def test_moe_router_topk_reads_hf_config(monkeypatch, hf_config, expected):
    import transformers

    import skyrl.backends.skyrl_train.inference_servers.utils as inference_utils

    def from_pretrained(*_args, **_kwargs):
        text_config = hf_config.get("text_config")
        fields = {k: v for k, v in hf_config.items() if k != "text_config"}
        return SimpleNamespace(**fields, text_config=SimpleNamespace(**text_config) if text_config else None)

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", from_pretrained)
    assert inference_utils._moe_router_topk("model") == expected


def test_resolve_policy_model_name_uses_served_model_name():
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = "base-model"
    cfg.generator.inference_engine.served_model_name = "served-alias"

    assert resolve_policy_model_name(cfg) == "served-alias"


def test_serialized_fp8_threads_the_wire_format_into_the_ignore_list(monkeypatch):
    """The engine ignore list is wire-specific, so the resolved wire must reach the spec.

    MXFP8 excludes more modules than blockwise (its kernels need a 32-aligned reduction dim
    and out_features >= 128), and the engine builds quantized modules for everything absent
    from this list, so a blockwise list under an MXFP8 wire builds modules the sync never
    feeds.
    """
    import skyrl.backends.skyrl_train.inference_servers.utils as inference_utils

    seen = []
    monkeypatch.setattr(
        inference_utils,
        "_serialized_fp8_engine_exclude_list",
        lambda _model_path, wire_format, _exclude_modules: seen.append(wire_format) or [],
    )
    cfg = SkyRLTrainConfig()
    ie_cfg = cfg.generator.inference_engine
    ie_cfg.fp8_weight_sync_mode = "mxfp8"
    engine_kwargs = {}

    _apply_serialized_fp8_weight_sync_defaults(ie_cfg, engine_kwargs, model_path="qwen35-test")

    assert seen == ["mxfp8"]
    assert engine_kwargs["quantization"] == "compressed-tensors"


def test_serialized_fp8_exclusions_reach_the_engine_ignore_list(monkeypatch):
    import transformers

    hf_config = SimpleNamespace(
        model_type="qwen3_5_moe",
        text_config=SimpleNamespace(
            model_type="qwen3_5_moe_text",
            layer_types=["linear_attention", "full_attention"],
            shared_expert_intermediate_size=512,
        ),
    )
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda *_args, **_kwargs: hf_config)
    cfg = SkyRLTrainConfig()
    ie_cfg = cfg.generator.inference_engine
    ie_cfg.fp8_weight_sync_mode = "blockwise"
    ie_cfg.fp8_weight_sync_exclude_modules = ["*.layers.1.self_attn.*", "*.layers.0.mlp.experts.*"]
    engine_kwargs = {}

    _apply_serialized_fp8_weight_sync_defaults(ie_cfg, engine_kwargs, model_path="qwen35-test")

    ignored = engine_kwargs["hf_overrides"]["quantization_config"]["ignored_layers"]
    assert set(ignored[-7:]) == {
        "model.language_model.layers.0.mlp.experts.0.gate_proj",
        "model.language_model.layers.0.mlp.experts.0.up_proj",
        "model.language_model.layers.0.mlp.experts.0.down_proj",
        "model.language_model.layers.1.self_attn.q_proj",
        "model.language_model.layers.1.self_attn.k_proj",
        "model.language_model.layers.1.self_attn.v_proj",
        "model.language_model.layers.1.self_attn.o_proj",
    }


class TestGetPDP2PConnectorName:
    """Tests for get_pd_p2p_connector_name."""

    def test_bare_nixl(self):
        assert get_pd_p2p_connector_name({"kv_connector": "NixlConnector"}) == "NixlConnector"

    def test_bare_mooncake(self):
        assert get_pd_p2p_connector_name({"kv_connector": "MooncakeConnector"}) == "MooncakeConnector"

    def test_multiconnector_resolves_single_p2p(self):
        kv_config = {
            "kv_connector": "MultiConnector",
            "kv_connector_extra_config": {
                "connectors": [
                    {"kv_connector": "MooncakeConnector"},
                    {"kv_connector": "MooncakeStoreConnector"},
                ]
            },
        }
        assert get_pd_p2p_connector_name(kv_config) == "MooncakeConnector"

    def test_multiconnector_zero_p2p_raises(self):
        kv_config = {
            "kv_connector": "MultiConnector",
            "kv_connector_extra_config": {"connectors": [{"kv_connector": "MooncakeStoreConnector"}]},
        }
        with pytest.raises(ValueError, match="exactly one P2P transfer connector"):
            get_pd_p2p_connector_name(kv_config)

    def test_multiconnector_two_p2p_raises(self):
        kv_config = {
            "kv_connector": "MultiConnector",
            "kv_connector_extra_config": {
                "connectors": [
                    {"kv_connector": "NixlConnector"},
                    {"kv_connector": "MooncakeConnector"},
                ]
            },
        }
        with pytest.raises(ValueError, match="exactly one P2P transfer connector"):
            get_pd_p2p_connector_name(kv_config)

    def test_unsupported_bare_connector_raises(self):
        with pytest.raises(ValueError, match="Unsupported kv_connector for PD"):
            get_pd_p2p_connector_name({"kv_connector": "SharedStorageConnector"})


class TestGetPDCLIArgs:
    """Tests for get_pd_cli_args role kwargs handling."""

    def test_role_init_kwargs_applied(self):
        args = Namespace()
        role_kwargs = {
            "all2all_backend": "deepep_low_latency",
            "kv_transfer_config": {"kv_connector": "MooncakeConnector"},
        }
        out = get_pd_cli_args(args, role="prefill", role_init_kwargs=role_kwargs)
        assert out.all2all_backend == "deepep_low_latency"
        # Base args namespace is not mutated (deep-copied inside).
        assert not hasattr(args, "all2all_backend")

    def test_kv_role_defaults_to_kv_both(self):
        args = Namespace(kv_transfer_config={"kv_connector": "NixlConnector"})
        out = get_pd_cli_args(args, role="prefill")
        assert out.kv_transfer_config["kv_role"] == "kv_both"

    def test_kv_role_preserved_when_set(self):
        args = Namespace()
        role_kwargs = {"kv_transfer_config": {"kv_connector": "MooncakeConnector", "kv_role": "kv_producer"}}
        out = get_pd_cli_args(args, role="prefill", role_init_kwargs=role_kwargs)
        assert out.kv_transfer_config["kv_role"] == "kv_producer"

        role_kwargs = {"kv_transfer_config": {"kv_connector": "MooncakeConnector", "kv_role": "kv_consumer"}}
        out = get_pd_cli_args(args, role="decode", role_init_kwargs=role_kwargs)
        assert out.kv_transfer_config["kv_role"] == "kv_consumer"

    @pytest.mark.parametrize("role_additional_config", [{"other": 1}, '{"other": 1}'])
    def test_role_additional_config_merges_with_base(self, role_additional_config):
        # The base args carry the derived MoE expert LoRA slot rank; a role's own
        # additional_config must not drop it.
        args = Namespace(
            additional_config={"max_expert_lora_rank": 8},
            kv_transfer_config={"kv_connector": "NixlConnector"},
        )
        out = get_pd_cli_args(args, role="decode", role_init_kwargs={"additional_config": role_additional_config})
        assert out.additional_config == {"max_expert_lora_rank": 8, "other": 1}
        assert args.additional_config == {"max_expert_lora_rank": 8}

    def test_missing_kv_transfer_config_raises(self):
        args = Namespace()
        with pytest.raises(ValueError, match="kv_transfer_config must be set when enable_pd=True"):
            get_pd_cli_args(args, role="decode")
