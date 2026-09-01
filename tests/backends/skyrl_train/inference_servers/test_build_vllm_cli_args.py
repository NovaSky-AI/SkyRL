"""Tests for build_vllm_cli_args on GPU-less hosts."""

from argparse import Namespace

import pytest

from skyrl.backends.skyrl_train.inference_servers.utils import (
    build_vllm_cli_args,
    get_pd_cli_args,
    get_pd_p2p_connector_name,
    resolve_policy_model_name,
)
from skyrl.train.config import SkyRLTrainConfig


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


def test_resolve_policy_model_name_uses_served_model_name():
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = "base-model"
    cfg.generator.inference_engine.served_model_name = "served-alias"

    assert resolve_policy_model_name(cfg) == "served-alias"


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

    def test_missing_kv_transfer_config_raises(self):
        args = Namespace()
        with pytest.raises(ValueError, match="kv_transfer_config must be set when enable_pd=True"):
            get_pd_cli_args(args, role="decode")
