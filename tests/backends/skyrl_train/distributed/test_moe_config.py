"""Tests for MoE config fields on MegatronConfig dataclass."""

from skyrl.train.config.config import MegatronConfig, build_nested_dataclass


class TestMegatronConfigMoEFields:
    """Verify the 5 new MoE config fields exist with correct types and defaults."""

    def test_moe_fields_exist(self):
        cfg = MegatronConfig()
        assert hasattr(cfg, "moe_token_dispatcher_type")
        assert hasattr(cfg, "moe_router_load_balancing_type")
        assert hasattr(cfg, "moe_grouped_gemm")
        assert hasattr(cfg, "moe_router_score_function")
        assert hasattr(cfg, "moe_router_enable_expert_bias")

    def test_moe_field_defaults(self):
        cfg = MegatronConfig()
        assert cfg.moe_token_dispatcher_type == "alltoall"
        assert cfg.moe_router_load_balancing_type == "none"
        assert cfg.moe_grouped_gemm is False
        assert cfg.moe_router_score_function is None
        assert cfg.moe_router_enable_expert_bias is None

    def test_moe_fields_override(self):
        cfg = MegatronConfig(
            moe_token_dispatcher_type="allgather",
            moe_router_load_balancing_type="aux_loss",
            moe_grouped_gemm=True,
            moe_router_score_function="sigmoid",
            moe_router_enable_expert_bias=True,
        )
        assert cfg.moe_token_dispatcher_type == "allgather"
        assert cfg.moe_router_load_balancing_type == "aux_loss"
        assert cfg.moe_grouped_gemm is True
        assert cfg.moe_router_score_function == "sigmoid"
        assert cfg.moe_router_enable_expert_bias is True

    def test_moe_config_from_dict(self):
        """MoE fields should survive dict -> dataclass round-trip."""
        d = {
            "moe_token_dispatcher_type": "alltoall",
            "moe_router_load_balancing_type": "none",
            "moe_grouped_gemm": True,
            "moe_router_score_function": "sigmoid",
            "moe_router_enable_expert_bias": True,
        }
        cfg = build_nested_dataclass(MegatronConfig, d)
        assert cfg.moe_grouped_gemm is True
        assert cfg.moe_router_score_function == "sigmoid"
        assert cfg.moe_router_enable_expert_bias is True

    def test_backward_compatible_defaults(self):
        """Default values must match the old hardcoded values for backward compat."""
        cfg = MegatronConfig()
        assert cfg.moe_token_dispatcher_type == "alltoall"
        assert cfg.moe_router_load_balancing_type == "none"

    def test_parallelism_fields_unchanged(self):
        """Existing parallelism fields should still work."""
        cfg = MegatronConfig(
            tensor_model_parallel_size=4,
            pipeline_model_parallel_size=2,
            expert_model_parallel_size=8,
        )
        assert cfg.tensor_model_parallel_size == 4
        assert cfg.pipeline_model_parallel_size == 2
        assert cfg.expert_model_parallel_size == 8
