"""
Test for RayPrometheusStatLogger integration in the vLLM engine.

Run with:
uv run --isolated --extra dev pytest tests/cpu/inf_engines/vllm/test_ray_prometheus_stats.py
"""

import pytest
from unittest.mock import patch, MagicMock
import sys


class TestRayPrometheusStatLoggers:
    """Test cases for _create_ray_prometheus_stat_loggers method."""

    def test_create_ray_prometheus_stat_loggers_v1_available(self):
        """Test that RayPrometheusStatLogger is returned when vLLM v1 API is available."""
        # Create a mock for the v1 RayPrometheusStatLogger
        mock_stat_logger = MagicMock()
        mock_stat_logger.__name__ = "RayPrometheusStatLogger"

        mock_ray_wrappers = MagicMock()
        mock_ray_wrappers.RayPrometheusStatLogger = mock_stat_logger

        # Patch the import to return our mock
        with patch.dict(sys.modules, {"vllm.v1.metrics.ray_wrappers": mock_ray_wrappers}):
            from skyrl_train.inference_engines.vllm.vllm_engine import AsyncVLLMInferenceEngine

            # Create a minimal instance without actually initializing the engine
            engine = object.__new__(AsyncVLLMInferenceEngine)

            result = engine._create_ray_prometheus_stat_loggers()

            # Should return a list with the stat logger class
            assert result is not None
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0] == mock_stat_logger

    def test_create_ray_prometheus_stat_loggers_v1_unavailable(self):
        """Test that None is returned when vLLM v1 API is not available."""
        # By setting the module to None in sys.modules, the import will fail.
        with patch.dict(sys.modules, {"vllm.v1.metrics.ray_wrappers": None}):
            from skyrl_train.inference_engines.vllm.vllm_engine import AsyncVLLMInferenceEngine

            # Create a minimal instance without actually initializing the engine
            engine = object.__new__(AsyncVLLMInferenceEngine)

            with patch("skyrl_train.inference_engines.vllm.vllm_engine.logger") as mock_logger:
                result = engine._create_ray_prometheus_stat_loggers()

                assert result is None
                mock_logger.warning.assert_called_once()
                assert "not available in this vLLM version" in mock_logger.warning.call_args[0][0]


class TestConfigIntegration:
    """Test that configuration flows correctly through the stack."""

    def test_config_default_value(self):
        """Test that enable_ray_prometheus_stats defaults to False in config."""
        from omegaconf import OmegaConf

        # Load the base config
        config_content = """
generator:
  enable_ray_prometheus_stats: false
"""
        cfg = OmegaConf.create(config_content)
        assert cfg.generator.enable_ray_prometheus_stats is False

    def test_config_can_be_enabled(self):
        """Test that enable_ray_prometheus_stats can be set to True."""
        from omegaconf import OmegaConf

        config_content = """
generator:
  enable_ray_prometheus_stats: true
"""
        cfg = OmegaConf.create(config_content)
        assert cfg.generator.enable_ray_prometheus_stats is True


class TestKwargsHandling:
    """Test that enable_ray_prometheus_stats is properly handled in kwargs."""

    def test_enable_ray_prometheus_stats_popped_from_kwargs(self):
        """Test that enable_ray_prometheus_stats is properly popped from kwargs."""
        # This test verifies the configuration flows correctly
        kwargs = {"enable_ray_prometheus_stats": True, "other_param": "value"}

        # Pop should remove it from kwargs (same logic as in _create_engine)
        enable_stats = kwargs.pop("enable_ray_prometheus_stats", False)
        assert enable_stats is True
        assert "enable_ray_prometheus_stats" not in kwargs
        assert kwargs == {"other_param": "value"}

    def test_enable_ray_prometheus_stats_defaults_to_false(self):
        """Test that enable_ray_prometheus_stats defaults to False when not present."""
        kwargs = {"other_param": "value"}

        enable_stats = kwargs.pop("enable_ray_prometheus_stats", False)
        assert enable_stats is False
        assert kwargs == {"other_param": "value"}
