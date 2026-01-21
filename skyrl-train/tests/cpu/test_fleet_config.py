"""Tests for Fleet task configuration in SkyRL.

These tests verify that the Hydra config schema includes fleet_task settings.
"""

import os

import yaml


class TestFleetTaskConfig:
    """Tests for fleet_task config in skyrl_gym_config."""

    def test_fleet_task_config_exists(self):
        """Test that fleet_task config exists in default.yaml."""
        config_path = os.path.join(
            os.path.dirname(__file__),
            "../../skyrl_train/config/skyrl_gym_config/default.yaml",
        )

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        assert "fleet_task" in config, "fleet_task section missing from config"

    def test_fleet_task_config_has_required_fields(self):
        """Test that fleet_task config has all required fields."""
        config_path = os.path.join(
            os.path.dirname(__file__),
            "../../skyrl_train/config/skyrl_gym_config/default.yaml",
        )

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        fleet_task = config["fleet_task"]

        # Required fields
        assert "tasks_file" in fleet_task, "tasks_file field missing"
        assert "api_key" in fleet_task, "api_key field missing"
        assert "ttl_seconds" in fleet_task, "ttl_seconds field missing"

    def test_fleet_task_config_default_values(self):
        """Test that fleet_task config has correct default values."""
        config_path = os.path.join(
            os.path.dirname(__file__),
            "../../skyrl_train/config/skyrl_gym_config/default.yaml",
        )

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        fleet_task = config["fleet_task"]

        # Default values
        assert fleet_task["tasks_file"] is None, "tasks_file should default to None"
        assert fleet_task["api_key"] is None, "api_key should default to None"
        assert fleet_task["ttl_seconds"] == 600, "ttl_seconds should default to 600"
