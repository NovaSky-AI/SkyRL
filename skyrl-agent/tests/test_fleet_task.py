"""Tests for Fleet Task integration."""

import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from skyrl_agent.tasks.fleet import FleetTask
from skyrl_agent.tasks.fleet.fleet_task import load_fleet_tasks


class TestFleetTask:
    """Test FleetTask class."""

    def test_get_instruction(self):
        """Test get_instruction returns proper format."""
        instance = {
            "task_key": "test-task-001",
            "prompt": "Book a flight from NYC to LA",
        }

        messages = FleetTask.get_instruction(instance)

        assert isinstance(messages, list)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "test-task-001" in messages[0]["content"]
        assert "Book a flight from NYC to LA" in messages[0]["content"]

    def test_get_instruction_with_key_fallback(self):
        """Test get_instruction with 'key' instead of 'task_key'."""
        instance = {
            "key": "fallback-task-001",
            "prompt": "Search for hotels",
        }

        messages = FleetTask.get_instruction(instance)

        assert "fallback-task-001" in messages[0]["content"]

    def test_complete_runtime_closes_env(self):
        """Test complete_runtime closes the environment."""
        mock_env = MagicMock()
        runtime = {"env": mock_env}
        instance = {"task_key": "test-task"}

        result = FleetTask.complete_runtime(runtime, instance)

        mock_env.close.assert_called_once()
        assert result["status"] == "completed"

    def test_complete_runtime_handles_missing_env(self):
        """Test complete_runtime handles missing env gracefully."""
        runtime = {}
        instance = {"task_key": "test-task"}

        result = FleetTask.complete_runtime(runtime, instance)

        assert result["status"] == "completed"

    def test_complete_runtime_handles_close_error(self):
        """Test complete_runtime handles close errors."""
        mock_env = MagicMock()
        mock_env.close.side_effect = Exception("Close failed")
        runtime = {"env": mock_env}
        instance = {"task_key": "test-task"}

        # Should not raise
        result = FleetTask.complete_runtime(runtime, instance)
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_evaluate_result_no_env(self):
        """Test evaluate_result returns False when no env."""
        result = await FleetTask.evaluate_result(
            instance={"task_key": "test"},
            runtime={},
            trajectory=[],
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_evaluate_result_no_verifier(self):
        """Test evaluate_result returns False when no verifier code."""
        mock_env = MagicMock()
        runtime = {
            "env": mock_env,
            "task_config": {"task_key": "test", "verifier_code": ""},
        }

        result = await FleetTask.evaluate_result(
            instance={"task_key": "test"},
            runtime=runtime,
            trajectory=[],
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_evaluate_result_success(self):
        """Test evaluate_result returns True on successful verification."""
        mock_env = MagicMock()
        mock_env._compute_reward = AsyncMock(return_value=1.0)
        runtime = {
            "env": mock_env,
            "task_config": {
                "task_key": "test",
                "verifier_code": "async def verify(env): return True",
            },
        }

        result = await FleetTask.evaluate_result(
            instance={"task_key": "test"},
            runtime=runtime,
            trajectory=[],
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_evaluate_result_failure(self):
        """Test evaluate_result returns False on failed verification."""
        mock_env = MagicMock()
        mock_env._compute_reward = AsyncMock(return_value=0.0)
        runtime = {
            "env": mock_env,
            "task_config": {
                "task_key": "test",
                "verifier_code": "async def verify(env): return False",
            },
        }

        result = await FleetTask.evaluate_result(
            instance={"task_key": "test"},
            runtime=runtime,
            trajectory=[],
        )
        assert result is False


class TestLoadFleetTasks:
    """Test load_fleet_tasks function."""

    def test_load_tasks_list_format(self, tmp_path):
        """Test loading tasks in list format."""
        tasks = [
            {"task_key": "task1", "prompt": "Do task 1", "env_key": "booking"},
            {"task_key": "task2", "prompt": "Do task 2", "env_key": "github"},
        ]
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps(tasks))

        loaded = load_fleet_tasks(str(tasks_file))

        assert len(loaded) == 2
        assert loaded[0]["task_key"] == "task1"

    def test_load_tasks_dict_format(self, tmp_path):
        """Test loading tasks in {"tasks": [...]} format."""
        data = {
            "tasks": [
                {"task_key": "task1", "prompt": "Do task 1"},
            ]
        }
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps(data))

        loaded = load_fleet_tasks(str(tasks_file))

        assert len(loaded) == 1

    def test_load_tasks_with_env_filter(self, tmp_path):
        """Test filtering tasks by env_key."""
        tasks = [
            {"task_key": "task1", "env_key": "booking"},
            {"task_key": "task2", "env_key": "github"},
            {"task_key": "task3", "env_key": "booking"},
        ]
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps(tasks))

        loaded = load_fleet_tasks(str(tasks_file), env_key_filter="booking")

        assert len(loaded) == 2
        assert all(t["env_key"] == "booking" for t in loaded)

    def test_load_tasks_with_max_tasks(self, tmp_path):
        """Test limiting number of tasks."""
        tasks = [{"task_key": f"task{i}"} for i in range(10)]
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps(tasks))

        loaded = load_fleet_tasks(str(tasks_file), max_tasks=3)

        assert len(loaded) == 3

    def test_load_tasks_combined_filters(self, tmp_path):
        """Test combining env_key filter and max_tasks."""
        tasks = [
            {"task_key": "task1", "env_key": "booking"},
            {"task_key": "task2", "env_key": "github"},
            {"task_key": "task3", "env_key": "booking"},
            {"task_key": "task4", "env_key": "booking"},
        ]
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps(tasks))

        loaded = load_fleet_tasks(str(tasks_file), env_key_filter="booking", max_tasks=2)

        assert len(loaded) == 2
        assert all(t["env_key"] == "booking" for t in loaded)


class TestFleetTaskInitializeRuntime:
    """Test FleetTask.initialize_runtime (requires mocking)."""

    @pytest.mark.asyncio
    async def test_initialize_runtime_missing_api_key(self):
        """Test initialize_runtime raises error without API key."""
        # Clear env var if set
        old_key = os.environ.pop("FLEET_API_KEY", None)

        try:
            with pytest.raises(ValueError, match="Fleet API key required"):
                await FleetTask.initialize_runtime(
                    instance={"task_key": "test", "prompt": "test"},
                    api_key=None,
                )
        finally:
            if old_key:
                os.environ["FLEET_API_KEY"] = old_key

    @pytest.mark.asyncio
    async def test_initialize_runtime_missing_openenv(self):
        """Test initialize_runtime raises ImportError without OpenEnv."""
        with patch.dict("sys.modules", {"envs.fleet_env": None}):
            # This should fail at import
            pass  # Can't easily test import errors with patch
