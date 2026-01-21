"""Unit tests for Fleet task environment."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import DictConfig

from integrations.fleet.env import (
    load_tasks_from_json,
    parse_tool_call,
    FleetTaskEnv,
    _TASK_CACHE,
)


class TestLoadTasksFromJson:
    """Tests for load_tasks_from_json function."""

    def setup_method(self):
        """Clear cache before each test."""
        _TASK_CACHE.clear()

    def test_load_array_format(self, tmp_path):
        """Test loading tasks from array format JSON."""
        tasks_file = tmp_path / "tasks.json"
        tasks_data = [
            {"key": "task-1", "prompt": "Do task 1", "env_id": "env-a"},
            {"key": "task-2", "prompt": "Do task 2", "env_id": "env-b"},
        ]
        tasks_file.write_text(json.dumps(tasks_data))

        result = load_tasks_from_json(str(tasks_file))

        assert "task-1" in result
        assert "task-2" in result
        assert result["task-1"]["prompt"] == "Do task 1"
        assert result["task-2"]["env_id"] == "env-b"

    def test_load_object_format(self, tmp_path):
        """Test loading tasks from object format with 'tasks' key."""
        tasks_file = tmp_path / "tasks.json"
        tasks_data = {
            "tasks": [
                {"task_key": "task-a", "prompt": "Prompt A"},
                {"task_key": "task-b", "prompt": "Prompt B"},
            ]
        }
        tasks_file.write_text(json.dumps(tasks_data))

        result = load_tasks_from_json(str(tasks_file))

        assert "task-a" in result
        assert "task-b" in result

    def test_caching(self, tmp_path):
        """Test that tasks are cached after first load."""
        tasks_file = tmp_path / "tasks.json"
        tasks_data = [{"key": "task-1", "prompt": "Test"}]
        tasks_file.write_text(json.dumps(tasks_data))

        # First load
        result1 = load_tasks_from_json(str(tasks_file))

        # Modify file (but cache should be used)
        tasks_file.write_text(json.dumps([{"key": "task-new", "prompt": "New"}]))

        # Second load should return cached data
        result2 = load_tasks_from_json(str(tasks_file))

        assert result1 is result2
        assert "task-1" in result2
        assert "task-new" not in result2

    def test_invalid_format(self, tmp_path):
        """Test error on invalid JSON format."""
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps({"invalid": "format"}))

        with pytest.raises(ValueError, match="Invalid JSON format"):
            load_tasks_from_json(str(tasks_file))

    def test_key_priority(self, tmp_path):
        """Test that 'key' takes priority over 'task_key'."""
        tasks_file = tmp_path / "tasks.json"
        tasks_data = [
            {"key": "primary-key", "task_key": "secondary-key", "prompt": "Test"},
        ]
        tasks_file.write_text(json.dumps(tasks_data))

        result = load_tasks_from_json(str(tasks_file))

        assert "primary-key" in result
        assert "secondary-key" not in result


class TestParseToolCall:
    """Tests for parse_tool_call function."""

    def test_tool_call_tag(self):
        """Test parsing <tool_call> format."""
        action = 'I will search. <tool_call>{"name": "search", "arguments": {"query": "test"}}</tool_call>'
        result = parse_tool_call(action)

        assert result is not None
        assert result["name"] == "search"
        assert result["arguments"] == {"query": "test"}

    def test_function_call_tag(self):
        """Test parsing <function_call> format."""
        action = '<function_call>{"name": "compute", "arguments": {"x": 42}}</function_call>'
        result = parse_tool_call(action)

        assert result is not None
        assert result["name"] == "compute"
        assert result["arguments"] == {"x": 42}

    def test_tool_key_normalization(self):
        """Test that 'tool' key is normalized to 'name'."""
        action = '<tool_call>{"tool": "my_tool", "params": {"a": 1}}</tool_call>'
        result = parse_tool_call(action)

        assert result is not None
        assert result["name"] == "my_tool"
        assert result["arguments"] == {"a": 1}

    def test_multiline_json(self):
        """Test parsing multiline JSON in tool call."""
        action = """<tool_call>
{
    "name": "complex_tool",
    "arguments": {
        "list": [1, 2, 3],
        "nested": {"key": "value"}
    }
}
</tool_call>"""
        result = parse_tool_call(action)

        assert result is not None
        assert result["name"] == "complex_tool"
        assert result["arguments"]["list"] == [1, 2, 3]

    def test_no_tool_call(self):
        """Test that None is returned when no tool call found."""
        action = "I don't know how to help with that."
        result = parse_tool_call(action)

        assert result is None

    def test_invalid_json(self):
        """Test handling of invalid JSON in tags."""
        action = '<tool_call>{"name": invalid json}</tool_call>'
        result = parse_tool_call(action)

        assert result is None

    def test_empty_arguments(self):
        """Test tool call with no arguments."""
        action = '<tool_call>{"name": "get_time", "arguments": {}}</tool_call>'
        result = parse_tool_call(action)

        assert result is not None
        assert result["name"] == "get_time"
        assert result["arguments"] == {}

    def test_missing_name(self):
        """Test that missing name returns None."""
        action = '<tool_call>{"arguments": {"x": 1}}</tool_call>'
        result = parse_tool_call(action)

        assert result is None


class TestFleetTaskEnvInit:
    """Tests for FleetTaskEnv initialization."""

    def setup_method(self):
        """Clear cache before each test."""
        _TASK_CACHE.clear()

    def test_missing_task_key(self, tmp_path):
        """Test error when task_key is not provided."""
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps([{"key": "test", "prompt": "Test"}]))

        env_config = DictConfig({"tasks_file": str(tasks_file)})

        with pytest.raises(ValueError, match="task_key must be provided"):
            FleetTaskEnv(env_config, extras={})

    def test_missing_tasks_file(self):
        """Test error when tasks_file is not provided."""
        env_config = DictConfig({})

        with pytest.raises(ValueError, match="tasks_file must be provided"):
            FleetTaskEnv(env_config, extras={"task_key": "test"})

    def test_task_not_found(self, tmp_path):
        """Test error when task_key doesn't exist in file."""
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps([{"key": "other-task", "prompt": "Test"}]))

        env_config = DictConfig({"tasks_file": str(tasks_file)})

        with pytest.raises(ValueError, match="Task 'missing-task' not found"):
            FleetTaskEnv(env_config, extras={"task_key": "missing-task"})

    @patch.dict(os.environ, {"FLEET_API_KEY": "test-api-key"})
    def test_successful_init(self, tmp_path):
        """Test successful initialization."""
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps([{"key": "my-task", "prompt": "Do something", "env_id": "test-env"}]))

        env_config = DictConfig({"tasks_file": str(tasks_file)})
        env = FleetTaskEnv(env_config, extras={"task_key": "my-task"})

        assert env.task_key == "my-task"
        assert env.task_config["prompt"] == "Do something"
        assert env.api_key == "test-api-key"
        assert env.max_turns == 50  # default

    @patch.dict(os.environ, {"FLEET_API_KEY": "env-key"})
    def test_api_key_from_config(self, tmp_path):
        """Test that config api_key takes priority over env var."""
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps([{"key": "task", "prompt": "Test"}]))

        env_config = DictConfig({"tasks_file": str(tasks_file), "api_key": "config-key"})
        env = FleetTaskEnv(env_config, extras={"task_key": "task"})

        assert env.api_key == "config-key"

    def test_max_turns_from_extras(self, tmp_path):
        """Test that max_turns can be set via extras."""
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps([{"key": "task", "prompt": "Test"}]))

        env_config = DictConfig({"tasks_file": str(tasks_file)})
        env = FleetTaskEnv(env_config, extras={"task_key": "task", "max_turns": 10})

        assert env.max_turns == 10


class TestFleetTaskEnvMetrics:
    """Tests for FleetTaskEnv metrics methods."""

    def setup_method(self):
        """Clear cache before each test."""
        _TASK_CACHE.clear()

    @patch.dict(os.environ, {"FLEET_API_KEY": "test-key"})
    def test_get_metrics(self, tmp_path):
        """Test get_metrics returns correct structure."""
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps([{"key": "task-1", "prompt": "Test", "env_id": "github"}]))

        env_config = DictConfig({"tasks_file": str(tasks_file)})
        env = FleetTaskEnv(env_config, extras={"task_key": "task-1"})
        env.turns = 5

        metrics = env.get_metrics()

        assert metrics["task_key"] == "task-1"
        assert metrics["env_key"] == "github"
        assert metrics["turns"] == 5

    def test_aggregate_metrics_empty(self):
        """Test aggregate_metrics with empty list."""
        result = FleetTaskEnv.aggregate_metrics([])
        assert result == {}

    def test_aggregate_metrics(self):
        """Test aggregate_metrics calculates correctly."""
        metrics = [
            {"task_key": "t1", "env_key": "github", "turns": 10},
            {"task_key": "t2", "env_key": "github", "turns": 20},
            {"task_key": "t3", "env_key": "booking", "turns": 15},
        ]

        result = FleetTaskEnv.aggregate_metrics(metrics)

        assert result["avg_turns"] == 15.0
        assert result["total_episodes"] == 3
        assert result["env_distribution"]["github"] == 2
        assert result["env_distribution"]["booking"] == 1


class TestFleetTaskEnvInitMethod:
    """Tests for FleetTaskEnv.init() method."""

    def setup_method(self):
        """Clear cache before each test."""
        _TASK_CACHE.clear()

    @patch("integrations.fleet.env.OpenEnvFleetTaskEnv")
    @patch.dict(os.environ, {"FLEET_API_KEY": "test-key"})
    def test_init_creates_fleet_env(self, mock_fleet_env_class, tmp_path):
        """Test that init() creates and resets the Fleet environment."""
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps([{"key": "task-1", "prompt": "Search for flights", "env_id": "booking"}]))

        # Mock the Fleet environment
        mock_fleet_env = MagicMock()
        mock_fleet_env.reset.return_value = {"tools": [], "observation": {}}
        mock_fleet_env_class.return_value = mock_fleet_env

        env_config = DictConfig({"tasks_file": str(tasks_file)})
        env = FleetTaskEnv(env_config, extras={"task_key": "task-1"})

        # Call init
        chat_history, metadata = env.init([])

        # Verify Fleet env was created
        mock_fleet_env_class.assert_called_once()
        mock_fleet_env.reset.assert_called_once()

        # Verify return values
        assert len(chat_history) == 1
        assert chat_history[0]["role"] == "user"
        assert "Search for flights" in chat_history[0]["content"]
        assert metadata["task_key"] == "task-1"
        assert metadata["env_key"] == "booking"

    @patch("integrations.fleet.env.OpenEnvFleetTaskEnv")
    @patch.dict(os.environ, {"FLEET_API_KEY": "test-key"})
    def test_init_includes_tools_info(self, mock_fleet_env_class, tmp_path):
        """Test that init() includes tool information in prompt."""
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps([{"key": "task-1", "prompt": "Do something", "env_id": "test"}]))

        # Mock with tools
        mock_fleet_env = MagicMock()
        mock_fleet_env.reset.return_value = {
            "tools": [
                {"function": {"name": "search"}},
                {"function": {"name": "click"}},
            ],
            "observation": {},
        }
        mock_fleet_env_class.return_value = mock_fleet_env

        env_config = DictConfig({"tasks_file": str(tasks_file)})
        env = FleetTaskEnv(env_config, extras={"task_key": "task-1"})

        chat_history, metadata = env.init([])

        # Verify tools are mentioned in prompt
        assert "search" in chat_history[0]["content"]
        assert "click" in chat_history[0]["content"]
        assert "<tool_call>" in chat_history[0]["content"]
        assert "<done>" in chat_history[0]["content"]

        # Verify tools in metadata
        assert len(metadata["tools"]) == 2


class TestFleetTaskEnvStep:
    """Tests for FleetTaskEnv.step() method."""

    def setup_method(self):
        """Clear cache before each test."""
        _TASK_CACHE.clear()

    @patch("integrations.fleet.env.OpenEnvFleetTaskEnv")
    @patch("integrations.fleet.env._run_async")
    @patch.dict(os.environ, {"FLEET_API_KEY": "test-key"})
    def test_step_with_tool_call(self, mock_run_async, mock_fleet_env_class, tmp_path):
        """Test step() with a valid tool call."""
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps([{"key": "task-1", "prompt": "Test", "env_id": "test"}]))

        # Setup mocks
        mock_fleet_env = MagicMock()
        mock_fleet_env.reset.return_value = {"tools": [], "observation": {}}
        mock_fleet_env_class.return_value = mock_fleet_env

        mock_run_async.return_value = (
            {"observation": "Search results: found 5 items"},
            0.0,
            False,
            {"done_reason": None},
        )

        env_config = DictConfig({"tasks_file": str(tasks_file)})
        env = FleetTaskEnv(env_config, extras={"task_key": "task-1"})
        env.init([])

        # Step with tool call
        action = '<tool_call>{"name": "search", "arguments": {"q": "test"}}</tool_call>'
        result = env.step(action)

        # Verify step was called
        mock_run_async.assert_called_once()

        # Verify result
        assert result["done"] is False
        assert len(result["observations"]) == 1
        assert "Search results" in result["observations"][0]["content"]
        assert result["metadata"]["tool_call"]["name"] == "search"

    @patch("integrations.fleet.env.OpenEnvFleetTaskEnv")
    @patch("integrations.fleet.env._run_async")
    @patch.dict(os.environ, {"FLEET_API_KEY": "test-key"})
    def test_step_with_done_signal(self, mock_run_async, mock_fleet_env_class, tmp_path):
        """Test step() when agent signals done."""
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps([{"key": "task-1", "prompt": "Test", "env_id": "test"}]))

        mock_fleet_env = MagicMock()
        mock_fleet_env.reset.return_value = {"tools": [], "observation": {}}
        mock_fleet_env_class.return_value = mock_fleet_env

        mock_run_async.return_value = (
            {"observation": {}},
            1.0,  # Reward on completion
            True,
            {"done_reason": "agent_done"},
        )

        env_config = DictConfig({"tasks_file": str(tasks_file)})
        env = FleetTaskEnv(env_config, extras={"task_key": "task-1"})
        env.init([])

        # Step with done signal
        result = env.step("Task completed! <done>")

        assert result["done"] is True
        assert result["reward"] == 1.0

    @patch("integrations.fleet.env.OpenEnvFleetTaskEnv")
    @patch("integrations.fleet.env._run_async")
    @patch.dict(os.environ, {"FLEET_API_KEY": "test-key"})
    def test_step_max_turns(self, mock_run_async, mock_fleet_env_class, tmp_path):
        """Test step() when max turns reached."""
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps([{"key": "task-1", "prompt": "Test", "env_id": "test"}]))

        mock_fleet_env = MagicMock()
        mock_fleet_env.reset.return_value = {"tools": [], "observation": {}}
        mock_fleet_env_class.return_value = mock_fleet_env

        mock_run_async.return_value = (
            {"observation": {}},
            0.0,
            False,
            {},
        )

        env_config = DictConfig({"tasks_file": str(tasks_file)})
        env = FleetTaskEnv(env_config, extras={"task_key": "task-1", "max_turns": 2})
        env.init([])

        # First step
        env.step("action 1")
        assert env.turns == 1

        # Second step (max turns)
        result = env.step("action 2")

        assert result["done"] is True
        assert result["metadata"]["done_reason"] == "max_turns"
        assert result["observations"] == []

    @patch("integrations.fleet.env.OpenEnvFleetTaskEnv")
    @patch("integrations.fleet.env._run_async")
    @patch.dict(os.environ, {"FLEET_API_KEY": "test-key"})
    def test_step_no_tool_call(self, mock_run_async, mock_fleet_env_class, tmp_path):
        """Test step() when no tool call is found."""
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps([{"key": "task-1", "prompt": "Test", "env_id": "test"}]))

        mock_fleet_env = MagicMock()
        mock_fleet_env.reset.return_value = {"tools": [], "observation": {}}
        mock_fleet_env_class.return_value = mock_fleet_env

        mock_run_async.return_value = (
            {"observation": None},
            0.0,
            False,
            {},
        )

        env_config = DictConfig({"tasks_file": str(tasks_file)})
        env = FleetTaskEnv(env_config, extras={"task_key": "task-1"})
        env.init([])

        # Step without tool call
        result = env.step("I'm not sure what to do")

        assert "No tool call found" in result["observations"][0]["content"]


class TestFleetTaskEnvClose:
    """Tests for FleetTaskEnv.close() method."""

    def setup_method(self):
        """Clear cache before each test."""
        _TASK_CACHE.clear()

    @patch("integrations.fleet.env.OpenEnvFleetTaskEnv")
    @patch.dict(os.environ, {"FLEET_API_KEY": "test-key"})
    def test_close(self, mock_fleet_env_class, tmp_path):
        """Test close() calls fleet_env.close()."""
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps([{"key": "task-1", "prompt": "Test", "env_id": "test"}]))

        mock_fleet_env = MagicMock()
        mock_fleet_env.reset.return_value = {"tools": [], "observation": {}}
        mock_fleet_env_class.return_value = mock_fleet_env

        env_config = DictConfig({"tasks_file": str(tasks_file)})
        env = FleetTaskEnv(env_config, extras={"task_key": "task-1"})
        env.init([])

        env.close()

        mock_fleet_env.close.assert_called_once()
        assert env.fleet_env is None

    @patch("integrations.fleet.env.OpenEnvFleetTaskEnv")
    @patch.dict(os.environ, {"FLEET_API_KEY": "test-key"})
    def test_close_handles_error(self, mock_fleet_env_class, tmp_path, capsys):
        """Test close() handles errors gracefully."""
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps([{"key": "task-1", "prompt": "Test", "env_id": "test"}]))

        mock_fleet_env = MagicMock()
        mock_fleet_env.reset.return_value = {"tools": [], "observation": {}}
        mock_fleet_env.close.side_effect = Exception("Connection error")
        mock_fleet_env_class.return_value = mock_fleet_env

        env_config = DictConfig({"tasks_file": str(tasks_file)})
        env = FleetTaskEnv(env_config, extras={"task_key": "task-1"})
        env.init([])

        # Should not raise
        env.close()

        # Should print warning
        captured = capsys.readouterr()
        assert "Warning" in captured.out
        assert env.fleet_env is None
