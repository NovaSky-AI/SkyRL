"""Tests for VLLMRouter."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from skyrl.backends.skyrl_train.inference_servers.vllm_router import VLLMRouter


def test_not_installed_raises_import_error():
    """start() raises ImportError when vllm-router binary is not on PATH."""
    router = VLLMRouter(["http://localhost:8000"], port=9999)
    with patch("shutil.which", return_value=None):
        with pytest.raises(ImportError, match="vllm-router binary not found"):
            router.start()


def test_empty_servers_raises_value_error():
    """start() raises ValueError when no servers are provided."""
    router = VLLMRouter([], port=9999)
    with pytest.raises(ValueError, match="No servers available"):
        router.start()


def test_build_cmd_default():
    """CLI command is built correctly with default settings."""
    router = VLLMRouter(
        ["http://w1:8000", "http://w2:8000"],
        host="0.0.0.0",
        port=30000,
    )
    cmd = router._build_cmd()
    assert cmd[:1] == ["vllm-router"]
    assert "--host" in cmd and cmd[cmd.index("--host") + 1] == "0.0.0.0"
    assert "--port" in cmd and cmd[cmd.index("--port") + 1] == "30000"
    assert "--policy" in cmd and cmd[cmd.index("--policy") + 1] == "consistent_hash"
    assert "http://w1:8000" in cmd
    assert "http://w2:8000" in cmd


def test_process_exit_during_health_check():
    """start() raises RuntimeError if the process exits before becoming healthy."""
    router = VLLMRouter(["http://localhost:8000"], port=9999)

    mock_process = MagicMock(spec=subprocess.Popen)
    mock_process.poll.return_value = 1  # process exited with code 1
    mock_process.returncode = 1
    mock_process.stdout = MagicMock()
    mock_process.stderr = MagicMock()
    # Make readline return empty immediately so drain threads don't block
    mock_process.stdout.readline.return_value = b""
    mock_process.stderr.readline.return_value = b""

    with patch("shutil.which", return_value="/usr/bin/vllm-router"):
        with patch("subprocess.Popen", return_value=mock_process):
            with pytest.raises(RuntimeError, match="process exited with code 1"):
                router.start()


def test_shutdown_terminates_process():
    """shutdown() sends SIGTERM to the subprocess."""
    router = VLLMRouter(["http://localhost:8000"])

    mock_process = MagicMock(spec=subprocess.Popen)
    mock_process.poll.return_value = None  # still running
    mock_process.wait.return_value = 0
    router._process = mock_process

    router.shutdown()

    mock_process.terminate.assert_called_once()
    mock_process.wait.assert_called_once_with(timeout=5)


def test_shutdown_kills_on_timeout():
    """shutdown() escalates to SIGKILL if SIGTERM doesn't work."""
    router = VLLMRouter(["http://localhost:8000"])

    mock_process = MagicMock(spec=subprocess.Popen)
    mock_process.poll.return_value = None
    mock_process.wait.side_effect = [subprocess.TimeoutExpired("vllm-router", 5), None]
    router._process = mock_process

    router.shutdown()

    mock_process.terminate.assert_called_once()
    mock_process.kill.assert_called_once()
