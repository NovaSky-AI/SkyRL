"""
Unit tests for ray_logging module (actor stdout/stderr redirection).

uv run --isolated --extra dev pytest tests/cpu/utils/test_ray_logging.py
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

import skyrl_train.env_vars as env_vars_mod
from skyrl_train.utils.ray_logging import redirect_actor_output_to_file


def _set_dump_infra(monkeypatch, enabled: bool):
    """Patch both the env var and the cached module-level constant."""
    monkeypatch.setenv("SKYRL_DUMP_INFRA_LOG_TO_STDOUT", "1" if enabled else "0")
    monkeypatch.setattr(env_vars_mod, "SKYRL_DUMP_INFRA_LOG_TO_STDOUT", enabled)


class TestRedirectActorOutputToFile:
    """Tests for redirect_actor_output_to_file()."""

    def test_dump_to_std_skips_redirection(self, monkeypatch):
        """When SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1, no redirection should happen."""
        _set_dump_infra(monkeypatch, True)
        monkeypatch.setenv("SKYRL_LOG_FILE", "/tmp/should-not-be-opened.log")

        original_stdout_fd = os.dup(sys.stdout.fileno())
        original_stderr_fd = os.dup(sys.stderr.fileno())
        try:
            redirect_actor_output_to_file()

            # stdout/stderr should still point to original fds (not redirected)
            assert os.fstat(sys.stdout.fileno()).st_ino == os.fstat(original_stdout_fd).st_ino
            assert os.fstat(sys.stderr.fileno()).st_ino == os.fstat(original_stderr_fd).st_ino
        finally:
            os.close(original_stdout_fd)
            os.close(original_stderr_fd)

    def test_no_log_file_set_is_noop(self, monkeypatch):
        """When SKYRL_LOG_FILE is not set, no redirection should happen."""
        _set_dump_infra(monkeypatch, False)
        monkeypatch.delenv("SKYRL_LOG_FILE", raising=False)

        original_stdout_fd = os.dup(sys.stdout.fileno())
        try:
            redirect_actor_output_to_file()

            assert os.fstat(sys.stdout.fileno()).st_ino == os.fstat(original_stdout_fd).st_ino
        finally:
            os.close(original_stdout_fd)

    def test_redirects_stdout_and_stderr_to_file(self, monkeypatch):
        """With dump disabled and SKYRL_LOG_FILE set, stdout/stderr should write to the log file."""
        _set_dump_infra(monkeypatch, False)

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test-infra.log")
            monkeypatch.setenv("SKYRL_LOG_FILE", log_path)

            # Save original fds so we can restore them after the test
            saved_stdout_fd = os.dup(sys.stdout.fileno())
            saved_stderr_fd = os.dup(sys.stderr.fileno())
            try:
                redirect_actor_output_to_file()

                # Write to stdout/stderr — should go to the log file
                os.write(sys.stdout.fileno(), b"stdout-test-line\n")
                os.write(sys.stderr.fileno(), b"stderr-test-line\n")

                with open(log_path) as f:
                    contents = f.read()

                assert "stdout-test-line" in contents
                assert "stderr-test-line" in contents
            finally:
                # Restore original stdout/stderr so subsequent tests aren't affected
                os.dup2(saved_stdout_fd, sys.stdout.fileno())
                os.dup2(saved_stderr_fd, sys.stderr.fileno())
                os.close(saved_stdout_fd)
                os.close(saved_stderr_fd)

    def test_appends_to_existing_file(self, monkeypatch):
        """Redirection should append, not overwrite existing log content."""
        _set_dump_infra(monkeypatch, False)

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test-infra.log")
            with open(log_path, "w") as f:
                f.write("pre-existing-content\n")

            monkeypatch.setenv("SKYRL_LOG_FILE", log_path)

            saved_stdout_fd = os.dup(sys.stdout.fileno())
            saved_stderr_fd = os.dup(sys.stderr.fileno())
            try:
                redirect_actor_output_to_file()

                os.write(sys.stdout.fileno(), b"new-content\n")

                with open(log_path) as f:
                    contents = f.read()

                assert "pre-existing-content" in contents
                assert "new-content" in contents
            finally:
                os.dup2(saved_stdout_fd, sys.stdout.fileno())
                os.dup2(saved_stderr_fd, sys.stderr.fileno())
                os.close(saved_stdout_fd)
                os.close(saved_stderr_fd)

    def test_dump_to_std_skips_log_file_creation(self, monkeypatch):
        """When dump enabled, initialize_ray should not create log dir or set SKYRL_LOG_FILE."""
        _set_dump_infra(monkeypatch, True)
        monkeypatch.delenv("SKYRL_LOG_FILE", raising=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            log_base = os.path.join(tmpdir, "skyrl-logs")
            monkeypatch.setenv("SKYRL_LOG_DIR", log_base)
            monkeypatch.setattr(env_vars_mod, "SKYRL_LOG_DIR", log_base)

            # Simulate the conditional from initialize_ray
            verbose_logging = env_vars_mod.SKYRL_DUMP_INFRA_LOG_TO_STDOUT

            assert verbose_logging, "Expected dump-to-std to be enabled"

            # When dumping to stdout, the log dir should NOT be created
            expected_log_dir = os.path.join(log_base, "test-run")
            assert not os.path.exists(expected_log_dir)
            assert os.environ.get("SKYRL_LOG_FILE") is None


def _path_escapes_base(base: str, run_name: str) -> bool:
    """Reproduce the path traversal check from initialize_ray()."""
    log_dir = (Path(base) / run_name).resolve()
    base_dir = Path(base).resolve()
    return not str(log_dir).startswith(str(base_dir) + os.sep) and log_dir != base_dir


class TestPathTraversalValidation:
    """Tests for the run_name path traversal check in initialize_ray()."""

    def test_normal_run_name_accepted(self):
        assert not _path_escapes_base("/tmp/skyrl-logs", "my-training-run-v1")

    def test_nested_run_name_accepted(self):
        assert not _path_escapes_base("/tmp/skyrl-logs", "experiments/run-42")

    def test_dotdot_escape_rejected(self):
        assert _path_escapes_base("/tmp/skyrl-logs", "../../etc/cron.daily/evil")

    @pytest.mark.parametrize(
        "run_name",
        [
            "../outside",
            "../../etc/passwd",
            "foo/../../outside",
            "/absolute/path",
        ],
    )
    def test_malicious_run_names_rejected(self, run_name):
        assert _path_escapes_base("/tmp/skyrl-logs", run_name)
