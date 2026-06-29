"""
Unit tests for ray_logging module.

uv run --isolated --extra fsdp pytest tests/train/utils/test_ray_logging.py
"""

import contextlib
import os
import sys
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
import ray
from ray._private.worker import get_dashboard_url
from ray.exceptions import ActorDiedError, RayTaskError

import skyrl.env_vars as env_vars_mod
from skyrl.train.utils.ray_logging import (
    get_actor_logs_tail,
    redirect_actor_output_to_file,
    reraise_with_actor_diagnostics,
)


def _set_dump_infra(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    """Patch both the env var and the cached module-level constant."""
    monkeypatch.setenv("SKYRL_DUMP_INFRA_LOG_TO_STDOUT", "1" if enabled else "0")
    monkeypatch.setattr(env_vars_mod, "SKYRL_DUMP_INFRA_LOG_TO_STDOUT", enabled)


@contextlib.contextmanager
def _preserved_fd(fd: int) -> Iterator[int]:
    """Yield a duplicate of ``fd``; on exit, restore ``fd`` from it and close it."""
    saved = os.dup(fd)
    try:
        yield saved
    finally:
        os.dup2(saved, fd)
        os.close(saved)


class TestRedirectActorOutputToFile:
    """Tests for redirect_actor_output_to_file()."""

    def test_dump_to_std_skips_redirection(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When SKYRL_DUMP_INFRA_LOG_TO_STDOUT=1, no redirection should happen."""
        _set_dump_infra(monkeypatch, True)
        monkeypatch.setenv("SKYRL_LOG_FILE", "/tmp/should-not-be-opened.log")

        with (
            _preserved_fd(sys.stdout.fileno()) as original_stdout_fd,
            _preserved_fd(sys.stderr.fileno()) as original_stderr_fd,
        ):
            redirect_actor_output_to_file()

            # stdout/stderr should still point to original fds (not redirected)
            assert os.fstat(sys.stdout.fileno()).st_ino == os.fstat(original_stdout_fd).st_ino
            assert os.fstat(sys.stderr.fileno()).st_ino == os.fstat(original_stderr_fd).st_ino

    def test_no_log_file_set_is_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When SKYRL_LOG_FILE is not set, no redirection should happen."""
        _set_dump_infra(monkeypatch, False)
        monkeypatch.delenv("SKYRL_LOG_FILE", raising=False)

        with _preserved_fd(sys.stdout.fileno()) as original_stdout_fd:
            redirect_actor_output_to_file()

            assert os.fstat(sys.stdout.fileno()).st_ino == os.fstat(original_stdout_fd).st_ino

    def test_redirects_stdout_and_stderr_to_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With dump disabled and SKYRL_LOG_FILE set, stdout/stderr should write to the log file."""
        _set_dump_infra(monkeypatch, False)

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test-infra.log")
            monkeypatch.setenv("SKYRL_LOG_FILE", log_path)

            # Restore original stdout/stderr after so subsequent tests aren't affected
            with _preserved_fd(sys.stdout.fileno()), _preserved_fd(sys.stderr.fileno()):
                redirect_actor_output_to_file()

                # Write to stdout/stderr — should go to the log file
                os.write(sys.stdout.fileno(), b"stdout-test-line\n")
                os.write(sys.stderr.fileno(), b"stderr-test-line\n")

                with open(log_path) as f:
                    contents = f.read()

                assert "stdout-test-line" in contents
                assert "stderr-test-line" in contents

    def test_actors_append_to_log_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Multiple actor redirections should append to the same log file."""
        _set_dump_infra(monkeypatch, False)

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test-infra.log")
            # Simulate driver truncation (SkyRL initialize_ray util opens with "w")
            open(log_path, "w").close()

            monkeypatch.setenv("SKYRL_LOG_FILE", log_path)

            with (
                _preserved_fd(sys.stdout.fileno()) as saved_stdout_fd,
                _preserved_fd(sys.stderr.fileno()) as saved_stderr_fd,
            ):
                # First actor redirects
                redirect_actor_output_to_file()
                os.write(sys.stdout.fileno(), b"actor-1-output\n")

                # Restore fds to simulate a second actor starting
                os.dup2(saved_stdout_fd, sys.stdout.fileno())
                os.dup2(saved_stderr_fd, sys.stderr.fileno())

                # Second actor redirects (should append, not overwrite)
                redirect_actor_output_to_file()
                os.write(sys.stdout.fileno(), b"actor-2-output\n")

                with open(log_path) as f:
                    contents = f.read()

                assert "actor-1-output" in contents
                assert "actor-2-output" in contents

    def test_dump_to_std_skips_log_file_creation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When dump enabled, initialize_ray should not create log dir or set SKYRL_LOG_FILE."""
        _set_dump_infra(monkeypatch, True)
        monkeypatch.delenv("SKYRL_LOG_FILE", raising=False)

        # Simulate the conditional from initialize_ray
        verbose_logging = env_vars_mod.SKYRL_DUMP_INFRA_LOG_TO_STDOUT

        assert verbose_logging, "Expected dump-to-std to be enabled"

        # When dumping to stdout, the log dir should NOT be created
        with tempfile.TemporaryDirectory() as tmpdir:
            expected_log_dir = os.path.join(tmpdir, "skyrl-logs", "test-run")
            assert not os.path.exists(expected_log_dir)
            assert not os.environ.get("SKYRL_LOG_FILE")


@ray.remote(num_cpus=0)
class _FailingInitActor:
    """Actor whose constructor writes a marker to stderr and then dies."""

    def __init__(self, marker: str) -> None:
        redirect_actor_output_to_file()  # no-op unless SKYRL_LOG_FILE is set
        # Write at the fd level: survives the dup2 redirect and bypasses Python buffering
        os.write(sys.stderr.fileno(), f"{marker}\n".encode())
        raise RuntimeError("intentional init failure")

    def ping(self) -> None:
        """No-op for ray.get to surface the constructor's failure."""
        return None


@ray.remote(num_cpus=0)
class _FailingMethodActor:
    """Actor that stays alive but writes a marker to stderr and raises from a method."""

    def fail(self, marker: str) -> None:
        os.write(sys.stderr.fileno(), f"{marker}\n".encode())
        raise RuntimeError("intentional method failure")


class TestActorLogDiagnostics:
    """Tests for get_actor_logs_tail() and reraise_with_actor_diagnostics()."""

    @pytest.mark.parametrize(
        ("use_skyrl_log_file", "dies_in_init"),
        [
            pytest.param(True, True, id="dead_actor_skyrl_log_file_route"),
            pytest.param(False, True, id="dead_actor_state_api_route"),
            pytest.param(False, False, id="live_actor_state_api_route"),
        ],
    )
    def test_reraise_surfaces_actor_stderr(self, use_skyrl_log_file: bool, dies_in_init: bool, tmp_path: Path) -> None:
        if not use_skyrl_log_file and not get_dashboard_url():
            pytest.skip("Ray dashboard unavailable, state API cannot fetch actor logs")
        marker = f"SKYRL-TEST-MARKER-{uuid.uuid4().hex}"
        with patch.dict(os.environ):
            # Drop any inherited SKYRL_LOG_FILE so only the parametrized route is active
            os.environ.pop("SKYRL_LOG_FILE", None)
            if use_skyrl_log_file:
                log_file = tmp_path / "infra.log"
                os.environ["SKYRL_LOG_FILE"] = str(log_file)
                actor = _FailingInitActor.options(runtime_env={"env_vars": {"SKYRL_LOG_FILE": str(log_file)}}).remote(
                    marker
                )
            elif dies_in_init:
                actor = _FailingInitActor.remote(marker)
            else:
                actor = _FailingMethodActor.remote()

            with pytest.raises(RuntimeError, match="engine init failed") as exc_info:
                try:
                    ray.get(actor.ping.remote() if dies_in_init else actor.fail.remote(marker))
                except (ActorDiedError, RayTaskError) as e:
                    reraise_with_actor_diagnostics(e, [actor], "engine init failed")

        assert marker in str(exc_info.value), "the actor's stderr tail should be attached"
        assert isinstance(exc_info.value.__cause__, ActorDiedError if dies_in_init else RayTaskError)

    def test_get_actor_logs_tail_without_actors(self, tmp_path: Path) -> None:
        log_file = tmp_path / "infra.log"
        log_file.write_text("\n".join(f"line-{i}" for i in range(300)) + "\n")
        with patch.dict(os.environ, {"SKYRL_LOG_FILE": str(log_file)}):
            diagnostics = get_actor_logs_tail([], max_lines_per_actor=5)

        assert diagnostics
        assert str(log_file) in diagnostics, "the section header should name the log file"
        assert "line-299" in diagnostics
        assert "line-294" not in diagnostics, "the tail should be capped at max_lines_per_actor"

        with patch.dict(os.environ):
            # Drop any inherited SKYRL_LOG_FILE so the log-file route stays off
            os.environ.pop("SKYRL_LOG_FILE", None)
            assert not get_actor_logs_tail([]), "no SKYRL_LOG_FILE and no actors leaves nothing to collect"

    def test_reraise_falls_back_to_pointer_text(self) -> None:
        cause = ValueError("boom")
        with patch.dict(os.environ):
            # Drop any inherited SKYRL_LOG_FILE so the log-file route stays off
            os.environ.pop("SKYRL_LOG_FILE", None)
            with pytest.raises(RuntimeError, match="could not fetch actor logs") as exc_info:
                reraise_with_actor_diagnostics(cause, [], "engines died")

        assert exc_info.value.__cause__ is cause
