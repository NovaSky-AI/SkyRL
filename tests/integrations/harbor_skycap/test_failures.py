"""Failure classification without a sandbox, model, or Harbor installation."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.train_integrations.harbor_skycap.failures import (
    AGENT_LOG_TAIL_BYTES,
    FailureClassifierError,
    TrialFailure,
    classify_failure,
    default_failure_classifier,
    load_failure_classifier,
)

pytestmark = pytest.mark.integrations
MODULE = "examples.train_integrations.harbor_skycap.failures"


def result(exception=None, message="", trial_uri=None):
    return SimpleNamespace(
        exception_info=SimpleNamespace(exception_type=exception, exception_message=message) if exception else None,
        trial_uri=trial_uri,
    )


@pytest.mark.parametrize(
    "exception,expected",
    [
        (None, None),
        ("ContextLengthExceededError", TrialFailure.CONTEXT_LENGTH),
        ("ContextWindowExceededError", TrialFailure.CONTEXT_LENGTH),
        ("AgentTimeoutError", TrialFailure.AGENT_TIMEOUT),
        ("EnvironmentStartTimeoutError", None),
        ("VerifierTimeoutError", None),
        ("NonZeroAgentExitCodeError", None),
        ("UnknownHarnessError", None),
    ],
)
def test_structured_exceptions(exception, expected):
    assert default_failure_classifier(result(exception)) is expected


@pytest.mark.parametrize(
    "message", ["context_length_exceeded", "context\n_length_exceeded", "ContextWindowExceededError"]
)
def test_installed_agent_message(message):
    assert default_failure_classifier(result("NonZeroAgentExitCodeError", message)) is TrialFailure.CONTEXT_LENGTH
    # An infrastructure failure mentioning the same text is not a context-limit failure.
    assert default_failure_classifier(result("EnvironmentStartTimeoutError", message)) is None


def test_log_fallback_is_bounded_and_handles_unreadable_files(tmp_path, monkeypatch):
    agent = tmp_path / "trial with spaces" / "agent"
    agent.mkdir(parents=True)
    bad = agent / "a-unreadable.txt"
    bad.write_text("context_length_exceeded")
    log = agent / "b-harness.txt"
    log.write_bytes(b"context_length_exceeded\n" + b"x" * AGENT_LOG_TAIL_BYTES)
    open_file = Path.open

    def open_or_deny(path, *args, **kwargs):
        if path == bad:
            raise PermissionError("unreadable log")
        return open_file(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_or_deny)
    trial = result("NonZeroAgentExitCodeError", trial_uri=agent.parent.as_uri())
    assert default_failure_classifier(trial) is None
    with log.open("ab") as f:
        f.write(b"\xff context\n_length_exceeded")
    assert default_failure_classifier(trial) is TrialFailure.CONTEXT_LENGTH


@pytest.mark.parametrize("uri", [None, "", "s3://bucket/trial", "file:///missing/trial"])
def test_unavailable_logs(uri):
    assert default_failure_classifier(result("NonZeroAgentExitCodeError", trial_uri=uri)) is None


def test_only_installed_agent_exit_reads_logs(tmp_path):
    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / "harness.txt").write_text("context_length_exceeded")
    for exception in (None, "EnvironmentStartTimeoutError", "VerifierTimeoutError"):
        assert default_failure_classifier(result(exception, trial_uri=tmp_path.as_uri())) is None


def test_load_classifier():
    assert load_failure_classifier(None) is default_failure_classifier
    assert load_failure_classifier(f"{MODULE}:default_failure_classifier") is default_failure_classifier


@pytest.mark.parametrize("spec", ["", "default", ":function", "module:"])
def test_bad_import_syntax(spec):
    with pytest.raises(ValueError, match="pkg.module:function"):
        load_failure_classifier(spec)


def test_missing_module_and_function():
    with pytest.raises(ModuleNotFoundError):
        load_failure_classifier("missing_harbor_classifier:classify")
    with pytest.raises(AttributeError):
        load_failure_classifier(f"{MODULE}:missing_classifier")


async def async_classifier(result):
    return None


@pytest.mark.parametrize("spec", [f"{MODULE}:AGENT_LOG_TAIL_BYTES", f"{__name__}:async_classifier"])
def test_invalid_callable(spec):
    with pytest.raises(TypeError, match="synchronous callable"):
        load_failure_classifier(spec)


@pytest.mark.parametrize("value", [None, *TrialFailure])
def test_classifier_contract(value):
    assert classify_failure(lambda trial: value, result()) is value


@pytest.mark.parametrize("value", [False, "context_length", "complete", {"kind": "context_length"}])
def test_invalid_classifier_output(value):
    with pytest.raises(FailureClassifierError, match="expected TrialFailure or None"):
        classify_failure(lambda trial: value, result())


def test_classifier_exception_retains_cause():
    def broken(trial):
        raise RuntimeError("harness record is malformed")

    with pytest.raises(FailureClassifierError, match="harness record is malformed") as caught:
        classify_failure(broken, result())
    assert isinstance(caught.value.__cause__, RuntimeError)
