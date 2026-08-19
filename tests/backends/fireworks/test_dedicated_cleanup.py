import argparse
from types import SimpleNamespace

import pytest

from skyrl.backends.fireworks import cleanup


class _TrainerManager:
    def __init__(self) -> None:
        self.states = ["JOB_STATE_RUNNING", "JOB_STATE_DELETED", "JOB_STATE_DELETED"]
        self.deleted: list[str] = []
        self.closed = False

    def try_get(self, resource_id: str):
        del resource_id
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return {"state": state}

    def delete(self, resource_id: str) -> None:
        self.deleted.append(resource_id)

    def close(self) -> None:
        self.closed = True


class _DeploymentManager:
    def __init__(self) -> None:
        self.states = ["READY", "DELETING", None]
        self.deleted: list[str] = []
        self.closed = False

    def get(self, resource_id: str):
        del resource_id
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return None if state is None else SimpleNamespace(state=state)

    def delete(self, resource_id: str) -> None:
        self.deleted.append(resource_id)

    def close(self) -> None:
        self.closed = True


def test_cleanup_waits_until_deployment_is_absent(monkeypatch) -> None:
    trainer = _TrainerManager()
    deployment = _DeploymentManager()
    sleeps: list[int] = []
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setattr(cleanup, "TrainerJobManager", lambda *, api_key: trainer)
    monkeypatch.setattr(cleanup, "DeploymentManager", lambda *, api_key: deployment)
    monkeypatch.setattr(cleanup.time, "sleep", sleeps.append)

    success = cleanup.cleanup_and_audit(
        trainer_job_id="skyrl-smoke-test-trainer",
        deployment_id="skyrl-smoke-test-rollout",
    )

    assert success is True
    assert trainer.deleted == ["skyrl-smoke-test-trainer"]
    assert deployment.deleted == ["skyrl-smoke-test-rollout"]
    assert sleeps == [5]
    assert trainer.closed is True
    assert deployment.closed is True


@pytest.mark.parametrize("resource_id", ["unrelated", "accounts/test/skyrl-smoke-id"])
def test_cleanup_rejects_non_smoke_or_full_resource_names(resource_id: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        cleanup._resource_id(resource_id)
