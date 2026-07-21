"""Exact-ID cleanup for bounded Fireworks dedicated smoke runs."""

from __future__ import annotations

import argparse
import os
import time
from typing import Any

from fireworks.training.sdk import DeploymentManager, TrainerJobManager

_SAFE_PREFIX = "skyrl-smoke-"
_TRAINER_TERMINAL = {
    "JOB_STATE_CANCELLED",
    "JOB_STATE_COMPLETED",
    "JOB_STATE_DELETED",
    "JOB_STATE_FAILED",
}
_DEPLOYMENT_TERMINAL = {"DELETED", "FAILED"}


def _resource_id(value: str) -> str:
    if not value.startswith(_SAFE_PREFIX):
        raise argparse.ArgumentTypeError(
            f"cleanup IDs must start with {_SAFE_PREFIX!r}"
        )
    if "/" in value:
        raise argparse.ArgumentTypeError(
            "pass the resource ID, not a full resource name"
        )
    return value


def _trainer_state(manager: TrainerJobManager, resource_id: str) -> str | None:
    row = manager.try_get(resource_id)
    return None if row is None else str(row.get("state", ""))


def _deployment_state(manager: DeploymentManager, resource_id: str) -> str | None:
    row = manager.get(resource_id)
    return None if row is None else str(row.state)


def cleanup_and_audit(
    *, trainer_job_id: str, deployment_id: str, attempts: int = 12
) -> bool:
    """Delete only the named smoke resources and wait for terminal states."""

    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("FIREWORKS_API_KEY is required for cleanup")

    trainer = TrainerJobManager(api_key=api_key)
    deployment = DeploymentManager(api_key=api_key)
    try:
        deployment_state = _deployment_state(deployment, deployment_id)
        if (
            deployment_state is not None
            and deployment_state not in _DEPLOYMENT_TERMINAL
        ):
            print(
                f"Deleting dedicated deployment {deployment_id} (state={deployment_state})",
                flush=True,
            )
            deployment.delete(deployment_id)

        trainer_state = _trainer_state(trainer, trainer_job_id)
        if trainer_state is not None and trainer_state not in _TRAINER_TERMINAL:
            print(
                f"Stopping dedicated trainer {trainer_job_id} (state={trainer_state})",
                flush=True,
            )
            trainer.delete(trainer_job_id)

        for attempt in range(1, attempts + 1):
            trainer_state = _trainer_state(trainer, trainer_job_id)
            deployment_state = _deployment_state(deployment, deployment_id)
            trainer_safe = trainer_state is None or trainer_state in _TRAINER_TERMINAL
            deployment_safe = (
                deployment_state is None or deployment_state in _DEPLOYMENT_TERMINAL
            )
            print(
                f"Cleanup audit {attempt}/{attempts}: trainer={trainer_state or 'absent'}, "
                f"deployment={deployment_state or 'absent'}",
                flush=True,
            )
            if trainer_safe and deployment_safe:
                return True
            time.sleep(5)
        return False
    finally:
        trainer.close()
        deployment.close()


def _parse_args() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer-job-id", required=True, type=_resource_id)
    parser.add_argument("--deployment-id", required=True, type=_resource_id)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not cleanup_and_audit(
        trainer_job_id=args.trainer_job_id, deployment_id=args.deployment_id
    ):
        raise RuntimeError(
            "Dedicated Fireworks resources did not reach terminal states"
        )
