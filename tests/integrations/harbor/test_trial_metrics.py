"""Harbor's per-attempt trial metrics, and the plain Harbor generator logging them without skycap."""

from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from omegaconf import OmegaConf

pytest.importorskip("harbor")

from examples.train_integrations.harbor import harbor_generator  # noqa: E402
from examples.train_integrations.harbor.entrypoints.main_harbor import (  # noqa: E402
    HARBOR_DEFAULT_CONFIG,
)
from examples.train_integrations.harbor.harbor_generator import (  # noqa: E402
    HarborGenerator,
)
from examples.train_integrations.harbor.trial_metrics import (  # noqa: E402
    TrialAttempts,
    _seconds,
    trial_metrics,
)
from skyrl.train.generators.base import TrajectoryID  # noqa: E402

pytestmark = pytest.mark.integrations

EPOCH = datetime(2026, 1, 1)
SETUP, AGENT, VERIFY, START_TIMEOUT = 5.0, 10.0, 2.0, 600.0


def timing(start: float, seconds: float) -> SimpleNamespace:
    return SimpleNamespace(
        started_at=EPOCH + timedelta(seconds=start), finished_at=EPOCH + timedelta(seconds=start + seconds)
    )


def result(exception: str | None = None, **phases: Any) -> SimpleNamespace:
    """A Harbor ``TrialResult``: rewarded 1.0 with one turn of rollout details unless it has an exception."""
    return SimpleNamespace(
        exception_info=SimpleNamespace(exception_type=exception) if exception else None,
        verifier_result=None if exception else SimpleNamespace(rewards={"reward": 1.0}),
        agent_result=SimpleNamespace(
            rollout_details=[{"prompt_token_ids": [[1, 2]], "completion_token_ids": [[3]], "logprobs": [[-0.5]]}],
            metadata={"n_episodes": 1},
        ),
        environment_setup=phases.get("environment_setup"),
        agent_execution=phases.get("agent_execution"),
        verifier=phases.get("verifier"),
    )


def completed() -> SimpleNamespace:
    return result(
        environment_setup=timing(0, SETUP),
        agent_execution=timing(SETUP, AGENT),
        verifier=timing(SETUP + AGENT, VERIFY),
    )


def test_a_phase_without_both_timestamps_has_no_duration() -> None:
    assert _seconds(SimpleNamespace(started_at=EPOCH, finished_at=None)) is None
    assert _seconds(None) is None
    assert _seconds(timing(0, 3)) == 3.0


def test_harbors_exception_is_the_cause_of_an_attempt_that_also_raised_after() -> None:
    trial = TrialAttempts()
    trial.start()
    trial.record(result("EnvironmentStartTimeoutError", environment_setup=timing(0, START_TIMEOUT)))
    trial.fail(RuntimeError("raised after Harbor returned"))
    trial.start()
    trial.fail(ValueError("raised before Harbor returned"))

    metrics = trial_metrics([trial])
    assert metrics["generate/harbor/num_attempts"] == 2
    assert metrics["generate/harbor/num_retried_attempts"] == 1
    assert metrics["generate/harbor/num_failed_attempts"] == 2
    assert metrics["generate/harbor/num_failed_attempts/EnvironmentStartTimeoutError"] == 1
    assert metrics["generate/harbor/num_failed_attempts/ValueError"] == 1
    assert "generate/harbor/num_failed_attempts/RuntimeError" not in metrics
    # The attempt that raised before Harbor returned has no times; the start timeout counts up to its failure.
    assert metrics["generate/harbor/environment_setup_time_max"] == START_TIMEOUT
    assert "generate/harbor/agent_execution_time_mean" not in metrics


def test_a_healthy_step_charts_zero_failures() -> None:
    trial = TrialAttempts()
    trial.start()
    trial.record(completed())

    metrics = trial_metrics([trial, TrialAttempts()])
    assert metrics["generate/harbor/num_attempts"] == 1
    assert metrics["generate/harbor/num_retried_attempts"] == 0
    assert metrics["generate/harbor/num_failed_attempts"] == 0
    assert not any(k.startswith("generate/harbor/num_failed_attempts/") for k in metrics)
    for phase, seconds in (("environment_setup", SETUP), ("agent_execution", AGENT), ("verifier", VERIFY)):
        for stat in ("mean", "p90", "max"):
            assert metrics[f"generate/harbor/{phase}_time_{stat}"] == pytest.approx(seconds)


class FakeTrial:
    """Replaces Harbor's ``Trial``: each task path's attempts return its results in turn."""

    scripts: dict[str, list[SimpleNamespace]] = {}

    def __init__(self, path: str) -> None:
        self.path = path

    @classmethod
    async def create(cls, config: Any) -> "FakeTrial":
        return cls(str(config.task.path))

    async def run(self) -> SimpleNamespace:
        return self.scripts[self.path].pop(0)


@pytest.mark.asyncio
async def test_the_plain_harbor_generator_logs_a_start_timeout_rescued_by_a_retry(monkeypatch) -> None:
    FakeTrial.scripts = {
        "slow_start": [result("EnvironmentStartTimeoutError", environment_setup=timing(0, START_TIMEOUT)), completed()],
        "linear": [completed()],
    }
    monkeypatch.setattr(harbor_generator, "Trial", FakeTrial)

    async def finish_session(session_id: str) -> None:
        pass

    client = SimpleNamespace(get_endpoint_url=lambda: "http://engine", finish_session=finish_session)
    cfg = OmegaConf.create(
        {
            "inference_engine": {"served_model_name": "policy"},
            "step_wise_trajectories": True,
            "merge_stepwise_output": True,
            "apply_overlong_filtering": False,
            "rate_limit": None,
        }
    )
    with open(HARBOR_DEFAULT_CONFIG) as f:
        harbor_cfg = yaml.safe_load(f)
    generator = HarborGenerator(cfg, harbor_cfg, client, tokenizer=None, max_seq_len=1024)

    out = await generator.generate(
        {
            "prompts": ["slow_start", "linear"],
            "trajectory_ids": [TrajectoryID(instance_id=p, repetition_id=0) for p in ("slow_start", "linear")],
        },
        disable_tqdm=True,
    )

    assert out["rewards"] == [1.0, 1.0]
    metrics = out["rollout_metrics"]
    # The trajectory-level counts can't see it: the retry rescued the trial.
    assert metrics["generate/num_error_trajectories"] == 0
    assert metrics["generate/num_masked_instances"] == 0
    assert metrics["generate/harbor/num_attempts"] == 3
    assert metrics["generate/harbor/num_retried_attempts"] == 1
    assert metrics["generate/harbor/num_failed_attempts/EnvironmentStartTimeoutError"] == 1
    assert metrics["generate/harbor/environment_setup_time_max"] == pytest.approx(START_TIMEOUT)
    assert metrics["generate/harbor/agent_execution_time_mean"] == pytest.approx(AGENT)
