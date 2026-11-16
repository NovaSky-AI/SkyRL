"""Which failed Harbor trials are masked, and the plain Harbor generator's rows for them."""

import pytest

pytest.importorskip("harbor")

from examples.train_integrations.harbor.entrypoints.main_harbor import (  # noqa: E402
    HarborGeneratorConfig,
)
from examples.train_integrations.harbor.harbor_generator import (  # noqa: E402
    HarborTrajectoryOutput,
    build_step_wise_generator_output,
)
from skyrl.integration.harbor.masking import mask_failed_trials  # noqa: E402
from skyrl.train.generators.base import TrajectoryID  # noqa: E402

pytestmark = pytest.mark.integrations

WHOLE = {"num_masked_instances": 1, "num_instance_with_timeouts_unmasked": 0, "num_traj_masked_individually": 0}


@pytest.mark.parametrize(
    "stop_reasons, threshold, masked, counts",
    [
        # One timeout below the threshold is masked by itself; its siblings train.
        (
            ["complete", "complete", "agent_timeout"],
            2,
            [False, False, True],
            {"num_masked_instances": 0, "num_instance_with_timeouts_unmasked": 1, "num_traj_masked_individually": 1},
        ),
        # At the threshold the whole instance is masked; 1 does that on any timeout.
        (["complete", "agent_timeout", "agent_timeout"], 2, [True] * 3, WHOLE),
        (["complete", "complete", "agent_timeout"], 1, [True] * 3, WHOLE),
        # A failure masks its instance whatever the threshold.
        (["complete", "error", "agent_timeout"], 3, [True] * 3, WHOLE),
        # A context-length stop is no failure.
        (
            ["complete", "context_length"],
            2,
            [False, False],
            {"num_masked_instances": 0, "num_instance_with_timeouts_unmasked": 0, "num_traj_masked_individually": 0},
        ),
    ],
)
def test_failed_trials_mask_themselves_or_their_instance(stop_reasons, threshold, masked, counts) -> None:
    # A second, healthy instance in the batch is never touched.
    trials = [("a", stop_reason) for stop_reason in stop_reasons] + [("b", "complete")]

    got, got_counts = mask_failed_trials(trials, threshold)

    assert got == masked + [False]
    assert {name: got_counts[name] for name in counts} == counts
    assert got_counts["num_timeout_trajectories"] == stop_reasons.count("agent_timeout")
    assert got_counts["num_error_trajectories"] == stop_reasons.count("error")


@pytest.mark.parametrize("threshold, rewards, loss_masks", [(1, [0.0, 0.0], [[0], [0]]), (2, [0.0, 1.0], [[0], [1]])])
def test_the_plain_harbor_generator_masks_a_lone_timeout_by_itself(threshold, rewards, loss_masks) -> None:
    turn = [{"prompt_token_ids": [[1, 2]], "completion_token_ids": [[3]], "logprobs": [[-0.5]]}]
    timed_out = HarborTrajectoryOutput(trajectory_id=TrajectoryID("a", 0), stop_reason="agent_timeout")
    passed = HarborTrajectoryOutput(trajectory_id=TrajectoryID("a", 1), rollout_details=turn, reward=1.0)

    out = build_step_wise_generator_output(
        [timed_out, passed], overlong_filtering=False, timeout_mask_instance_threshold=threshold
    )

    assert out["rewards"] == rewards and out["loss_masks"] == loss_masks


def test_the_harbor_generator_config_still_fills_the_generator_defaults() -> None:
    assert HarborGeneratorConfig().eval_sampling_params is not None


def test_the_timeout_threshold_must_be_at_least_one() -> None:
    with pytest.raises(ValueError, match="timeout_mask_instance_threshold must be at least 1"):
        HarborGeneratorConfig(timeout_mask_instance_threshold=0)
