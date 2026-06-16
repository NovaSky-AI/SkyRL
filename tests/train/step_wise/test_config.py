"""CPU tests for step-wise training config validation.

Run:
    uv run --isolated --extra dev --extra skyrl-train pytest tests/train/step_wise/
"""

from unittest.mock import patch

import pytest

from skyrl.train.utils.utils import validate_cfg
from tests.train.util import example_dummy_config


@pytest.fixture
def dummy_config():
    return example_dummy_config()


@pytest.mark.parametrize(
    ("estimator", "should_raise"),
    [
        ("gae", True),
        ("reinforce++", True),
        ("grpo", False),
        ("rloo", False),
        ("maxrl", False),
    ],
)
@patch("skyrl.train.utils.utils.validate_batch_sizes", new=lambda cfg: None)
@patch("skyrl.train.utils.utils.validate_generator_cfg", new=lambda cfg: None)
def test_validate_cfg_step_wise_estimator_compatibility(dummy_config, estimator, should_raise):
    """``validate_cfg`` must reject step-wise training with temporal estimators (GAE, REINFORCE++)
    and accept it with outcome-based estimators (GRPO, RLOO, MAXRL).

    Step-wise training collapses each trajectory to a single scalar advantage broadcast uniformly
    to every step's response tokens. The temporal credit assignment that GAE / REINFORCE++ produce
    is lost in that collapse, so we refuse the combination at startup.

    ``validate_batch_sizes`` and ``validate_generator_cfg`` are patched to no-ops so the test
    exercises only the step-wise compatibility check on the minimal dummy config.
    """
    dummy_config.generator.step_wise_trajectories = True
    dummy_config.trainer.algorithm.advantage_estimator = estimator
    if estimator == "gae":
        dummy_config.trainer.critic.model.path = "dummy-critic-path"

    if should_raise:
        with pytest.raises(ValueError, match="not supported with step_wise_trajectories"):
            validate_cfg(dummy_config)
    else:
        validate_cfg(dummy_config)
