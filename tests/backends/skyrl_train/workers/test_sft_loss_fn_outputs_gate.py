"""CPU coverage for the per-request ``return_per_token_outputs`` gate.

These tests drive worker loss-build methods with CPU mocks and pin default,
skip, config-pop, and RL-ungated behavior. GPU parity lives in
``tests/backends/skyrl_train/gpu/gpu_ci/test_training_step.py``.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from skyrl.backends.skyrl_train.utils.ppo_utils import PolicyLossRegistry
from skyrl.backends.skyrl_train.workers.worker import PolicyWorkerBase
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.dataset.replay_buffer import Experience

NUM_ACTIONS = 4
BATCH_SIZE = 2
SEQ_LEN = 6


@pytest.fixture(scope="module", autouse=True)
def _repopulate_policy_loss_registry() -> None:
    """Restore defaults after registry tests reset global state."""
    PolicyLossRegistry.repopulate_registry()


def _make_cpu_policy_worker() -> PolicyWorkerBase:
    """Construct a CPU PolicyWorkerBase with mocked distributed deps."""
    cfg = SkyRLTrainConfig()
    cfg.trainer.algorithm.policy_loss_type = "cross_entropy"
    cfg.generator.sampling_params.temperature = 1.0
    cfg.trainer.algorithm.temperature = 1.0

    worker = PolicyWorkerBase(
        cfg=cfg.trainer,
        world_size=1,
        rank=0,
        local_rank=0,
        master_addr="localhost",
        master_port=12345,
        sequence_parallel_size=1,
    )
    worker.strategy = MagicMock()
    worker.scheduler = MagicMock()
    worker.scheduler.get_last_lr.return_value = [1e-4]
    worker.optimizer = MagicMock()
    return worker


def _patch_model(worker: PolicyWorkerBase, action_log_probs: torch.Tensor) -> None:
    """Make ``worker.model(...)`` return canned logprobs."""
    model = MagicMock()
    model.return_value = (action_log_probs, {"entropy": torch.zeros_like(action_log_probs)})
    worker.model = model


def _make_experience() -> Experience:
    """Small CPU Experience with valid_len == NUM_ACTIONS."""
    return Experience(
        sequences=torch.randint(0, 100, (BATCH_SIZE, SEQ_LEN)),
        action_log_probs=None,
        base_action_log_probs=None,
        rollout_logprobs=None,
        values=None,
        returns=None,
        advantages=torch.zeros(BATCH_SIZE, NUM_ACTIONS),
        attention_mask=torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long),
        loss_mask=torch.ones(BATCH_SIZE, NUM_ACTIONS, dtype=torch.long),
        action_mask=None,
        rollout_expert_indices=None,
        num_actions=NUM_ACTIONS,
        info={},
    )


def _run_forward_backward_micro(return_per_token_outputs=True, loss_fn_config=None):
    """Drive the train cross_entropy build on CPU."""
    worker = _make_cpu_policy_worker()
    action_log_probs = torch.full((BATCH_SIZE, NUM_ACTIONS), -0.5)
    _patch_model(worker, action_log_probs)
    experience = _make_experience()
    with patch("torch.cuda.current_device", return_value="cpu"), patch("torch.autocast", MagicMock()):
        return worker._forward_backward_micro(
            experience,
            microbatch_weight=1.0,
            loss_fn="cross_entropy",
            loss_fn_config=loss_fn_config,
            return_per_token_outputs=return_per_token_outputs,
        )


def _run_forward_micro_with_loss(return_per_token_outputs=True, loss_fn_config=None):
    """Drive the eval cross_entropy build on CPU."""
    worker = _make_cpu_policy_worker()
    action_log_probs = torch.full((BATCH_SIZE, NUM_ACTIONS), -0.5)
    _patch_model(worker, action_log_probs)
    experience = _make_experience()
    with patch("torch.cuda.current_device", return_value="cpu"), patch("torch.autocast", MagicMock()):
        return worker._forward_micro_with_loss(
            experience,
            loss_fn="cross_entropy",
            loss_fn_config=loss_fn_config,
            return_per_token_outputs=return_per_token_outputs,
        )


def _run_forward_backward_micro_rl(return_per_token_outputs=True):
    """Drive the PPO path, which must ignore the SFT-only gate."""
    worker = _make_cpu_policy_worker()
    worker.cfg.algorithm.policy_loss_type = "regular"
    # Disable extra PPO terms so this test isolates loss_fn_outputs.
    worker.cfg.algorithm.use_kl_loss = False
    worker.cfg.algorithm.use_entropy_loss = False
    worker.mesh_rank = MagicMock()
    worker.mesh_rank.dp_size = 1

    action_log_probs = torch.full((BATCH_SIZE, NUM_ACTIONS), -0.5)
    model = MagicMock()
    # The RL branch slices entropy from the full sequence.
    model.return_value = (action_log_probs, {"entropy": torch.zeros(BATCH_SIZE, SEQ_LEN)})
    worker.model = model

    experience = _make_experience()
    experience.action_log_probs = torch.full((BATCH_SIZE, NUM_ACTIONS), -0.5)
    experience.advantages = torch.ones(BATCH_SIZE, NUM_ACTIONS)
    with patch("torch.cuda.current_device", return_value="cpu"), patch("torch.autocast", MagicMock()):
        return worker._forward_backward_micro(
            experience,
            microbatch_weight=1.0,
            loss_fn="regular",
            return_per_token_outputs=return_per_token_outputs,
        )


class TestForwardBackwardMicroGate:
    def test_default_keeps_per_token_outputs(self):
        """Default: every sequence carries logprobs + NLL."""
        status = _run_forward_backward_micro(loss_fn_config=None)
        outputs = status["loss_fn_outputs"]
        assert len(outputs) == BATCH_SIZE
        for out in outputs:
            assert len(out["logprobs"]) == NUM_ACTIONS
            assert len(out["elementwise_loss"]) == NUM_ACTIONS

    def test_explicit_true_keeps_per_token_outputs(self):
        status = _run_forward_backward_micro(return_per_token_outputs=True)
        outputs = status["loss_fn_outputs"]
        assert len(outputs) == BATCH_SIZE
        for out in outputs:
            assert len(out["logprobs"]) == NUM_ACTIONS
            assert len(out["elementwise_loss"]) == NUM_ACTIONS

    def test_false_skips_per_token_outputs(self):
        """False: one empty dict per sequence."""
        status = _run_forward_backward_micro(return_per_token_outputs=False)
        outputs = status["loss_fn_outputs"]
        assert len(outputs) == BATCH_SIZE
        for out in outputs:
            assert out == {}

    def test_loss_and_metrics_identical_across_flag(self):
        """Skipping per-token outputs must not perturb consumed metrics."""
        kept = _run_forward_backward_micro(return_per_token_outputs=True)
        skipped = _run_forward_backward_micro(return_per_token_outputs=False)
        assert kept["loss"] == skipped["loss"]
        assert kept["response_length"] == skipped["response_length"]
        assert kept["lr"] == skipped["lr"]

    def test_gate_composes_with_loss_fn_config_override(self):
        """The gate is independent of loss_fn_config; overrides still merge."""
        # A real override key confirms legitimate config merge still runs while
        # the explicit gate suppresses per-token outputs.
        status = _run_forward_backward_micro(return_per_token_outputs=False, loss_fn_config={"eps_clip_low": 0.1})
        assert status["loss_fn_outputs"] == [{} for _ in range(BATCH_SIZE)]


class TestForwardMicroWithLossGate:
    def test_default_keeps_per_token_outputs(self):
        status = _run_forward_micro_with_loss(loss_fn_config=None)
        outputs = status["loss_fn_outputs"]
        assert len(outputs) == BATCH_SIZE
        for out in outputs:
            assert len(out["logprobs"]) == NUM_ACTIONS
            assert len(out["elementwise_loss"]) == NUM_ACTIONS

    def test_false_skips_per_token_outputs(self):
        status = _run_forward_micro_with_loss(return_per_token_outputs=False)
        outputs = status["loss_fn_outputs"]
        assert len(outputs) == BATCH_SIZE
        for out in outputs:
            assert out == {}

    def test_loss_identical_across_flag(self):
        kept = _run_forward_micro_with_loss(return_per_token_outputs=True)
        skipped = _run_forward_micro_with_loss(return_per_token_outputs=False)
        assert kept["loss"] == skipped["loss"]
        assert kept["response_length"] == skipped["response_length"]


class TestForwardBackwardMicroRLPathUngated:
    def test_rl_builds_logprobs_with_flag_true(self):
        status = _run_forward_backward_micro_rl(return_per_token_outputs=True)
        outputs = status["loss_fn_outputs"]
        assert len(outputs) == BATCH_SIZE
        for out in outputs:
            assert len(out["logprobs"]) == NUM_ACTIONS

    def test_rl_builds_logprobs_even_when_flag_false(self):
        """The SFT-only gate must not empty PPO outputs."""
        status = _run_forward_backward_micro_rl(return_per_token_outputs=False)
        outputs = status["loss_fn_outputs"]
        assert len(outputs) == BATCH_SIZE
        for out in outputs:
            assert len(out["logprobs"]) == NUM_ACTIONS

    def test_rl_loss_fn_outputs_identical_across_flag(self):
        kept = _run_forward_backward_micro_rl(return_per_token_outputs=True)
        skipped = _run_forward_backward_micro_rl(return_per_token_outputs=False)
        assert kept["loss_fn_outputs"] == skipped["loss_fn_outputs"]
        assert kept["final_loss"] == skipped["final_loss"]
