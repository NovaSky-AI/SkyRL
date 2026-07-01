"""CPU coverage for the per-request ``return_per_token_outputs`` gate.

These tests drive worker loss-build methods with CPU mocks and pin default,
skip, config-pop, and RL-ungated behavior. GPU parity lives in
``tests/backends/skyrl_train/gpu/gpu_ci/test_training_step.py``.
"""

from unittest.mock import MagicMock, patch

import torch

from skyrl.backends.skyrl_train.workers.worker import PolicyWorkerBase
from skyrl.backends.skyrl_train.workers.worker_utils import RETURN_PER_TOKEN_OUTPUTS_KEY
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.dataset.replay_buffer import Experience

NUM_ACTIONS = 4
BATCH_SIZE = 2
SEQ_LEN = 6


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


def _run_forward_backward_micro(loss_fn_config):
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
        )


def _run_forward_micro_with_loss(loss_fn_config):
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
        )


def _run_forward_backward_micro_rl(loss_fn_config):
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
            loss_fn_config=loss_fn_config,
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
        status = _run_forward_backward_micro(loss_fn_config={RETURN_PER_TOKEN_OUTPUTS_KEY: True})
        outputs = status["loss_fn_outputs"]
        assert len(outputs) == BATCH_SIZE
        for out in outputs:
            assert len(out["logprobs"]) == NUM_ACTIONS
            assert len(out["elementwise_loss"]) == NUM_ACTIONS

    def test_false_skips_per_token_outputs(self):
        """False: one empty dict per sequence."""
        status = _run_forward_backward_micro(loss_fn_config={RETURN_PER_TOKEN_OUTPUTS_KEY: False})
        outputs = status["loss_fn_outputs"]
        assert len(outputs) == BATCH_SIZE
        for out in outputs:
            assert out == {}

    def test_loss_and_metrics_identical_across_flag(self):
        """Skipping per-token outputs must not perturb consumed metrics."""
        kept = _run_forward_backward_micro(loss_fn_config={RETURN_PER_TOKEN_OUTPUTS_KEY: True})
        skipped = _run_forward_backward_micro(loss_fn_config={RETURN_PER_TOKEN_OUTPUTS_KEY: False})
        assert kept["loss"] == skipped["loss"]
        assert kept["response_length"] == skipped["response_length"]
        assert kept["lr"] == skipped["lr"]

    def test_flag_popped_before_algorithm_config_merge(self):
        """The reserved flag is removed before AlgorithmConfig validation."""
        # A real override key confirms legitimate config merge still runs.
        status = _run_forward_backward_micro(loss_fn_config={RETURN_PER_TOKEN_OUTPUTS_KEY: False, "eps_clip_low": 0.1})
        assert status["loss_fn_outputs"] == [{} for _ in range(BATCH_SIZE)]

    def test_caller_loss_fn_config_not_mutated(self):
        """Popping the flag must not mutate the caller-provided dict."""
        cfg_dict = {RETURN_PER_TOKEN_OUTPUTS_KEY: False}
        _run_forward_backward_micro(loss_fn_config=cfg_dict)
        assert cfg_dict == {RETURN_PER_TOKEN_OUTPUTS_KEY: False}


class TestForwardMicroWithLossGate:
    def test_default_keeps_per_token_outputs(self):
        status = _run_forward_micro_with_loss(loss_fn_config=None)
        outputs = status["loss_fn_outputs"]
        assert len(outputs) == BATCH_SIZE
        for out in outputs:
            assert len(out["logprobs"]) == NUM_ACTIONS
            assert len(out["elementwise_loss"]) == NUM_ACTIONS

    def test_false_skips_per_token_outputs(self):
        status = _run_forward_micro_with_loss(loss_fn_config={RETURN_PER_TOKEN_OUTPUTS_KEY: False})
        outputs = status["loss_fn_outputs"]
        assert len(outputs) == BATCH_SIZE
        for out in outputs:
            assert out == {}

    def test_loss_identical_across_flag(self):
        kept = _run_forward_micro_with_loss(loss_fn_config={RETURN_PER_TOKEN_OUTPUTS_KEY: True})
        skipped = _run_forward_micro_with_loss(loss_fn_config={RETURN_PER_TOKEN_OUTPUTS_KEY: False})
        assert kept["loss"] == skipped["loss"]
        assert kept["response_length"] == skipped["response_length"]


class TestForwardBackwardMicroRLPathUngated:
    def test_rl_builds_logprobs_with_flag_true(self):
        status = _run_forward_backward_micro_rl(loss_fn_config={RETURN_PER_TOKEN_OUTPUTS_KEY: True})
        outputs = status["loss_fn_outputs"]
        assert len(outputs) == BATCH_SIZE
        for out in outputs:
            assert len(out["logprobs"]) == NUM_ACTIONS

    def test_rl_builds_logprobs_even_when_flag_false(self):
        """The SFT-only gate must not empty PPO outputs."""
        status = _run_forward_backward_micro_rl(loss_fn_config={RETURN_PER_TOKEN_OUTPUTS_KEY: False})
        outputs = status["loss_fn_outputs"]
        assert len(outputs) == BATCH_SIZE
        for out in outputs:
            assert len(out["logprobs"]) == NUM_ACTIONS

    def test_rl_loss_fn_outputs_identical_across_flag(self):
        kept = _run_forward_backward_micro_rl(loss_fn_config={RETURN_PER_TOKEN_OUTPUTS_KEY: True})
        skipped = _run_forward_backward_micro_rl(loss_fn_config={RETURN_PER_TOKEN_OUTPUTS_KEY: False})
        assert kept["loss_fn_outputs"] == skipped["loss_fn_outputs"]
        assert kept["final_loss"] == skipped["final_loss"]
