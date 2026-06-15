"""CPU-only tests for the per-request ``return_per_token_outputs`` gate.

The SFT cross_entropy path builds a per-sequence ``loss_fn_outputs`` list (each
token's logprob + elementwise NLL) for consumers that read them (Tinker API, RL).
SkyRL's own ``SFTTrainer`` reads only ``output.metrics`` and never the per-token
outputs, so it opts out via ``loss_fn_config={"return_per_token_outputs": False}``,
which skips the ``.tolist()`` loops and the detached D2H copies.

These tests pin the contract WITHOUT CUDA by driving the real FSDP worker loss
builds (``_forward_backward_micro`` / ``_forward_micro_with_loss``) with a mocked
model + strategy on CPU (``torch.cuda.current_device`` patched to ``"cpu"``):

* default (flag absent / True): per-token ``logprobs`` + ``elementwise_loss`` are
  populated;
* flag False: per-token outputs are empty dicts, while ``loss`` /
  ``response_length`` are byte-identical to the default path;
* the flag is popped before the algorithm-config merge, so an arbitrary
  (non-AlgorithmConfig) key never reaches ``from_dict_config``, and a real PPO
  override key alongside the flag is still applied;
* the RL (non-cross_entropy) else-branch is *not* gated: it always builds
  ``logprobs`` regardless of the flag (the gate only wraps the cross_entropy
  branch), so passing ``return_per_token_outputs=False`` on a PPO loss must not
  zero out the RL ``loss_fn_outputs``.

The on-GPU Megatron / FSDP parity (flag default-True vs False produce identical
loss + consumed metrics, outputs populated vs empty) lives in the gpu_ci suite:
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
    """Construct a PolicyWorkerBase on CPU with mocked distributed deps.

    Mirrors ``tests/train/test_trainer.py::test_forward_backward_batch_calculations``:
    we never touch CUDA / a real process group, only the loss-build logic.
    """
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
    """Make ``worker.model(...)`` return canned ``(action_log_probs, output)``.

    ``.train()`` / ``.eval()`` on the mock are no-ops; the cross_entropy branch
    only consumes ``action_log_probs`` (output entropy is unused for SFT).
    """
    model = MagicMock()
    model.return_value = (action_log_probs, {"entropy": torch.zeros_like(action_log_probs)})
    worker.model = model


def _make_experience() -> Experience:
    """A small CPU Experience with an all-ones loss mask (valid_len == NUM_ACTIONS)."""
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
    """Drive the real ``_forward_backward_micro`` cross_entropy build on CPU."""
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
    """Drive the real (eval) ``_forward_micro_with_loss`` cross_entropy build on CPU."""
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
    """Drive the real RL (non-cross_entropy) else-branch of ``_forward_backward_micro``.

    Uses the ``regular`` PPO loss so ``resolved_loss_name != "cross_entropy"`` and the
    gated cross_entropy block is skipped entirely. The RL else-branch always builds
    ``logprobs`` and is independent of ``return_per_token_outputs`` — this pins that
    the gate does not leak into the RL path.
    """
    worker = _make_cpu_policy_worker()
    worker.cfg.algorithm.policy_loss_type = "regular"
    # Disable the KL/entropy terms so this test isolates the per-token-output gate from
    # the RL path: with them off, ``base_action_log_probs`` is unused and the entropy term
    # is a zero tensor. (``use_kl_loss`` defaults to ``True``, which would otherwise feed
    # ``base_action_log_probs=None`` into ``compute_approx_kl``.) ``dp_size`` feeds the
    # grad-sum correction factor.
    worker.cfg.algorithm.use_kl_loss = False
    worker.cfg.algorithm.use_entropy_loss = False
    worker.mesh_rank = MagicMock()
    worker.mesh_rank.dp_size = 1

    action_log_probs = torch.full((BATCH_SIZE, NUM_ACTIONS), -0.5)
    model = MagicMock()
    # entropy spans the full sequence; the RL branch slices [:, -num_actions - 1 : -1].
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


# ---------------------------------------------------------------------------
# forward_backward (train) path
# ---------------------------------------------------------------------------
class TestForwardBackwardMicroGate:
    def test_default_keeps_per_token_outputs(self):
        """Flag absent (default True): every sequence carries logprobs + NLL."""
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
        """Flag False: one empty dict per sequence (no logprobs / NLL keys)."""
        status = _run_forward_backward_micro(loss_fn_config={RETURN_PER_TOKEN_OUTPUTS_KEY: False})
        outputs = status["loss_fn_outputs"]
        assert len(outputs) == BATCH_SIZE
        for out in outputs:
            assert out == {}

    def test_loss_and_metrics_identical_across_flag(self):
        """Skipping the per-token build must not perturb loss / response_length."""
        kept = _run_forward_backward_micro(loss_fn_config={RETURN_PER_TOKEN_OUTPUTS_KEY: True})
        skipped = _run_forward_backward_micro(loss_fn_config={RETURN_PER_TOKEN_OUTPUTS_KEY: False})
        assert kept["loss"] == skipped["loss"]
        assert kept["response_length"] == skipped["response_length"]
        assert kept["lr"] == skipped["lr"]

    def test_flag_popped_before_algorithm_config_merge(self):
        """The non-AlgorithmConfig flag must never reach ``from_dict_config``;
        a real PPO override key passed alongside it is still applied."""
        # If the flag leaked into the config merge, ``from_dict_config`` key
        # validation would raise here. A real override key (``eps_clip_low``)
        # confirms the merge still runs for legitimate keys.
        status = _run_forward_backward_micro(loss_fn_config={RETURN_PER_TOKEN_OUTPUTS_KEY: False, "eps_clip_low": 0.1})
        assert status["loss_fn_outputs"] == [{} for _ in range(BATCH_SIZE)]

    def test_caller_loss_fn_config_not_mutated(self):
        """Popping the flag must not mutate the caller-provided dict."""
        cfg_dict = {RETURN_PER_TOKEN_OUTPUTS_KEY: False}
        _run_forward_backward_micro(loss_fn_config=cfg_dict)
        assert cfg_dict == {RETURN_PER_TOKEN_OUTPUTS_KEY: False}


# ---------------------------------------------------------------------------
# forward (eval / Tinker) path — shares the same gate
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# RL (non-cross_entropy) else-branch — NOT gated; always builds logprobs
# ---------------------------------------------------------------------------
class TestForwardBackwardMicroRLPathUngated:
    def test_rl_builds_logprobs_with_flag_true(self):
        status = _run_forward_backward_micro_rl(loss_fn_config={RETURN_PER_TOKEN_OUTPUTS_KEY: True})
        outputs = status["loss_fn_outputs"]
        assert len(outputs) == BATCH_SIZE
        for out in outputs:
            assert len(out["logprobs"]) == NUM_ACTIONS

    def test_rl_builds_logprobs_even_when_flag_false(self):
        """The gate only wraps the cross_entropy branch; the RL else-branch must
        still populate per-sequence logprobs even with the flag set False."""
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
