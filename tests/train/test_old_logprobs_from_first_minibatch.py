"""
Tests for ``trainer.old_logprobs_from_first_minibatch``.

uv run --isolated --extra dev pytest tests/train/test_old_logprobs_from_first_minibatch.py
"""

from unittest.mock import MagicMock

import pytest
import torch

from skyrl.backends.skyrl_train.distributed.dispatch import WorkerOutput
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils.utils import validate_cfg
from tests.train.util import example_dummy_config


class _DummyDataset:
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return "dummy"

    def collate_fn(self, batch):
        return batch


def _make_trainer(cfg):
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    trainer = RayPPOTrainer(
        cfg=cfg,
        tracker=None,
        tokenizer=tokenizer,
        train_dataset=_DummyDataset(),
        eval_dataset=None,
        inference_engine_client=None,
        generator=MagicMock(),
    )
    trainer.dispatch = MagicMock()
    trainer.dispatch.get_lcm_dp_size.return_value = 1
    return trainer


def _batch(num_samples: int, response_len: int, boundaries):
    data = TrainingInputBatch(
        {
            "sequences": torch.randint(0, 100, (num_samples, response_len + 2)),
            "attention_mask": torch.ones(num_samples, response_len + 2, dtype=torch.long),
            "loss_mask": torch.ones(num_samples, response_len),
            "rollout_logprobs": torch.full((num_samples, response_len), -1.0),
        }
    )
    data.metadata = {"response_length": response_len, "policy_mini_batch_boundaries": boundaries}
    return data


def _staged_forward(response_len):
    """Mock ``stage_data``/``forward_from_staged``: each staged chunk is a mini-batch, and the
    forward returns logprobs equal to -(global row index + 1) for every token."""

    def stage_data(model, data, boundaries):
        return [[(start, end)] for start, end in boundaries]

    def forward_from_staged(model, chunk_refs):
        (start, end) = chunk_refs[0]
        return WorkerOutput(loss_fn_outputs=[{"logprobs": [-(i + 1.0)] * response_len} for i in range(start, end)])

    return stage_data, forward_from_staged


@pytest.mark.parametrize(
    "boundaries, expected_forward_minibatches",
    [([(0, 4)], []), ([(0, 2), (2, 4)], [(2, 4)]), ([(0, 2), (2, 3), (3, 4)], [(2, 3), (3, 4)])],
)
def test_policy_forward_skips_first_minibatch(boundaries, expected_forward_minibatches):
    cfg = example_dummy_config()
    cfg.trainer.old_logprobs_from_first_minibatch = True
    trainer = _make_trainer(cfg)
    stage_data, forward_from_staged = _staged_forward(response_len=3)
    trainer.dispatch.stage_data.side_effect = stage_data
    trainer.dispatch.forward_from_staged.side_effect = forward_from_staged

    out = trainer.fwd_logprobs_values_reward(_batch(4, 3, boundaries))

    forwarded = [call.args[1][0] for call in trainer.dispatch.forward_from_staged.call_args_list]
    assert forwarded == expected_forward_minibatches
    trainer.dispatch.forward.assert_not_called()
    first_end = boundaries[0][1]
    logprobs = out["action_log_probs"]
    assert logprobs.shape == (4, 3)
    assert torch.all(logprobs[:first_end] == 0)
    for row in range(first_end, 4):
        assert torch.all(logprobs[row] == -(row + 1.0))
    # The rollout/train diff metric covers only rows with forward-pass logprobs.
    if first_end < 4:
        rows = torch.arange(first_end, 4, dtype=torch.float32)
        expected_mean = (rows + 1.0 - 1.0).abs().mean().item()
        assert trainer.all_metrics["policy/rollout_train_logprobs_abs_diff_mean"] == pytest.approx(expected_mean)
    else:
        assert "policy/rollout_train_logprobs_abs_diff_mean" not in trainer.all_metrics


def test_training_step_flags_only_first_policy_minibatch():
    cfg = example_dummy_config()
    cfg.trainer.old_logprobs_from_first_minibatch = True
    trainer = _make_trainer(cfg)
    trainer.dispatch.stage_data.return_value = [["mb0"], ["mb1"], ["mb2"]]
    trainer.dispatch.forward_backward_from_staged.return_value = WorkerOutput(metrics={"policy_loss": 0.0})
    trainer.dispatch.optim_step.return_value = None
    trainer._normalize_advantages = lambda data, boundaries, prompt_boundaries: data

    data = _batch(6, 3, [(0, 2), (2, 4), (4, 6)])
    trainer._execute_training_step("policy", data)

    calls = trainer.dispatch.forward_backward_from_staged.call_args_list
    assert [c.kwargs.get("old_logprobs_from_current", False) for c in calls] == [True, False, False]


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"strategy": "fsdp"}, "only supported by the Megatron backend"),
        ({"update_epochs_per_batch": 2}, "update_epochs_per_batch=1"),
        ({"algorithm.use_kl_in_reward": True, "algorithm.use_kl_loss": False}, "use_kl_in_reward=false"),
    ],
)
def test_validate_cfg_rejects_unsupported(overrides, match):
    cfg = example_dummy_config()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.old_logprobs_from_first_minibatch = True
    for key, value in overrides.items():
        target = cfg.trainer
        *parents, leaf = key.split(".")
        for p in parents:
            target = getattr(target, p)
        setattr(target, leaf, value)
    with pytest.raises(ValueError, match=match):
        validate_cfg(cfg)
