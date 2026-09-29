"""Exercise worker accumulation with a small differentiable CPU model."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from skyrl.backends.skyrl_train.training_batch import (
    TrainingInputBatch,
    pad_training_input_batch,
)
from skyrl.backends.skyrl_train.utils.ppo_utils import ppo_critic_loss, ppo_policy_loss
from skyrl.backends.skyrl_train.workers.worker import CriticWorkerBase, PolicyWorkerBase
from skyrl.train.config import TrainerConfig


class TinyModel(torch.nn.Module):
    def __init__(self, critic=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.15, dtype=torch.float64))
        self.critic = critic

    def forward(self, sequences, num_actions, **kwargs):
        features = sequences.to(torch.float64) / 10
        scores = self.weight * features[:, -num_actions:]
        if self.critic:
            return scores, {}
        logits = torch.stack((self.weight * features, -self.weight * features, torch.zeros_like(features)), dim=-1)
        log_probs = logits.log_softmax(dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        return log_probs[:, -num_actions:, 0], {"entropy": entropy}


class LocalStrategy:
    def backward(self, loss, model, optimizer):
        loss.backward()

    def all_reduce(self, metrics, op, group):
        return metrics

    def optimizer_step(self, optimizer, model, scheduler, name):
        grad_norm = model.weight.grad.detach().abs()
        optimizer.step()
        return grad_norm


def make_batch():
    data = TrainingInputBatch(
        {
            "sequences": torch.arange(1, 25).reshape(4, 6),
            "attention_mask": torch.tensor([[1] * 6, [0, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1]]),
            "loss_mask": torch.tensor([[1, 1, 1], [1, 0, 0], [1, 1, 0], [0, 0, 0]]),
            "response_mask": torch.ones(4, 3, dtype=torch.long),
            "action_log_probs": torch.full((4, 3), -0.2, dtype=torch.float64),
            "base_action_log_probs": torch.full((4, 3), -0.5, dtype=torch.float64),
            "advantages": torch.full((4, 3), 0.1, dtype=torch.float64),
            "values": torch.zeros(4, 3, dtype=torch.float64),
            "returns": torch.ones(4, 3, dtype=torch.float64),
        }
    )
    data.metadata = {"response_length": 3}
    return data


def run_worker(critic, micro_size, token_budget, pad_size=0):
    worker_type = CriticWorkerBase if critic else PolicyWorkerBase
    worker = object.__new__(worker_type)
    worker.cfg = TrainerConfig()
    worker.cfg.algorithm.policy_loss_type = "regular"
    worker.cfg.algorithm.use_kl_loss = True
    worker.cfg.algorithm.kl_loss_coef = 0.2
    worker.cfg.algorithm.use_entropy_loss = True
    worker.cfg.algorithm.entropy_loss_coef = 0.1
    worker.cfg.micro_train_batch_size_per_gpu = micro_size
    worker.cfg.max_tokens_per_microbatch = token_budget
    worker.mesh_rank = SimpleNamespace(dp_size=1)
    worker.device_mesh = SimpleNamespace(get_group=lambda name: None)
    worker.model = TinyModel(critic)
    worker.optimizer = torch.optim.SGD(worker.model.parameters(), lr=0.01)
    worker.scheduler = SimpleNamespace(get_last_lr=lambda: [0.01])
    worker.strategy = LocalStrategy()
    worker.policy_loss_fn = ppo_policy_loss
    worker.critic_loss_fn = ppo_critic_loss
    data = make_batch()
    if pad_size:
        data = pad_training_input_batch(data, pad_size)
    with patch("torch.cuda.current_device", return_value="cpu"), patch("torch.autocast", return_value=nullcontext()):
        result = worker.forward_backward(data)
        worker.optim_step()
    gradient = worker.model.weight.grad.clone()
    return result, gradient, worker.model.weight.detach().clone()


@pytest.mark.parametrize("critic", [False, True])
@pytest.mark.parametrize("micro_size, token_budget", [(1, -1), (2, -1), (1, 11), (1, 100)])
def test_worker_loss_gradient_and_update_match_full_minibatch(critic, micro_size, token_budget):
    baseline, baseline_gradient, baseline_weight = run_worker(critic, 4, -1)
    packed, packed_gradient, packed_weight = run_worker(critic, micro_size, token_budget)
    keys = ["critic_loss"] if critic else ["policy_loss", "policy_kl", "policy_entropy", "final_loss"]
    for key in keys:
        assert packed.metrics[key] == pytest.approx(baseline.metrics[key], rel=1e-6, abs=1e-7), key
    torch.testing.assert_close(packed_gradient, baseline_gradient)
    torch.testing.assert_close(packed_weight, baseline_weight)
    assert packed.loss_fn_outputs == baseline.loss_fn_outputs


@pytest.mark.parametrize("micro_size, token_budget", [(1, -1), (2, -1), (1, 11)])
def test_padded_batch_returns_one_output_per_input_row(micro_size, token_budget):
    # Callers such as the Tinker backend trim their own padding rows by count.
    baseline, baseline_gradient, _ = run_worker(False, 4, -1)
    padded, padded_gradient, _ = run_worker(False, micro_size, token_budget, pad_size=2)
    assert len(padded.loss_fn_outputs) == len(baseline.loss_fn_outputs) + 2
    assert padded.loss_fn_outputs[:-2] == baseline.loss_fn_outputs
    torch.testing.assert_close(padded_gradient, baseline_gradient)
