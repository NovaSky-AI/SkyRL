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
from skyrl.backends.skyrl_train.utils.loss_normalization import (
    MinibatchLossNormalization,
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


def make_worker(critic, micro_size, token_budget, model=None):
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
    worker.model = model if model is not None else TinyModel(critic)
    worker.optimizer = torch.optim.SGD(worker.model.parameters(), lr=0.01)
    worker.scheduler = SimpleNamespace(get_last_lr=lambda: [0.01])
    worker.strategy = LocalStrategy()
    worker.policy_loss_fn = ppo_policy_loss
    worker.critic_loss_fn = ppo_critic_loss
    return worker


def run_worker(critic, micro_size, token_budget, pad_size=0, calls=None):
    """Train one optimizer step, over ``calls`` (row slices sharing the step's denominators) if given."""
    worker = make_worker(critic, micro_size, token_budget)
    data = make_batch()
    if pad_size:
        data = pad_training_input_batch(data, pad_size)
    with patch("torch.cuda.current_device", return_value="cpu"), patch("torch.autocast", return_value=nullcontext()):
        if calls is None:
            result = worker.forward_backward(data)
        else:
            normalization = MinibatchLossNormalization.from_batch(data)
            results = [
                worker.forward_backward(data[start:end], loss_normalization=normalization) for start, end in calls
            ]
            result = SimpleNamespace(
                metrics={key: sum(r.metrics[key] for r in results) for key in results[0].metrics},
                loss_fn_outputs=[output for r in results for output in r.loss_fn_outputs],
            )
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


@pytest.mark.parametrize("critic", [False, True])
@pytest.mark.parametrize("calls", [[(0, 1), (1, 4)], [(0, 2), (2, 3), (3, 4)]])
def test_calls_sharing_the_step_normalization_match_one_call(critic, calls):
    baseline, baseline_gradient, baseline_weight = run_worker(critic, 4, -1)
    split, gradient, weight = run_worker(critic, 1, -1, calls=calls)
    keys = ["critic_loss"] if critic else ["policy_loss", "policy_kl", "policy_entropy", "final_loss"]
    for key in keys:
        assert split.metrics[key] == pytest.approx(baseline.metrics[key], rel=1e-6, abs=1e-7), key
    torch.testing.assert_close(gradient, baseline_gradient)
    torch.testing.assert_close(weight, baseline_weight)


class InferenceTinyModel(TinyModel):
    """``TinyModel`` that also answers the inference call (``return_output=False``)."""

    def forward(self, sequences, num_actions, *args, return_output=True, **kwargs):
        log_probs, output = super().forward(sequences, num_actions, **kwargs)
        return (log_probs, output) if return_output else log_probs


@pytest.fixture(scope="module")
def local_ray():
    import ray

    if not ray.is_initialized():
        ray.init(num_cpus=2, include_dashboard=False, object_store_memory=200 * 1024**2)
    yield


def make_shard():
    """``make_batch``'s token rows, stored as a step-wise trajectory of two rows and two single-row ones."""
    import numpy as np

    from skyrl.backends.skyrl_train.rollout_shard import RolloutShard
    from skyrl.train.dataset.preprocess import RowCollation
    from skyrl.train.generators.base import StoredRolloutGroup

    data = make_batch()
    rows = {"prompt_token_ids": [], "response_ids": [], "loss_masks": [], "rewards": []}
    for row in range(data.batch_size):
        tokens = data["sequences"][row][data["attention_mask"][row].bool()].numpy().astype(np.int32)
        rows["prompt_token_ids"].append(tokens[:-3])
        rows["response_ids"].append(tokens[-3:])
        rows["loss_masks"].append(data["loss_mask"][row].numpy().astype(np.int8))
        rows["rewards"].append(1.0)
    stored = [
        t
        for a, b in [(0, 2), (2, 3), (3, 4)]
        for t in StoredRolloutGroup.from_output({key: value[a:b] for key, value in rows.items()}).trajectories
    ]
    return RolloutShard(
        payload_refs=tuple(t.payload for t in stored),
        payload_rows=tuple(t.num_rows for t in stored),
        prompt_lengths=tuple(len(p) for p in rows["prompt_token_ids"]),
        response_lengths=(3,) * data.batch_size,
        row_advantages=(0.1, -0.3, 0.2, 0.4),
        collation=RowCollation(pad_token_id=0, max_seq_len=None, require_rollout_logprobs=False),
    )


@pytest.mark.parametrize("token_budget", [7, 11, 100])
def test_a_stored_shard_trains_as_its_rows_collated_whole_would(local_ray, token_budget):
    from skyrl.backends.skyrl_train.rollout_shard import build_training_batch
    from skyrl.backends.skyrl_train.workers.worker_utils import select_forward_inputs

    def worker():
        trainer = make_worker(False, 1, token_budget, model=InferenceTinyModel())
        # Streamed training refuses a KL loss: rollouts carry no reference logprobs.
        trainer.cfg.algorithm.use_kl_loss = False
        return trainer

    shard = make_shard()
    data = build_training_batch(shard)
    normalization = MinibatchLossNormalization.from_batch(data)
    with patch("torch.cuda.current_device", return_value="cpu"), patch("torch.autocast", return_value=nullcontext()):
        baseline_worker = worker()
        data["action_log_probs"] = baseline_worker._forward_logprobs(select_forward_inputs(data))
        baseline = baseline_worker.forward_backward(data, loss_normalization=normalization)
        baseline_worker.optim_step()

        shard_worker = worker()
        per_token_flags = []
        micro = shard_worker._forward_backward_micro

        def spy(*args, **kwargs):
            per_token_flags.append(kwargs["return_per_token_outputs"])
            return micro(*args, **kwargs)

        shard_worker._forward_backward_micro = spy
        result = shard_worker.forward_backward_rollout_shard(
            shard, loss_normalization=normalization, compute_old_logprobs=True
        )
        shard_worker.optim_step()

    for key in ["policy_loss", "policy_entropy", "final_loss", "num_microbatches"]:
        assert result.metrics[key] == baseline.metrics[key], key
    assert torch.equal(shard_worker.model.weight.grad, baseline_worker.model.weight.grad)
    # Nothing per token is built or goes back: the trainer of a streamed step reads only metrics.
    assert per_token_flags and not any(per_token_flags)
    assert result.loss_fn_outputs == []
