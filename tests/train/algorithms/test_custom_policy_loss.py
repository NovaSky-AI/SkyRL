import pytest
import torch

from examples.train.algorithms.custom_policy_loss.main_custom_policy_loss import (
    compute_reinforce_policy_loss,
)
from skyrl.backends.skyrl_train.utils.ppo_utils import (
    apply_loss_reduction_to_advantages_minibatch,
)
from skyrl.train.config import AlgorithmConfig


@pytest.mark.parametrize("rollout_logprobs", [None, torch.tensor([[-4.0, -5.0, -6.0]])])
def test_custom_policy_loss_accepts_worker_arguments(rollout_logprobs):
    loss, metrics = compute_reinforce_policy_loss(
        torch.tensor([[-1.0, -2.0, -3.0]]),
        torch.tensor([[-2.0, -3.0, -4.0]]),
        torch.tensor([[1.0, 2.0, 4.0]]),
        config=AlgorithmConfig(),
        loss_mask=torch.tensor([[1.0, 1.0, 0.0]]),
        rollout_logprobs=rollout_logprobs,
    )

    assert loss.item() == pytest.approx(5.0)
    assert metrics == {"clip_ratio": 0.0}


@pytest.mark.parametrize(
    "loss_mask, expected_loss, expected_gradient",
    [
        (None, 17.0, [[-1.0, -2.0, -4.0]]),
        (torch.tensor([[1.0, 1.0, 0.0]]), 5.0, [[-1.0, -2.0, 0.0]]),
        (torch.zeros(1, 3), 0.0, [[0.0, 0.0, 0.0]]),
    ],
)
def test_custom_policy_loss_sum_and_gradients(loss_mask, expected_loss, expected_gradient):
    log_probs = torch.tensor([[-1.0, -2.0, -3.0]], requires_grad=True)

    loss, _ = compute_reinforce_policy_loss(
        log_probs,
        log_probs.detach(),
        torch.tensor([[1.0, 2.0, 4.0]]),
        AlgorithmConfig(),
        loss_mask=loss_mask,
    )
    loss.backward()

    assert loss.item() == pytest.approx(expected_loss)
    torch.testing.assert_close(log_probs.grad, torch.tensor(expected_gradient))


def test_custom_policy_loss_preserves_minibatch_normalization():
    log_probs = torch.tensor([[-1.0, -2.0, -3.0], [-2.0, -4.0, -1.0]], requires_grad=True)
    loss_mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    advantages = apply_loss_reduction_to_advantages_minibatch(
        torch.tensor([[1.0, 2.0, 4.0], [3.0, 5.0, 7.0]]),
        loss_mask,
        loss_reduction="token_mean",
        micro_batch_size=1,
        max_seq_len=3,
    )
    config = AlgorithmConfig()
    losses = []
    for index in range(2):
        loss, _ = compute_reinforce_policy_loss(
            log_probs[index : index + 1],
            log_probs.detach()[index : index + 1],
            advantages[index : index + 1],
            config,
            loss_mask=loss_mask[index : index + 1],
        )
        losses.append(loss)
    loss = torch.stack(losses).sum()
    loss.backward()

    assert loss.item() == pytest.approx(4.5)
    torch.testing.assert_close(log_probs.grad, torch.tensor([[-0.25, -0.5, 0.0], [-0.75, 0.0, -1.75]]))
