"""
Run with:
uv run --isolated --extra dev pytest tests/cpu/utils/test_ppo_utils.py
"""

import torch
import math
import pytest
from skyrl_train.utils.ppo_utils import (
    compute_approx_kl,
    compute_gae_advantage_return,
    compute_grpo_outcome_advantage,
    AdaptiveKLController,
    FixedKLController,
)
import numpy as np


@pytest.fixture
def dummy_data():
    log_probs = torch.tensor([[0.2, 0.3, 0.5]])
    log_probs_base = torch.tensor([[0.1, 0.2, 0.4]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])  # last value masked out
    return log_probs, log_probs_base, mask


@pytest.fixture
def advantage_test_data():
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    values = torch.tensor([[0.5, 1.0, 1.5]])
    response_mask = torch.tensor([[1.0, 1.0, 1.0]])
    index = np.array(["0", "0", "0"])
    return rewards, values, response_mask, index


def test_compute_approx_kl(dummy_data):
    log_probs, log_probs_base, mask = dummy_data
    kl = compute_approx_kl(log_probs, log_probs_base, mask)

    expected_kl = (log_probs - log_probs_base) * mask
    assert torch.allclose(kl, expected_kl), "KL approximation should be log-prob diff masked"

    kl_k3 = compute_approx_kl(log_probs, log_probs_base, mask, use_kl_estimator_k3=True)
    log_ratio = log_probs - log_probs_base
    expected_k3 = (torch.exp(-log_ratio) - 1 + log_ratio) * mask
    assert torch.allclose(kl_k3, expected_k3, atol=1e-4), "k3 estimator is not correct"


def test_compute_grpo_outcome_advantage(advantage_test_data):
    rewards, _, response_mask, index = advantage_test_data

    adv, ret = compute_grpo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=index,
    )

    assert adv.shape == rewards.shape
    assert ret.shape == rewards.shape
    assert torch.allclose(adv, ret), "Advantages and returns should be equal with GRPO"


def test_compute_gae_advantage_return(advantage_test_data):
    rewards, values, response_mask, index = advantage_test_data

    adv, ret = compute_gae_advantage_return(
        token_level_rewards=rewards,
        values=values,
        response_mask=response_mask,
        gamma=1.0,
        lambd=1.0,  # no discounting for simplicity
    )

    expected_ret = torch.tensor([[6.0, 5.0, 3.0]])

    # The advantages will be whitened, so we just check the shape and that they're not all zeros
    assert adv.shape == rewards.shape
    assert not torch.allclose(adv, torch.zeros_like(adv))
    assert ret.shape == expected_ret.shape
    assert torch.allclose(ret, expected_ret, atol=1e-5)


def test_compute_gae_advantage_return_with_masking(advantage_test_data):
    rewards, values, _, _ = advantage_test_data
    response_mask = torch.tensor([[1.0, 0.0, 1.0]])  # Mask out the second token

    adv, ret = compute_gae_advantage_return(
        token_level_rewards=rewards,
        values=values,
        response_mask=response_mask,
        gamma=1.0,
        lambd=1.0,  # no discounting for simplicity
    )

    # The returns should be reversed cumulative rewards
    expected_ret = torch.tensor([[6.0, 5.0, 3.0]])
    expected_adv = torch.tensor([[0.7071, 0.1768, -0.7071]])

    assert torch.allclose(ret, expected_ret, atol=1e-5)
    assert torch.allclose(adv, expected_adv, atol=1e-4)


def test_compute_gae_advantage_return_gamma(advantage_test_data):
    rewards, values, response_mask, _ = advantage_test_data

    _, ret = compute_gae_advantage_return(
        token_level_rewards=rewards,
        values=values,
        response_mask=response_mask,
        gamma=0.5,
        lambd=1.0,
    )

    expected_ret = torch.tensor([[2.7500, 3.5000, 3.0000]])
    assert torch.allclose(ret, expected_ret, atol=1e-5)


def test_compute_gae_advantage_return_lam(advantage_test_data):
    rewards, values, response_mask, _ = advantage_test_data

    _, ret = compute_gae_advantage_return(
        token_level_rewards=rewards,
        values=values,
        response_mask=response_mask,
        lambd=0.5,
        gamma=1.0,
    )

    expected_ret = torch.tensor([[3.6250, 4.2500, 3.0000]])
    assert torch.allclose(ret, expected_ret, atol=1e-5)


def test_adaptive_kl_controller_update():
    controller = AdaptiveKLController(init_kl_coef=0.2, target=0.1, horizon=100)
    controller.update(current=0.2, n_steps=10)

    # Expected error: (0.2 / 0.1 - 1) = 1 → clipped to 0.2
    # Mult = 1 + 0.2 * 10 / 100 = 1.02
    expected = 0.2 * 1.02
    assert math.isclose(controller.value, expected, rel_tol=1e-5)


def test_fixed_kl_controller():
    controller = FixedKLController(kl_coef=0.1)
    controller.update(current=1.0, n_steps=10)
    assert controller.value == 0.1  # Should remain unchanged
