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
    compute_advantages_and_returns,
    AdaptiveKLController,
    FixedKLController,
    AdvantageEstimatorRegistry,
    register_advantage_estimator,
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


def test_advantage_estimator_registration():
    """Test that we can register and retrieve a custom estimator."""

    # Create a simple dummy estimator
    def dummy_estimator(**kwargs):
        return torch.zeros_like(kwargs["token_level_rewards"]), torch.zeros_like(kwargs["token_level_rewards"])

    # Register it
    AdvantageEstimatorRegistry.register("dummy", dummy_estimator)

    # Check it's retrievable
    retrieved_func = AdvantageEstimatorRegistry.get("dummy")
    assert retrieved_func == dummy_estimator

    # Check it's in the available list
    assert "dummy" in AdvantageEstimatorRegistry.list_available()

    # Clean up
    AdvantageEstimatorRegistry.unregister("dummy")


def test_duplicate_registration_error():
    """Test that registering the same name twice raises an error."""

    def estimator1(**kwargs):
        return None, None

    def estimator2(**kwargs):
        return None, None

    # Register first one
    AdvantageEstimatorRegistry.register("duplicate_test", estimator1)

    # Try to register second one with same name - should fail
    with pytest.raises(ValueError, match="already registered"):
        AdvantageEstimatorRegistry.register("duplicate_test", estimator2)

    # Clean up
    AdvantageEstimatorRegistry.unregister("duplicate_test")


def test_unknown_estimator_error():
    """Test that getting an unknown estimator raises error."""
    with pytest.raises(ValueError, match="Unknown estimator.*Available:"):
        AdvantageEstimatorRegistry.get("nonexistent_estimator")


def test_decorator_registration():
    """Test that the decorator works for registration."""

    @register_advantage_estimator("decorated_estimator")
    def my_custom_estimator(**kwargs):
        return torch.ones_like(kwargs["token_level_rewards"]), torch.ones_like(kwargs["token_level_rewards"])

    # Check it was registered
    assert "decorated_estimator" in AdvantageEstimatorRegistry.list_available()

    # Check we can retrieve it
    retrieved = AdvantageEstimatorRegistry.get("decorated_estimator")
    assert retrieved == my_custom_estimator

    # Clean up
    AdvantageEstimatorRegistry.unregister("decorated_estimator")


def test_custom_estimator_integration(advantage_test_data):
    """Test that compute_advantages_and_returns works with custom estimators."""
    rewards, values, response_mask, index = advantage_test_data

    # Register a simple custom estimator
    @register_advantage_estimator("simple_test")
    def simple_estimator(**kwargs):
        # Just return the rewards as both advantages and returns
        r = kwargs["token_level_rewards"]
        return r, r

    # Use it in the main function
    adv, ret = compute_advantages_and_returns(
        token_level_rewards=rewards, response_mask=response_mask, index=index, adv_estimator="simple_test"
    )

    assert torch.allclose(adv, rewards)
    assert torch.allclose(ret, rewards)

    # Clean up
    AdvantageEstimatorRegistry.unregister("simple_test")


def test_unregister_estimator():
    """Test that we can unregister estimators."""

    def dummy_estimator(**kwargs):
        return torch.zeros_like(kwargs["token_level_rewards"]), torch.zeros_like(kwargs["token_level_rewards"])

    # Register it
    AdvantageEstimatorRegistry.register("unregister_test", dummy_estimator)
    assert "unregister_test" in AdvantageEstimatorRegistry.list_available()

    # Unregister it
    AdvantageEstimatorRegistry.unregister("unregister_test")
    assert "unregister_test" not in AdvantageEstimatorRegistry.list_available()


def test_unregister_nonexistent_error():
    """Test that unregistering a nonexistent estimator raises error."""
    with pytest.raises(ValueError, match="not registered"):
        AdvantageEstimatorRegistry.unregister("nonexistent_estimator")
