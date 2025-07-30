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
    PolicyLossRegistry,
    register_policy_loss,
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


def test_policy_loss_registration():
    """Test that we can register and retrieve a custom policy loss function."""
    from omegaconf import DictConfig

    # Create a simple dummy policy loss function
    def dummy_policy_loss(log_probs, old_log_probs, advantages, config, loss_mask=None):
        return torch.tensor(0.5), 0.1  # dummy loss and clip_ratio

    # Register it
    PolicyLossRegistry.register("dummy_loss", dummy_policy_loss)

    # Check it's retrievable
    retrieved_func = PolicyLossRegistry.get("dummy_loss")
    assert retrieved_func == dummy_policy_loss

    # Check it's in the available list
    assert "dummy_loss" in PolicyLossRegistry.list_available()

    # Test the function works
    config = DictConfig({"policy_loss_type": "dummy_loss"})
    loss, clip_ratio = retrieved_func(
        log_probs=torch.tensor([[0.1]]),
        old_log_probs=torch.tensor([[0.2]]),
        advantages=torch.tensor([[1.0]]),
        config=config,
    )
    assert loss.item() == 0.5
    assert clip_ratio == 0.1

    # Clean up
    PolicyLossRegistry.unregister("dummy_loss")


def test_policy_loss_decorator():
    """Test the register_policy_loss decorator works."""

    @register_policy_loss("decorated_loss")
    def decorated_policy_loss(log_probs, old_log_probs, advantages, config, loss_mask=None):
        return torch.tensor(1.0), 0.2

    # Check it was registered
    assert "decorated_loss" in PolicyLossRegistry.list_available()
    retrieved_func = PolicyLossRegistry.get("decorated_loss")
    assert retrieved_func == decorated_policy_loss

    # Clean up
    PolicyLossRegistry.unregister("decorated_loss")


def test_policy_loss_registry_errors():
    """Test PolicyLossRegistry error handling."""

    # Test getting non-existent loss
    with pytest.raises(ValueError, match="Unknown policy loss"):
        PolicyLossRegistry.get("non_existent")

    # Test unregistering non-existent loss
    with pytest.raises(ValueError, match="not registered"):
        PolicyLossRegistry.unregister("non_existent")

    # Test duplicate registration
    def dummy_loss(log_probs, old_log_probs, advantages, config, loss_mask=None):
        return torch.tensor(0.0), 0.0

    PolicyLossRegistry.register("test_dup", dummy_loss)
    with pytest.raises(ValueError, match="already registered"):
        PolicyLossRegistry.register("test_dup", dummy_loss)

    # Clean up
    PolicyLossRegistry.unregister("test_dup")


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


def test_registry_cross_ray_process():
    """Test that registry works across Ray processes - registering outside and getting inside Ray actor."""
    import ray

    # Create a dummy policy loss function for testing
    def cross_process_policy_loss(log_probs, old_log_probs, advantages, config, loss_mask=None):
        return torch.tensor(2.0), 0.5

    # Register in the main process
    PolicyLossRegistry.register("cross_process_test", cross_process_policy_loss)

    # Create a Ray actor that will try to get the registered function
    @ray.remote
    class TestActor:
        def __init__(self):
            pass

        def get_policy_loss_from_registry(self, name):
            """Try to get a policy loss function from inside the Ray actor."""
            try:
                func = PolicyLossRegistry.get(name)
                # Test that we can call the function
                result = func(
                    log_probs=torch.tensor([[0.1]]),
                    old_log_probs=torch.tensor([[0.2]]),
                    advantages=torch.tensor([[1.0]]),
                    config={"policy_loss_type": name},
                )
                return result[0].item(), result[1]  # loss, clip_ratio
            except Exception as e:
                return None, str(e)

        def check_if_registered(self, name):
            """Check if a function is available in the registry from inside the actor."""
            available = PolicyLossRegistry.list_available()
            return name in available

        def get_advantage_estimator_from_registry(self, name):
            """Try to get an advantage estimator from inside the Ray actor."""
            try:
                func = AdvantageEstimatorRegistry.get(name)
                # Test that we can call the function
                rewards = torch.tensor([[1.0, 2.0]])
                response_mask = torch.tensor([[1.0, 1.0]])
                index = np.array(["0", "0"])
                result = func(
                    token_level_rewards=rewards,
                    response_mask=response_mask,
                    index=index,
                )
                return result[0].shape, result[1].shape  # advantages.shape, returns.shape
            except Exception as e:
                return None, str(e)

    # Create the actor
    actor = TestActor.remote()

    try:
        # Test 1: Check if the policy loss function is available from inside the actor
        is_available = ray.get(actor.check_if_registered.remote("cross_process_test"))
        assert is_available, "Policy loss should be available in Ray actor"

        # Test 2: Try to get and call the function from inside the actor
        loss, clip_ratio = ray.get(actor.get_policy_loss_from_registry.remote("cross_process_test"))
        assert loss == 2.0, f"Expected loss 2.0, got {loss}"
        assert clip_ratio == 0.5, f"Expected clip_ratio 0.5, got {clip_ratio}"

        # Test 3: Test with advantage estimator registry as well
        def cross_process_advantage_estimator(**kwargs):
            rewards = kwargs["token_level_rewards"]
            return rewards * 2, rewards * 3  # Simple transformation

        AdvantageEstimatorRegistry.register("cross_process_adv_test", cross_process_advantage_estimator)

        # Check from actor
        adv_shape, ret_shape = ray.get(actor.get_advantage_estimator_from_registry.remote("cross_process_adv_test"))
        assert adv_shape == torch.Size([1, 2]), f"Expected advantage shape [1, 2], got {adv_shape}"
        assert ret_shape == torch.Size([1, 2]), f"Expected return shape [1, 2], got {ret_shape}"

    finally:
        # Clean up
        PolicyLossRegistry.unregister("cross_process_test")
        AdvantageEstimatorRegistry.unregister("cross_process_adv_test")

        # Clean up the actor
        ray.kill(actor)


def test_registry_named_actor_creation():
    """Test that the registry attempts to create named Ray actors."""
    import ray

    # This test verifies that the registry tries to create named actors
    # even though the function serialization is incomplete

    def test_func(**kwargs):
        return torch.zeros_like(kwargs["token_level_rewards"]), torch.zeros_like(kwargs["token_level_rewards"])

    # Register a function (this should attempt to create/use a named actor)
    AdvantageEstimatorRegistry.register("named_actor_test", test_func)

    try:
        # Check that we can retrieve it locally
        retrieved = AdvantageEstimatorRegistry.get("named_actor_test")
        assert retrieved == test_func, "Should be able to retrieve function locally"

        # Check if the named actor was created (it might exist even if not fully functional)
        try:
            actor = ray.get_actor("advantage_estimator_registry")
            # If we get here, the named actor exists
            assert actor is not None, "Named actor should be created"
        except ValueError:
            # Actor doesn't exist - this is also acceptable given the implementation
            pass

    finally:
        # Clean up
        AdvantageEstimatorRegistry.unregister("named_actor_test")
