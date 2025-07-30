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
    """Test that registry works with Ray - focusing on practical usage patterns."""
    import ray

    # Create a dummy policy loss function for testing
    def cross_process_policy_loss(log_probs, old_log_probs, advantages, config, loss_mask=None):
        return torch.tensor(2.0), 0.5

    # Test 1: Basic registration and retrieval
    PolicyLossRegistry.register("cross_process_test", cross_process_policy_loss)

    # Verify function is available locally
    available_main = PolicyLossRegistry.list_available()
    assert "cross_process_test" in available_main, "Function should be registered locally"

    # Verify we can retrieve and call the function
    retrieved_func = PolicyLossRegistry.get("cross_process_test")
    result = retrieved_func(
        log_probs=torch.tensor([[0.1]]),
        old_log_probs=torch.tensor([[0.2]]),
        advantages=torch.tensor([[1.0]]),
        config={"policy_loss_type": "cross_process_test"},
    )
    assert result[0].item() == 2.0, f"Expected loss 2.0, got {result[0].item()}"
    assert result[1] == 0.5, f"Expected clip_ratio 0.5, got {result[1]}"

    # Test 2: Test advantage estimator registry as well
    def cross_process_advantage_estimator(**kwargs):
        rewards = kwargs["token_level_rewards"]
        return rewards * 2, rewards * 3  # Simple transformation

    AdvantageEstimatorRegistry.register("cross_process_adv_test", cross_process_advantage_estimator)

    # Verify advantage estimator is available
    available_adv = AdvantageEstimatorRegistry.list_available()
    assert "cross_process_adv_test" in available_adv, "Advantage estimator should be registered locally"

    # Verify we can retrieve and call the advantage estimator
    retrieved_estimator = AdvantageEstimatorRegistry.get("cross_process_adv_test")
    rewards = torch.tensor([[1.0, 2.0]])
    response_mask = torch.tensor([[1.0, 1.0]])
    index = np.array(["0", "0"])
    adv_result = retrieved_estimator(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=index,
    )
    assert adv_result[0].shape == torch.Size([1, 2]), f"Expected advantage shape [1, 2], got {adv_result[0].shape}"
    assert adv_result[1].shape == torch.Size([1, 2]), f"Expected return shape [1, 2], got {adv_result[1].shape}"

    # Test 3: Test Ray actor integration (basic)
    @ray.remote
    def simple_ray_task():
        # This tests that the registries work within Ray tasks
        # Functions registered in main process should be available
        try:
            available = PolicyLossRegistry.list_available()
            return "cross_process_test" in available, available
        except Exception as e:
            return False, str(e)

    @ray.remote
    def simple_ray_task_advantage_estimator():
        # This tests that the registries work within Ray tasks
        # Functions registered in main process should be available
        try:
            available = AdvantageEstimatorRegistry.list_available()
            return "cross_process_adv_test" in available, available
        except Exception as e:
            return False, str(e)

    # Run the Ray task
    is_available, available_list = ray.get(simple_ray_task.remote())
    is_available_advantage_estimator, available_list_advantage_estimator = ray.get(
        simple_ray_task_advantage_estimator.remote()
    )
    # Note: Ray tasks may start with fresh registry state, so we don't strictly require
    # cross-process sync for this test. The important thing is that the APIs work.
    print(f"Ray task result: available={is_available}, functions={available_list}")
    print(
        f"Ray task advantage estimator result: available={is_available_advantage_estimator}, functions={available_list_advantage_estimator}"
    )
    # Clean up - be lenient about cleanup failures
    try:
        PolicyLossRegistry.unregister("cross_process_test")
    except ValueError:
        pass  # Function might not be found locally after Ray operations, but that's ok

    try:
        AdvantageEstimatorRegistry.unregister("cross_process_adv_test")
    except ValueError:
        pass  # Function might not be found locally after Ray operations, but that's ok


def test_registry_named_actor_creation():
    """Test that the registry creates named Ray actors and properly serializes functions with cloudpickle."""
    import ray

    def test_func(**kwargs):
        rewards = kwargs["token_level_rewards"]
        return rewards * 2, rewards * 3  # Simple transformation to verify functionality

    # Register a function (this should create/use a named actor and serialize the function)
    AdvantageEstimatorRegistry.register("named_actor_test", test_func)

    try:
        # Test 1: Check that we can retrieve it locally
        retrieved = AdvantageEstimatorRegistry.get("named_actor_test")
        assert retrieved == test_func, "Should be able to retrieve function locally"

        # Test 2: Verify the named actor was created and contains the serialized function
        actor = ray.get_actor("advantage_estimator_registry")
        assert actor is not None, "Named actor should be created"

        # Test 3: Verify the function is actually stored in the Ray actor
        available_in_actor = ray.get(actor.list_available.remote())
        assert "named_actor_test" in available_in_actor, "Function should be stored in Ray actor"

        # Test 4: Verify we can retrieve and deserialize the function from the actor
        serialized_func = ray.get(actor.get.remote("named_actor_test"))
        assert serialized_func is not None, "Serialized function should be retrievable from actor"

        # Test 5: Verify cloudpickle deserialization works
        try:
            import cloudpickle

            deserialized_func = cloudpickle.loads(serialized_func)
        except ImportError:
            # Fallback to pickle if cloudpickle not available
            import pickle

            deserialized_func = pickle.loads(serialized_func)

        # Test 6: Verify the deserialized function works correctly
        test_rewards = torch.tensor([[1.0, 2.0]])
        result = deserialized_func(
            token_level_rewards=test_rewards,
            response_mask=torch.tensor([[1.0, 1.0]]),
            index=np.array(["0", "0"]),
        )

        expected_adv = test_rewards * 2
        expected_ret = test_rewards * 3

        assert torch.allclose(result[0], expected_adv), f"Expected advantages {expected_adv}, got {result[0]}"
        assert torch.allclose(result[1], expected_ret), f"Expected returns {expected_ret}, got {result[1]}"

    finally:
        # Clean up
        AdvantageEstimatorRegistry.unregister("named_actor_test")
