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


def test_base_function_registry_registration_and_retrieval():
    """Test basic registration and retrieval functionality of BaseFunctionRegistry."""

    def dummy_function(**kwargs):
        return torch.zeros_like(kwargs["token_level_rewards"]), torch.zeros_like(kwargs["token_level_rewards"])

    # Register function
    AdvantageEstimatorRegistry.register("test_basic", dummy_function)

    # Test retrieval
    retrieved_func = AdvantageEstimatorRegistry.get("test_basic")
    assert retrieved_func == dummy_function

    # Test it's in available list
    assert "test_basic" in AdvantageEstimatorRegistry.list_available()

    # Clean up
    AdvantageEstimatorRegistry.unregister("test_basic")


def test_base_function_registry_error_handling():
    """Test error handling in BaseFunctionRegistry."""

    def dummy_function(**kwargs):
        return None, None

    # Test getting non-existent function
    with pytest.raises(ValueError, match="Unknown advantage estimator"):
        AdvantageEstimatorRegistry.get("non_existent")

    # Test unregistering non-existent function
    with pytest.raises(ValueError, match="not registered"):
        AdvantageEstimatorRegistry.unregister("non_existent")

    # Test duplicate registration
    AdvantageEstimatorRegistry.register("test_dup", dummy_function)
    with pytest.raises(ValueError, match="already registered"):
        AdvantageEstimatorRegistry.register("test_dup", dummy_function)

    # Clean up
    AdvantageEstimatorRegistry.unregister("test_dup")


def test_base_registry_unregister():
    """Test unregistration functionality."""

    def dummy_function(**kwargs):
        return torch.zeros_like(kwargs["token_level_rewards"]), torch.zeros_like(kwargs["token_level_rewards"])

    # Register and verify
    AdvantageEstimatorRegistry.register("test_unregister", dummy_function)
    assert "test_unregister" in AdvantageEstimatorRegistry.list_available()

    # Unregister and verify
    AdvantageEstimatorRegistry.unregister("test_unregister")
    assert "test_unregister" not in AdvantageEstimatorRegistry.list_available()


def test_advantage_estimator_registry_specific():
    """Test AdvantageEstimatorRegistry-specific functionality."""

    @register_advantage_estimator("test_decorator")
    def decorated_estimator(**kwargs):
        return torch.ones_like(kwargs["token_level_rewards"]), torch.ones_like(kwargs["token_level_rewards"])

    # Test decorator worked
    assert "test_decorator" in AdvantageEstimatorRegistry.list_available()
    retrieved = AdvantageEstimatorRegistry.get("test_decorator")
    assert retrieved == decorated_estimator

    # Test integration with compute_advantages_and_returns
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    response_mask = torch.tensor([[1.0, 1.0, 1.0]])
    index = np.array(["0", "0", "0"])

    adv, ret = compute_advantages_and_returns(
        token_level_rewards=rewards, response_mask=response_mask, index=index, adv_estimator="test_decorator"
    )

    assert torch.allclose(adv, torch.ones_like(rewards))
    assert torch.allclose(ret, torch.ones_like(rewards))

    # Clean up
    AdvantageEstimatorRegistry.unregister("test_decorator")


def test_policy_loss_registry_specific():
    """Test PolicyLossRegistry-specific functionality."""
    from omegaconf import DictConfig

    @register_policy_loss("test_policy_decorator")
    def decorated_policy_loss(log_probs, old_log_probs, advantages, config, loss_mask=None):
        return torch.tensor(1.5), 0.3

    # Test decorator worked
    assert "test_policy_decorator" in PolicyLossRegistry.list_available()
    retrieved = PolicyLossRegistry.get("test_policy_decorator")
    assert retrieved == decorated_policy_loss

    # Test function execution
    config = DictConfig({"policy_loss_type": "test_policy_decorator"})
    loss, clip_ratio = retrieved(
        log_probs=torch.tensor([[0.1]]),
        old_log_probs=torch.tensor([[0.2]]),
        advantages=torch.tensor([[1.0]]),
        config=config,
    )
    assert loss.item() == 1.5
    assert clip_ratio == 0.3

    # Test error message includes "Policy loss"
    with pytest.raises(ValueError, match="Unknown policy loss"):
        PolicyLossRegistry.get("non_existent_policy")

    # Clean up
    PolicyLossRegistry.unregister("test_policy_decorator")


def test_registry_cross_ray_process():
    """Test that registry works with Ray and that functions can be retrieved and called from different processes"""
    try:
        import ray
        from omegaconf import DictConfig

        if not ray.is_initialized():
            ray.init()

        # Create test functions
        def test_policy_loss(log_probs, old_log_probs, advantages, config, loss_mask=None):
            return torch.tensor(2.0), 0.5

        def test_advantage_estimator(**kwargs):
            rewards = kwargs["token_level_rewards"]
            return rewards * 2, rewards * 3

        # Test basic registration and retrieval
        PolicyLossRegistry.register("cross_process_test", test_policy_loss)
        AdvantageEstimatorRegistry.register("cross_process_adv_test", test_advantage_estimator)

        # Test Ray integration
        @ray.remote
        def test_ray_registry_access():
            policy_loss = PolicyLossRegistry.get("cross_process_test")
            adv_estimator = AdvantageEstimatorRegistry.get("cross_process_adv_test")

            loss, clip_ratio = policy_loss(
                log_probs=torch.tensor([[0.1]]),
                old_log_probs=torch.tensor([[0.2]]),
                advantages=torch.tensor([[1.0]]),
                config=DictConfig({"policy_loss_type": "cross_process_test"}),
            )

            adv, ret = adv_estimator(
                token_level_rewards=torch.tensor([[1.0, 2.0]]),
                response_mask=torch.tensor([[1.0, 1.0]]),
                index=np.array(["0", "0"]),
            )
            return loss, clip_ratio, adv, ret

        # Run Ray task
        loss, clip_ratio, adv, ret = ray.get(test_ray_registry_access.remote())
        assert loss.item() == 2.0
        assert clip_ratio == 0.5
        assert adv.shape == torch.Size([1, 2])
        assert ret.shape == torch.Size([1, 2])
    finally:
        # Clean up
        PolicyLossRegistry.unregister("cross_process_test")
        AdvantageEstimatorRegistry.unregister("cross_process_adv_test")
        PolicyLossRegistry._ray_actor = None
        PolicyLossRegistry._synced_to_actor = False
        AdvantageEstimatorRegistry._ray_actor = None
        AdvantageEstimatorRegistry._synced_to_actor = False


def test_registry_named_actor_creation():
    """Test that the registry creates named Ray actors and properly serializes functions."""
    try:
        import ray

        if not ray.is_initialized():
            ray.init()

        def test_func(**kwargs):
            rewards = kwargs["token_level_rewards"]
            return rewards * 2, rewards * 3

        # Register function (should create/use named actor)
        AdvantageEstimatorRegistry.register("named_actor_test", test_func)

        # Verify local retrieval works
        retrieved = AdvantageEstimatorRegistry.get("named_actor_test")
        assert retrieved == test_func

        # Verify named actor exists and contains function
        actor = ray.get_actor("advantage_estimator_registry")
        assert actor is not None

        available_in_actor = ray.get(actor.list_available.remote())
        assert "named_actor_test" in available_in_actor

        # Verify function serialization/deserialization
        serialized_func = ray.get(actor.get.remote("named_actor_test"))
        assert serialized_func is not None

        import cloudpickle

        deserialized_func = cloudpickle.loads(serialized_func)

        # Test deserialized function works
        test_rewards = torch.tensor([[1.0, 2.0]])
        result = deserialized_func(
            token_level_rewards=test_rewards,
            response_mask=torch.tensor([[1.0, 1.0]]),
            index=np.array(["0", "0"]),
        )

        assert torch.allclose(result[0], test_rewards * 2)
        assert torch.allclose(result[1], test_rewards * 3)

    finally:
        # Clean up
        AdvantageEstimatorRegistry.unregister("named_actor_test")
        AdvantageEstimatorRegistry._ray_actor = None
        AdvantageEstimatorRegistry._synced_to_actor = False
