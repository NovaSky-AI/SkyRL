# This code is adapted from VERL
# https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py
# The original copyright is reproduced below:
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from enum import Enum
from typing import Callable, Dict, List, Tuple, Union, Optional

import torch
import numpy as np

from omegaconf import DictConfig
from skyrl_train.training_batch import TrainingInputBatch
from jaxtyping import Float

import ray

# Import cloudpickle for function serialization
try:
    import cloudpickle
except ImportError:
    # Fallback to pickle if cloudpickle is not available
    import pickle as cloudpickle


# Shared registry actor class for both policy loss and advantage estimator registries
@ray.remote
class RegistryActor:
    """Shared Ray actor for managing function registries across processes."""

    def __init__(self):
        self.registry = {}

    def register(self, name: str, func_serialized: bytes):
        """Register a serialized function."""
        self.registry[name] = func_serialized

    def get(self, name: str):
        """Get a serialized function by name."""
        return self.registry.get(name)

    def list_available(self):
        """List all available function names."""
        return list(self.registry.keys())

    def unregister(self, name: str):
        """Unregister a function by name."""
        return self.registry.pop(name, None)


class AdvantageEstimator(Enum):
    GAE = "gae"
    GRPO = "grpo"

    def __str__(self):
        return self.value


class AdvantageEstimatorRegistry:
    """
    Registry for advantage estimator functions.

    This registry allows users to register custom advantage estimators without modifying
    the skyrl_train package. Custom estimators can be registered by calling
    AdvantageEstimatorRegistry.register() directly or by using the @register_advantage_estimator
    decorator.

    See examples/algorithm/custom_advantage_estimator for a simple example of how to
    register and use custom advantage estimators.
    """

    _estimators: Dict[str, Callable] = {}
    _ray_actor = None

    @classmethod
    def _get_or_create_actor(cls):
        """Get or create the Ray actor for managing the registry using get_if_exists."""
        if cls._ray_actor is None:
            try:
                # Use get_if_exists to create actor only if it doesn't exist
                cls._ray_actor = RegistryActor.options(name="advantage_estimator_registry", get_if_exists=True).remote()
            except Exception:
                # If Ray is not available, return None
                cls._ray_actor = None
        return cls._ray_actor

    @classmethod
    def _sync_with_actor(cls):
        """Sync local registry with Ray actor if Ray is available."""
        try:
            actor = cls._get_or_create_actor()
            if actor is None:
                return

            available = ray.get(actor.list_available.remote())

            # Sync any new functions from actor to local registry
            for name in available:
                if name not in cls._estimators:
                    func_serialized = ray.get(actor.get.remote(name))
                    if func_serialized is not None:
                        # Deserialize the function
                        try:
                            func = cloudpickle.loads(func_serialized)
                            cls._estimators[name] = func
                        except Exception:
                            # If deserialization fails, skip this function
                            pass

        except Exception:
            # Ray not available or other error, continue with local registry
            pass

    @classmethod
    def register(cls, name: Union[str, AdvantageEstimator], func: Callable):
        """Register an advantage estimator function."""
        # Convert enum to string if needed
        if isinstance(name, AdvantageEstimator):
            name = name.value

        if name in cls._estimators:
            raise ValueError(f"Estimator '{name}' already registered")

        # Store in local registry
        cls._estimators[name] = func

        # Try to sync with Ray actor if available
        try:
            actor = cls._get_or_create_actor()
            if actor is not None:
                # Serialize the function using cloudpickle
                func_serialized = cloudpickle.dumps(func)
                ray.get(actor.register.remote(name, func_serialized))
        except Exception:
            # Ray not available or other error, continue with local registry
            pass

    @classmethod
    def get(cls, name: str) -> Callable:
        """Get an estimator function by name."""
        # Try to sync with actor first
        cls._sync_with_actor()

        if name not in cls._estimators:
            available = list(cls._estimators.keys())
            raise ValueError(f"Unknown estimator '{name}'. Available: {available}")
        return cls._estimators[name]

    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered estimators."""
        # Try to sync with actor first
        cls._sync_with_actor()
        return list(cls._estimators.keys())

    @classmethod
    def unregister(cls, name: Union[str, AdvantageEstimator]):
        """Unregister an advantage estimator function. Useful for testing."""
        # Convert enum to string if needed
        if isinstance(name, AdvantageEstimator):
            name = name.value

        if name not in cls._estimators:
            raise ValueError(f"Estimator '{name}' not registered")

        # Remove from local registry
        del cls._estimators[name]

        # Try to sync with Ray actor if available
        try:
            actor = cls._get_or_create_actor()
            if actor is not None:
                ray.get(actor.unregister.remote(name))
        except Exception:
            # Ray not available or other error, continue with local registry
            pass


def register_advantage_estimator(name: Union[str, AdvantageEstimator]):
    """Decorator to register an advantage estimator function."""

    def decorator(func: Callable):
        AdvantageEstimatorRegistry.register(name, func)
        return func

    return decorator


class PolicyLossType(Enum):
    REGULAR = "regular"
    DUAL_CLIP = "dual_clip"

    def __str__(self):
        return self.value


class PolicyLossRegistry:
    """
    Registry for policy loss functions.

    This registry allows users to register custom policy loss functions without modifying
    the skyrl_train package. Custom functions can be registered by calling
    PolicyLossRegistry.register() directly or by using the @register_policy_loss
    decorator.

    See examples/algorithm/custom_policy_loss for a simple example of how to
    register and use custom policy loss functions.
    """

    _loss_functions: Dict[str, Callable] = {}
    _ray_actor = None

    @classmethod
    def _get_or_create_actor(cls):
        """Get or create the Ray actor for managing the registry using get_if_exists."""
        if cls._ray_actor is None:
            try:
                # Use get_if_exists to create actor only if it doesn't exist
                cls._ray_actor = RegistryActor.options(name="policy_loss_registry", get_if_exists=True).remote()
            except Exception:
                # If Ray is not available, return None
                cls._ray_actor = None
        return cls._ray_actor

    @classmethod
    def _sync_with_actor(cls):
        """Sync local registry with Ray actor if Ray is available."""
        try:
            actor = cls._get_or_create_actor()
            if actor is None:
                return

            available = ray.get(actor.list_available.remote())

            # Sync any new functions from actor to local registry
            for name in available:
                if name not in cls._loss_functions:
                    func_serialized = ray.get(actor.get.remote(name))
                    if func_serialized is not None:
                        # Deserialize the function
                        try:
                            func = cloudpickle.loads(func_serialized)
                            cls._loss_functions[name] = func
                        except Exception:
                            # If deserialization fails, skip this function
                            pass

        except Exception:
            # Ray not available or other error, continue with local registry
            pass

    @classmethod
    def register(cls, name: Union[str, PolicyLossType], func: Callable):
        """Register a policy loss function."""
        # Convert enum to string if needed
        if isinstance(name, PolicyLossType):
            name = name.value

        if name in cls._loss_functions:
            raise ValueError(f"Policy loss '{name}' already registered")

        # Store in local registry
        cls._loss_functions[name] = func

        # Try to sync with Ray actor if available
        try:
            actor = cls._get_or_create_actor()
            if actor is not None:
                # Serialize the function using cloudpickle
                func_serialized = cloudpickle.dumps(func)
                ray.get(actor.register.remote(name, func_serialized))
        except Exception:
            # Ray not available or other error, continue with local registry
            pass

    @classmethod
    def get(cls, name: str) -> Callable:
        """Get a policy loss function by name."""
        # Try to sync with actor first
        cls._sync_with_actor()

        if name not in cls._loss_functions:
            available = list(cls._loss_functions.keys())
            raise ValueError(f"Unknown policy loss '{name}'. Available: {available}")
        return cls._loss_functions[name]

    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered policy loss functions."""
        # Try to sync with actor first
        cls._sync_with_actor()
        return list(cls._loss_functions.keys())

    @classmethod
    def unregister(cls, name: Union[str, PolicyLossType]):
        """Unregister a policy loss function. Useful for testing."""
        # Convert enum to string if needed
        if isinstance(name, PolicyLossType):
            name = name.value

        if name not in cls._loss_functions:
            raise ValueError(f"Policy loss '{name}' not registered")

        # Remove from local registry
        del cls._loss_functions[name]

        # Try to sync with Ray actor if available
        try:
            actor = cls._get_or_create_actor()
            if actor is not None:
                ray.get(actor.unregister.remote(name))
        except Exception:
            # Ray not available or other error, continue with local registry
            pass


def register_policy_loss(name: Union[str, PolicyLossType]):
    """Decorator to register a policy loss function."""

    def decorator(func: Callable):
        PolicyLossRegistry.register(name, func)
        return func

    return decorator


@register_policy_loss(PolicyLossType.REGULAR)
@register_policy_loss(PolicyLossType.DUAL_CLIP)
def ppo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    config: DictConfig,
    loss_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    assert config.policy_loss_type in ["regular", "dual_clip"], "loss_type must be either 'regular' or 'dual_clip'"
    loss_reduction = config.loss_reduction
    assert loss_reduction in [
        "token_mean",
        "sequence_mean",
    ], "loss_reduction must be either 'token_mean' or 'sequence_mean'"

    ratio = (log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1 - config.eps_clip_low, 1 + config.eps_clip_high) * advantages
    loss = -torch.min(surr1, surr2)
    clip_ratio = masked_mean((-surr2 > -surr1).float(), loss_mask).mean().detach().item()
    clip_pg_losses1 = loss
    if config.policy_loss_type == "dual_clip":
        pg_losses3 = -advantages * config.clip_ratio_c
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        loss = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    loss = reduce_loss(loss, loss_mask, loss_reduction)
    return loss, clip_ratio


def reduce_loss(loss: torch.Tensor, loss_mask: Optional[torch.Tensor], loss_reduction: str) -> torch.Tensor:
    if loss_reduction == "token_mean":
        # sum over *all* valid tokens, divide by total valid-token count
        loss = masked_mean(loss, loss_mask)
    elif loss_reduction == "sequence_mean":
        # per-sequence token-mean (dim=-1), then batch-mean
        loss = masked_mean(loss, loss_mask, dim=-1).mean()
    else:
        raise ValueError(f"Invalid loss reduction type: {loss_reduction}")
    return loss


# TODO (erictang000): unused right now, but will be useful as we add more algorithm support
class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        target = self.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass


def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: Optional[int] = None) -> torch.Tensor:
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim).clamp(min=1.0)


@torch.no_grad()
def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    use_kl_estimator_k3: bool = False,
    use_abs_kl: bool = False,
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        action_mask: Mask for actions.
    """

    log_ratio = log_probs - log_probs_base

    # The k3 estimator is the non negative kl approximation in
    # http://joschu.net/blog/kl-approx.html
    # Besides non negative, it is also unbiased and have lower variance.
    if use_kl_estimator_k3:
        log_ratio = -log_ratio
        log_ratio = log_ratio.exp() - 1 - log_ratio

    if use_abs_kl:
        log_ratio = log_ratio.abs()

    if loss_mask is not None:
        log_ratio = log_ratio * loss_mask

    return log_ratio


@torch.no_grad()
def normalize_advantages_dict(data: TrainingInputBatch) -> TrainingInputBatch:
    """Normalizes the advantages in the data batch.

    Expects:
        - `["advantages"]`: Float[torch.Tensor, "batch_size seqlen"]
        - `["response_mask"]`: Float[torch.Tensor, "batch_size seqlen"]
    """
    advantages: Float[torch.Tensor, "batch_size seqlen"] = data["advantages"]
    response_masks: Float[torch.Tensor, "batch_size seqlen"] = data["response_mask"]
    num_actions: float = response_masks.sum()
    # mean
    mean: float = advantages.mean()
    # std
    std: float = ((advantages - mean).pow(2) * response_masks).sum()
    rstd: float = (std / num_actions).clamp(min=1e-8).rsqrt()

    data["advantages"] = (advantages - mean) * rstd
    return data


# NOTE (erictang000): below ported from verl
def masked_var(values, mask, unbiased=True):
    """Compute variance of tensor with masked values."""
    mean = masked_mean(values, mask)
    centered_values = values - mean
    variance = masked_mean(centered_values**2, mask)
    if unbiased:
        mask_sum = mask.sum()
        if mask_sum == 0:
            raise ValueError("At least one element in the mask has to be 1.")
        # note that if mask_sum == 1, then there is a division by zero issue
        # to avoid it you just need to use a larger minibatch_size
        if mask_sum == 1:
            raise ValueError("The sum of the mask is one, which can cause a division by zero.")
        bessel_correction = mask_sum / (mask_sum - 1)
        variance = variance * bessel_correction
    return variance


def masked_whiten(values, mask, shift_mean=True):
    """Whiten values with masked values."""
    mean, var = masked_mean(values, mask), masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


@register_advantage_estimator(AdvantageEstimator.GAE)
def compute_gae_advantage_return(
    token_level_rewards: Float[torch.Tensor, "batch_size seqlen"],
    values: Float[torch.Tensor, "batch_size seqlen"],
    response_mask: Float[torch.Tensor, "batch_size seqlen"],
    gamma: float,
    lambd: float,
    **kwargs,
) -> Tuple[Float[torch.Tensor, "batch_size seqlen"], Float[torch.Tensor, "batch_size seqlen"]]:
    """
    Compute advantage and return for GAE.

    Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py
    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = masked_whiten(advantages, response_mask)
    return advantages, returns


@register_advantage_estimator(AdvantageEstimator.GRPO)
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    **kwargs,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward (with only one scalar reward for each response).

    Expects:
        - token_level_rewards: Float[torch.Tensor, "batch_size seqlen"]
        - response_mask: Float[torch.Tensor, "batch_size seqlen"]
        - index: np.ndarray (batch_size)
        - epsilon: float
        - norm_adv_by_std_in_grpo: bool

    Returns:
        - advantages: Float[torch.Tensor, "batch_size seqlen"]
        - returns: Float[torch.Tensor, "batch_size seqlen"]
    """
    # this assumes response-level rewards
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_advantages_and_returns(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    adv_estimator: Union[str, AdvantageEstimator],
    values: Optional[torch.Tensor] = None,
    norm_adv_by_std_in_grpo: bool = True,
    gamma=1.0,
    lambd=1.0,
):
    if isinstance(adv_estimator, AdvantageEstimator):
        estimator_name = adv_estimator.value
    else:
        estimator_name = adv_estimator

    estimator_func = AdvantageEstimatorRegistry.get(estimator_name)

    return estimator_func(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        values=values,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        gamma=gamma,
        lambd=lambd,
    )
