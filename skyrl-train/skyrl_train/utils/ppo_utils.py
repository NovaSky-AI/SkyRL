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

import torch
import numpy as np
from typing import Optional, Tuple
from skyrl_train.training_batch import TrainingInputBatch
from jaxtyping import Float
from collections import defaultdict


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
    return (tensor * mask).sum(axis=dim) / (mask.sum(axis=dim) + 1e-8)


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


def compute_gae_advantage_return(
    token_level_rewards: Float[torch.Tensor, "batch_size seqlen"],
    values: Float[torch.Tensor, "batch_size seqlen"],
    response_mask: Float[torch.Tensor, "batch_size seqlen"],
    gamma: float,
    lambd: float,
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


def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
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
    adv_estimator: str,
    values: Optional[torch.Tensor] = None,
    norm_adv_by_std_in_grpo: bool = True,
    gamma=1.0,
    lambd=1.0,
):
    if adv_estimator == "gae":
        advantages, returns = compute_gae_advantage_return(
            token_level_rewards=token_level_rewards,
            values=values,
            response_mask=response_mask,
            gamma=gamma,
            lambd=lambd,
        )
    elif adv_estimator == "grpo":
        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
    else:
        raise ValueError(f"Invalid adv_estimator: {adv_estimator}")
    return advantages, returns
