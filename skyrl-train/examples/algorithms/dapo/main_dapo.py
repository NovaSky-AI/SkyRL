"""
uv run --isolated --extra vllm -m examples.algorithms.dapo.main_dapo
"""

import ray
import hydra
import torch
import numpy as np
from omegaconf import DictConfig
from skyrl_train.utils import initialize_ray
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_train.utils.ppo_utils import AdvantageEstimatorRegistry, compute_grpo_outcome_advantage


# Custom advantage estimator to implement soft overlong punishment for DAPO
def compute_grpo_with_soft_overlong_punishment(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, **kwargs
):
    """
    Applies soft overlong punishment to the token-level rewards and then computes GRPO advantages.

    Args:
        token_level_rewards: (batch_size, seqlen) tensor of token-level rewards
        response_mask: (batch_size, seqlen) tensor of response mask
        index: (batch_size) tensor of prompt indices

    Returns:
        advantages: (batch_size, seqlen) tensor of advantages
        returns: (batch_size, seqlen) tensor of returns
    """
    # this assumes response-level rewards
    scores = token_level_rewards.sum(dim=-1)

    # Overlong punishment params - hardcoded for this script for now
    # TODO (erictang000): make these configurable (in general for all custom registries)
    max_resp_length = 1024  # this is generator.sampling_params.max_generate_length in the `run_dapo_gsm8k.sh` script
    overlong_buffer_len = 512  # overlong buffer is last 512 tokens of the response as an example
    overlong_penalty_factor = (
        1.0  # reward penalty increases linearly from 0 to 1.0 as the response length enters the overlong buffer
    )

    # add soft overlong punishment
    lengths = response_mask.sum(dim=-1)
    buffer_start_idx = max_resp_length - overlong_buffer_len
    # apply penalty
    penalty_mask = lengths > buffer_start_idx
    penalty = (lengths[penalty_mask] - buffer_start_idx) / overlong_buffer_len * overlong_penalty_factor
    scores[penalty_mask] -= penalty
    # for responses that have length >= max_resp_length, overlong filtering is already applied in the config
    # by setting apply_overlong_filtering=true

    # reconstruct response-level rewards in format expected in compute_grpo_outcome_advantage
    new_token_level_rewards = torch.zeros_like(token_level_rewards)
    new_token_level_rewards[:, -1] = scores

    # compute GRPO advantages
    advantages, returns = compute_grpo_outcome_advantage(
        new_token_level_rewards, response_mask, index, epsilon=1e-6, norm_adv_by_std_in_grpo=True
    )

    return advantages, returns


# Register our custom advantage estimator
AdvantageEstimatorRegistry.register("grpo_with_soft_overlong_punishment", compute_grpo_with_soft_overlong_punishment)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    exp = BasePPOExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
