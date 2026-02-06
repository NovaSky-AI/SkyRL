"""
uv run --isolated --extra vllm -m examples.algorithm.custom_policy_loss.main_custom_policy_loss
"""

import sys

import ray
import torch
from typing import Optional
from skyrl_train.config import SkyRLConfig, AlgorithmConfig
from skyrl_train.utils import initialize_ray
from skyrl_train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl_train.utils.ppo_utils import PolicyLossRegistry


# Example of custom policy loss: "reinforce"
def compute_reinforce_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    config: AlgorithmConfig,
    loss_mask: Optional[torch.Tensor] = None,
):
    """
    Simple REINFORCE baseline - basic policy gradient that will enable learning.
    """
    # Classic REINFORCE: minimize -log_prob * advantage
    loss = (-log_probs * advantages).mean()

    # Return loss and dummy clip_ratio (no clipping in REINFORCE)
    return loss, 0.0


# Register the custom policy loss
PolicyLossRegistry.register("reinforce", compute_reinforce_policy_loss)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLConfig):
    exp = BasePPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = SkyRLConfig.from_cli_overrides(sys.argv[1:])
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)

    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
