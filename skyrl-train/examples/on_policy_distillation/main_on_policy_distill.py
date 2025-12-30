import torch
import ray
from omegaconf import DictConfig
from skyrl_train.entrypoints.main_base import BasePPOExp
import hydra
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils import initialize_ray
from skyrl_train.entrypoints.main_base import config_dir, validate_cfg
from skyrl_train.utils.ppo_utils import (
    register_advantage_estimator,
    register_policy_loss,
    reduce_loss,
)
from skyrl_train.training_batch import TrainingInputBatch


class OnPolicyDistillationTrainer(RayPPOTrainer):
    """
    Custom trainer for On Policy Distillation.

    Overrides the apply_reward_kl_penalty method to set the rewards just to the kl penalty
    """

    def apply_reward_kl_penalty(
        self,
        data: TrainingInputBatch,
    ) -> TrainingInputBatch:
        """Computes the KL penalty and sets the rewards to the KL penalty"""
        loss_masks_all: torch.Tensor = data["loss_mask"]
        teacher_action_log_probs: torch.Tensor = data["base_action_log_probs"]
        action_log_probs: torch.Tensor = data["action_log_probs"]

        # set rewards to the KL penalty
        # note: tinker seems to use k1 or k2: https://github.com/thinking-machines-lab/tinker-cookbook/blob/3dd0463472dda5847efee80010b50514fa3068ef/tinker_cookbook/rl/metrics.py#L40
        rewards = -(action_log_probs - teacher_action_log_probs) * loss_masks_all
        data["rewards"] = rewards
        return data


# Using the decorator
@register_advantage_estimator("no_op")
def compute_no_op_advantage(token_level_rewards: torch.Tensor, **kwargs):
    # just pass through the rewards
    return token_level_rewards, token_level_rewards


@register_policy_loss("importance_sampling")
def compute_importance_sampling_policy_loss(
    log_probs, old_log_probs, advantages, config, loss_mask=None, rollout_logprobs=None, **kwargs
):
    # as defined here: https://tinker-docs.thinkingmachines.ai/losses#policy-gradient-importance_sampling
    loss = -torch.exp(log_probs - old_log_probs) * advantages

    loss = reduce_loss(loss, loss_mask, "seq_mean_token_sum_norm", config.max_seq_len)
    # return loss and a dummy clip ratio value as we aren't clipping here
    return loss, 0.0


class OnPolicyDistillationExp(BasePPOExp):
    def get_trainer(self, *args, **kwargs):
        return OnPolicyDistillationTrainer(*args, **kwargs)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    exp = OnPolicyDistillationExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
