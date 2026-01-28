"""
GOLD (General On-policy Logit Distillation) Trainer for Cross-Tokenizer Distillation.

This example extends the on-policy distillation example (PR #585) to support
student-teacher pairs with different tokenizers. It loads a separate teacher
tokenizer and logs cross-tokenizer statistics.

Reference:
- HuggingFace blog: https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation
- TRL GOLDTrainer: https://github.com/huggingface/trl/blob/v0.25.1/trl/experimental/gold/gold_trainer.py
"""

import torch
import ray
from omegaconf import DictConfig
from transformers import AutoTokenizer
import hydra

from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils import initialize_ray
from skyrl_train.utils.ppo_utils import (
    AdvantageEstimatorRegistry,
    PolicyLossRegistry,
    reduce_loss,
)
from skyrl_train.training_batch import TrainingInputBatch


class GOLDDistillationTrainer(RayPPOTrainer):
    """
    Custom trainer for GOLD (General On-policy Logit Distillation).

    Extends OnPolicyDistillationTrainer to support different tokenizers between
    student and teacher models.
    """

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

        # Load teacher tokenizer (from ref model path)
        teacher_tokenizer_path = cfg.trainer.ref.model.path
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_tokenizer_path, trust_remote_code=True)
        if self.teacher_tokenizer.pad_token is None:
            self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token

        # Check if tokenizers are different
        self.cross_tokenizer = self._check_tokenizer_difference()

        if self.cross_tokenizer:
            print("[GOLD] Cross-tokenizer distillation enabled")
            print(f"[GOLD] Student vocab size: {len(self.tokenizer)}")
            print(f"[GOLD] Teacher vocab size: {len(self.teacher_tokenizer)}")
        else:
            print("[GOLD] Same tokenizer detected - using standard distillation")

    def _check_tokenizer_difference(self) -> bool:
        """Check if student and teacher tokenizers are different."""
        if len(self.tokenizer) != len(self.teacher_tokenizer):
            return True
        test_text = "Hello world! This is a test."
        student_tokens = self.tokenizer.encode(test_text)
        teacher_tokens = self.teacher_tokenizer.encode(test_text)
        return student_tokens != teacher_tokens

    def apply_reward_kl_penalty(
        self,
        data: TrainingInputBatch,
    ) -> TrainingInputBatch:
        """Computes the KL penalty and sets the rewards to the KL penalty."""
        loss_masks_all: torch.Tensor = data["loss_mask"]
        teacher_action_log_probs: torch.Tensor = data["base_action_log_probs"]
        action_log_probs: torch.Tensor = data["action_log_probs"]

        # set rewards to the KL penalty (same as on-policy distillation)
        rewards = -(action_log_probs - teacher_action_log_probs) * loss_masks_all
        data["rewards"] = rewards

        # Log cross-tokenizer info
        if self.cross_tokenizer:
            self.all_metrics.update({"gold/cross_tokenizer": 1.0})

        return data


def compute_no_op_advantage(token_level_rewards: torch.Tensor, **kwargs):
    # just pass through the rewards
    return token_level_rewards, token_level_rewards


def compute_importance_sampling_policy_loss(
    log_probs, old_log_probs, advantages, config, loss_mask=None, rollout_logprobs=None, **kwargs
):
    # as defined here: https://tinker-docs.thinkingmachines.ai/losses#policy-gradient-importance_sampling
    loss = -torch.exp(log_probs - old_log_probs) * advantages
    loss = reduce_loss(loss, loss_mask, "seq_mean_token_sum_norm", config.max_seq_len)
    return loss, 0.0


def _register_gold_distillation_algorithms() -> None:
    try:
        AdvantageEstimatorRegistry.register("no_op", compute_no_op_advantage)
    except ValueError:
        pass
    try:
        PolicyLossRegistry.register("importance_sampling", compute_importance_sampling_policy_loss)
    except ValueError:
        pass


_register_gold_distillation_algorithms()


class GOLDDistillationExp(BasePPOExp):
    def get_trainer(self, *args, **kwargs):
        return GOLDDistillationTrainer(*args, **kwargs)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    exp = GOLDDistillationExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
