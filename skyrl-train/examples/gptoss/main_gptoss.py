import ray
import hydra
import torch
import numpy as np
from collections import defaultdict

from omegaconf import DictConfig
from skyrl_train.utils import initialize_ray
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_train.generators.base import GeneratorInterface
from examples.gptoss.gpt_oss_generator_step_wise import GPTOSSGenerator
from examples.gptoss.gpt_oss_trainer import GPTOSSTrainer
from skyrl_train.utils.ppo_utils import AdvantageEstimatorRegistry


# Example of custom advantage estimator: "simple_baseline"
def compute_advantages_step_wise(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    values: torch.Tensor,
    grpo_norm_by_std,
    gamma,
    lambd,
    config,
    trajectory_ids,
    is_last_step,
    **kwargs,
):
    """
    A custom advantage estimator where the inputs are represented as step level turns
    """
    scores = token_level_rewards.sum(dim=-1)
    id2score = defaultdict(list)  # str -> list
    id2mean = {}
    id2std = {}
    traj_id_to_steps = defaultdict(list)

    epsilon = 1e-6
    # calculcate for the last steps only and then broadcast to all steps

    # last_step_scores = scores[is_last_step]
    # advantages, returns = compute_grpo_outcome_advantage(
    #     last_step_scores,
    #     response_mask[is_last_step],
    #     index[is_last_step],
    #     values[is_last_step],
    #     epsilon=1e-6,
    #     grpo_norm_by_std=grpo_norm_by_std,
    # )

    # new_advantages = torch.zeros_like(scores)
    # new_advantages[is_last_step] = advantages
    # traj_ids = torch.tensor([f"{trajectory_ids[i].instance_id}_{trajectory_ids[i].repetition_id}" for i in range(len(trajectory_ids))])

    # new_advantages = new_advantages.=(0, )

    # breakpoint()
    with torch.no_grad():

        for i in range(len(token_level_rewards)):
            trajectory_id = trajectory_ids[i]
            traj_id_to_steps[f"{trajectory_id.instance_id}_{trajectory_id.repetition_id}"].append((scores[i], i))

        for key, entries in traj_id_to_steps.items():
            instance_id: str = key.split("_")[0]
            # assume last entry is the last turn
            id2score[instance_id].append(entries[-1][0])
            if config.get("use_same_reward_all_steps", False):
                for _, i in entries:
                    scores[i] = scores[entries[-1][-1]]

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        # grpo: id2score ->
        for i in range(len(scores)):
            id_ = trajectory_ids[i].instance_id
            if grpo_norm_by_std:
                scores[i] = (scores[i] - id2mean[id_]) / (id2std[id_] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[id_]
        scores = scores.unsqueeze(-1) * response_mask

        advantages = scores
        returns = advantages.clone()

        return advantages, returns


# Register the custom advantage estimator
AdvantageEstimatorRegistry.register("step_wise", compute_advantages_step_wise)


class GPTOSSExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """Initializes the generator.

        Returns:
            GeneratorInterface: The generator.
        """

        return GPTOSSGenerator(
            generator_cfg=cfg.generator,
            skyrl_gym_cfg=cfg.environment.skyrl_gym,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            model_name=cfg.trainer.policy.model.path,
        )

    def get_trainer(
        self,
        cfg,
        tracker,
        tokenizer,
        train_dataset,
        eval_dataset,
        inference_engine_client,
        generator: GeneratorInterface,
        colocate_pg,
    ):
        """Initializes the trainer.

        Returns:
            GPTOSSTrainer: The trainer.
        """
        return GPTOSSTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):

    # make sure that the training loop is not run on the head node.
    exp = GPTOSSExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
