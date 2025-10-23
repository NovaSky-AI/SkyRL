import ray
import hydra

from omegaconf import DictConfig
from skyrl_train.utils import initialize_ray
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_train.generators.base import GeneratorInterface
from examples.step_wise.step_wise_generator import StepWiseGenerator
from examples.step_wise.step_wise_trainer import StepWiseTrainer


class StepWiseExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """Initializes the generator.

        Returns:
            StepWiseGenerator: The generator.
        """

        return StepWiseGenerator(
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
            StepWiseTrainer: The trainer.
        """
        return StepWiseTrainer(
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
    exp = StepWiseExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
