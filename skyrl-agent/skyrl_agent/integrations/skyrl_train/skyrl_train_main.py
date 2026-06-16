from dataclasses import dataclass
import hydra
import sys
import os
from omegaconf import DictConfig
from skyrl.train.config import GeneratorConfig, SkyRLTrainConfig, make_config
from skyrl.train.generators.base import GeneratorInterface, GeneratorInput, GeneratorOutput
from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl.train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl.train.utils import initialize_ray
import ray

from skyrl_agent import AutoAgentRunner

from .trainer import SkyRLAgentPPOTrainer


@dataclass
class SkyRLAgentGeneratorConfig(GeneratorConfig):
    task: str = ""


SkyRLAgentConfig = make_config(generator_cls=SkyRLAgentGeneratorConfig)


class SkyRLAgentGenerator(GeneratorInterface):
    def __init__(self, generator_cfg: SkyRLAgentGeneratorConfig, llm_endpoint_client: InferenceEngineClient, tokenizer):
        # read the skyagent task yaml
        skyagent_task_yaml_path = generator_cfg.task
        if not os.path.exists(skyagent_task_yaml_path):
            raise FileNotFoundError(f"Task YAML not found: {skyagent_task_yaml_path}")

        self.agent_generator = AutoAgentRunner.from_task(
            skyagent_task_yaml_path, infer_engine=llm_endpoint_client, tokenizer=tokenizer
        )

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        val_mode = input_batch["batch_metadata"].training_phase == "eval"
        return await self.agent_generator.run(input_batch, val_mode=val_mode)


class SkyRLAgentPPOExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, llm_endpoint_client):
        generator = SkyRLAgentGenerator(
            generator_cfg=cfg.generator, llm_endpoint_client=llm_endpoint_client, tokenizer=tokenizer
        )
        return generator

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
            RayPPOTrainer: The trainer.
        """
        return SkyRLAgentPPOTrainer(
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
def skyrl_entrypoint(cfg: SkyRLAgentConfig):
    # make sure that the training loop is not run on the head node.
    exp = SkyRLAgentPPOExp(cfg)
    exp.run()


def main() -> None:
    # Parse CLI args and build typed config
    cfg = SkyRLAgentConfig.from_cli_overrides(sys.argv[1:])

    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)
    task = skyrl_entrypoint.remote(cfg)
    try:
        ray.get(task)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, shutting down...")
        ray.cancel(task)
        raise


if __name__ == "__main__":
    main()
