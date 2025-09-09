import hydra
from omegaconf import DictConfig
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_train.utils import initialize_ray
import ray
from integrations.verifiers.verifiers_generator import VerifiersGenerator
from transformers import PreTrainedTokenizer
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient


class VerifiersEntrypoint(BasePPOExp):
    def get_generator(
        self, cfg: DictConfig, tokenizer: PreTrainedTokenizer, inference_engine_client: InferenceEngineClient
    ):
        return VerifiersGenerator(
            generator_cfg=cfg.generator,
            tokenizer=tokenizer,
            model_name=cfg.trainer.policy.model.path,
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    exp = VerifiersEntrypoint(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Validate config args.
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
