import ray
import sys

from skyrl_train.config import SkyRLGymConfig, make_config
from skyrl_train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl_train.utils import initialize_ray

from .mini_swe_generator import MiniSweAgentGenerator, MiniSWEGeneratorConfig


MiniSWEConfig = make_config(generator_cls=MiniSWEGeneratorConfig)


class MiniSWEPPOExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        generator = MiniSweAgentGenerator(
            generator_cfg=cfg.generator,
            skyrl_gym_cfg=SkyRLGymConfig(max_env_workers=0),
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            model_name=self.cfg.trainer.policy.model.path,
        )
        return generator


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    # make sure that the training loop is not run on the head node.
    exp = MiniSWEPPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = MiniSWEConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
