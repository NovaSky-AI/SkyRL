import sys

from skyrl_train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl_train.utils import initialize_ray
import ray
from integrations.verifiers.verifiers_generator import VerifiersGenerator
from transformers import PreTrainedTokenizer
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.config import SkyRLTrainConfig


class VerifiersEntrypoint(BasePPOExp):
    def get_generator(
        self,
        cfg: SkyRLTrainConfig,
        tokenizer: PreTrainedTokenizer,
        inference_engine_client: InferenceEngineClient,
    ):
        return VerifiersGenerator(
            generator_cfg=cfg.generator,
            tokenizer=tokenizer,
            model_name=cfg.trainer.policy.model.path,
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    exp = VerifiersEntrypoint(cfg)
    exp.run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    # Validate config args.
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
