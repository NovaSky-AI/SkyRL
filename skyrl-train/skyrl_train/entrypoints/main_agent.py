"""

uv run --isolated --extra vllm --extra swebench -m skyrl.entrypoints.main_agent

Multi-node Training:
on master node, first run `ray start --head`
then on other nodes, run `ray start --address='<master-node-ip>:<master-node-port>'`

"""

from skyrl_train.utils import validate_cfg

from skyrl_train.utils.utils import initialize_ray
from omegaconf import DictConfig
import ray

import hydra

from skyrl_train.generators.swe_agent.sweagent_generator import OHAgentGenerator, OHAgentConfig
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir


class AgentPPOExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, llm_endpoint_client):
        print(cfg.swebench_config)
        print("type:", type(cfg.swebench_config))
        return OHAgentGenerator(
            cfg.generator,
            OHAgentConfig(**cfg.swebench_config),
            llm_endpoint_client,
            tokenizer,
            max_prompt_length=cfg.trainer.max_prompt_length,
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    # make sure that the training loop is not run on the head node.
    exp = AgentPPOExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
