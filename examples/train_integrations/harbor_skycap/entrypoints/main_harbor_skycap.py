"""Train on Harbor tasks, with skycap capturing each rollout's exact tokens.

The sibling ``main_harbor`` with two changes: a pool of skycap servers (Ray
actors, see ``skyrl.integration.skycap.servers``) starts in front of the inference router, and the
generator points each trial at its own trajectory on one of them. Agents that
run inside their sandbox (mini-swe-agent, ...) reach it through
``skycap.exposure``.

    uv run --isolated --extra fsdp --extra harbor --extra skycap \\
        -m examples.train_integrations.harbor_skycap.entrypoints.main_harbor_skycap \\
        trainer.policy.model.path=Qwen/Qwen3-8B generator.inference_engine.served_model_name=policy \\
        generator.step_wise_trajectories=true data.train_data="['/path/to/harbor/tasks']" ...
"""

import sys

import ray

from skyrl.integration.harbor.config import load_trial_config
from skyrl.integration.harbor.skycap_exp import HarborSkycapConfig, HarborSkycapExp
from skyrl.integration.skycap.config import exposure_type
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    HarborSkycapExp(cfg).run()


def main() -> None:
    cfg = HarborSkycapConfig.from_cli_overrides(sys.argv[1:])
    cfg.harbor_trial_config = load_trial_config(cfg.harbor_trial_config)
    validate_cfg(cfg)
    exposure_type(cfg)
    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError("trainer.algorithm.max_seq_len must be set for Harbor training")
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
