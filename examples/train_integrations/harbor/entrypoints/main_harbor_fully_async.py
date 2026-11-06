"""
Fully-async entrypoint for training on Harbor tasks.

Reuses HarborExp's generator/dataset overrides and swaps in
``FullyAsyncRayPPOTrainer``. This is the moral equivalent of
``examples/train/fully_async/main_fully_async.py`` for harbor.
"""

import sys

import ray

from skyrl.integration.harbor.config import HarborSkyRLConfig, load_trial_config
from skyrl.integration.harbor.exp import HarborFullyAsyncExp
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    exp = HarborFullyAsyncExp(cfg)
    exp.run()


def main() -> None:
    cfg = HarborSkyRLConfig.from_cli_overrides(sys.argv[1:])
    cfg.harbor_trial_config = load_trial_config(cfg.harbor_trial_config)

    validate_cfg(cfg)
    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError(
            "trainer.algorithm.max_seq_len must be explicitly set for Harbor training; "
            "it is required to truncate responses to the maximum allowed length."
        )
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
