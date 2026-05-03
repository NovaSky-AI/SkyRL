"""
Main entrypoint for fully async Harbor training with ThunderAgent routing.
"""

import sys
from pathlib import Path
from typing import Dict

import ray
import yaml

from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray

from .main_thunder_agent import FullyAsyncThunderAgentExp
from .training_config import ThunderAgentHarborConfig


HARBOR_DEFAULT_CONFIG = Path(__file__).parent / "harbor_trial_config" / "default.yaml"


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Merge overrides into base dict recursively, modifying base in-place."""
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


class HarborThunderAgentFullyAsyncExp(FullyAsyncThunderAgentExp):
    """Harbor fully-async experiment that routes inference through ThunderAgent."""

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        from .skyrl_integration.harbor_generator import ThunderAgentHarborGenerator

        return ThunderAgentHarborGenerator(
            generator_cfg=cfg.generator,
            harbor_cfg=cfg.harbor_trial_config,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            max_seq_len=cfg.trainer.algorithm.max_seq_len,
        )

    def get_train_dataset(self):
        from .skyrl_integration.harbor_dataset import ThunderAgentHarborDataset

        prompts_dataset = ThunderAgentHarborDataset(
            data_files=self.cfg.data.train_data,
            max_tasks=self.cfg.max_train_tasks,
        )
        assert (
            len(prompts_dataset) >= self.cfg.trainer.train_batch_size
        ), f"dataset should be at least as large as `train_batch_size` {self.cfg.trainer.train_batch_size}, got size {len(prompts_dataset)}"
        return prompts_dataset

    def get_eval_dataset(self):
        from .skyrl_integration.harbor_dataset import ThunderAgentHarborDataset

        if self.cfg.trainer.eval_interval > 0 and self.cfg.data.val_data:
            return ThunderAgentHarborDataset(
                data_files=self.cfg.data.val_data,
                max_tasks=self.cfg.max_eval_tasks,
            )
        return None


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    exp = HarborThunderAgentFullyAsyncExp(cfg)
    exp.run()


def main() -> None:
    cfg = ThunderAgentHarborConfig.from_cli_overrides(sys.argv[1:])

    # Load harbor defaults and merge CLI overrides on top
    if HARBOR_DEFAULT_CONFIG.exists():
        with open(HARBOR_DEFAULT_CONFIG) as f:
            defaults = yaml.safe_load(f)
        cfg.harbor_trial_config = _deep_merge(defaults, cfg.harbor_trial_config)

    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
