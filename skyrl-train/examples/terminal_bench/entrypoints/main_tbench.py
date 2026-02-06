"""
Main entrypoint for training on terminal bench tasks.
"""

import ray
import sys
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

from skyrl_train.entrypoints.main_base import BasePPOExp
from skyrl_train.config import SkyRLConfig
from skyrl_train.utils import validate_cfg
from skyrl_train.utils.utils import initialize_ray
from examples.terminal_bench.generator.terminal_bench_generator import TerminalBenchGenerator
from examples.terminal_bench.dataset import TerminalBenchTaskDataset

TERMINAL_BENCH_DEFAULT_CONFIG = Path(__file__).parent.parent / "terminal_bench_config" / "default.yaml"


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Merge overrides into base dict recursively, modifying base in-place."""
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


@dataclass
class TerminalBenchSkyRLConfig(SkyRLConfig):
    """SkyRLConfig with terminal bench configuration."""

    terminal_bench_config: Dict[str, Any] = field(default_factory=dict)


class TerminalBenchExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """
        Initializes the TerminalBenchGenerator.
        """
        return TerminalBenchGenerator(
            generator_cfg=cfg.generator,
            terminal_bench_cfg=cfg.terminal_bench_config,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
        )

    def get_train_dataset(self):
        """Initializes the training dataset."""
        prompts_dataset = TerminalBenchTaskDataset(
            data_files=self.cfg.data.train_data,
        )
        assert (
            len(prompts_dataset) >= self.cfg.trainer.train_batch_size
        ), f"dataset should be atleast as large as `train_batch_size` {self.cfg.trainer.train_batch_size}, got size {len(prompts_dataset)}"
        return prompts_dataset

    def get_eval_dataset(self):
        """Initializes the evaluation dataset."""
        if self.cfg.trainer.eval_interval > 0 and self.cfg.data.val_data:
            prompts_dataset = TerminalBenchTaskDataset(
                data_files=self.cfg.data.val_data,
            )
            return prompts_dataset
        return None


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    exp = TerminalBenchExp(cfg)
    exp.run()


def main() -> None:
    cfg = TerminalBenchSkyRLConfig.from_cli_overrides(sys.argv[1:])

    # Load terminal bench defaults and merge CLI overrides on top
    with open(TERMINAL_BENCH_DEFAULT_CONFIG) as f:
        defaults = yaml.safe_load(f)
    cfg.terminal_bench_config = _deep_merge(defaults, cfg.terminal_bench_config)

    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
