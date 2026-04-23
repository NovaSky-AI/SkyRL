"""Eval-only entry point for the Recursive Language Model (RLM) environment.

Mirrors ``skyrl.train.entrypoints.main_generate`` but uses ``RLMConfig`` so the
RLM-specific generator fields (``child_openrouter_model``, etc.) are recognized
on the CLI.
"""

import asyncio
import sys

import ray
from loguru import logger

from skyrl.train.config import make_config
from skyrl.train.entrypoints.main_generate import EvalOnlyEntrypoint
from skyrl.train.utils.utils import initialize_ray, validate_generator_cfg

from .rlm_config import RLMGeneratorConfig


RLMConfig = make_config(generator_cls=RLMGeneratorConfig)


@ray.remote(num_cpus=1)
def eval_entrypoint(cfg) -> dict:
    exp = EvalOnlyEntrypoint(cfg)
    inference_engine_client = exp.get_inference_client()
    return asyncio.run(exp.run(inference_engine_client))


def main() -> None:
    cfg = RLMConfig.from_cli_overrides(sys.argv[1:])
    validate_generator_cfg(cfg)
    initialize_ray(cfg)
    metrics = ray.get(eval_entrypoint.remote(cfg))
    logger.info(f"Metrics from eval only run: {metrics}")


if __name__ == "__main__":
    main()
