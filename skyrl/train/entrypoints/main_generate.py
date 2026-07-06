"""
Main entrypoint for evaluation-only.
"""

import asyncio
import sys
from typing import Any

import ray
from loguru import logger

from skyrl.backends.skyrl_train.inference_servers.base import InferenceEngineInterface
from skyrl.train.config import InferenceEngineConfig, SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import (
    BasePPOExp,
)
from skyrl.train.evaluate import evaluate, evaluate_step_wise
from skyrl.train.utils.trainer_utils import build_dataloader
from skyrl.train.utils.utils import initialize_ray, validate_generator_cfg
from skyrl.utils.tok import get_tokenizer


def _get_fireworks_inference_client(ie_cfg: InferenceEngineConfig, tokenizer) -> InferenceEngineInterface:
    """Build the external Fireworks client: no local engines, no vllm import, no control plane."""
    try:
        from skyrl.backends.skyrl_train.inference_servers.fireworks_client import (
            FireworksInferenceClient,
        )
    except ImportError as e:
        raise ImportError(
            "backend='fireworks' requires the fireworks-ai SDK. Install the `fireworks` extra, "
            "e.g. `uv run --isolated --extra fireworks ...`."
        ) from e

    return FireworksInferenceClient(
        model_name=ie_cfg.served_model_name,
        tokenizer=tokenizer,
        base_url=ie_cfg.external_proxy_url,
        api_key=ie_cfg.api_key,
    )


class EvalOnlyEntrypoint(BasePPOExp):
    def get_train_dataset(self):
        """Override to avoid requiring a train dataset for eval-only runs."""
        return None

    def get_tokenizer(self):
        """Load the tokenizer from ``hf_tokenizer_name`` for the eval-only ``fireworks`` backend
        (which serves a model independent of ``trainer.policy.model.path``); defer to BasePPOExp
        otherwise."""
        ie_cfg = self.cfg.generator.inference_engine
        if ie_cfg.backend == "fireworks":
            return get_tokenizer(
                ie_cfg.hf_tokenizer_name,
                trust_remote_code=True,
                use_fast=not self.cfg.trainer.disable_fast_tokenizer,
                padding_side="left",
            )
        return super().get_tokenizer()

    def get_inference_client(self) -> InferenceEngineInterface:
        """Additionally allow the eval-only ``fireworks`` backend; defer to BasePPOExp otherwise."""
        if self.cfg.generator.inference_engine.backend == "fireworks":
            return _get_fireworks_inference_client(self.cfg.generator.inference_engine, self.tokenizer)
        return super().get_inference_client()

    async def run(self, inference_engine_client: InferenceEngineInterface) -> dict[str, Any]:
        assert self.eval_dataset is not None, "The evaluation only entrypoint requires an eval dataset is provided"

        await inference_engine_client.wake_up()
        generator = self.get_generator(self.cfg, self.tokenizer, inference_engine_client)

        eval_fn = evaluate_step_wise if self.cfg.generator.step_wise_trajectories else evaluate
        results: dict[str, Any] = await eval_fn(
            eval_dataloader=build_dataloader(self.cfg, self.eval_dataset, is_train=False),
            generator=generator,
            cfg=self.cfg,
            global_step=None,
            tokenizer=self.tokenizer,
        )

        tracker = self.get_tracker()
        tracker.log(results, step=0, commit=True)

        return results


@ray.remote(num_cpus=1)
def eval_entrypoint(cfg: SkyRLTrainConfig) -> dict:
    exp = EvalOnlyEntrypoint(cfg)
    # Build the inference client from a sync context so _get_new_inference_client
    # can run its own asyncio.run() for the colocated-mode sleep step.
    inference_engine_client = exp.get_inference_client()
    return asyncio.run(exp.run(inference_engine_client))


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_generator_cfg(cfg)
    initialize_ray(cfg)
    metrics = ray.get(eval_entrypoint.remote(cfg))
    logger.info(f"Metrics from eval only run: {metrics}")


if __name__ == "__main__":
    main()
