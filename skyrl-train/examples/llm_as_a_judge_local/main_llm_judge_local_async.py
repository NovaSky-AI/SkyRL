"""
Async entrypoint for LLM-as-a-Judge with a **local** vLLM reward model.

Combines the async trainer (generation runs one step ahead of training)
with the local reward inference service.

Layout (3 GPUs, disaggregated — async requires colocate_all=false):
  - GPU 1: FSDP policy model (training)
  - GPU 2: vLLM inference engine (generation, runs ahead by 1 step)
  - GPU 3: Frozen vLLM reward engine (scoring)

Usage:
    bash examples/llm_as_a_judge_local/run_llm_judge_local_async.sh
"""

import os
import ray
import hydra
import asyncio
import importlib
from omegaconf import DictConfig

from skyrl_train.utils import initialize_ray
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_gym.envs import register

# `async` is a Python keyword — use importlib to import from examples.async
_async_module = importlib.import_module("examples.async.async_trainer")
AsyncRayPPOTrainer = _async_module.AsyncRayPPOTrainer

# Reuse the reward service setup from the sync entrypoint
from examples.llm_as_a_judge_local.main_llm_judge_local import start_reward_service


class AsyncLLMJudgePPOExp(BasePPOExp):
    """Async PPO experiment with LLM-as-a-Judge reward model."""

    def get_trainer(
        self,
        cfg,
        tracker,
        tokenizer,
        train_dataset,
        eval_dataset,
        inference_engine_client,
        generator,
        colocate_pg,
    ):
        return AsyncRayPPOTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )

    def run(self):
        trainer = self._setup_trainer()
        asyncio.run(trainer.train())


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig, hf_token: str = ""):
    """Remote entrypoint: start reward service, register env, run async training.

    With colocate_all=false, the FSDP policy, inference engine, and reward
    engine each get their own GPU — Ray's scheduler handles placement.
    """
    # 1. Start reward service
    start_reward_service(cfg, hf_token=hf_token)

    # 2. Register env
    register(
        id="llm_as_a_judge_local",
        entry_point="examples.llm_as_a_judge_local.llm_judge_local_env:GSM8kLLMJudgeLocalEnv",
    )

    # 3. Run async training
    exp = AsyncLLMJudgePPOExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    validate_cfg(cfg)
    initialize_ray(cfg)

    hf_token = os.environ.get("HF_TOKEN", "") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN", ""
    )

    # Propagate PG timeout to worker via runtime_env (env var is read at
    # import time in skyrl_train.env_vars, so it must be set before the
    # worker process loads that module).
    pg_timeout = os.environ.get("SKYRL_RAY_PG_TIMEOUT_IN_S", "600")
    entrypoint = skyrl_entrypoint.options(
        runtime_env={"env_vars": {"SKYRL_RAY_PG_TIMEOUT_IN_S": pg_timeout}}
    )
    ray.get(entrypoint.remote(cfg, hf_token=hf_token))


if __name__ == "__main__":
    main()
