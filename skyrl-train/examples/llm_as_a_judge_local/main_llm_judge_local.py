"""
Entrypoint for LLM-as-a-Judge with a **local** vLLM reward model.

This script:
1. Starts a local reward inference service as a Ray actor
   (``RewardInferenceService`` wrapping ``FrozenRewardInferenceClient``,
   a subclass of ``InferenceEngineClient`` without weight sync)
2. Waits for it to be healthy
3. Registers the ``llm_as_a_judge_local`` environment
4. Runs standard SkyRL training

Layout (default: 2 × 1 GPU nodes):
  - Node 1: FSDP policy + vLLM inference (colocated, sleep/wake)
  - Node 2: Frozen vLLM reward engine (scoring)

No changes to SkyRL core are required.  The frozen engine creation
logic is entirely contained in ``reward_inference.py``.

Usage:
    bash examples/llm_as_a_judge_local/run_llm_judge_local.sh
"""

import os
import ray
import hydra
import omegaconf
from omegaconf import DictConfig, OmegaConf
from loguru import logger

from skyrl_train.utils import initialize_ray
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_gym.envs import register


def _cleanup_stale_reward_services() -> None:
    """Kill any leftover reward actors from previous runs.

    Previous runs may leave detached actors in other Ray namespaces that hold
    GPUs.  We look up actors by class name in *every* known namespace and kill
    them.
    """
    import subprocess
    import json
    import time

    for class_name in ["RewardInferenceService"]:
        try:
            result = subprocess.run(
                [
                    "ray", "list", "actors",
                    "--filter", "state=ALIVE",
                    "--filter", f"class_name={class_name}",
                    "-f", "json",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except subprocess.TimeoutExpired:
            logger.warning(
                f"Timeout while listing Ray actors for cleanup. "
                f"Skipping cleanup for {class_name}."
            )
            continue
        try:
            actors = json.loads(result.stdout) if result.stdout.strip() else []
        except json.JSONDecodeError:
            actors = []

        if not actors:
            continue

        logger.info(
            f"Found {len(actors)} stale {class_name} actor(s) — cleaning up"
        )
        for a in actors:
            name = a.get("name", "")
            ns = a.get("ray_namespace", "")
            try:
                handle = ray.get_actor(name, namespace=ns)
                ray.kill(handle, no_restart=True)
                logger.info(
                    f"  Killed stale {class_name} in namespace {ns[:12]}.. "
                    f"(job {a.get('job_id', '')})"
                )
            except Exception as e:
                logger.warning(
                    f"  Could not kill {class_name} in ns {ns[:12]}..: {e}"
                )

    # Give actors a moment to release GPU resources
    time.sleep(2)


def start_reward_service(cfg: DictConfig, hf_token: str = "") -> None:
    """Start the local reward model as a ``RewardInferenceService`` actor.

    Uses ``FrozenRewardInferenceClient`` (a subclass of
    ``InferenceEngineClient``) with frozen vLLM engines (no weight sync),
    getting automatic load balancing, placement-group GPU scheduling, and
    proper Ray lifecycle management.

    The actor is named ``"reward_inference_service"`` so environments can
    look it up by name — no base_url injection needed.

    Before starting, any stale reward actors from previous runs are killed
    to avoid GPU conflicts.
    """
    from examples.llm_as_a_judge_local.reward_inference import (
        RewardInferenceService,
    )

    # Read reward-model settings from config (with sensible defaults)
    try:
        reward_cfg = OmegaConf.to_container(
            cfg.environment.skyrl_gym.llm_as_a_judge_local,
            resolve=True,
        )
    except (AttributeError, KeyError, omegaconf.errors.ConfigAttributeError) as e:
        logger.warning(f"Could not resolve reward config: {e}. Using defaults.")
        reward_cfg = {}

    model = reward_cfg.get("model", "Qwen/Qwen2.5-1.5B-Instruct")
    num_engines = int(reward_cfg.get("num_reward_engines", 1))
    tp_size = int(reward_cfg.get("reward_tp_size", 1))
    gpu_mem = float(reward_cfg.get("gpu_memory_utilization", 0.85))
    max_model_len = int(reward_cfg.get("max_model_len", 4096))

    # Clean up leftover actors from previous runs
    _cleanup_stale_reward_services()

    logger.info(
        f"Starting reward inference service: model={model}, "
        f"engines={num_engines}, tp={tp_size}, max_model_len={max_model_len}"
    )

    # Pass HF_TOKEN via runtime_env (needed for gated models)
    actor_env_vars = {}
    effective_token = (
        hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    if effective_token:
        actor_env_vars["HF_TOKEN"] = effective_token
        actor_env_vars["HUGGING_FACE_HUB_TOKEN"] = effective_token

    service = RewardInferenceService.options(
        name="reward_inference_service",
        lifetime="detached",
        runtime_env={"env_vars": actor_env_vars} if actor_env_vars else {},
    ).remote(
        model=model,
        num_engines=num_engines,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
    )

    # Block until the service is ready
    healthy = ray.get(service.health_check.remote())
    if not healthy:
        raise RuntimeError(
            "Reward inference service started but health check failed"
        )

    model_name = ray.get(service.get_model_name.remote())
    logger.info(
        f"Reward inference service ready: {model_name} × {num_engines} engine(s)"
    )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig, hf_token: str = ""):
    """Remote entrypoint: start reward service, register env, run training.

    Runs on a worker (not the head node).  With ``colocate_all=false`` the
    FSDP workers, inference engine, and reward engine each get their own
    GPU on separate nodes — Ray's scheduler handles placement automatically.
    """
    # 1. Start the reward service (engines get their own GPUs via placement groups)
    start_reward_service(cfg, hf_token=hf_token)

    # 2. Register the environment (no core modifications needed)
    #    The environment auto-discovers the reward service by actor name —
    #    no base_url injection required.
    register(
        id="llm_as_a_judge_local",
        entry_point="examples.llm_as_a_judge_local.llm_judge_local_env:GSM8kLLMJudgeLocalEnv",
    )

    # 3. Run training
    exp = BasePPOExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    validate_cfg(cfg)
    initialize_ray(cfg)

    # Capture HF_TOKEN on the head node and pass it explicitly.
    hf_token = os.environ.get("HF_TOKEN", "") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN", ""
    )

    ray.get(skyrl_entrypoint.remote(cfg, hf_token=hf_token))


if __name__ == "__main__":
    main()
