"""Produce paired PiSSA initialization artifacts with Megatron."""

import argparse
import json
from pathlib import Path

import ray
import torch

from examples.train.pissa.pissa_init_worker import PiSSAInitWorker
from skyrl.backends.skyrl_train.workers.worker import PPORayActorGroup
from skyrl.train.config import SkyRLTrainConfig, get_config_as_dict
from skyrl.train.utils.utils import initialize_ray
from skyrl.utils.log import logger
from skyrl.utils.tok import get_tokenizer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config-overrides", type=json.loads, default={})
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.rank <= 0:
        raise ValueError("--rank must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    overrides = dict(args.config_overrides)
    overrides.update(
        {
            "trainer.strategy": "megatron",
            "trainer.policy.model.path": args.base_model,
            "trainer.policy.model.lora.rank": args.rank,
            "trainer.policy.model.lora.alpha": args.rank,
            "trainer.policy.model.lora.target_modules": "all-linear",
            "trainer.policy.model.lora.exclude_modules": None,
            "trainer.policy.model.lora.share_expert_adapters": False,
            "trainer.placement.colocate_all": False,
        }
    )
    cfg = SkyRLTrainConfig.from_cli_overrides(overrides)
    tokenizer = get_tokenizer(args.base_model)

    try:
        initialize_ray(cfg)
        policy = PPORayActorGroup(
            cfg.trainer,
            cfg.trainer.placement.policy_num_nodes,
            cfg.trainer.placement.policy_num_gpus_per_node,
            PiSSAInitWorker,
            num_gpus_per_actor=1,
            colocate_all=False,
            sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
            record_memory=cfg.trainer.policy.record_memory,
        )
        ray.get(policy.async_init_model(args.base_model, num_training_steps=1_000_000_000))
        ray.get(policy.async_run_ray_method("pass_through", "_set_pad_token_id", tokenizer.pad_token_id))
        ray.get(policy.async_run_ray_method("pass_through", "prime_optimizer_state"))

        checkpoint_dir = args.output_dir / "global_step_0"
        ray.get(
            policy.async_run_ray_method(
                "pass_through",
                "save_checkpoint",
                str(checkpoint_dir / "policy"),
                tokenizer,
            )
        )
        torch.save(
            {"global_step": 0, "config": get_config_as_dict(cfg)},
            checkpoint_dir / "trainer_state.pt",
        )
        ray.get(
            policy.async_run_ray_method(
                "pass_through",
                "save_pissa_residual",
                str(args.output_dir / "residual_base"),
                tokenizer,
            )
        )
        logger.info(f"PiSSA initialization artifacts saved to {args.output_dir}")
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
