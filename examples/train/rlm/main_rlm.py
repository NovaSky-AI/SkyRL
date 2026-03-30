"""
Train an RLM (Recursive Language Model) with SkyRL.

Generates dataset, then trains a model to interact with long contexts
through a REPL environment.

Usage:
    # 1. Generate the dataset
    python -m examples.train.rlm.rlm_dataset --output_dir ~/data/rlm_niah

    # 2. Train (single-node, FSDP)
    uv run --isolated --extra fsdp -m examples.train.rlm.main_rlm \
        data.train_data="['~/data/rlm_niah/train.parquet']" \
        data.val_data="['~/data/rlm_niah/validation.parquet']" \
        trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
        environment.env_class=rlm \
        generator.max_turns=10 \
        generator.use_conversation_multi_turn=true
"""

import sys

import ray
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl.train.utils import initialize_ray


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    exp = BasePPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
