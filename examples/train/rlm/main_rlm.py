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

    # Enable LEASH adaptive length penalty:
        trainer.algorithm.leash.use_leash=true \
        trainer.algorithm.leash.lambda_init=0.2 \
        trainer.algorithm.leash.lambda_lr=0.05 \
        trainer.algorithm.leash.target_length=4096
"""

import sys
from typing import List

import ray
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl.train.generators.base import GeneratorOutput
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import initialize_ray


class RLMTrainer(RayPPOTrainer):
    """RayPPOTrainer subclass that applies the LEASH adaptive length penalty."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.leash_lambda: float = self.cfg.trainer.algorithm.leash.lambda_init

    def postprocess_generator_output(self, generator_output: GeneratorOutput, uids: List[str]) -> GeneratorOutput:
        # Base processing: converts response-level rewards to per-token rewards.
        generator_output = super().postprocess_generator_output(generator_output, uids)

        leash_cfg = self.cfg.trainer.algorithm.leash
        if not leash_cfg.use_leash:
            return generator_output

        # LEASH length penalty is not compatible with step-wise trajectories: in that mode,
        # response_ids contains one entry per step (not the full episode), so the length
        # measures a single step rather than total assistant tokens across the episode.
        # Disable the penalty automatically to avoid silently incorrect behavior.
        if self.cfg.generator.step_wise_trajectories:
            return generator_output

        L_t = leash_cfg.target_length or self.cfg.generator.sampling_params.max_generate_length
        lam = self.leash_lambda

        per_token_rewards: List[List[float]] = generator_output["rewards"]
        loss_masks: List[List[int]] = generator_output["loss_masks"]

        # Use the number of assistant-generated tokens (loss_mask == 1) as the response length.
        # response_ids for multi-turn includes observation tokens between turns (loss_mask == 0),
        # so len(response_ids) would overcount. sum(loss_mask) gives only assistant turn tokens.
        violations = []
        for reward_seq, loss_mask in zip(per_token_rewards, loss_masks):
            assistant_token_count = sum(loss_mask)
            violation = max(0.0, assistant_token_count / L_t - 1.0)
            violations.append(violation)
            # Penalty is applied at the last token position (where the response reward lives).
            reward_seq[-1] = max(0.0, min(1.0, reward_seq[-1] - lam * violation))

        # Dual variable update: λ_{k+1} = clip(λ_k + α * J_P, λ_min, λ_max)
        J_P = sum(violations) / len(violations) if violations else 0.0
        self.leash_lambda = max(leash_cfg.lambda_min, min(leash_cfg.lambda_max, lam + leash_cfg.lambda_lr * J_P))

        self.all_metrics.update(
            {
                "leash/lambda": self.leash_lambda,
                "leash/J_P": J_P,
            }
        )

        generator_output["rewards"] = per_token_rewards
        return generator_output


class RLMPPOExp(BasePPOExp):
    """BasePPOExp subclass that uses RLMTrainer."""

    def get_trainer(self, cfg, tracker, tokenizer, train_dataset, eval_dataset, inference_engine_client, generator, colocate_pg):
        return RLMTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    exp = RLMPPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
