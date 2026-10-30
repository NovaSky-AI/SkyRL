"""SDPO generator, trainer, reusable experiment class, and CLI/Ray launcher.

Recipes inherit SDPOExp and pass their subclass to main().
"""

import os
import random
import sys
from copy import deepcopy

import ray
import torch

from skyrl.backends.skyrl_train.training_batch import (
    TensorList,
    TrainingInputBatch,
    pad_training_input_batch,
)
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.generators.base import GeneratorInterface
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import initialize_ray, validate_cfg
from skyrl.train.utils.trainer_utils import finalize_minibatch_rollout_logprob_diff_std

from .sdpo_config import SDPOConfig, validate_sdpo_config
from .utils import (
    append_teacher_hint,
    build_teacher_prompts,
    collate_teacher_inputs,
    encode_visual_teacher_prompt,
    extract_expert_traces,
    pad_bool_mask,
    srpo_route_masks,
    student_answer_loss_mask,
)


class SDPOGenerator(GeneratorInterface):
    def __init__(self, generator, config):
        self.generator = generator
        self.config = config

    def __getattr__(self, name):
        return getattr(self.generator, name)

    async def generate(self, input_batch, **kwargs):
        # Training has already captured traces in SDPOTrainer. Evaluation
        # calls the generator directly and must strip them here as well.
        student_input, _ = extract_expert_traces(input_batch, self.config)
        return await self.generator.generate(student_input, **kwargs)


class SDPOTrainer(RayPPOTrainer):
    def _remove_tail_data(self, entries):
        # Keep every configured prompt. Conversion pads uneven DP batches with
        # zero-loss rows and separately masks padded teacher supervision.
        return entries

    def build_models(self, PolicyWorker, CriticWorker, RefWorker):
        validate_sdpo_config(self.cfg)
        if self.cfg.trainer.strategy == "megatron":
            from .megatron.megatron_worker import PolicyWorker as SDPOWorker
        else:
            from .fsdp_worker import PolicyWorker as SDPOWorker
        return super().build_models(SDPOWorker, CriticWorker, RefWorker)

    async def generate(self, input_batch):
        student_input, expert_traces = extract_expert_traces(input_batch, self.cfg.trainer.distillation)
        output = await super().generate(student_input)
        responses = output["response_ids"]
        self.all_metrics["generate/response_length/mean"] = (
            sum(map(len, responses)) / len(responses) if responses else 0.0
        )
        # Dynamic sampling and step-wise expansion are rejected by validation,
        # so rows remain paired with their original chat messages.
        self._sdpo_prompts = deepcopy(input_batch["prompts"])
        self._sdpo_expert_traces = expert_traces
        self._sdpo_raw_rewards = [sum(r) if isinstance(r, list) else float(r) for r in output["rewards"]]
        return output

    def convert_to_training_input(self, generator_output, uids):
        config = self.cfg.trainer.distillation
        n = len(generator_output["response_ids"])
        if (generator_output.get("pixel_values") is not None and not config.vision_language) or generator_output.get(
            "rollout_expert_indices"
        ) is not None:
            raise ValueError("SDPO image inputs require vision_language=true; router replay is unsupported")
        if config.rollout_is_clip is not None and generator_output.get("rollout_logprobs") is None:
            raise ValueError("distillation.rollout_is_clip requires rollout log-probabilities")
        feedback = generator_output.get("feedback")
        if feedback is None:
            metrics = generator_output.get("env_metrics") or [{} for _ in range(n)]
            feedback = [m.get(config.feedback_key) if m else None for m in metrics]
        metrics = generator_output.get("env_metrics") or [{} for _ in range(n)]
        feedback_images = [m.get("feedback_images", []) if m else [] for m in metrics]
        feedback_videos = [m.get("feedback_video") if m else None for m in metrics]
        encoded_visual = {}
        visual_encoder = None
        if config.include_environment_feedback and (any(feedback_images) or any(feedback_videos)):
            if not config.vision_language or generator_output.get("pixel_values") is None:
                raise ValueError("Visual teacher feedback requires multimodal student inputs")
            if config.overflow == "truncate_right":
                raise ValueError("Visual teacher feedback supports overflow=skip or error, not truncation")
            if not hasattr(self, "_sdpo_processor"):
                from skyrl.utils.tok import get_processor

                self._sdpo_processor = get_processor(self.cfg.trainer.policy.model.path)

            def visual_encoder(i, messages, enriched, has_feedback):
                if not has_feedback or not (feedback_images[i] or feedback_videos[i]):
                    return append_teacher_hint(
                        self.tokenizer,
                        list(generator_output["prompt_token_ids"][i]),
                        enriched,
                        config.chat_template_kwargs,
                    )
                encoded = encode_visual_teacher_prompt(
                    self._sdpo_processor,
                    messages,
                    enriched,
                    feedback_images[i],
                    config.chat_template_kwargs,
                    video=feedback_videos[i],
                )
                encoded_visual[i] = encoded
                return encoded["input_ids"][0].tolist()

        reasons = generator_output.get("stop_reasons") or [None] * n
        teacher = build_teacher_prompts(
            self.tokenizer,
            self._sdpo_prompts,
            generator_output["prompt_token_ids"],
            generator_output["response_ids"],
            self._sdpo_raw_rewards,
            uids,
            feedback,
            config,
            multimodal_encoder=visual_encoder,
            expert_traces=getattr(self, "_sdpo_expert_traces", None),
            rng=random.Random(f"{self.cfg.trainer.seed}:{getattr(self, 'global_step', 0)}"),
            demonstration_allowed=[
                not config.filter_ineligible_demonstrations or (reason != "length" and any(mask))
                for reason, mask in zip(reasons, generator_output["loss_masks"])
            ],
        )
        teacher_ids, teacher_attention = collate_teacher_inputs(
            teacher.token_ids,
            generator_output["response_ids"],
            self.tokenizer.pad_token_id,
        )
        if not config.train_on_reasoning_tokens:
            # Leave original tokens and teacher context intact. Mask before base
            # collation so padding/alignment and SRPO routing preserve exclusions.
            masks = [
                student_answer_loss_mask(self.tokenizer, response, prompt, mask)
                for response, prompt, mask in zip(
                    generator_output["response_ids"],
                    generator_output["prompt_token_ids"],
                    generator_output["loss_masks"],
                    strict=True,
                )
            ]
            before = sum(sum(mask) for mask in generator_output["loss_masks"])
            after = sum(sum(mask) for mask in masks)
            self.all_metrics["sdpo/reasoning_masked_token_fraction"] = (before - after) / max(before, 1)
            generator_output = {**generator_output, "loss_masks": masks}
        data = super().convert_to_training_input(generator_output, uids)
        pad = data.batch_size - n
        self.all_metrics["sdpo/video_feedback_fraction"] = sum(
            bool(feedback_videos[i]) and i in encoded_visual and teacher.eligible[i] for i in range(n)
        ) / max(n, 1)
        self.all_metrics["sdpo/visual_feedback_fraction"] = 0.0
        teacher_vision = {}
        if visual_encoder is not None:
            for key in ("pixel_values", "image_grid_thw"):
                teacher_vision["teacher_" + key] = TensorList(
                    [
                        (
                            encoded_visual[i][key]
                            if i in encoded_visual and teacher.eligible[i]
                            else generator_output[key][i]
                        )
                        for i in range(n)
                    ]
                )
            for key in ("pixel_values_videos", "video_grid_thw"):
                reference = next(
                    (row[key] for i, row in encoded_visual.items() if teacher.eligible[i] and key in row),
                    None,
                )
                if reference is not None:
                    teacher_vision["teacher_" + key] = TensorList(
                        [
                            (
                                encoded_visual[i][key]
                                if i in encoded_visual and teacher.eligible[i] and key in encoded_visual[i]
                                else reference.new_empty((0, *reference.shape[1:]))
                            )
                            for i in range(n)
                        ]
                    )
            self.all_metrics["sdpo/visual_feedback_fraction"] = sum(
                i in encoded_visual and teacher.eligible[i] for i in range(n)
            ) / max(n, 1)
        teacher_batch = pad_training_input_batch(
            TrainingInputBatch(
                {
                    **teacher_vision,
                    "teacher_sequences": teacher_ids,
                    "teacher_attention_mask": teacher_attention,
                    "distillation_mask": torch.tensor(teacher.eligible),
                }
            ),
            pad,
        )
        # The generic helper copies row zero for ordinary fields. Padded
        # teacher rows must never acquire supervision from that copied row.
        teacher_batch["distillation_mask"][n:] = False
        for key, value in teacher_batch.items():
            data[key] = value
        original_loss_mask = data["loss_mask"].float()
        data["loss_mask"] = data["loss_mask"].float() * data["distillation_mask"][:, None]
        self.all_metrics.update({f"sdpo/{k}": v for k, v in teacher.metrics.items()})
        self.all_metrics["loss/avg_final_rewards"] = sum(self._sdpo_raw_rewards) / max(n, 1)
        if config.use_sample_routing:
            data = self._apply_sample_routing(data, n, original_loss_mask)
        return data

    def _apply_sample_routing(self, data, n, original_loss_mask):
        """Select the objective without reactivating filtered response tokens."""
        sdpo_mask, grpo_mask = srpo_route_masks(
            self._sdpo_raw_rewards,
            data["distillation_mask"][:n],
            self.cfg.trainer.distillation.success_reward_threshold,
            positive_rewards=self.cfg.trainer.distillation.srpo_route_positive_rewards,
            success_only=self.cfg.trainer.distillation.srpo_route_success_only,
        )
        data["sdpo_mask"] = pad_bool_mask(sdpo_mask, data.batch_size)
        data["grpo_mask"] = pad_bool_mask(grpo_mask, data.batch_size)
        active = data["sdpo_mask"] | data["grpo_mask"]
        data["loss_mask"] = original_loss_mask * active[:, None]
        denominator = max(n, 1)
        self.all_metrics["sdpo/srpo_sdpo_fraction"] = sdpo_mask.float().mean().item()
        self.all_metrics["sdpo/srpo_grpo_fraction"] = grpo_mask.float().mean().item()
        self.all_metrics["sdpo/srpo_teacher_available_fraction"] = data["distillation_mask"][:n].float().mean().item()
        self.all_metrics["sdpo/srpo_skipped_fraction"] = (~(sdpo_mask | grpo_mask)).float().mean().item()
        self.all_metrics["sdpo/srpo_positive_reward_fraction"] = (
            sum(float(r > 0) for r in self._sdpo_raw_rewards) / denominator
        )
        self.all_metrics["sdpo/srpo_correct_fraction"] = (
            sum(float(r >= self.cfg.trainer.distillation.success_reward_threshold) for r in self._sdpo_raw_rewards)
            / denominator
        )
        return data

    def fwd_logprobs_values_reward(self, training_input):
        # Distillation scores both branches in the worker. GRPO rows still need
        # π_old when the sampler did not return logprobs.
        if not self.cfg.trainer.distillation.use_sample_routing or training_input.get("rollout_logprobs") is not None:
            return training_input
        return RayPPOTrainer.fwd_logprobs_values_reward(self, training_input)

    def compute_advantages_and_returns(self, data):
        if not self.cfg.trainer.distillation.use_sample_routing:
            return data
        # Group-relative advantages use every rollout, including SDPO-routed
        # failures, so correct samples keep a baseline. The GRPO loss zeros
        # SDPO rows after the estimator runs.
        if "values" not in data:
            data["values"] = None
        data = RayPPOTrainer.compute_advantages_and_returns(self, data)
        data["advantages"] = data["advantages"] * data["grpo_mask"].unsqueeze(-1).to(dtype=data["advantages"].dtype)
        return data

    def _execute_training_step(self, model, data):
        if model != "policy":
            raise ValueError("SDPO does not train a critic")
        # Bypass PPO advantage normalization and consume the entire rollout
        # batch in one optimizer step. Dispatch supplies model on/offloading.
        status = self.dispatch.forward_backward("policy", data, return_per_token_outputs=False)
        norm = self.dispatch.optim_step("policy")
        metrics = dict(status.metrics)
        finalize_minibatch_rollout_logprob_diff_std(metrics)
        if norm is not None:
            metrics["grad_norm"] = norm
        return metrics


class SDPOExp(BasePPOExp):
    """Base for SDPO recipes; reuse SkyRL dataset hooks and add callbacks.

    Subclasses can set config_cls, extend validate_config/setup, override
    get_train_dataset/get_eval_dataset, and return training_callbacks. They
    inherit teacher isolation, worker wiring, and the standard training loop.
    """

    config_cls = SDPOConfig

    @classmethod
    def validate_config(cls, cfg):
        validate_sdpo_config(cfg)
        validate_cfg(cfg)

    @classmethod
    def setup(cls, cfg):
        """Register recipe environments inside the Ray entrypoint process."""

    def training_callbacks(self):
        """Return recipe callbacks without replacing the SDPO trainer."""
        return ()

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return SDPOGenerator(
            super().get_generator(cfg, tokenizer, inference_engine_client),
            cfg.trainer.distillation,
        )

    def get_inference_client(self):
        # Scope HTTP ports to this job: concurrent jobs must not discover or
        # send weight updates to another job's inference servers.
        from skyrl.backends.skyrl_train.inference_servers import setup

        start_port = os.environ.get("SDPO_VLLM_START_PORT")
        if start_port is None:
            return super().get_inference_client()
        previous = setup.VLLM_START_PORT
        setup.VLLM_START_PORT = int(start_port)
        try:
            return super().get_inference_client()
        finally:
            setup.VLLM_START_PORT = previous

    def get_trainer(self, *args, **kwargs):
        trainer = SDPOTrainer(*args, **kwargs)
        for callback in self.training_callbacks():
            trainer.add_callback(callback)
        return trainer


@ray.remote(num_cpus=1)
def _run_on_ray(experiment_cls, cfg):
    experiment_cls.setup(cfg)
    experiment_cls(cfg).run()


def main(experiment_cls=SDPOExp, argv=None):
    """Launch an SDPOExp subclass with its declared typed configuration."""
    cfg = experiment_cls.config_cls.from_cli_overrides(sys.argv[1:] if argv is None else argv)
    experiment_cls.validate_config(cfg)
    initialize_ray(cfg)
    ray.get(_run_on_ray.remote(experiment_cls, cfg))


if __name__ == "__main__":
    main()
