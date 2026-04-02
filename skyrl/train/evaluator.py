from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch
from loguru import logger
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.inference_engines.utils import (
    get_sampling_params_for_backend,
)
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.generators.base import (
    GeneratorInput,
    GeneratorInterface,
    GeneratorOutput,
    TrainingPhase,
)
from skyrl.train.generators.utils import (
    concatenate_generator_outputs,
    get_metrics_from_generator_output,
    prepare_generator_input as prepare_generator_input_impl,
)
from skyrl.train.utils import Timer
from skyrl.train.utils.logging_utils import log_example
from skyrl.train.utils.trainer_utils import (
    calculate_per_dataset_metrics,
    dump_per_dataset_eval_results,
    validate_generator_output as validate_generator_output_impl,
)


class EvaluationHooks:
    cfg: SkyRLTrainConfig
    generator: GeneratorInterface
    tokenizer: AutoTokenizer

    def prepare_generator_input(
        self,
        prompts: List[Any],
        training_phase: TrainingPhase,
        global_step: int | None,
    ) -> tuple[GeneratorInput, List[str]]:
        """Prepare generator input for the configured training phase."""
        if training_phase == "eval":
            n_samples_per_prompt = self.cfg.generator.eval_n_samples_per_prompt
            sampling_params = self.cfg.generator.eval_sampling_params
        else:
            n_samples_per_prompt = self.cfg.generator.n_samples_per_prompt
            sampling_params = self.cfg.generator.sampling_params

        return prepare_generator_input_impl(
            prompts,
            n_samples_per_prompt,
            get_sampling_params_for_backend(self.cfg.generator.inference_engine.backend, sampling_params),
            self.cfg.environment.env_class,
            training_phase,
            global_step,
        )

    def validate_generator_output(
        self,
        input_batch: GeneratorInput,
        generator_output: GeneratorOutput,
    ) -> None:
        """Validate generator output against the current config."""
        step_wise = self.cfg.generator.step_wise_trajectories or generator_output.get("is_last_step") is not None
        validate_generator_output_impl(
            len(input_batch["prompts"]),
            generator_output,
            step_wise=step_wise,
        )

    def get_eval_metadata(
        self,
        generator_input: GeneratorInput,
        uids: List[str],
        generator_output: GeneratorOutput,
    ) -> tuple[List[str], List[Dict[str, Any]], List[str]]:
        """Return metadata aligned with the rows in ``generator_output``."""
        if not self.cfg.generator.step_wise_trajectories:
            env_extras = generator_input["env_extras"] or [{} for _ in generator_input["env_classes"]]
            return list(generator_input["env_classes"]), list(env_extras), list(uids)

        assert generator_input["trajectory_ids"] is not None, "Step-wise evaluation requires input trajectory_ids"
        assert generator_output["trajectory_ids"] is not None, "Step-wise evaluation requires output trajectory_ids"

        env_extras = generator_input["env_extras"] or [{} for _ in generator_input["trajectory_ids"]]
        traj_id_to_input = {
            (traj_id.instance_id, traj_id.repetition_id): {"env_class": env_class, "env_extras": env_extra}
            for traj_id, env_class, env_extra in zip(
                generator_input["trajectory_ids"], generator_input["env_classes"], env_extras
            )
        }

        output_env_classes: List[str] = []
        output_env_extras: List[Dict[str, Any]] = []
        output_uids: List[str] = []
        for traj_id in generator_output["trajectory_ids"]:
            key = (traj_id.instance_id, traj_id.repetition_id)
            assert key in traj_id_to_input, f"Trajectory ID {traj_id.to_string()} not found in input"
            output_env_classes.append(traj_id_to_input[key]["env_class"])
            output_env_extras.append(traj_id_to_input[key]["env_extras"])
            output_uids.append(traj_id.instance_id)

        return output_env_classes, output_env_extras, output_uids

    @torch.no_grad()
    async def evaluate(
        self,
        eval_dataloader: StatefulDataLoader,
        global_step: int | None,
    ) -> Dict[str, float]:
        """Run evaluation for non-step-wise generation."""
        generator_outputs: List[GeneratorOutput] = []
        concat_all_envs: List[str] = []
        concat_env_extras: List[Dict[str, Any]] = []
        concat_uids: List[str] = []

        pbar = tqdm(total=len(eval_dataloader), initial=0, desc="Evaluation Progress")
        for _, prompts in enumerate(eval_dataloader):
            pbar.update(1)
            generator_input, uids = self.prepare_generator_input(prompts, "eval", global_step)
            generator_output = await self.generator.generate(generator_input)
            self.validate_generator_output(generator_input, generator_output)
            generator_outputs.append(generator_output)
            eval_envs, eval_env_extras, eval_uids = self.get_eval_metadata(generator_input, uids, generator_output)
            concat_all_envs.extend(eval_envs)
            concat_env_extras.extend(eval_env_extras)
            concat_uids.extend(eval_uids)

        concat_generator_outputs = concatenate_generator_outputs(generator_outputs)
        concat_data_sources = [env_extra.get("data_source") for env_extra in concat_env_extras]
        vis = self.tokenizer.decode(generator_output["response_ids"][0])
        log_example(
            logger,
            prompt=generator_input["prompts"][0],
            response=vis,
            reward=generator_output["rewards"][0],
        )

        eval_metrics = calculate_per_dataset_metrics(
            concat_generator_outputs,
            concat_uids,
            concat_data_sources,
            self.cfg.generator.eval_n_samples_per_prompt,
        )

        overall_metrics = get_metrics_from_generator_output(concat_generator_outputs, concat_uids)
        eval_metrics.update(
            {
                "eval/all/avg_score": overall_metrics["avg_score"],
                f"eval/all/pass_at_{self.cfg.generator.eval_n_samples_per_prompt}": overall_metrics["pass_at_n"],
                "eval/all/mean_positive_reward": overall_metrics["mean_positive_reward"],
            }
        )

        for key, value in concat_generator_outputs["rollout_metrics"].items():
            eval_metrics[f"eval/all/{key}"] = value

        if self.cfg.trainer.dump_eval_results:
            with Timer("dump_eval_results"):
                data_save_dir = (
                    Path(self.cfg.trainer.export_path)
                    / "dumped_evals"
                    / ("eval_only" if global_step is None else f"global_step_{global_step}_evals")
                )
                data_save_dir.mkdir(parents=True, exist_ok=True)
                dump_per_dataset_eval_results(
                    data_save_dir,
                    self.tokenizer,
                    concat_generator_outputs,
                    concat_data_sources,
                    concat_all_envs,
                    concat_env_extras,
                    eval_metrics,
                )

        return eval_metrics

    @torch.no_grad()
    async def evaluate_step_wise(
        self,
        eval_dataloader: StatefulDataLoader,
        global_step: int | None,
    ) -> Dict[str, float]:
        """Run evaluation for step-wise generation."""
        generator_outputs: List[GeneratorOutput] = []
        concat_all_envs: List[str] = []
        concat_env_extras: List[Dict[str, Any]] = []
        concat_uids: List[str] = []

        pbar = tqdm(total=len(eval_dataloader), initial=0, desc="Evaluation Progress")
        for _, prompts in enumerate(eval_dataloader):
            pbar.update(1)
            generator_input, uids = self.prepare_generator_input(prompts, "eval", global_step)
            generator_output = await self.generator.generate(generator_input)
            self.validate_generator_output(generator_input, generator_output)
            eval_envs, eval_env_extras, eval_uids = self.get_eval_metadata(generator_input, uids, generator_output)
            concat_all_envs.extend(eval_envs)
            concat_env_extras.extend(eval_env_extras)
            concat_uids.extend(eval_uids)
            generator_outputs.append(generator_output)

        concat_generator_outputs = concatenate_generator_outputs(generator_outputs)

        concat_data_sources = [env_extra.get("data_source") for env_extra in concat_env_extras]
        vis = self.tokenizer.decode(generator_output["response_ids"][0])
        logger.info(f"Eval output example: {vis}")

        generator_output_last_step = defaultdict(list)
        is_last_step_mask = concat_generator_outputs["is_last_step"]
        for key in concat_generator_outputs:
            if isinstance(concat_generator_outputs[key], list):
                assert len(concat_generator_outputs[key]) == len(
                    is_last_step_mask
                ), f"Length mismatch: {len(concat_generator_outputs[key])} != {len(is_last_step_mask)} for key {key}"
                generator_output_last_step[key] = [
                    val for val, is_last_step in zip(concat_generator_outputs[key], is_last_step_mask) if is_last_step
                ]
        uids_last_step = [uid for uid, is_last_step in zip(concat_uids, is_last_step_mask) if is_last_step]
        data_sources_last_step = [
            data_source for data_source, is_last_step in zip(concat_data_sources, is_last_step_mask) if is_last_step
        ]

        eval_metrics = calculate_per_dataset_metrics(
            generator_output_last_step,
            uids_last_step,
            data_sources_last_step,
            self.cfg.generator.eval_n_samples_per_prompt,
        )
        overall_metrics = get_metrics_from_generator_output(generator_output_last_step, uids_last_step)
        eval_metrics.update(
            {
                "eval/all/avg_score": overall_metrics["avg_score"],
                f"eval/all/pass_at_{self.cfg.generator.eval_n_samples_per_prompt}": overall_metrics["pass_at_n"],
                "eval/all/mean_positive_reward": overall_metrics["mean_positive_reward"],
            }
        )

        if self.cfg.trainer.dump_eval_results:
            with Timer("dump_eval_results"):
                data_save_dir = (
                    Path(self.cfg.trainer.export_path)
                    / "dumped_evals"
                    / ("eval_only" if global_step is None else f"global_step_{global_step}_evals")
                )
                data_save_dir.mkdir(parents=True, exist_ok=True)
                dump_per_dataset_eval_results(
                    data_save_dir,
                    self.tokenizer,
                    concat_generator_outputs,
                    concat_data_sources,
                    concat_all_envs,
                    concat_env_extras,
                    eval_metrics,
                )

        return eval_metrics


class StandaloneEvaluator(EvaluationHooks):
    def __init__(
        self,
        cfg: SkyRLTrainConfig,
        generator: GeneratorInterface,
        tokenizer: AutoTokenizer,
    ):
        self.cfg = cfg
        self.generator = generator
        self.tokenizer = tokenizer
