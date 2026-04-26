import asyncio
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import ray
import yaml
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import get_config_path
from minisweagent.models import get_model
from minisweagent.run.utils.save import save_traj

from .mini_swe_utils import evaluate_trajectory, get_sb_environment
from .stepwise_utils import (
    MiniSWEStepOutput,
    build_stepwise_outputs_from_messages,
    build_stepwise_sampling_params,
)
from skyrl.backends.skyrl_train.inference_engines.base import ConversationType
from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl.backends.skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl.train.config import GeneratorConfig, SkyRLGymConfig
from skyrl.train.generators.base import (
    BatchMetadata,
    GeneratorInput,
    GeneratorOutput,
    TrajectoryID,
    TrainingPhase,
)
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator
from skyrl.train.generators.utils import apply_overlong_filtering, get_rollout_metrics


@dataclass
class MiniSWEGeneratorConfig(GeneratorConfig):
    """Extended generator config with Mini-SWE-Agent-specific fields."""

    batched: bool = False
    step_wise_trajectories: bool = True
    miniswe_config_path: str = ""
    miniswe_traj_dir: str = ""

    def __post_init__(self):
        super().__post_init__()


class DefaultAgentWithReminder(DefaultAgent):
    def get_observation(self, response: dict) -> dict:
        """Execute the action and return the output."""
        output = self.execute_action(self.parse_action(response))
        observation = self.render_template(self.config.action_observation_template, output=output)
        remaining = self.config.step_limit - self.model.n_calls

        if remaining == 1:
            observation = f"{observation}\nREMINDER: You only have 1 turn left. Please provide the final answer"
        elif remaining > 1:
            observation = f"{observation}\nREMINDER: You have {remaining} turns left to arrive at the solution."

        self.add_message("user", observation)
        return output


@ray.remote(num_cpus=0.01)
def init_and_run(
    instance: dict,
    litellm_model_name: str,
    sweagent_config: dict,
    generator_cfg: GeneratorConfig,
    data_source: str,
    sampling_params: dict,
    trajectory_id: TrajectoryID,
    global_step: int,
    training_phase: TrainingPhase,
):
    from loguru import logger

    model_config = sweagent_config.get("model", {})
    # Use new sampling parameters
    # Can also have custom sampling parameters per trajectory (ex: custom max tokens)
    model_config.setdefault("model_kwargs", {}).update(sampling_params)
    model = get_model(litellm_model_name, model_config)

    agent = None
    env = None
    extra_info = None
    result = None
    reward = 0
    error = None
    try:
        env = get_sb_environment(sweagent_config, instance, data_source)
        agent = DefaultAgentWithReminder(model, env, **sweagent_config.get("agent", {}))
        exit_status, result = agent.run(instance["problem_statement"])  # type: ignore[arg-type]
    except Exception as e:
        logger.error(f"Error processing instance {instance['instance_id']}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        error = str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        # Create trajectory directory with proper structure: step_{global_step}/{train/eval}
        path = Path(generator_cfg.miniswe_traj_dir) / f"step_{global_step}" / training_phase
        path.mkdir(parents=True, exist_ok=True)
        # Use instance_id and repetition_id for meaningful filename: {instance_id}_{repetition_id}.json
        instance_id = instance["instance_id"]
        filename = f"{instance_id}_{trajectory_id.repetition_id}.json"
        path = path / filename
        if agent is not None:
            eval_error = None
            try:
                result = evaluate_trajectory(instance, result, sweagent_config, data_source)
                reward = int(result["resolved"])
                eval_error = result["eval_error"]
                if eval_error:
                    error = eval_error
                    logger.debug(f"Error during evaluation {eval_error}")
            except Exception as e:
                logger.debug(f"Error during evaluation {e}")
                logger.debug(f"traceback: {traceback.format_exc()}")
                eval_error = str(e)
                error = str(e)

            save_traj(
                agent,
                path,
                exit_status=exit_status,
                result=result,
                extra_info=extra_info,
                reward=reward,
                eval_error=eval_error,
            )  # type: ignore[arg-type]

    return (agent.messages if agent is not None else [], reward, error)


class MiniSweAgentGenerator(SkyRLGymGenerator):
    def __init__(
        self,
        generator_cfg: GeneratorConfig,
        skyrl_gym_cfg: SkyRLGymConfig,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
        model_name: str,
    ):

        # Call parent constructor first
        super().__init__(generator_cfg, skyrl_gym_cfg, inference_engine_client, tokenizer)

        self.http_server_inference_engine_client_host = generator_cfg.inference_engine.http_endpoint_host

        self.http_server_inference_engine_client_port = generator_cfg.inference_engine.http_endpoint_port

        self.base_url = (
            f"http://{self.http_server_inference_engine_client_host}:{self.http_server_inference_engine_client_port}"
        )
        self.generator_cfg = generator_cfg
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.litellm_model_name = "openai/" + self.model_name

        if not self.generator_cfg.step_wise_trajectories:
            raise ValueError("MiniSWEAgentGenerator requires `generator.step_wise_trajectories=true`.")
        if self.generator_cfg.batched:
            raise ValueError("MiniSWEAgentGenerator does not support `generator.batched=true`.")
        if not self.generator_cfg.inference_engine.enable_http_endpoint:
            raise ValueError(
                "MiniSWEAgentGenerator requires `generator.inference_engine.enable_http_endpoint=true`."
            )
        if self.generator_cfg.chat_template.name_or_path is not None:
            raise NotImplementedError("MiniSWEAgentGenerator doesn't support custom chat template")

    async def minisweagent_agent_loop(
        self,
        prompt: ConversationType,
        env_extras: Dict[str, Any],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Dict[str, Any],
        trajectory_id: TrajectoryID,
        batch_metadata: BatchMetadata,
    ) -> Optional[list[MiniSWEStepOutput]]:
        sampling_params = build_stepwise_sampling_params(sampling_params, trajectory_id)
        sweagent_config = yaml.safe_load(get_config_path(self.generator_cfg.miniswe_config_path).read_text())

        # NOTE (sumanthrh): Input `prompt` is not used here because mini-swe-agent uses a similar entry from the `instance` obj
        messages, reward, error = await init_and_run.remote(
            env_extras["instance"],
            self.litellm_model_name,
            sweagent_config,
            self.generator_cfg,
            env_extras["data_source"],
            sampling_params,
            trajectory_id,
            batch_metadata.global_step,
            batch_metadata.training_phase,
        )
        if not len(messages):
            if error:
                return None
            return None

        step_outputs = build_stepwise_outputs_from_messages(
            messages=messages,
            reward=reward,
            trajectory_id=trajectory_id,
            max_seq_len=max_tokens + max_input_length,
        )
        if not step_outputs:
            raise ValueError(
                "Found no assistant responses with exact token metadata. Please ensure Mini-SWE-Agent is using "
                "SkyRL's HTTP endpoint and exact token IDs/logprobs are enabled."
            )
        if any(step_output.rollout_logprobs is None for step_output in step_outputs):
            raise ValueError("MiniSWEAgentGenerator requires per-token rollout logprobs for every assistant turn.")
        return step_outputs

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """
        Generate trajectories for the input batch.

        Returns outputs in the same order as the input batch.
        Args:
            input_batch: GeneratorInput
        Returns:
            GeneratorOutput
        """
        prompts = input_batch["prompts"]
        env_extras = input_batch["env_extras"]
        trajectory_ids = input_batch["trajectory_ids"]
        batch_metadata = input_batch["batch_metadata"]
        if trajectory_ids is None:
            raise ValueError("MiniSWEAgentGenerator requires `trajectory_ids` for step-wise training.")
        if batch_metadata is None:
            raise ValueError("MiniSWEAgentGenerator requires `batch_metadata` to save Mini-SWE trajectories.")
        max_tokens = self.generator_cfg.sampling_params.max_generate_length
        max_input_length = self.generator_cfg.max_input_length
        sampling_params = get_sampling_params_for_backend(
            self.generator_cfg.inference_engine.backend, self.generator_cfg.sampling_params
        )

        tasks = []

        for i in range(len(prompts)):
            tasks.append(
                self.minisweagent_agent_loop(
                    prompts[i],
                    env_extras[i],
                    max_tokens=max_tokens,
                    max_input_length=max_input_length,
                    sampling_params=sampling_params,
                    trajectory_id=trajectory_ids[i],
                    batch_metadata=batch_metadata,
                )
            )

        all_outputs = await asyncio.gather(*tasks)

        step_outputs = [
            step_output
            for trajectory_steps in all_outputs
            if trajectory_steps is not None
            for step_output in trajectory_steps
        ]
        if not step_outputs:
            raise ValueError(
                "Found no valid responses for this step. This means that generation failed for all trajectories, likely due to errors in environment setup."
            )

        responses = [step_output.response_ids for step_output in step_outputs]
        rewards = [step_output.rewards for step_output in step_outputs]
        stop_reasons = [step_output.stop_reason for step_output in step_outputs]
        loss_masks = [step_output.loss_mask for step_output in step_outputs]
        prompt_token_ids = [step_output.prompt_token_ids for step_output in step_outputs]
        rollout_logprobs = []
        for step_output in step_outputs:
            assert step_output.rollout_logprobs is not None
            rollout_logprobs.append(step_output.rollout_logprobs)
        out_trajectory_ids = [step_output.trajectory_id for step_output in step_outputs]
        is_last_step = [step_output.is_last_step for step_output in step_outputs]

        rollout_metrics = get_rollout_metrics(responses, rewards)

        if self.generator_cfg.zero_reward_on_non_stop:
            rewards = self._zero_reward_if_not_stop(rewards, stop_reasons)

        if self.generator_cfg.apply_overlong_filtering:
            loss_masks = apply_overlong_filtering(loss_masks, stop_reasons)

        generator_output: GeneratorOutput = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": rollout_logprobs,
            "trajectory_ids": out_trajectory_ids,
            "rollout_expert_indices": None,
            "is_last_step": is_last_step,
        }

        return generator_output
