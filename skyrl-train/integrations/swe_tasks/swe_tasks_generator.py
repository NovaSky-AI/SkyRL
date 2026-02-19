"""
SWE Tasks Generator for SkyRL.

Adapts the MiniSweAgentGenerator pattern for our swe-task-generator task format.
Uses mini-swe-agent for multi-step Docker execution, with our custom Docker images
from Docker Hub and eval_script from task.json.
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from omegaconf import DictConfig
import yaml
import traceback
import ray
from pathlib import Path

from minisweagent.models import get_model
from minisweagent.agents.default import DefaultAgent
from minisweagent.run.utils.save import save_traj
from minisweagent.config import get_config_path
from .env import evaluate_trajectory, get_swe_task_environment

from skyrl_train.generators.skyrl_gym_generator import SkyRLGymGenerator, GeneratorOutput, GeneratorInput
from skyrl_train.generators.base import TrajectoryID, TrainingPhase, BatchMetadata
from skyrl_train.inference_engines.base import ConversationType
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.generators.utils import (
    get_rollout_metrics,
    get_response_ids_and_loss_mask_from_messages,
)


class DefaultAgentWithReminder(DefaultAgent):
    """Agent that reminds the model of remaining turns."""

    def get_observation(self, response: dict) -> dict:
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
    generator_cfg: DictConfig,
    sampling_params: dict,
    trajectory_id: TrajectoryID,
    global_step: int,
    training_phase: TrainingPhase,
    vllm_base_url: str = "http://127.0.0.1:8000",
):
    """
    Ray remote task: initialize environment and run agent trajectory.

    Adapted from mini_swe_agent but uses our task instance format:
    - image_name from task.json (Docker Hub)
    - eval_script from task.json
    """
    import os
    from loguru import logger

    # Disable minisweagent cost tracking for local models not in litellm pricing DB
    os.environ["MSWEA_COST_TRACKING"] = "ignore_errors"

    model_config = sweagent_config.get("model", {})
    model_config.setdefault("model_kwargs", {}).update(sampling_params)
    # Point litellm to the local vLLM inference server
    model_config["model_kwargs"]["api_base"] = f"{vllm_base_url}/v1"
    model_config["model_kwargs"]["api_key"] = "dummy"
    model = get_model(litellm_model_name, model_config)

    agent = None
    env = None
    extra_info = None
    result = None
    reward = 0
    error = None

    try:
        env = get_swe_task_environment(sweagent_config, instance)
        agent = DefaultAgentWithReminder(model, env, **sweagent_config.get("agent", {}))
        exit_status, result = agent.run(instance["problem_statement"])
    except Exception as e:
        logger.error(f"Error processing instance {instance['instance_id']}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        error = str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        # Save trajectory for debugging
        path = Path(generator_cfg.swe_tasks_traj_dir) / f"step_{global_step}" / training_phase
        path.mkdir(parents=True, exist_ok=True)
        instance_id = instance["instance_id"]
        filename = f"{instance_id}_{trajectory_id.repetition_id}.json"
        path = path / filename

        if agent is not None:
            eval_error = None
            try:
                result = evaluate_trajectory(instance, result, sweagent_config)
                reward = int(result["resolved"])
                eval_error = result["eval_error"]
                if eval_error:
                    error = eval_error
                    logger.debug(f"Eval error: {eval_error}")
            except Exception as e:
                logger.debug(f"Eval error: {e}")
                logger.debug(f"Traceback: {traceback.format_exc()}")
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
            )

    return (agent.messages if agent is not None else [], reward, error)


class SWETasksGenerator(SkyRLGymGenerator):
    """
    Custom generator for swe-task-generator tasks.

    Uses mini-swe-agent for multi-step Docker execution with our custom
    Docker images and eval scripts. The reward is binary:
      1.0 = eval_script passes (agent fixed the bug)
      0.0 = eval_script fails (agent did not fix the bug)
    """

    def __init__(
        self,
        generator_cfg: DictConfig,
        skyrl_gym_cfg: DictConfig,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
        model_name: str,
    ):
        super().__init__(generator_cfg, skyrl_gym_cfg, inference_engine_client, tokenizer, model_name)

        self.http_endpoint_host = generator_cfg.get("http_endpoint_host", "127.0.0.1")
        self.http_endpoint_port = generator_cfg.get("http_endpoint_port", 8001)
        self.base_url = f"http://{self.http_endpoint_host}:{self.http_endpoint_port}"
        self.generator_cfg = generator_cfg
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.litellm_model_name = "openai/" + self.model_name

        if self.generator_cfg.chat_template.name_or_path is not None:
            raise NotImplementedError("SWETasksGenerator doesn't support custom chat template")

    async def swe_tasks_agent_loop(
        self,
        prompt: ConversationType,
        env_extras: Dict[str, Any],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Dict[str, Any],
        trajectory_id: TrajectoryID,
        batch_metadata: BatchMetadata,
    ) -> Tuple[List[int], float, str, List[int], List[int], Optional[List[int]]]:
        """Run a single agent trajectory on a SWE task instance."""

        sweagent_config = yaml.safe_load(get_config_path(self.generator_cfg.swe_tasks_config_path).read_text())

        # Build instance dict from env_extras (which come from the dataset row)
        instance = {
            "instance_id": env_extras["instance_id"],
            "problem_statement": env_extras["problem_statement"],
            "eval_script": env_extras["eval_script"],
            "image_name": env_extras["image_name"],
            "gold_patch": env_extras.get("gold_patch", ""),
        }

        messages, reward, error = await init_and_run.remote(
            instance,
            self.litellm_model_name,
            sweagent_config,
            self.generator_cfg,
            sampling_params,
            trajectory_id,
            batch_metadata.global_step,
            batch_metadata.training_phase,
            vllm_base_url=self.base_url,
        )

        if not len(messages):
            return None, None, None, None, None, None

        # First two messages are system + user (same as mini_swe_agent)
        response_messages = messages[2:]

        for message in messages[:2]:
            assert message["role"] in ("system", "user")

        initial_input_ids = self.tokenizer.apply_chat_template(messages[:2], add_generation_prompt=False, tokenize=True)
        initial_prompt_length = len(initial_input_ids)

        # Remove trailing user messages (final git diff capture)
        if not response_messages:
            # No response beyond system+user — agent produced nothing
            return None, None, None, None, None, None
        last_idx = len(response_messages) - 1
        while last_idx >= 0 and response_messages[last_idx]["role"] == "user":
            last_idx -= 1
        if last_idx < 0:
            # Only user messages, no assistant messages at all
            return None, None, None, None, None, None
        response_messages = response_messages[: last_idx + 1]

        response_ids, loss_mask, _ = get_response_ids_and_loss_mask_from_messages(
            response_messages,
            self.tokenizer,
            assistant_logprobs=None,
        )

        prompt_ids = initial_input_ids
        max_response_tokens = max_tokens + max_input_length - initial_prompt_length

        stop_reason = "complete"
        if len(response_ids) > max_response_tokens:
            stop_reason = "length"

        response_ids = response_ids[:max_response_tokens]
        loss_mask = loss_mask[:max_response_tokens]

        return (response_ids, reward, stop_reason, loss_mask, prompt_ids, None)

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """Generate trajectories for the input batch."""
        prompts = input_batch["prompts"]
        env_extras = input_batch["env_extras"]
        trajectory_ids = input_batch["trajectory_ids"]
        batch_metadata = input_batch["batch_metadata"]
        max_tokens = self.generator_cfg.sampling_params.max_generate_length
        max_input_length = self.generator_cfg.max_input_length
        sampling_params = get_sampling_params_for_backend(
            self.generator_cfg.backend, self.generator_cfg.sampling_params
        )

        tasks = []
        for i in range(len(prompts)):
            tasks.append(
                self.swe_tasks_agent_loop(
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

        responses = [output[0] for output in all_outputs if output[0] is not None]
        rewards = [output[1] for output in all_outputs if output[0] is not None]
        stop_reasons = [output[2] for output in all_outputs if output[0] is not None]
        loss_masks = [output[3] for output in all_outputs if output[0] is not None]
        prompt_token_ids = [output[4] for output in all_outputs if output[0] is not None]

        if not len(responses):
            raise ValueError("No valid responses. All trajectories failed — check environment setup.")

        rollout_metrics = get_rollout_metrics(responses, rewards)

        generator_output: GeneratorOutput = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": None,
        }

        return generator_output
