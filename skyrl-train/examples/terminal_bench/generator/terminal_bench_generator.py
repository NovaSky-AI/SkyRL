import asyncio
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional
from loguru import logger
from uuid import uuid4
from skyrl_train.generators.base import GeneratorInterface, GeneratorInput, GeneratorOutput, TrajectoryID
from skyrl_train.generators.utils import get_rollout_metrics, get_response_ids_and_loss_mask_from_messages
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.base import ConversationType
from omegaconf import DictConfig, OmegaConf
from harbor.trial.trial import Trial
from harbor.models.trial.config import TrialConfig

# Suppress LiteLLM verbose logging

import litellm
import logging

litellm.suppress_debug_info = True  # Suppress the "Provider List" output
litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

# We have N retries for each trial, if one of the rollout (out of n_samples_per_prompt) fails
# after N attemptes, we skip this prompt altogether.
MAX_NUM_RETRIES_PER_TRIAL = 2


@dataclass
class TerminalBenchAgentOutput:
    response_ids: List[int]
    reward: float
    stop_reason: str
    loss_mask: List[int]
    prompt_ids: List[int]
    trajectory_id: TrajectoryID
    summarization_count: Optional[int] = None


class TerminalBenchGenerator(GeneratorInterface):
    def __init__(
        self,
        generator_cfg: DictConfig,
        terminal_bench_cfg: DictConfig,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
    ):
        """
        Args:
            generator_cfg: DictConfig object containing the generator configuration
            terminal_bench_cfg: DictConfig object containing the terminal bench configuration
            inference_engine_client: InferenceEngineClient object for interacting with the inference engines
            tokenizer: tokenizer object for encoding and decoding text
        """
        self.base_url = f"http://{generator_cfg.http_endpoint_host}:{generator_cfg.http_endpoint_port}"
        self.generator_cfg = generator_cfg
        self.tokenizer = tokenizer
        self.model_name = generator_cfg.model_name

        # Harbor config template - users can specify any Harbor TrialConfig options in YAML or command line.
        # SkyRL injects: model_name and api_base (once at init), task.path and session_id (per trial)
        self._harbor_config_template = OmegaConf.to_container(terminal_bench_cfg, resolve=True)

        # Set model_name and api_base once (constant across all trials)
        assert generator_cfg.served_model_name is not None, "served_model_name must be set"
        assert (
            "/" not in generator_cfg.served_model_name
        ), "served_model_name must not contain '/', Harbor expects hosted_vllm/{model_name}"
        self._harbor_config_template.setdefault("agent", {})[
            "model_name"
        ] = f"hosted_vllm/{generator_cfg.served_model_name}"
        self._harbor_config_template["agent"].setdefault("kwargs", {})["api_base"] = f"{self.base_url}/v1"

        logger.info(
            f"TerminalBenchGenerator initialized with Harbor config. "
            f"Agent: {self._harbor_config_template.get('agent', {}).get('name')}, "
            f"Trials dir: {self._harbor_config_template.get('trials_dir', 'trials')}"
        )

        # Read custom chat template
        custom_chat_template_path = generator_cfg.engine_init_kwargs.get("chat_template", None)
        if custom_chat_template_path:
            with open(custom_chat_template_path, "r") as f:
                self.custom_chat_template_content = f.read()
            logger.info(
                f"TerminalBenchGenerator initialized with custom chat template read from: {custom_chat_template_path}"
            )
        else:
            self.custom_chat_template_content = None

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        tasks = []
        for i in range(len(input_batch["prompts"])):
            tasks.append(
                self.terminal_bench_agent_loop(
                    prompt=input_batch["prompts"][i],
                    trajectory_id=input_batch["trajectory_ids"][i],
                )
            )

        all_outputs: List[TerminalBenchAgentOutput] = await asyncio.gather(*tasks)

        # For a group of trajectories (n_samples_per_prompt trajectories for the same prompt), if one
        # of the trajectories fails, we skip the entire group. We also skip the group for rollout metric aggregation
        failed_instance_ids = set()
        num_failed_trajectories = 0  # per-trajectory, rather than per-instance
        successful_outputs: List[TerminalBenchAgentOutput] = []  # only for metrics purpose
        for output in all_outputs:
            if output.stop_reason == "error":
                failed_instance_ids.add(output.trajectory_id.instance_id)
                num_failed_trajectories += 1

        for output in all_outputs:
            if output.trajectory_id.instance_id in failed_instance_ids:
                output.response_ids = [0]
                output.stop_reason = "error"
                output.loss_mask = [0]
                output.prompt_ids = [0]
                output.reward = 0
            else:
                successful_outputs.append(output)

        # Calculate rollout metrics for successful outputs
        if len(successful_outputs) > 0:
            rollout_metrics = get_rollout_metrics(
                [output.response_ids for output in successful_outputs],
                [output.reward for output in successful_outputs],
            )
            rollout_metrics["generate/trajectories_summarized"] = sum(
                1 for output in successful_outputs if output.summarization_count > 0
            )
            rollout_metrics["generate/trajectories_truncated"] = sum(
                1 for output in successful_outputs if output.stop_reason == "length"
            )
        else:
            rollout_metrics = {}
        rollout_metrics["generate/num_failed_instances"] = len(failed_instance_ids)
        rollout_metrics["generate/num_failed_trajectories"] = num_failed_trajectories

        generator_output: GeneratorOutput = {
            "prompt_token_ids": [output.prompt_ids for output in all_outputs],
            "response_ids": [output.response_ids for output in all_outputs],
            "rewards": [output.reward for output in all_outputs],
            "loss_masks": [output.loss_mask for output in all_outputs],
            "stop_reasons": [output.stop_reason for output in all_outputs],
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": None,
        }

        return generator_output

    async def terminal_bench_agent_loop(
        self,
        prompt: ConversationType,
        trajectory_id: TrajectoryID,
    ) -> TerminalBenchAgentOutput:
        """
        Run a single terminal_bench agent.
        """
        # Build TrialConfig from template, only override task.path and session_id per trial
        config = deepcopy(self._harbor_config_template)
        config["task"] = {"path": prompt}
        config["agent"]["kwargs"]["session_id"] = uuid4().hex
        trial_config = TrialConfig.model_validate(config)

        trial = Trial(trial_config)

        # Run the trial to get `rewards`, `chat_history`, and `summarization_count`
        successful = False
        reward = None
        chat_history = None
        summarization_count = None
        for i in range(MAX_NUM_RETRIES_PER_TRIAL):
            prefix = f"Trajectory {trajectory_id} attempt {i+1}/{MAX_NUM_RETRIES_PER_TRIAL}"
            results = None
            try:
                results = await trial.run()
                if not results.verifier_result:
                    logger.warning(f"{prefix} failed: Exception info: {results.exception_info}. Results: {results}")
                    continue

                reward = results.verifier_result.rewards["reward"]
                chat_history = results.agent_result.metadata["all_messages"]
                summarization_count = results.agent_result.metadata["summarization_count"]
                if len(chat_history) > 1 and chat_history[0]["role"] == "user":
                    successful = True
                    logger.info(f"{prefix} successful: Results: {results.agent_result.metadata}")
                    break
                else:
                    logger.warning(
                        f"{prefix} failed: Agent did not return a chat history with a user message. chat_history: {chat_history}\n\nResults: {results}"
                    )
            except Exception as e:
                logger.warning(f"{prefix} failed: Error running trial: {e}. Results: {results}")
                continue

        if not successful:
            # We make loss mask 0 so it does not contribute to model updates
            logger.warning(
                f"Trajectory {trajectory_id} failed after {MAX_NUM_RETRIES_PER_TRIAL} attempts, will set loss mask to [0]."
            )
            return TerminalBenchAgentOutput(
                response_ids=[0],
                reward=0,
                stop_reason="error",
                loss_mask=[0],
                prompt_ids=[0],
                trajectory_id=trajectory_id,
            )

        # Use the first message as the prompt. We assume to be no systems messages.
        assert chat_history[0]["role"] == "user", "The first message should be a user message"
        prompt = [chat_history[0]]
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=False,  # the message below will add it themselves
            tokenize=True,
            chat_template=self.custom_chat_template_content,
        )
        initial_prompt_length = len(prompt_ids)

        # Process response messages (everything after the first message)
        response_messages = chat_history[1:]
        assistant_logprobs = getattr(results.agent_result, "output_logprobs", None)
        response_ids, loss_mask, rollout_logprobs = get_response_ids_and_loss_mask_from_messages(
            response_messages, self.tokenizer, assistant_logprobs, chat_template=self.custom_chat_template_content
        )

        # Determine stop reason
        max_response_tokens = (
            self.generator_cfg.sampling_params.max_generate_length
            + self.generator_cfg.max_input_length
            - initial_prompt_length
        )
        stop_reason = "complete"  # Default for trial completion
        if len(response_ids) > max_response_tokens:
            stop_reason = "length"

        # Truncate to maximum allowed length
        response_ids = response_ids[:max_response_tokens]
        loss_mask = loss_mask[:max_response_tokens]
        return TerminalBenchAgentOutput(
            response_ids=response_ids,
            reward=reward,
            stop_reason=stop_reason,
            loss_mask=loss_mask,
            prompt_ids=prompt_ids,
            trajectory_id=trajectory_id,
            summarization_count=summarization_count,
        )
