import asyncio
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Union
from loguru import logger
from uuid import uuid4
from skyrl.train.generators.base import GeneratorInterface, GeneratorInput, GeneratorOutput, TrajectoryID
from skyrl.train.generators.utils import get_rollout_metrics, get_response_ids_and_loss_mask_from_messages
from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl.backends.skyrl_train.inference_engines.base import ConversationType
from skyrl.train.utils.rate_limiter import create_rate_limiter
from tqdm import tqdm
from omegaconf import DictConfig
from harbor.trial.trial import Trial
from harbor.models.trial.config import TrialConfig
from harbor.models.trial.result import TrialResult

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
class HarborAgentOutput:
    response_ids: List[int]
    reward: Union[float, List[float]]
    stop_reason: str
    loss_mask: List[int]
    prompt_ids: List[int]
    trajectory_id: TrajectoryID
    summarization_count: Optional[int] = None
    num_turns: Optional[int] = None
    rollout_logprobs: Optional[List[float]] = None


@dataclass
class HarborStepWiseOutput:
    """Step-wise output from a single Harbor trajectory.

    Each step_output corresponds to one agent turn (LLM call),
    using the exact token IDs and logprobs from vLLM (no re-tokenization).
    """

    step_outputs: List[HarborAgentOutput] = field(default_factory=list)
    trajectory_id: Optional[TrajectoryID] = None
    summarization_count: Optional[int] = None
    num_turns: Optional[int] = None


class HarborGenerator(GeneratorInterface):
    def __init__(
        self,
        generator_cfg: DictConfig,
        harbor_cfg: DictConfig,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
        max_seq_len: int,
    ):
        """
        Args:
            generator_cfg: DictConfig object containing the generator configuration
            harbor_cfg: DictConfig object containing the Harbor configuration
            inference_engine_client: InferenceEngineClient object for interacting with the inference engines
            tokenizer: tokenizer object for encoding and decoding text
            max_seq_len: Maximum total sequence length (prompt + response). Used to truncate responses.
        """
        ie_cfg = generator_cfg.inference_engine
        self.base_url = f"http://{ie_cfg.http_endpoint_host}:{ie_cfg.http_endpoint_port}"
        self.generator_cfg = generator_cfg
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.step_wise = getattr(generator_cfg, "step_wise_trajectories", False)

        # Harbor config template - users can specify any Harbor TrialConfig options in YAML or command line.
        # SkyRL injects: model_name and api_base (once at init), task.path and session_id (per trial)
        self._harbor_trial_config_template = deepcopy(harbor_cfg)

        # Set model_name and api_base once (constant across all trials)
        assert ie_cfg.served_model_name is not None, "served_model_name must be set"
        assert (
            "/" not in ie_cfg.served_model_name
        ), "served_model_name must not contain '/', Harbor expects hosted_vllm/{model_name}"
        self._harbor_trial_config_template.setdefault("agent", {})[
            "model_name"
        ] = f"hosted_vllm/{ie_cfg.served_model_name}"
        self._harbor_trial_config_template["agent"].setdefault("kwargs", {})["api_base"] = f"{self.base_url}/v1"

        # Config post-processings
        agent_kwargs = self._harbor_trial_config_template["agent"].get("kwargs", {})

        # Summarization is not supproted yet
        if agent_kwargs.get("enable_summarize", False):
            raise ValueError(
                "step_wise_trajectories=True is incompatible with enable_summarize=True. "
                "Summarization invalidates rollout_details. Set enable_summarize=false."
            )

        # Step-wise training requires collect_rollout_details to get per-turn token IDs and logprobs
        if self.step_wise and not agent_kwargs.get("collect_rollout_details", False):
            logger.warning(
                "step_wise_trajectories=True but collect_rollout_details is not enabled in Harbor config. "
                "Enabling it automatically."
            )
            self._harbor_trial_config_template["agent"]["kwargs"]["collect_rollout_details"] = True

        logger.info(
            f"HarborGenerator initialized with Harbor config. "
            f"Agent: {self._harbor_trial_config_template.get('agent', {}).get('name')}, "
            f"Trials dir: {self._harbor_trial_config_template.get('trials_dir', 'trials')}, "
            f"Step-wise: {self.step_wise}"
        )

        # Read custom chat template
        custom_chat_template_path = ie_cfg.engine_init_kwargs.get("chat_template", None)
        if custom_chat_template_path:
            with open(custom_chat_template_path, "r") as f:
                self.custom_chat_template_content = f.read()
            logger.info(f"HarborGenerator initialized with custom chat template read from: {custom_chat_template_path}")
        else:
            self.custom_chat_template_content = None

        # Initialize rate limiter from generator config (not part of Harbor TrialConfig)
        rate_limit_config = getattr(generator_cfg, "rate_limit", None)
        self._rate_limiter = create_rate_limiter(rate_limit_config)

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        prompts = input_batch["prompts"]
        trajectory_ids = input_batch["trajectory_ids"]

        if trajectory_ids is None:
            raise ValueError("`trajectory_ids` is required in the input batch")
        if len(prompts) != len(trajectory_ids):
            raise ValueError(
                f"Prompt count ({len(prompts)}) doesn't match " f"trajectory_ids count ({len(trajectory_ids)})"
            )

        all_outputs: List[Union[HarborAgentOutput, HarborStepWiseOutput]] = [None] * len(prompts)  # type: ignore[list-item]
        progress = tqdm(
            total=len(prompts),
            desc="Generating Trajectories",
            miniters=max(1, len(prompts) // 10),
            mininterval=5,
        )

        async def _worker(idx, prompt, trajectory_id):
            result = await self.harbor_agent_loop(prompt=prompt, trajectory_id=trajectory_id)
            all_outputs[idx] = result
            progress.update(1)

        try:
            async with asyncio.TaskGroup() as tg:
                for idx, (prompt, trajectory_id) in enumerate(zip(prompts, trajectory_ids)):
                    tg.create_task(_worker(idx, prompt, trajectory_id))
        finally:
            progress.close()

        if self.step_wise:
            return self._build_step_wise_generator_output(all_outputs, trajectory_ids)
        else:
            all_outputs, rollout_metrics = self._mask_failed_instances_and_compute_metrics(all_outputs)
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

    def _build_step_wise_generator_output(
        self,
        all_outputs: List[Union[HarborAgentOutput, HarborStepWiseOutput]],
        trajectory_ids: List[TrajectoryID],
    ) -> GeneratorOutput:
        """Flatten step-wise outputs into the GeneratorOutput format.

        Each multi-turn trajectory becomes N separate (prompt, response) samples,
        with `is_last_step` marking the final step of each trajectory.
        """
        # First, identify failed instances (same logic as non-step-wise)
        timeout_instance_ids = set()
        error_instance_ids = set()
        all_instance_ids = set()
        num_timeout_trajectories = 0
        num_error_trajectories = 0

        for output in all_outputs:
            if isinstance(output, HarborStepWiseOutput):
                tid = output.trajectory_id
                # Check if any step indicates failure
                stop_reasons = [s.stop_reason for s in output.step_outputs] if output.step_outputs else ["error"]
                last_stop = stop_reasons[-1] if stop_reasons else "error"
            else:
                tid = output.trajectory_id
                last_stop = output.stop_reason

            all_instance_ids.add(tid.instance_id)
            if last_stop == "agent_timeout":
                num_timeout_trajectories += 1
                timeout_instance_ids.add(tid.instance_id)
            elif last_stop == "error":
                num_error_trajectories += 1
                error_instance_ids.add(tid.instance_id)

        masked_instance_ids = timeout_instance_ids | error_instance_ids

        # Flatten step-wise outputs
        responses = []
        rewards = []
        stop_reasons = []
        loss_masks = []
        prompt_token_ids = []
        is_last_step_list = []
        out_trajectory_ids = []
        rollout_logprobs_list = []
        successful_outputs_for_metrics = []

        for output in all_outputs:
            if isinstance(output, HarborStepWiseOutput):
                tid = output.trajectory_id
            else:
                tid = output.trajectory_id

            is_masked = tid.instance_id in masked_instance_ids

            if is_masked:
                # Emit a single zeroed-out step for masked instances
                responses.append([0])
                rewards.append([0.0])
                stop_reasons.append("error")
                loss_masks.append([0])
                prompt_token_ids.append([0])
                is_last_step_list.append(True)
                out_trajectory_ids.append(tid)
                rollout_logprobs_list.append([0.0])
                continue

            if isinstance(output, HarborStepWiseOutput):
                for j, step in enumerate(output.step_outputs):
                    is_last = j == len(output.step_outputs) - 1
                    responses.append(step.response_ids)
                    rewards.append(step.reward)
                    stop_reasons.append(step.stop_reason)
                    loss_masks.append(step.loss_mask)
                    prompt_token_ids.append(step.prompt_ids)
                    is_last_step_list.append(is_last)
                    out_trajectory_ids.append(tid)
                    rollout_logprobs_list.append(step.rollout_logprobs)

                    if is_last:
                        successful_outputs_for_metrics.append(step)
            else:
                # Non-step-wise fallback (e.g., failed trial returned HarborAgentOutput)
                responses.append(output.response_ids)
                rewards.append(output.reward if isinstance(output.reward, list) else [float(output.reward)])
                stop_reasons.append(output.stop_reason)
                loss_masks.append(output.loss_mask)
                prompt_token_ids.append(output.prompt_ids)
                is_last_step_list.append(True)
                out_trajectory_ids.append(tid)
                rollout_logprobs_list.append(output.rollout_logprobs if output.rollout_logprobs is not None else [0.0] * len(output.response_ids))
                successful_outputs_for_metrics.append(output)

        # Compute rollout metrics from successful last-step outputs
        if successful_outputs_for_metrics:
            # For metrics, use only last-step rewards (scalar form)
            metric_response_ids = [o.response_ids for o in successful_outputs_for_metrics]
            metric_rewards = []
            for o in successful_outputs_for_metrics:
                if isinstance(o.reward, list):
                    metric_rewards.append(sum(o.reward))
                else:
                    metric_rewards.append(o.reward)
            rollout_metrics = get_rollout_metrics(metric_response_ids, metric_rewards)

            # Add Harbor-specific metrics from original step-wise outputs
            summarization_counts = []
            num_turns_list = []
            context_exceeded = 0
            for output in all_outputs:
                if isinstance(output, HarborStepWiseOutput) and output.trajectory_id.instance_id not in masked_instance_ids:
                    if output.summarization_count is not None:
                        summarization_counts.append(output.summarization_count)
                    if output.num_turns is not None:
                        num_turns_list.append(output.num_turns)
                    if output.step_outputs and output.step_outputs[-1].stop_reason == "context_length":
                        context_exceeded += 1

            rollout_metrics["generate/trajectories_summarized"] = sum(1 for c in summarization_counts if c > 0)
            rollout_metrics["generate/trajectories_context_length_exceeded"] = context_exceeded
            if num_turns_list:
                rollout_metrics["generate/avg_num_turns"] = sum(num_turns_list) / len(num_turns_list)
        else:
            rollout_metrics = {}

        rollout_metrics["generate/num_timeout_trajectories"] = num_timeout_trajectories
        rollout_metrics["generate/num_error_trajectories"] = num_error_trajectories
        rollout_metrics["generate/num_masked_instances"] = len(masked_instance_ids)

        logger.info(
            f"\n# of masked instances: {len(masked_instance_ids)} / {len(all_instance_ids)}\n"
            f"# of timeout trajectories: {num_timeout_trajectories}\n"
            f"# of error trajectories: {num_error_trajectories}\n"
            f"# of flattened step-samples: {len(responses)}"
        )

        # --- Normalize prompt/response lengths to prevent padding OOM ---
        # The padding function pads all prompts to max(all_prompts) and responses to max(all_responses).
        # In step-wise mode, different steps have different prompt/response ratios (early turns have
        # short prompts, late turns have long prompts). Without normalization, the padded sequence
        # = max_prompt + max_response can far exceed max_seq_len, causing OOM.
        # Fix: truncate prompts from the LEFT so that max_prompt + max_response ≤ max_seq_len.
        max_response_len = max(len(r) for r in responses) if responses else 0
        max_prompt_budget = max(0, self.max_seq_len - max_response_len)
        num_prompts_truncated = 0
        for i in range(len(prompt_token_ids)):
            if len(prompt_token_ids[i]) > max_prompt_budget:
                excess = len(prompt_token_ids[i]) - max_prompt_budget
                prompt_token_ids[i] = prompt_token_ids[i][excess:]  # Truncate from left
                num_prompts_truncated += 1
        if num_prompts_truncated > 0:
            logger.info(
                f"Truncated {num_prompts_truncated} prompts from left to fit "
                f"max_prompt_budget={max_prompt_budget} (max_response={max_response_len}, max_seq_len={self.max_seq_len})"
            )

        # Check if any rollout_logprobs are available (non-zero-length lists)
        has_logprobs = any(lp is not None and len(lp) > 0 for lp in rollout_logprobs_list)

        if has_logprobs:
            # Ensure all entries are lists (replace None with zero-filled lists matching response length)
            for i, lp in enumerate(rollout_logprobs_list):
                if lp is None:
                    rollout_logprobs_list[i] = [0.0] * len(responses[i])

        generator_output: GeneratorOutput = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": rollout_logprobs_list if has_logprobs else None,
            "is_last_step": is_last_step_list,
            "trajectory_ids": out_trajectory_ids,
        }

        return generator_output

    @staticmethod
    def _mask_failed_instances_and_compute_metrics(
        all_outputs: List[HarborAgentOutput],
    ) -> tuple[List[HarborAgentOutput], dict]:
        """Mutates all_outputs in-place: zeros out every output belonging to a failed instance.

        For a group of trajectories (n_samples_per_prompt for the same prompt),
        if one trajectory fails we skip training the entire group.

        Returns:
            all_outputs: The same list, with failed-instance outputs zeroed out.
            rollout_metrics: Dict of rollout metrics for logging.
        """
        # Count failures by type before grouping overwrites stop_reason.
        num_timeout_trajectories = 0
        num_error_trajectories = 0
        timeout_instance_ids = set()
        error_instance_ids = set()
        all_instance_ids = set()
        for output in all_outputs:
            cur_instance_id = output.trajectory_id.instance_id
            all_instance_ids.add(cur_instance_id)
            if output.stop_reason == "agent_timeout":
                num_timeout_trajectories += 1
                timeout_instance_ids.add(cur_instance_id)
            elif output.stop_reason == "error":
                num_error_trajectories += 1
                error_instance_ids.add(cur_instance_id)

        masked_instance_ids = timeout_instance_ids | error_instance_ids

        # Zero-out all outputs belonging to any timeout or error instance so we skip training on them.
        successful_outputs: List[HarborAgentOutput] = []
        for output in all_outputs:
            if output.trajectory_id.instance_id in masked_instance_ids:
                output.response_ids = [0]
                output.stop_reason = "error"
                output.loss_mask = [0]
                output.prompt_ids = [0]
                output.reward = 0
            else:
                successful_outputs.append(output)

        # Rollout metrics for successful outputs.
        if len(successful_outputs) > 0:
            rollout_metrics = get_rollout_metrics(
                [output.response_ids for output in successful_outputs],
                [output.reward for output in successful_outputs],
            )
            rollout_metrics["generate/trajectories_summarized"] = sum(
                1 for output in successful_outputs if output.summarization_count > 0
            )
            rollout_metrics["generate/trajectories_context_length_exceeded"] = sum(
                1 for output in successful_outputs if output.stop_reason == "context_length"
            )
            rollout_metrics["generate/avg_num_turns"] = sum(output.num_turns for output in successful_outputs) / len(
                successful_outputs
            )
        else:
            rollout_metrics = {}

        # Failure metrics: timeout vs unknown error trajectories, and masked instances.
        rollout_metrics["generate/num_timeout_trajectories"] = num_timeout_trajectories
        rollout_metrics["generate/num_error_trajectories"] = num_error_trajectories
        rollout_metrics["generate/num_masked_instances"] = len(masked_instance_ids)

        logger.info(
            f"\n# of masked instances: {len(masked_instance_ids)} / {len(all_instance_ids)}\n"
            f"# of timeout trajectories: {num_timeout_trajectories}\n"
            f"# of error trajectories: {num_error_trajectories}"
        )

        return all_outputs, rollout_metrics

    async def harbor_agent_loop(
        self,
        prompt: ConversationType,
        trajectory_id: TrajectoryID,
    ) -> Union[HarborAgentOutput, HarborStepWiseOutput]:
        """
        Run a single harbor agent.

        Returns HarborStepWiseOutput when step_wise_trajectories=True,
        HarborAgentOutput otherwise (or on failure).
        """
        # Run the trial to get `reward`, `chat_history`, `summarization_count`, and `num_turns`
        reward = None
        chat_history = None
        summarization_count = None
        num_turns = None
        successful = False
        is_context_length_error = False
        is_agent_timeout_error = False
        results: Optional[TrialResult] = None
        for i in range(MAX_NUM_RETRIES_PER_TRIAL):
            prefix = f"Trajectory {trajectory_id} attempt {i+1}/{MAX_NUM_RETRIES_PER_TRIAL}"
            results = None
            try:
                # Create a fresh Trial each attempt so agent state is clean on retry.
                config = deepcopy(self._harbor_trial_config_template)
                config["task"] = {"path": prompt}
                config["agent"]["kwargs"]["session_id"] = uuid4().hex
                trial_config = TrialConfig.model_validate(config)
                trial = Trial(trial_config)

                async with self._rate_limiter:
                    results = await trial.run()

                # Parse exception type
                exc_type = results.exception_info.exception_type if results.exception_info else None
                is_context_length_error = exc_type == "ContextLengthExceededError"
                is_agent_timeout_error = exc_type == "AgentTimeoutError"

                # --- Determine reward ---
                if is_agent_timeout_error:
                    # AgentTimeoutError: not successful, no retry, loss-masked
                    logger.debug(f"{prefix} hit AgentTimeoutError (no retry). Results: {results}")
                    break
                elif is_context_length_error:
                    # ContextLengthExceededError: always train with reward=0.
                    logger.debug(
                        f"{prefix} hit ContextLengthExceededError, will train with reward=0. Results: {results}"
                    )
                    reward = 0
                elif not results.verifier_result:
                    # Does not have a verifier result, so it's not successful, will retry
                    logger.warning(f"{prefix} failed: Exception info: {results.exception_info}. Results: {results}")
                    continue
                else:
                    reward = results.verifier_result.rewards["reward"]

                # --- Extract chat history and check for success ---
                chat_history = results.agent_result.metadata["all_messages"]
                summarization_count = results.agent_result.metadata["summarization_count"]
                num_turns = results.agent_result.metadata["n_episodes"]
                if self.step_wise:
                    # For step-wise, success is defined as having rollout_details
                    if results.agent_result.rollout_details is not None and len(results.agent_result.rollout_details[0].get("prompt_token_ids", [])) > 0:
                        successful = True
                        logger.debug(f"{prefix} successful: reward={reward}. Results: {results}")
                        break
                    else:
                        logger.warning(
                            f"{prefix} failed: No rollout_details (or empty one). Results: {results}"
                        )
                else:
                    # For non-step-wise, success is defined as having a chat history starting with a user message
                    if len(chat_history) > 1 and chat_history[0]["role"] == "user":
                        successful = True
                        logger.debug(f"{prefix} successful: reward={reward}. Results: {results}")
                        break
                    else:
                        logger.warning(
                            f"{prefix} failed: Did not return a chat history with a user message. chat_history: {chat_history}\nResults: {results}"
                    )
            except Exception as e:
                logger.warning(f"{prefix} failed: Error running trial: {e}. Results: {results}")
                continue

        if not successful:
            # We make loss mask 0 so it does not contribute to model updates
            stop_reason = "agent_timeout" if is_agent_timeout_error else "error"
            error_message = f"Trajectory {trajectory_id} failed (stop_reason={stop_reason}), will set loss mask to [0]."
            if stop_reason == "error":
                error_message += f" Results: {results}"
            logger.warning(error_message)
            if self.step_wise:
                failed_step = HarborAgentOutput(
                    response_ids=[0],
                    reward=[0.0],
                    stop_reason=stop_reason,
                    loss_mask=[0],
                    prompt_ids=[0],
                    trajectory_id=trajectory_id,
                )
                return HarborStepWiseOutput(
                    step_outputs=[failed_step],
                    trajectory_id=trajectory_id,
                )
            return HarborAgentOutput(
                response_ids=[0],
                reward=0,
                stop_reason=stop_reason,
                loss_mask=[0],
                prompt_ids=[0],
                trajectory_id=trajectory_id,
            )

        # --- Step-wise path: use rollout_details from Harbor ---
        if self.step_wise:
            return self._build_step_wise_output(
                results=results,
                reward=reward,
                trajectory_id=trajectory_id,
                summarization_count=summarization_count,
                num_turns=num_turns,
                is_context_length_error=is_context_length_error,
            )

        # --- Non-step-wise path: re-tokenize chat history (original behavior) ---
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
        max_response_tokens = max(0, self.max_seq_len - initial_prompt_length)
        if is_context_length_error or len(response_ids) > max_response_tokens:
            stop_reason = "context_length"
        else:
            stop_reason = "complete"

        # Apply overlong filtering.
        # TODO(Charlie): should this also apply when the end reason is max_turns in Harbor?
        # Revisit. We would like to reuse `utils.py`'s implementation for overlong filtering.
        if self.generator_cfg.apply_overlong_filtering and stop_reason == "context_length":
            loss_mask = [0] * len(loss_mask)

        # Truncate to maximum allowed length.
        # NOTE(Charlie): though it shouldn't happen since it'd reach `ContextLengthExceededError`
        # from Harbor first. We do it anyway to be safe.
        response_ids = response_ids[:max_response_tokens]
        loss_mask = loss_mask[:max_response_tokens]

        return HarborAgentOutput(
            response_ids=response_ids,
            reward=reward,
            stop_reason=stop_reason,
            loss_mask=loss_mask,
            prompt_ids=prompt_ids,
            trajectory_id=trajectory_id,
            summarization_count=summarization_count,
            num_turns=num_turns,
        )

    def _build_step_wise_output(
        self,
        results: TrialResult,
        reward: float,
        trajectory_id: TrajectoryID,
        summarization_count: Optional[int],
        num_turns: Optional[int],
        is_context_length_error: bool,
    ) -> HarborStepWiseOutput:
        """Build a HarborStepWiseOutput from Harbor trial results using rollout_details.

        Uses the exact per-turn token IDs and logprobs from vLLM (no re-tokenization).
        This avoids retokenization drift and enables correct TIS computation.
        """
        # 1. Extract needed information
        rollout_details_list = results.agent_result.rollout_details
        assert rollout_details_list is not None, "rollout_details_list is required for step-wise training"

        # Use the first (main) rollout detail — this is the main agent's conversation.
        # Additional entries (index 1+) are subagent rollout details (e.g., summarization).
        main_rollout = rollout_details_list[0]
        prompt_token_ids_per_turn = main_rollout.get("prompt_token_ids", [])
        completion_token_ids_per_turn = main_rollout.get("completion_token_ids", [])
        logprobs_per_turn = main_rollout.get("logprobs", [])
        n_turns = len(completion_token_ids_per_turn)

        # 2. Validate the data
        # Assert no summarization occurred (not supported yet)
        if len(rollout_details_list) > 1:
            assert summarization_count == 0, (
                f"Trajectory {trajectory_id}: step_wise_trajectories=True but summarization occurred "
                f"({summarization_count} summarizations, {len(rollout_details_list)} rollout detail segments). "
                f"This is not supported. Set enable_summarize=false."
            )
        assert n_turns > 0, "rollout_details has no completion turns"
        assert len(prompt_token_ids_per_turn) == n_turns and len(logprobs_per_turn) == n_turns, (
            f"Trajectory {trajectory_id}: Expect prompt_token_ids, completion_token_ids, and "
            f"logprobs to have the same length, but respectively got: {len(prompt_token_ids_per_turn)} turns, ",
            f"{len(completion_token_ids_per_turn)} turns, and {len(logprobs_per_turn)} turns."
        )
        for turn_idx in range(n_turns):
            assert len(logprobs_per_turn[turn_idx]) == len(completion_token_ids_per_turn[turn_idx]), (
                f"Trajectory {trajectory_id} turn {turn_idx}: "
                f"logprobs length ({len(logprobs_per_turn[turn_idx])}) != "
                f"completion_ids length ({len(completion_token_ids_per_turn[turn_idx])}). "
            )

        step_outputs = []
        for turn_idx in range(n_turns):
            completion_ids = completion_token_ids_per_turn[turn_idx]
            turn_prompt_ids = prompt_token_ids_per_turn[turn_idx]
            turn_logprobs = logprobs_per_turn[turn_idx]
            is_last = turn_idx == n_turns - 1

            # Per-token reward: zeros for all but the last token of the LAST step
            turn_reward = [0.0] * len(completion_ids)
            if is_last and len(turn_reward) > 0:
                turn_reward[-1] = float(reward)

            # Determine stop reason for this step
            if is_context_length_error:
                turn_stop_reason = "context_length"
            else:
                turn_stop_reason = "complete"

            # Loss mask: all completion tokens are trainable (they are the model's generation)
            turn_loss_mask = [1] * len(completion_ids)
            # Apply overlong filtering.
            if self.generator_cfg.apply_overlong_filtering and turn_stop_reason == "context_length":
                turn_loss_mask = [0] * len(completion_ids)

            step_output = HarborAgentOutput(
                response_ids=completion_ids,
                reward=turn_reward,
                stop_reason=turn_stop_reason,
                loss_mask=turn_loss_mask,
                prompt_ids=turn_prompt_ids,
                trajectory_id=trajectory_id,
                rollout_logprobs=turn_logprobs,
                summarization_count=summarization_count if is_last else 0,
                num_turns=num_turns if is_last else 0,
            )
            step_outputs.append(step_output)

        return HarborStepWiseOutput(
            step_outputs=step_outputs,
            trajectory_id=trajectory_id,
            summarization_count=summarization_count,
            num_turns=num_turns,
        )
