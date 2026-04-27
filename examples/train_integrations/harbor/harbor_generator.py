import asyncio
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional
from loguru import logger
from uuid import uuid4
from skyrl.train.generators.base import GeneratorInterface, GeneratorInput, GeneratorOutput, TrajectoryID
from skyrl.train.generators.utils import get_rollout_metrics
from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl.backends.skyrl_train.inference_engines.base import ConversationType
from skyrl.train.utils.rate_limiter import create_rate_limiter
from tqdm import tqdm
from omegaconf import DictConfig
from harbor.trial.trial import Trial
from harbor.models.trial.config import TrialConfig
from harbor.models.agent.rollout_detail import RolloutDetail

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
class HarborTrajectoryOutput:
    """One trajectory's raw output from Harbor.

    Holds the entire ``rollout_details`` from ``agent_result``. Per-step interpretation
    (loss-mask / reward broadcast / overlong filtering) is done downstream in
    `build_step_wise_generator_output`.
    """

    trajectory_id: TrajectoryID
    # Entire rollout_details list as returned by harbor's agent_result. None for failed trajectories
    # (agent_timeout / error) that we will mask in `build_step_wise_generator_output`.
    rollout_details: Optional[List[RolloutDetail]] = None
    reward: float = 0.0
    num_turns: int = 0
    # One of: "complete", "context_length", "agent_timeout", "error". Used by
    # `build_step_wise_generator_output` to decide whether to skip the entire prompt group.
    stop_reason: str = "complete"


def build_step_wise_generator_output(
    trajectory_outputs: List[HarborTrajectoryOutput], overlong_filtering: bool
) -> GeneratorOutput:
    """Flatten per-trajectory rollout details into one entry per LLM turn.

    Steps for one trajectory are emitted contiguously and the last step has
    ``is_last_step=True``. Failures (timeout / unknown error / empty rollout
    details) are batched per ``instance_id``: if any rollout for prompt P
    failed, all rollouts for P are replaced with single zeroed-out
    placeholder steps.
    """
    # 1. Identify failed instances. If any rollout for prompt P failed, mask all rollouts for P (conservative).
    timeout_instance_ids = set()
    error_instance_ids = set()
    all_instance_ids = set()
    num_timeout_trajectories = 0
    num_error_trajectories = 0
    for traj in trajectory_outputs:
        instance_id = traj.trajectory_id.instance_id
        all_instance_ids.add(instance_id)
        if traj.stop_reason == "agent_timeout":
            num_timeout_trajectories += 1
            timeout_instance_ids.add(instance_id)
        elif traj.stop_reason == "error" or traj.rollout_details is None:
            num_error_trajectories += 1
            error_instance_ids.add(instance_id)
    masked_instance_ids = timeout_instance_ids | error_instance_ids

    # 2. Walk trajectories and emit one entry of GeneratorOutput per step.
    prompt_token_ids: List[List[int]] = []
    response_ids: List[List[int]] = []
    rewards: List[float] = []
    loss_masks: List[List[int]] = []
    stop_reasons: List[str] = []
    is_last_step_list: List[bool] = []
    out_trajectory_ids: List[TrajectoryID] = []
    rollout_logprobs_list: List[List[float]] = []

    successful_trajectories: List[HarborTrajectoryOutput] = []
    response_ids_for_metrics: List[List[int]] = []
    rewards_for_metrics: List[float] = []
    for traj in trajectory_outputs:
        tid = traj.trajectory_id

        # 2.1. For failed trajectories, set loss mask to [0] and stop reason to "error".
        if tid.instance_id in masked_instance_ids:
            prompt_token_ids.append([0])
            response_ids.append([0])
            rewards.append(0.0)
            loss_masks.append([0])
            stop_reasons.append("error")
            is_last_step_list.append(True)
            out_trajectory_ids.append(tid)
            rollout_logprobs_list.append([0.0])
            continue

        # 2.2. For successful trajectories, emit one entry per step.
        successful_trajectories.append(traj)

        # 2.3. Check rollout_details expected format.
        # Expect no summarization; rollout_details is a single linear chat segment from the main agent.
        # TODO(Charlie): Support summarization.
        assert len(traj.rollout_details) == 1, f"Expected exactly one rollout segment, got {len(traj.rollout_details)}."
        rollout_detail = traj.rollout_details[0]
        prompt_token_ids_per_turn = rollout_detail["prompt_token_ids"]
        completion_token_ids_per_turn = rollout_detail["completion_token_ids"]
        logprobs_per_turn = rollout_detail["logprobs"]
        n_turns = len(completion_token_ids_per_turn)
        assert len(prompt_token_ids_per_turn) == n_turns and len(logprobs_per_turn) == n_turns, (
            f"Malformed rollout_details (prompts={len(prompt_token_ids_per_turn)}, completions={n_turns}, "
            f"logprobs={len(logprobs_per_turn)})."
        )

        # 2.4. Emit one entry per step, following SkyRL's step-wise convention.
        for t in range(n_turns):
            comp_ids = completion_token_ids_per_turn[t]
            p_ids = prompt_token_ids_per_turn[t]
            lp = logprobs_per_turn[t]
            assert len(lp) == len(comp_ids), "logprobs and completion token ids must have the same length."

            # Record actual reward in last turn, and zeros for all other turns.
            is_last = t == n_turns - 1
            reward = traj.reward if is_last else 0.0

            # Loss mask.
            step_loss_mask = [1] * len(comp_ids)
            step_stop_reason = "complete"
            if traj.stop_reason == "context_length":
                step_stop_reason = "context_length"
                if overlong_filtering:
                    step_loss_mask = [0] * len(comp_ids)

            prompt_token_ids.append(p_ids)
            response_ids.append(comp_ids)
            rewards.append(reward)
            loss_masks.append(step_loss_mask)
            stop_reasons.append(step_stop_reason)
            is_last_step_list.append(is_last)
            out_trajectory_ids.append(tid)
            rollout_logprobs_list.append(lp)

        # 2.5. For trajectory-level metrics, record the last turn's prompt IDs and response IDs which
        # contains the entire trajectory.
        response_ids_for_metrics.append(prompt_token_ids_per_turn[-1] + completion_token_ids_per_turn[-1])
        rewards_for_metrics.append(traj.reward)

    # 3. Aggregate trajectory-level metrics for logging.
    if successful_trajectories:
        rollout_metrics = get_rollout_metrics(response_ids_for_metrics, rewards_for_metrics)
        rollout_metrics["generate/trajectories_context_length_exceeded"] = sum(
            1 for t in successful_trajectories if t.stop_reason == "context_length"
        )
        rollout_metrics["generate/avg_num_turns"] = sum(t.num_turns for t in successful_trajectories) / len(
            successful_trajectories
        )
    else:
        rollout_metrics = {}

    rollout_metrics["generate/num_timeout_trajectories"] = num_timeout_trajectories
    rollout_metrics["generate/num_error_trajectories"] = num_error_trajectories
    rollout_metrics["generate/num_masked_instances"] = len(masked_instance_ids)

    return GeneratorOutput(
        prompt_token_ids=prompt_token_ids,
        response_ids=response_ids,
        rewards=rewards,
        loss_masks=loss_masks,
        stop_reasons=stop_reasons,
        rollout_metrics=rollout_metrics,
        rollout_logprobs=rollout_logprobs_list,
        trajectory_ids=out_trajectory_ids,
        is_last_step=is_last_step_list,
    )


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

        if not getattr(generator_cfg, "step_wise_trajectories", False):
            raise ValueError(
                "HarborGenerator only supports step-wise training. " "Set generator.step_wise_trajectories=true."
            )
        if not getattr(generator_cfg, "merge_stepwise_output", False):
            logger.warning(
                "merge_stepwise_output=true is not set; will not merge step-wise outputs. This "
                "may result in much slower training."
            )

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

        # Step-wise needs per-turn token IDs and logprobs from vLLM via Harbor.
        agent_kwargs = self._harbor_trial_config_template["agent"]["kwargs"]
        if not agent_kwargs.get("collect_rollout_details", False):
            logger.warning("step_wise_trajectories=true requires collect_rollout_details=true; enabling automatically.")
            agent_kwargs["collect_rollout_details"] = True

        # Can support summarization in future.
        if agent_kwargs.get("enable_summarize", False):
            raise ValueError(
                "step_wise_trajectories=true is incompatible with enable_summarize=true. "
                "Set harbor_trial_config.agent.kwargs.enable_summarize=false."
            )

        logger.info(
            f"HarborGenerator initialized with Harbor config. "
            f"Agent: {self._harbor_trial_config_template.get('agent', {}).get('name')}, "
            f"Trials dir: {self._harbor_trial_config_template.get('trials_dir', 'trials')}"
        )

        rate_limit_config = getattr(generator_cfg, "rate_limit", None)
        self._rate_limiter = create_rate_limiter(rate_limit_config)

    async def generate(self, input_batch: GeneratorInput, disable_tqdm: bool = False) -> GeneratorOutput:
        prompts = input_batch["prompts"]
        trajectory_ids = input_batch["trajectory_ids"]

        if trajectory_ids is None:
            raise ValueError("`trajectory_ids` is required in the input batch")
        if len(prompts) != len(trajectory_ids):
            raise ValueError(
                f"Prompt count ({len(prompts)}) doesn't match trajectory_ids count ({len(trajectory_ids)})"
            )

        all_outputs: List[HarborTrajectoryOutput] = [None] * len(prompts)  # type: ignore[list-item]
        progress = tqdm(
            disable=disable_tqdm,  # disable for fully async training
            total=len(prompts),
            desc="Generating Trajectories",
            miniters=max(1, len(prompts) // 10),
            mininterval=5,
        )

        async def _worker(idx, prompt, trajectory_id):
            result = await self._harbor_agent_loop(prompt=prompt, trajectory_id=trajectory_id)
            all_outputs[idx] = result
            progress.update(1)

        try:
            async with asyncio.TaskGroup() as tg:
                for idx, (prompt, trajectory_id) in enumerate(zip(prompts, trajectory_ids)):
                    tg.create_task(_worker(idx, prompt, trajectory_id))
        finally:
            progress.close()

        return build_step_wise_generator_output(
            all_outputs, overlong_filtering=self.generator_cfg.apply_overlong_filtering
        )

    async def _harbor_agent_loop(
        self,
        prompt: ConversationType,
        trajectory_id: TrajectoryID,
    ) -> HarborTrajectoryOutput:
        """Run a single Harbor trial and return the rollout details plus a trajectory-level reward.
        Retries on unknown errors; context length errors train with reward=0; agent timeouts mask the trajectory.
        """
        reward = None
        results = None
        rollout_details = None
        num_turns = None
        successful = False
        is_context_length_error = False
        is_agent_timeout_error = False

        for i in range(MAX_NUM_RETRIES_PER_TRIAL):
            prefix = f"Trajectory {trajectory_id} attempt {i+1}/{MAX_NUM_RETRIES_PER_TRIAL}"
            results = None
            try:
                # Create a fresh Trial each attempt so agent state is clean on retry.
                config = deepcopy(self._harbor_trial_config_template)
                config["task"] = {"path": prompt}
                config["agent"]["kwargs"]["session_id"] = uuid4().hex
                trial_config = TrialConfig.model_validate(config)
                trial = await Trial.create(trial_config)

                async with self._rate_limiter:
                    results = await trial.run()

                # Parse exception type
                exc_type = results.exception_info.exception_type if results.exception_info else None
                is_context_length_error = exc_type == "ContextLengthExceededError"
                is_agent_timeout_error = exc_type == "AgentTimeoutError"

                # Determine reward.
                if is_agent_timeout_error:
                    # AgentTimeoutError: not successful, no retry, loss-masked
                    logger.debug(f"{prefix} hit AgentTimeoutError (no retry). Results: {results}")
                    break
                elif is_context_length_error:
                    # ContextLengthExceededError: always train with reward=0.
                    logger.debug(f"{prefix} hit ContextLengthExceededError, setting reward=0. Results: {results}")
                    reward = 0.0
                elif not results.verifier_result:
                    # Does not have a verifier result, so it's not successful, will retry
                    logger.warning(f"{prefix} failed: Exception info: {results.exception_info}. Results: {results}")
                    continue
                else:
                    reward = float(results.verifier_result.rewards["reward"])

                # Extract rollout details and check for success
                rollout_details = results.agent_result.rollout_details
                num_turns = results.agent_result.metadata["n_episodes"]

                if (
                    rollout_details
                    and len(rollout_details) >= 1
                    and len(rollout_details[0].get("completion_token_ids", [])) > 0
                ):
                    successful = True
                    logger.debug(f"{prefix} successful: reward={reward}.")
                    break
                else:
                    logger.warning(f"{prefix} failed: empty/missing rollout_details. Results: {results}")
            except Exception as e:
                logger.warning(f"{prefix} failed: Error running trial: {e}. Results: {results}")
                continue

        if not successful:
            stop_reason = "agent_timeout" if is_agent_timeout_error else "error"
            error_message = f"Trajectory {trajectory_id} failed (stop_reason={stop_reason}), will set loss mask to [0]."
            if stop_reason == "error":
                error_message += f" Results: {results}"
            logger.warning(error_message)
            return HarborTrajectoryOutput(trajectory_id=trajectory_id, rollout_details=None, stop_reason=stop_reason)
        else:
            return HarborTrajectoryOutput(
                trajectory_id=trajectory_id,
                rollout_details=rollout_details,
                reward=reward,
                num_turns=num_turns,
                stop_reason="context_length" if is_context_length_error else "complete",
            )
