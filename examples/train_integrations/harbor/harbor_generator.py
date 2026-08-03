import asyncio
import logging
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional
from uuid import uuid4

# Suppress LiteLLM verbose logging
import litellm
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from harbor.models.agent.rollout_detail import RolloutDetail
from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial
from skyrl.backends.skyrl_train.inference_servers.base import (
    ConversationType,
    InferenceEngineInterface,
)
from skyrl.train.generators.base import (
    GeneratorInput,
    GeneratorInterface,
    GeneratorOutput,
    TrajectoryID,
)
from skyrl.train.generators.tito import (
    Trace,
    TraceOutcome,
    build_trace_generator_output,
)
from skyrl.train.generators.tito.proxy import TITOProxy
from skyrl.train.generators.tito.renderer import PrimeRendererAdapter
from skyrl.train.utils.rate_limiter import create_rate_limiter

setattr(litellm, "suppress_debug_info", True)  # Suppress the "Provider List" output
litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

# We have N retries for each trial, if one of the rollout (out of n_samples_per_prompt) fails
# after N attemptes, we skip this prompt altogether.
MAX_NUM_RETRIES_PER_TRIAL = 2


@dataclass
class HarborTrajectoryOutput:
    """One trajectory's raw output from Harbor.

    The TITO trace is authoritative for training output. Harbor rollout details
    are retained only for optional call-by-call parity validation.
    """

    trajectory_id: TrajectoryID
    rollout_details: Optional[List[RolloutDetail]] = None
    trace: Optional[Trace] = None
    reward: float = 0.0
    num_turns: int = 0
    # One of: "complete", "context_length", "agent_timeout", "error".
    stop_reason: str = "complete"
    # End-to-end wall-clock time (seconds) to generate this trajectory. Optional: left as None if
    # timing was not recorded.
    e2e_time: Optional[float] = None


class HarborGenerator(GeneratorInterface):
    def __init__(
        self,
        generator_cfg: DictConfig,
        harbor_cfg: DictConfig,
        inference_engine_client: InferenceEngineInterface,
        tokenizer,
        max_seq_len: int,
    ):
        """
        Args:
            generator_cfg: DictConfig object containing the generator configuration
            harbor_cfg: DictConfig object containing the Harbor configuration
            inference_engine_client: inference engine client for interacting with the inference engines
            tokenizer: tokenizer object for encoding and decoding text
            max_seq_len: Maximum total sequence length (prompt + response). Used to truncate responses.
        """
        ie_cfg = generator_cfg.inference_engine
        self.base_url = inference_engine_client.get_endpoint_url()
        # Kept so we can notify the router when a session (trial attempt) ends,
        # which lets session-aware routing policies rebalance new trajectories.
        self.inference_engine_client = inference_engine_client
        self.generator_cfg = generator_cfg
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.renderer = PrimeRendererAdapter(tokenizer)
        self._validate_rollout_details = getattr(generator_cfg, "tito_validate_rollout_details", True)

        self._harbor_trial_config_template = deepcopy(harbor_cfg)

        # Mixed into the prefix-cache salt so distinct models / adapters don't share cache blocks.
        self._served_model_name = ie_cfg.served_model_name

        # Set model_name and api_base once (constant across all trials)
        assert ie_cfg.served_model_name is not None, "served_model_name must be set"
        assert (
            "/" not in ie_cfg.served_model_name
        ), "served_model_name must not contain '/', Harbor expects hosted_vllm/{model_name}"
        self._harbor_trial_config_template.setdefault("agent", {})[
            "model_name"
        ] = f"hosted_vllm/{ie_cfg.served_model_name}"
        self._harbor_trial_config_template["agent"].setdefault("kwargs", {})["api_base"] = f"{self.base_url}/v1"

        # Keep Harbor's own token bookkeeping as a parity oracle during rollout.
        agent_kwargs = self._harbor_trial_config_template["agent"]["kwargs"]
        if self._validate_rollout_details and not agent_kwargs.get("collect_rollout_details", False):
            logger.warning("TITO parity validation requires collect_rollout_details=true; enabling automatically.")
            agent_kwargs["collect_rollout_details"] = True
        if not agent_kwargs.get("interleaved_thinking", False):
            logger.warning(
                "TITO requires interleaved_thinking=true to preserve assistant history; enabling automatically."
            )
            agent_kwargs["interleaved_thinking"] = True

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

    def _compute_cache_salt(self) -> Optional[str]:
        """Derive a prefix-cache salt from the current policy version.

        Mirrors ``SkyRLGymGenerator._compute_cache_salt``: keyed on the engine's ``weight_version`` and
        served model name, called once per ``generate`` batch. Returns ``None`` when disabled or when the
        client exposes no weight version.
        """
        if not getattr(self.generator_cfg, "use_cache_salt", False):
            return None
        weight_version = getattr(self.inference_engine_client, "weight_version", None)
        if weight_version is None:
            return None
        version = f"{self._served_model_name}@" if self._served_model_name is not None else ""
        return f"{version}{weight_version}"

    async def generate(self, input_batch: GeneratorInput, disable_tqdm: bool = False) -> GeneratorOutput:
        prompts = input_batch["prompts"]
        trajectory_ids = input_batch["trajectory_ids"]

        if trajectory_ids is None:
            raise ValueError("`trajectory_ids` is required in the input batch")
        if len(prompts) != len(trajectory_ids):
            raise ValueError(
                f"Prompt count ({len(prompts)}) doesn't match trajectory_ids count ({len(trajectory_ids)})"
            )

        # Captured once so every trajectory shares the policy version at the start of the batch.
        cache_salt = self._compute_cache_salt()

        all_outputs: List[Optional[HarborTrajectoryOutput]] = [None] * len(prompts)
        progress = tqdm(
            disable=disable_tqdm,  # disable for fully async training
            total=len(prompts),
            desc="Generating Trajectories",
            miniters=max(1, len(prompts) // 10),
            mininterval=5,
        )

        proxy = TITOProxy(self.inference_engine_client, self.renderer)

        async def _worker(idx, prompt, trajectory_id):
            result = await self._harbor_agent_loop(
                proxy=proxy,
                prompt=prompt,
                trajectory_id=trajectory_id,
                cache_salt=cache_salt,
            )
            all_outputs[idx] = result
            progress.update(1)

        try:
            async with proxy.serving():
                async with asyncio.TaskGroup() as tg:
                    for idx, (prompt, trajectory_id) in enumerate(zip(prompts, trajectory_ids)):
                        tg.create_task(_worker(idx, prompt, trajectory_id))
        finally:
            progress.close()

        if any(output is None for output in all_outputs):
            raise RuntimeError("Harbor generation completed without producing every trajectory output")
        completed_outputs = [output for output in all_outputs if output is not None]
        outcomes = [
            TraceOutcome(
                trace=output.trace if output.trace is not None else self._empty_trace(),
                trajectory_id=output.trajectory_id,
                reward=output.reward,
                stop_reason=output.stop_reason,
                generation_time=output.e2e_time,
                num_turns=output.num_turns,
            )
            for output in completed_outputs
        ]
        return build_trace_generator_output(
            outcomes,
            overlong_filtering=self.generator_cfg.apply_overlong_filtering,
            step_wise=self.generator_cfg.step_wise_trajectories,
        )

    async def _harbor_agent_loop(
        self,
        proxy: TITOProxy,
        prompt: ConversationType,
        trajectory_id: TrajectoryID,
        cache_salt: Optional[str] = None,
    ) -> HarborTrajectoryOutput:
        """Run a single Harbor trial and return the rollout details plus a trajectory-level reward.
        Retries on unknown errors; context length errors train with reward=0; agent timeouts mask the trajectory.
        """
        agent_loop_start_time = time.monotonic()
        reward = None
        results = None
        rollout_details = None
        num_turns = None
        successful_trace = None
        successful = False
        is_context_length_error = False
        is_agent_timeout_error = False

        for i in range(MAX_NUM_RETRIES_PER_TRIAL):
            prefix = f"Trajectory {trajectory_id} attempt {i+1}/{MAX_NUM_RETRIES_PER_TRIAL}"
            results = None
            # Each attempt is a distinct router session; track it so it can be
            # released on completion/error/cancellation.
            session_id = uuid4().hex
            trace = Trace()
            try:
                # Create a fresh Trial each attempt so agent state is clean on retry.
                config = deepcopy(self._harbor_trial_config_template)
                config["task"] = {"path": prompt}
                config["agent"]["kwargs"]["session_id"] = session_id
                # Forward the salt via llm_kwargs.extra_body -> LiteLLM -> the vLLM request's top-level
                # `cache_salt` field. vLLM rejects an empty salt, so attach only when set.
                if cache_salt is not None:
                    llm_kwargs = config["agent"]["kwargs"].setdefault("llm_kwargs", {})
                    extra_body = llm_kwargs.setdefault("extra_body", {})
                    if not isinstance(extra_body, dict):
                        raise TypeError("harbor_trial_config.agent.kwargs.llm_kwargs.extra_body must be a mapping")
                    extra_body["cache_salt"] = cache_salt
                async with proxy.register(
                    trace,
                    router_session_id=session_id,
                    cache_salt=cache_salt,
                    model=self._served_model_name,
                ) as handle:
                    config["agent"]["kwargs"]["api_base"] = f"{handle.base_url}/v1"
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
                    verifier_result = results.verifier_result
                    if verifier_result is None:
                        raise RuntimeError("Harbor verifier result disappeared after validation")
                    verifier_rewards = verifier_result.rewards
                    if verifier_rewards is None or "reward" not in verifier_rewards:
                        raise RuntimeError("Harbor verifier result is missing reward")
                    reward = float(verifier_rewards["reward"])

                # Extract rollout details and check for success
                agent_result = results.agent_result
                if agent_result is None:
                    raise RuntimeError("Harbor result is missing agent_result")
                rollout_details = agent_result.rollout_details
                metadata = agent_result.metadata
                if metadata is None or "n_episodes" not in metadata:
                    raise RuntimeError("Harbor agent result is missing n_episodes metadata")
                num_turns = int(metadata["n_episodes"])

                if trace.committed_turns():
                    if getattr(self, "_validate_rollout_details", True):
                        self._validate_trace_parity(trace, rollout_details)
                    successful_trace = trace
                    successful = True
                    logger.debug(f"{prefix} successful: reward={reward}.")
                    break
                else:
                    logger.warning(f"{prefix} failed: proxy trace contains no committed turns. Results: {results}")
            except Exception as e:
                logger.warning(f"{prefix} failed: Error running trial: {e}. Results: {results}")
                continue

        if not successful:
            stop_reason = "agent_timeout" if is_agent_timeout_error else "error"
            error_message = f"Trajectory {trajectory_id} failed (stop_reason={stop_reason}), will set loss mask to [0]."
            if stop_reason == "error":
                error_message += f" Results: {results}"
            logger.warning(error_message)
            return HarborTrajectoryOutput(
                trajectory_id=trajectory_id,
                rollout_details=None,
                trace=self._empty_trace(),
                stop_reason=stop_reason,
                e2e_time=time.monotonic() - agent_loop_start_time,
            )
        else:
            if reward is None or num_turns is None or successful_trace is None:
                raise RuntimeError("Successful Harbor trajectory is missing required TITO output")
            return HarborTrajectoryOutput(
                trajectory_id=trajectory_id,
                rollout_details=rollout_details,
                trace=successful_trace,
                reward=reward,
                num_turns=num_turns,
                stop_reason="context_length" if is_context_length_error else "complete",
                e2e_time=time.monotonic() - agent_loop_start_time,
            )

    @staticmethod
    def _empty_trace() -> Trace:
        trace = Trace()
        trace.seal()
        return trace

    @staticmethod
    def _validate_trace_parity(trace: Trace, rollout_details: Optional[List[RolloutDetail]]) -> None:
        """Require exact per-call parity while Harbor rollout details are enabled."""
        if not rollout_details:
            raise ValueError("Harbor did not return rollout details for TITO parity validation")
        if len(rollout_details) != 1:
            raise ValueError(f"Expected one Harbor rollout segment, got {len(rollout_details)}")

        detail = rollout_details[0]
        prompt_ids = detail.get("prompt_token_ids", [])
        completion_ids = detail.get("completion_token_ids", [])
        logprobs = detail.get("logprobs", [])
        turns = trace.committed_turns()
        if not (len(prompt_ids) == len(completion_ids) == len(logprobs) == len(turns)):
            raise ValueError(
                "Harbor rollout detail lengths do not match committed TITO turns: "
                f"prompts={len(prompt_ids)}, completions={len(completion_ids)}, "
                f"logprobs={len(logprobs)}, turns={len(turns)}"
            )

        for index, turn in enumerate(turns):
            if list(turn.prompt_token_ids) != prompt_ids[index]:
                raise ValueError(f"TITO prompt IDs differ from Harbor rollout details at turn {index}")
            if list(turn.completion_ids) != completion_ids[index]:
                raise ValueError(f"TITO completion IDs differ from Harbor rollout details at turn {index}")
            if list(turn.completion_logprobs) != logprobs[index]:
                raise ValueError(f"TITO completion logprobs differ from Harbor rollout details at turn {index}")
