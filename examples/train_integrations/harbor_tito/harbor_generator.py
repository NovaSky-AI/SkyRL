"""Harbor generator backed by SkyRL's token-in/token-out proxy."""

import asyncio
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional
from uuid import uuid4

from harbor.models.agent.rollout_detail import RolloutDetail
from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from examples.train_integrations.harbor.harbor_generator import (
    MAX_NUM_RETRIES_PER_TRIAL,
    HarborGenerator,
    HarborTrajectoryOutput,
)
from skyrl.backends.skyrl_train.inference_servers.base import (
    ConversationType,
    InferenceEngineInterface,
)
from skyrl.train.generators.base import GeneratorInput, GeneratorOutput, TrajectoryID
from skyrl.train.generators.tito import (
    Trace,
    TraceOutcome,
    build_trace_generator_output,
)
from skyrl.train.generators.tito.proxy import TITOProxy
from skyrl.train.generators.tito.renderer import PrimeRendererAdapter
from skyrl.train.utils.rate_limiter import create_rate_limiter


@dataclass
class TITOHarborTrajectoryOutput(HarborTrajectoryOutput):
    """One Harbor outcome whose trace is authoritative for training."""

    trace: Optional[Trace] = None


class TITOHarborGenerator(HarborGenerator):
    """Run Harbor trials through a generator-owned TITO proxy."""

    def __init__(
        self,
        generator_cfg: DictConfig,
        harbor_cfg: DictConfig,
        inference_engine_client: InferenceEngineInterface,
        tokenizer,
        max_seq_len: int,
    ):
        ie_cfg = generator_cfg.inference_engine
        self.base_url = inference_engine_client.get_endpoint_url()
        self.inference_engine_client = inference_engine_client
        self.generator_cfg = generator_cfg
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.renderer = PrimeRendererAdapter(tokenizer)
        self._validate_rollout_details = getattr(generator_cfg, "tito_validate_rollout_details", True)
        self._harbor_trial_config_template = deepcopy(harbor_cfg)
        self._served_model_name = ie_cfg.served_model_name

        if ie_cfg.served_model_name is None:
            raise ValueError("served_model_name must be set")
        if "/" in ie_cfg.served_model_name:
            raise ValueError("served_model_name must not contain '/', Harbor expects hosted_vllm/{model_name}")
        self._harbor_trial_config_template.setdefault("agent", {})[
            "model_name"
        ] = f"hosted_vllm/{ie_cfg.served_model_name}"
        self._harbor_trial_config_template["agent"].setdefault("kwargs", {})["api_base"] = f"{self.base_url}/v1"

        agent_kwargs = self._harbor_trial_config_template["agent"]["kwargs"]
        if self._validate_rollout_details and not agent_kwargs.get("collect_rollout_details", False):
            logger.warning("TITO parity validation requires collect_rollout_details=true; enabling automatically.")
            agent_kwargs["collect_rollout_details"] = True
        if not agent_kwargs.get("interleaved_thinking", False):
            logger.warning(
                "TITO requires interleaved_thinking=true to preserve assistant history; enabling automatically."
            )
            agent_kwargs["interleaved_thinking"] = True
        if agent_kwargs.get("enable_summarize", False):
            raise ValueError(
                "TITO Harbor generation does not yet support enable_summarize=true. "
                "Set harbor_trial_config.agent.kwargs.enable_summarize=false."
            )

        logger.info(
            f"TITOHarborGenerator initialized. "
            f"Agent: {self._harbor_trial_config_template.get('agent', {}).get('name')}, "
            f"Trials dir: {self._harbor_trial_config_template.get('trials_dir', 'trials')}"
        )
        self._rate_limiter = create_rate_limiter(getattr(generator_cfg, "rate_limit", None))

    async def generate(self, input_batch: GeneratorInput, disable_tqdm: bool = False) -> GeneratorOutput:
        prompts = input_batch["prompts"]
        trajectory_ids = input_batch["trajectory_ids"]
        if trajectory_ids is None:
            raise ValueError("`trajectory_ids` is required in the input batch")
        if len(prompts) != len(trajectory_ids):
            raise ValueError(
                f"Prompt count ({len(prompts)}) doesn't match trajectory_ids count ({len(trajectory_ids)})"
            )

        cache_salt = self._compute_cache_salt()
        all_outputs: List[Optional[TITOHarborTrajectoryOutput]] = [None] * len(prompts)
        progress = tqdm(
            disable=disable_tqdm,
            total=len(prompts),
            desc="Generating TITO Harbor Trajectories",
            miniters=max(1, len(prompts) // 10),
            mininterval=5,
        )
        proxy = TITOProxy(self.inference_engine_client, self.renderer)

        async def _worker(idx, prompt, trajectory_id):
            output = await self._tito_harbor_agent_loop(
                proxy=proxy,
                prompt=prompt,
                trajectory_id=trajectory_id,
                cache_salt=cache_salt,
            )
            all_outputs[idx] = output
            progress.update(1)

        try:
            async with proxy.serving():
                async with asyncio.TaskGroup() as task_group:
                    for idx, (prompt, trajectory_id) in enumerate(zip(prompts, trajectory_ids)):
                        task_group.create_task(_worker(idx, prompt, trajectory_id))
        finally:
            progress.close()

        if any(output is None for output in all_outputs):
            raise RuntimeError("TITO Harbor generation did not produce every trajectory output")
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

    async def _tito_harbor_agent_loop(
        self,
        proxy: TITOProxy,
        prompt: ConversationType,
        trajectory_id: TrajectoryID,
        cache_salt: Optional[str] = None,
    ) -> TITOHarborTrajectoryOutput:
        agent_loop_start_time = time.monotonic()
        reward = None
        results = None
        rollout_details = None
        num_turns = None
        successful_trace = None
        successful = False
        is_context_length_error = False
        is_agent_timeout_error = False

        for attempt in range(MAX_NUM_RETRIES_PER_TRIAL):
            prefix = f"Trajectory {trajectory_id} attempt {attempt + 1}/{MAX_NUM_RETRIES_PER_TRIAL}"
            results = None
            session_id = uuid4().hex
            trace = Trace()
            try:
                config = deepcopy(self._harbor_trial_config_template)
                config["task"] = {"path": prompt}
                config["agent"]["kwargs"]["session_id"] = session_id
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
                    trial = await Trial.create(TrialConfig.model_validate(config))
                    async with self._rate_limiter:
                        results = await trial.run()

                exc_type = results.exception_info.exception_type if results.exception_info else None
                is_context_length_error = exc_type == "ContextLengthExceededError"
                is_agent_timeout_error = exc_type == "AgentTimeoutError"
                if is_agent_timeout_error:
                    logger.debug(f"{prefix} hit AgentTimeoutError (no retry). Results: {results}")
                    break
                if is_context_length_error:
                    logger.debug(f"{prefix} hit ContextLengthExceededError, setting reward=0. Results: {results}")
                    reward = 0.0
                elif not results.verifier_result:
                    logger.warning(f"{prefix} failed: Exception info: {results.exception_info}. Results: {results}")
                    continue
                else:
                    verifier_result = results.verifier_result
                    if verifier_result is None or verifier_result.rewards is None:
                        raise RuntimeError("Harbor verifier result is missing rewards")
                    reward = float(verifier_result.rewards["reward"])

                agent_result = results.agent_result
                if agent_result is None:
                    raise RuntimeError("Harbor result is missing agent_result")
                rollout_details = agent_result.rollout_details
                metadata = agent_result.metadata
                if metadata is None or "n_episodes" not in metadata:
                    raise RuntimeError("Harbor agent result is missing n_episodes metadata")
                num_turns = int(metadata["n_episodes"])

                if not trace.committed_turns():
                    logger.warning(f"{prefix} failed: proxy trace contains no committed turns. Results: {results}")
                    continue
                if getattr(self, "_validate_rollout_details", True):
                    self._validate_trace_parity(trace, rollout_details)
                successful_trace = trace
                successful = True
                logger.debug(f"{prefix} successful: reward={reward}.")
                break
            except Exception as error:
                logger.warning(f"{prefix} failed: Error running trial: {error}. Results: {results}")

        if not successful:
            stop_reason = "agent_timeout" if is_agent_timeout_error else "error"
            logger.warning(f"Trajectory {trajectory_id} failed (stop_reason={stop_reason}); masking it.")
            return TITOHarborTrajectoryOutput(
                trajectory_id=trajectory_id,
                rollout_details=None,
                trace=self._empty_trace(),
                stop_reason=stop_reason,
                e2e_time=time.monotonic() - agent_loop_start_time,
            )

        if reward is None or num_turns is None or successful_trace is None:
            raise RuntimeError("Successful Harbor trajectory is missing required TITO output")
        return TITOHarborTrajectoryOutput(
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
