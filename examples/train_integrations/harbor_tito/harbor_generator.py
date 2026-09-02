"""Harbor generator backed by SkyRL's token-in/token-out proxy."""

import asyncio
import json
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from harbor.models.agent.rollout_detail import RolloutDetail
from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial
from loguru import logger
from omegaconf import DictConfig
from pydantic import TypeAdapter
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
from skyrl.train.generators.tito.proxy import TITOProxy, TITOProxyConfig
from skyrl.train.generators.tito.renderer import PrimeRendererAdapter
from skyrl.train.utils.rate_limiter import create_rate_limiter


@dataclass
class TITOHarborTrajectoryOutput(HarborTrajectoryOutput):
    """One Harbor outcome whose trace is authoritative for training."""

    trace: Optional[Trace] = None
    summarization_count: int = 0


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
        renderer_config = None
        if raw_renderer_config := getattr(generator_cfg, "tito_renderer_config", None):
            from renderers import RendererConfig

            renderer_config = TypeAdapter(RendererConfig).validate_python(raw_renderer_config)
        self.renderer = PrimeRendererAdapter(
            tokenizer,
            config=renderer_config,
            chat_template_kwargs=getattr(generator_cfg, "chat_template_kwargs", None),
        )
        self._validate_rollout_details = getattr(generator_cfg, "tito_validate_rollout_details", True)
        trace_log_dir = getattr(generator_cfg, "tito_trace_log_dir", None)
        self._trace_log_dir = Path(trace_log_dir).expanduser() if trace_log_dir else None
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
        proxy = TITOProxy(
            self.inference_engine_client,
            self.renderer,
            config=TITOProxyConfig(
                max_model_len=self.max_seq_len,
                default_max_tokens=self.generator_cfg.sampling_params.max_generate_length,
            ),
        )

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
        summarization_count = 0
        successful_trace = None
        segment_transition_ids: List[List[int]] = []
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
                summarization_count = int(metadata.get("summarization_count", 0))

                if not trace.committed_turns():
                    logger.warning(f"{prefix} failed: proxy trace contains no committed turns. Results: {results}")
                    continue
                if getattr(self, "_validate_rollout_details", True):
                    segment_transition_ids = self._validate_trace_parity(trace, rollout_details)
                successful_trace = trace
                self._write_trace_log(
                    trace,
                    trajectory_id=trajectory_id,
                    session_id=session_id,
                    reward=reward,
                    summarization_count=summarization_count,
                    rollout_details=rollout_details,
                    segment_transition_ids=segment_transition_ids,
                )
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
            summarization_count=summarization_count,
            stop_reason="context_length" if is_context_length_error else "complete",
            e2e_time=time.monotonic() - agent_loop_start_time,
        )

    @staticmethod
    def _empty_trace() -> Trace:
        trace = Trace()
        trace.seal()
        return trace

    @staticmethod
    def _validate_trace_parity(
        trace: Trace,
        rollout_details: Optional[List[RolloutDetail]],
    ) -> List[List[int]]:
        if not rollout_details:
            raise ValueError("Harbor did not return rollout details for TITO parity validation")
        transitions = list(trace.transitions())
        unmatched = set(range(len(transitions)))
        segment_transition_ids: List[List[int]] = []

        for segment_index, detail in enumerate(rollout_details):
            prompts = detail.get("prompt_token_ids", [])
            completions = detail.get("completion_token_ids", [])
            logprobs = detail.get("logprobs", [])
            if not (len(prompts) == len(completions) == len(logprobs)):
                raise ValueError(
                    f"Malformed Harbor rollout segment {segment_index}: "
                    f"prompts={len(prompts)}, completions={len(completions)}, logprobs={len(logprobs)}"
                )

            matched_segment: List[int] = []
            for call_index, (prompt_ids, completion_ids, call_logprobs) in enumerate(
                zip(prompts, completions, logprobs)
            ):
                match = next(
                    (
                        transition_id
                        for transition_id in unmatched
                        if list(transitions[transition_id].prompt_token_ids) == prompt_ids
                        and list(transitions[transition_id].completion_ids) == completion_ids
                        and list(transitions[transition_id].completion_logprobs) == call_logprobs
                    ),
                    None,
                )
                if match is None:
                    raise ValueError(
                        f"Harbor rollout segment {segment_index} call {call_index} "
                        "does not match any committed TITO transition"
                    )
                unmatched.remove(match)
                matched_segment.append(match)
            segment_transition_ids.append(matched_segment)

        unexpected = [
            transition_id
            for transition_id in sorted(unmatched)
            if transitions[transition_id].stop_reason not in ("length", "max_tokens")
        ]
        if unexpected:
            raise ValueError(
                f"Non-truncated TITO transitions missing from Harbor rollout details: {unexpected}"
            )
        if unmatched:
            logger.info(
                "Allowing truncated TITO transitions omitted from Harbor rollout details: {}",
                sorted(unmatched),
            )
        return segment_transition_ids

    def _write_trace_log(
        self,
        trace: Trace,
        *,
        trajectory_id: TrajectoryID,
        session_id: str,
        reward: float,
        summarization_count: int,
        rollout_details: Optional[List[RolloutDetail]],
        segment_transition_ids: List[List[int]],
    ) -> None:
        trace_log_dir = getattr(self, "_trace_log_dir", None)
        if trace_log_dir is None:
            return
        trace_log_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "trajectory_id": trajectory_id.to_string(),
            "session_id": session_id,
            "reward": reward,
            "summarization_count": summarization_count,
            "harbor_rollout_segments": [
                {
                    "prompt_lengths": [len(ids) for ids in segment.get("prompt_token_ids", [])],
                    "completion_lengths": [len(ids) for ids in segment.get("completion_token_ids", [])],
                    "logprob_lengths": [len(values) for values in segment.get("logprobs", [])],
                }
                for segment in rollout_details or []
            ],
            "harbor_segment_transition_ids": segment_transition_ids,
            "trace": trace.to_debug_dict(),
        }
        path = trace_log_dir / f"{trajectory_id.to_string()}-{session_id}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
