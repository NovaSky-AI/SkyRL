"""ThunderAgent adapter for the shared Harbor generator."""

import asyncio
import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, cast
from uuid import uuid4

import httpx
from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from examples.train_integrations.harbor.harbor_generator import (
    MAX_NUM_RETRIES_PER_TRIAL,
    HarborAgentOutput,
    HarborGenerator,
)
from skyrl.backends.skyrl_train.inference_engines.base import ConversationType
from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import (
    InferenceEngineClient,
)
from skyrl.train.generators.base import GeneratorInput, GeneratorOutput, TrajectoryID
from skyrl.train.generators.utils import (
    encode_messages_subset,
    get_generation_prompt_ids,
    get_response_ids_and_loss_mask_from_messages,
)


@dataclass
class ThunderAgentHarborOutput(HarborAgentOutput):
    rollout_logprobs: Optional[List[float]] = None
    hard_verifier_failure: bool = False
    circuit_breaker_skipped: bool = False


class ThunderAgentHarborGenerator(HarborGenerator):
    """Run Harbor trials through ThunderAgent while reusing shared Harbor training semantics."""

    def __init__(
        self,
        generator_cfg: DictConfig,
        harbor_cfg: DictConfig,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
        max_seq_len: int,
    ):
        super().__init__(
            generator_cfg=generator_cfg,
            harbor_cfg=harbor_cfg,
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
        )

        proxy_url = getattr(inference_engine_client, "proxy_url", None)
        if proxy_url:
            self.base_url = proxy_url
            self._harbor_trial_config_template["agent"].setdefault("kwargs", {})["api_base"] = f"{self.base_url}/v1"

        thunderagent_disabled = os.getenv("SKYRL_DISABLE_THUNDERAGENT", "0") == "1"
        self._supports_program_release = proxy_url is not None and not thunderagent_disabled
        self._release_program_fn: Optional[Callable[[str], Any]] = getattr(
            inference_engine_client, "release_program", None
        )
        self._release_endpoint = f"{self.base_url}/programs/release"
        self._release_timeout_sec = max(1.0, float(os.getenv("THUNDERAGENT_RELEASE_TIMEOUT_SEC", "30")))
        self._release_max_attempts = max(1, int(os.getenv("THUNDERAGENT_RELEASE_MAX_ATTEMPTS", "4")))
        self._release_retry_backoff_sec = max(0.0, float(os.getenv("THUNDERAGENT_RELEASE_RETRY_BACKOFF_SEC", "0.5")))
        self._release_max_inflight = max(1, int(os.getenv("THUNDERAGENT_RELEASE_MAX_INFLIGHT", "64")))
        self._release_client: Optional[httpx.AsyncClient] = None
        self._release_semaphore: Optional[asyncio.Semaphore] = None

        self._hard_verifier_failure_types = {
            item.strip()
            for item in os.getenv(
                "HARBOR_HARD_FAILURE_EXCEPTION_TYPES",
                "RewardFileNotFoundError,VerifierTimeoutError",
            ).split(",")
            if item.strip()
        }
        self._task_circuit_breaker_enabled = self._get_bool_env("HARBOR_TASK_CIRCUIT_BREAKER_ENABLED", True)
        self._task_circuit_breaker_threshold = self._get_int_env("HARBOR_TASK_CIRCUIT_BREAKER_THRESHOLD", 2, minimum=1)
        self._task_hard_failure_streaks: dict[str, int] = defaultdict(int)
        self._task_circuit_breaker_open: set[str] = set()

        logger.info(
            "ThunderAgentHarborGenerator initialized. "
            f"api_base={self._harbor_trial_config_template.get('agent', {}).get('kwargs', {}).get('api_base')} "
            f"hard_failure_types={sorted(self._hard_verifier_failure_types)} "
            f"circuit_breaker_enabled={self._task_circuit_breaker_enabled} "
            f"threshold={self._task_circuit_breaker_threshold}"
        )

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """Parent generate(), plus Harbor rollout logprobs when training requests them."""
        prompts = input_batch["prompts"]
        trajectory_ids = input_batch["trajectory_ids"]
        sampling_params = input_batch.get("sampling_params")

        if trajectory_ids is None:
            raise ValueError("`trajectory_ids` is required in the input batch")
        if len(prompts) != len(trajectory_ids):
            raise ValueError(
                f"Prompt count ({len(prompts)}) doesn't match " f"trajectory_ids count ({len(trajectory_ids)})"
            )

        all_outputs: List[ThunderAgentHarborOutput] = [None] * len(prompts)  # type: ignore[list-item]
        progress = tqdm(
            total=len(prompts),
            desc="Generating Trajectories",
            miniters=max(1, len(prompts) // 10),
            mininterval=5,
        )

        async def _worker(idx, prompt, trajectory_id):
            result = await self.harbor_agent_loop(
                prompt=prompt,
                trajectory_id=trajectory_id,
                sampling_params=sampling_params,
            )
            all_outputs[idx] = result
            progress.update(1)

        try:
            async with asyncio.TaskGroup() as tg:
                for idx, (prompt, trajectory_id) in enumerate(zip(prompts, trajectory_ids)):
                    tg.create_task(_worker(idx, prompt, trajectory_id))
        finally:
            progress.close()

        all_outputs, rollout_metrics = self._mask_failed_instances_and_compute_metrics(all_outputs)
        rollout_metrics["generate/num_hard_verifier_failures"] = sum(
            1 for output in all_outputs if output.hard_verifier_failure
        )
        rollout_metrics["generate/num_circuit_breaker_skipped"] = sum(
            1 for output in all_outputs if output.circuit_breaker_skipped
        )

        has_logprobs = [output.rollout_logprobs is not None for output in all_outputs]
        if any(has_logprobs) and not all(has_logprobs):
            raise ValueError("Harbor outputs mixed null and non-null rollout_logprobs")
        rollout_logprobs = (
            [cast(List[float], output.rollout_logprobs) for output in all_outputs]
            if all_outputs and all(has_logprobs)
            else None
        )

        generator_output: GeneratorOutput = {
            "prompt_token_ids": [output.prompt_ids for output in all_outputs],
            "response_ids": [output.response_ids for output in all_outputs],
            "rewards": [output.reward for output in all_outputs],
            "loss_masks": [output.loss_mask for output in all_outputs],
            "stop_reasons": [output.stop_reason for output in all_outputs],
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": rollout_logprobs,
        }
        return generator_output

    @staticmethod
    def _mask_failed_instances_and_compute_metrics(
        all_outputs: List[ThunderAgentHarborOutput],
    ) -> tuple[List[ThunderAgentHarborOutput], dict]:
        masked_outputs, rollout_metrics = HarborGenerator._mask_failed_instances_and_compute_metrics(all_outputs)
        for output in masked_outputs:
            if output.stop_reason == "error" and output.rollout_logprobs is not None:
                output.rollout_logprobs = [0.0]
        return cast(List[ThunderAgentHarborOutput], masked_outputs), rollout_metrics

    @staticmethod
    def _get_bool_env(name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() not in {"0", "false", "no", "off"}

    @staticmethod
    def _get_int_env(name: str, default: int, minimum: int = 0) -> int:
        value = os.getenv(name)
        if value is None:
            return default
        try:
            return max(minimum, int(value))
        except ValueError:
            logger.warning(f"Invalid integer for {name}={value!r}; falling back to {default}")
            return default

    @staticmethod
    def _task_key(prompt: str) -> str:
        return str(prompt)

    @staticmethod
    def _task_label(prompt: str) -> str:
        prompt_str = str(prompt)
        return os.path.basename(prompt_str.rstrip("/")) or prompt_str

    def _is_hard_verifier_failure(self, exc_type: Optional[str]) -> bool:
        return exc_type in self._hard_verifier_failure_types

    def _record_non_hard_outcome(self, task_key: str) -> None:
        if task_key in self._task_circuit_breaker_open:
            return
        self._task_hard_failure_streaks.pop(task_key, None)

    def _record_hard_verifier_failure(self, task_key: str, task_label: str, exc_type: str) -> bool:
        if not self._task_circuit_breaker_enabled:
            return False
        self._task_hard_failure_streaks[task_key] += 1
        failure_count = self._task_hard_failure_streaks[task_key]
        if failure_count < self._task_circuit_breaker_threshold:
            return False
        if task_key not in self._task_circuit_breaker_open:
            self._task_circuit_breaker_open.add(task_key)
            logger.warning(
                f"Opening Harbor task circuit breaker for {task_label} after "
                f"{failure_count} consecutive {exc_type} failures"
            )
        return True

    @staticmethod
    def _should_collect_rollout_details(
        sampling_params: Optional[Dict[str, Any]],
    ) -> bool:
        if sampling_params is None:
            return False
        return sampling_params.get("logprobs") not in (None, False, 0)

    @staticmethod
    def _attach_trial_routing_ids(config: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        agent_kwargs = config.setdefault("agent", {}).setdefault("kwargs", {})
        agent_kwargs["session_id"] = session_id
        llm_call_kwargs = agent_kwargs.setdefault("llm_call_kwargs", {})
        extra_body = llm_call_kwargs.setdefault("extra_body", {})
        if not isinstance(extra_body, dict):
            raise TypeError("harbor_trial_config.agent.kwargs.llm_call_kwargs.extra_body must be a mapping")
        extra_body["program_id"] = session_id
        return config

    @classmethod
    def _apply_rollout_detail_request_to_trial_config(
        cls,
        config: Dict[str, Any],
        sampling_params: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if cls._should_collect_rollout_details(sampling_params):
            agent_kwargs = config.setdefault("agent", {}).setdefault("kwargs", {})
            agent_kwargs["collect_rollout_details"] = True
        return config

    def _ensure_release_client(self) -> httpx.AsyncClient:
        if self._release_client is None:
            self._release_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._release_timeout_sec),
                limits=httpx.Limits(
                    max_connections=max(64, self._release_max_inflight),
                    max_keepalive_connections=min(self._release_max_inflight, 64),
                ),
            )
        return self._release_client

    def _ensure_release_semaphore(self) -> asyncio.Semaphore:
        if self._release_semaphore is None:
            self._release_semaphore = asyncio.Semaphore(self._release_max_inflight)
        return self._release_semaphore

    async def _release_program_once(self, program_id: str) -> None:
        if self._release_program_fn is not None:
            await self._release_program_fn(program_id)
            return

        client = self._ensure_release_client()
        response = await client.post(self._release_endpoint, json={"program_id": program_id})
        if response.status_code in (200, 404):
            return
        body = response.text[:200].replace("\n", " ")
        raise RuntimeError(
            f"Program release returned status {response.status_code} " f"for program_id={program_id}. body={body!r}"
        )

    async def _release_program(self, program_id: Optional[str]) -> None:
        if not self._supports_program_release or not program_id:
            return
        semaphore = self._ensure_release_semaphore()
        async with semaphore:
            for attempt in range(1, self._release_max_attempts + 1):
                try:
                    await asyncio.wait_for(
                        self._release_program_once(program_id),
                        timeout=self._release_timeout_sec,
                    )
                    return
                except Exception as exc:
                    if attempt >= self._release_max_attempts:
                        logger.warning(
                            f"Failed to release program_id={program_id} after "
                            f"{attempt} attempts ({type(exc).__name__}: {exc!r})"
                        )
                        return
                    await asyncio.sleep(self._release_retry_backoff_sec * (2 ** (attempt - 1)))

    @staticmethod
    def _get_maybe_mapping_value(obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @classmethod
    def _extract_assistant_rollout_field(
        cls,
        agent_result: Any,
        rollout_field_name: str,
        direct_field_name: Optional[str] = None,
    ) -> Optional[List[List[int] | List[float]]]:
        if direct_field_name is not None:
            direct_field_value = cls._get_maybe_mapping_value(agent_result, direct_field_name)
            if direct_field_value is not None:
                return direct_field_value

        rollout_details = cls._get_maybe_mapping_value(agent_result, "rollout_details")
        if not rollout_details:
            return None

        flattened_values: List[List[int] | List[float]] = []
        for rollout_detail in rollout_details:
            cur_rollout_field = cls._get_maybe_mapping_value(rollout_detail, rollout_field_name)
            if not cur_rollout_field:
                continue
            if (
                isinstance(cur_rollout_field, list)
                and cur_rollout_field
                and all(isinstance(x, list) for x in cur_rollout_field)
            ):
                flattened_values.extend(cur_rollout_field)
            else:
                flattened_values.append(cur_rollout_field)
        return flattened_values or None

    def _get_response_ids_and_loss_mask_from_harbor_rollout(
        self,
        messages: ConversationType,
        assistant_completion_token_ids: List[List[int]],
        assistant_logprobs: Optional[List[List[float]]],
    ) -> tuple[List[int], List[int], Optional[List[float]]]:
        generation_prompt_ids = get_generation_prompt_ids(
            self.tokenizer, chat_template=self.custom_chat_template_content
        )
        response_ids: List[int] = []
        loss_mask: List[int] = []
        rollout_logprobs = None if assistant_logprobs is None else []
        assistant_msg_idx = 0

        for cur_message in messages:
            cur_token_ids = encode_messages_subset(
                [cur_message],
                self.tokenizer,
                chat_template=self.custom_chat_template_content,
            )
            if cur_message["role"] == "user":
                response_ids.extend(cur_token_ids)
                loss_mask.extend([0] * len(cur_token_ids))
                if rollout_logprobs is not None:
                    rollout_logprobs.extend([0.0] * len(cur_token_ids))
                continue

            if cur_message["role"] != "assistant":
                raise ValueError(f"Expected message role 'user' or 'assistant', got {cur_message['role']}")
            if assistant_msg_idx >= len(assistant_completion_token_ids):
                raise ValueError(
                    f"Missing completion token ids for assistant message #{assistant_msg_idx + 1}. "
                    f"Provided {len(assistant_completion_token_ids)} completion token id lists."
                )
            if cur_token_ids[: len(generation_prompt_ids)] != generation_prompt_ids:
                raise ValueError(
                    "Assistant message tokens should start with generation prompt. "
                    f"Expected {generation_prompt_ids}, got {cur_token_ids[: len(generation_prompt_ids)]}"
                )

            generated_token_ids = assistant_completion_token_ids[assistant_msg_idx]
            if self.tokenizer.eos_token_id in cur_token_ids:
                last_eos_token_index = len(cur_token_ids) - 1 - cur_token_ids[::-1].index(self.tokenizer.eos_token_id)
                tokens_after_eos = cur_token_ids[last_eos_token_index + 1 :]
            else:
                tokens_after_eos = []

            response_ids.extend(generation_prompt_ids)
            response_ids.extend(generated_token_ids)
            response_ids.extend(tokens_after_eos)

            loss_mask.extend([0] * len(generation_prompt_ids))
            loss_mask.extend([1] * len(generated_token_ids))
            loss_mask.extend([0] * len(tokens_after_eos))

            if rollout_logprobs is not None:
                if assistant_msg_idx >= len(assistant_logprobs):
                    raise ValueError(
                        f"Missing logprobs for assistant message #{assistant_msg_idx + 1}. "
                        f"Provided {len(assistant_logprobs)} logprob lists."
                    )
                cur_logprobs = assistant_logprobs[assistant_msg_idx]
                if len(cur_logprobs) != len(generated_token_ids):
                    raise ValueError(
                        f"Logprobs count ({len(cur_logprobs)}) does not match "
                        f"completion token count ({len(generated_token_ids)}) "
                        f"for assistant message #{assistant_msg_idx + 1}."
                    )
                rollout_logprobs.extend([0.0] * len(generation_prompt_ids))
                rollout_logprobs.extend(cur_logprobs)
                rollout_logprobs.extend([0.0] * len(tokens_after_eos))

            assistant_msg_idx += 1

        return response_ids, loss_mask, rollout_logprobs

    @staticmethod
    def _failed_output(
        trajectory_id: TrajectoryID,
        stop_reason: str,
        collect_rollout_details: bool,
        hard_verifier_failure: bool = False,
        circuit_breaker_skipped: bool = False,
    ) -> ThunderAgentHarborOutput:
        return ThunderAgentHarborOutput(
            response_ids=[0],
            reward=0,
            stop_reason=stop_reason,
            loss_mask=[0],
            prompt_ids=[0],
            trajectory_id=trajectory_id,
            rollout_logprobs=[0.0] if collect_rollout_details else None,
            hard_verifier_failure=hard_verifier_failure,
            circuit_breaker_skipped=circuit_breaker_skipped,
        )

    async def harbor_agent_loop(
        self,
        prompt: ConversationType,
        trajectory_id: TrajectoryID,
        sampling_params: Optional[Dict[str, Any]] = None,
    ) -> ThunderAgentHarborOutput:
        reward = None
        chat_history = None
        summarization_count = None
        num_turns = None
        successful = False
        is_context_length_error = False
        is_agent_timeout_error = False
        hard_verifier_failure = False
        collect_rollout_details = self._should_collect_rollout_details(sampling_params)
        task_key = self._task_key(prompt)
        task_label = self._task_label(prompt)

        if task_key in self._task_circuit_breaker_open:
            logger.warning(f"Skipping Harbor trial for {task_label}: task circuit breaker already open")
            return self._failed_output(
                trajectory_id,
                stop_reason="error",
                collect_rollout_details=collect_rollout_details,
                circuit_breaker_skipped=True,
            )

        for i in range(MAX_NUM_RETRIES_PER_TRIAL):
            prefix = f"Trajectory {trajectory_id} attempt {i + 1}/{MAX_NUM_RETRIES_PER_TRIAL}"
            results = None
            trial_session_id = None
            try:
                config = deepcopy(self._harbor_trial_config_template)
                config["task"] = {"path": prompt}
                config = self._apply_rollout_detail_request_to_trial_config(config, sampling_params)
                trial_session_id = uuid4().hex
                config = self._attach_trial_routing_ids(config, session_id=trial_session_id)
                collect_rollout_details = bool(config["agent"]["kwargs"].get("collect_rollout_details", False))

                trial_config = TrialConfig.model_validate(config)
                trial = Trial(trial_config)

                async with self._rate_limiter:
                    results = await trial.run()

                exc_type = results.exception_info.exception_type if results.exception_info else None
                is_context_length_error = exc_type == "ContextLengthExceededError"
                is_agent_timeout_error = exc_type == "AgentTimeoutError"

                if is_agent_timeout_error:
                    self._record_non_hard_outcome(task_key)
                    logger.debug(f"{prefix} hit AgentTimeoutError (no retry). Results: {results}")
                    break
                elif is_context_length_error:
                    self._record_non_hard_outcome(task_key)
                    logger.debug(
                        f"{prefix} hit ContextLengthExceededError, will train with reward=0. " f"Results: {results}"
                    )
                    reward = 0
                elif self._is_hard_verifier_failure(exc_type):
                    hard_verifier_failure = True
                    breaker_open = self._record_hard_verifier_failure(task_key, task_label, cast(str, exc_type))
                    logger.warning(f"{prefix} hit hard verifier failure {exc_type} (no retry). " f"Results: {results}")
                    if breaker_open:
                        logger.warning(f"{prefix} opened circuit breaker for {task_label}")
                    break
                elif not results.verifier_result:
                    self._record_non_hard_outcome(task_key)
                    logger.warning(f"{prefix} failed: Exception info: {results.exception_info}. " f"Results: {results}")
                    continue
                else:
                    self._record_non_hard_outcome(task_key)
                    reward = results.verifier_result.rewards["reward"]

                chat_history = results.agent_result.metadata["all_messages"]
                summarization_count = results.agent_result.metadata["summarization_count"]
                num_turns = results.agent_result.metadata["n_episodes"]
                if len(chat_history) > 1 and chat_history[0]["role"] == "user":
                    successful = True
                    logger.debug(f"{prefix} successful: reward={reward}. Results: {results}")
                    break

                logger.warning(
                    f"{prefix} failed: Did not return a chat history with a user message. "
                    f"chat_history: {chat_history}\nResults: {results}"
                )
            except Exception as exc:
                self._record_non_hard_outcome(task_key)
                logger.warning(f"{prefix} failed: Error running trial: {exc}. Results: {results}")
                continue
            finally:
                await self._release_program(trial_session_id)

        if not successful:
            stop_reason = "agent_timeout" if is_agent_timeout_error else "error"
            logger.warning(
                f"Trajectory {trajectory_id} failed (stop_reason={stop_reason}), " "will set loss mask to [0]."
            )
            return self._failed_output(
                trajectory_id,
                stop_reason=stop_reason,
                collect_rollout_details=collect_rollout_details,
                hard_verifier_failure=hard_verifier_failure,
            )

        assert chat_history[0]["role"] == "user", "First message should be user"
        prompt = [chat_history[0]]
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=False,
            tokenize=True,
            chat_template=self.custom_chat_template_content,
        )
        initial_prompt_length = len(prompt_ids)

        response_messages = chat_history[1:]
        assistant_logprobs = self._extract_assistant_rollout_field(
            results.agent_result, "logprobs", direct_field_name="output_logprobs"
        )
        assistant_completion_token_ids = self._extract_assistant_rollout_field(
            results.agent_result, "completion_token_ids"
        )
        if collect_rollout_details and (assistant_logprobs is None or assistant_completion_token_ids is None):
            raise ValueError(
                f"Harbor trial for trajectory {trajectory_id} did not return "
                "assistant logprobs/token ids despite collect_rollout_details=True."
            )

        if assistant_completion_token_ids is not None:
            response_ids, loss_mask, rollout_logprobs = self._get_response_ids_and_loss_mask_from_harbor_rollout(
                response_messages,
                assistant_completion_token_ids=cast(List[List[int]], assistant_completion_token_ids),
                assistant_logprobs=cast(Optional[List[List[float]]], assistant_logprobs),
            )
        else:
            response_ids, loss_mask, rollout_logprobs = get_response_ids_and_loss_mask_from_messages(
                response_messages,
                self.tokenizer,
                cast(Optional[List[List[float]]], assistant_logprobs),
                chat_template=self.custom_chat_template_content,
            )

        max_response_tokens = max(0, self.max_seq_len - initial_prompt_length)
        if is_context_length_error or len(response_ids) > max_response_tokens:
            stop_reason = "context_length"
        else:
            stop_reason = "complete"

        if self.generator_cfg.apply_overlong_filtering and stop_reason == "context_length":
            loss_mask = [0] * len(loss_mask)

        response_ids = response_ids[:max_response_tokens]
        loss_mask = loss_mask[:max_response_tokens]
        if rollout_logprobs is not None:
            rollout_logprobs = rollout_logprobs[:max_response_tokens]

        return ThunderAgentHarborOutput(
            response_ids=response_ids,
            reward=reward,
            stop_reason=stop_reason,
            loss_mask=loss_mask,
            prompt_ids=prompt_ids,
            trajectory_id=trajectory_id,
            summarization_count=summarization_count,
            num_turns=num_turns,
            rollout_logprobs=rollout_logprobs,
        )
