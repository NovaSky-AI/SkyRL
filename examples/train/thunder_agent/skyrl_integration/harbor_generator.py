"""ThunderAgent-aware Harbor generator for R2EGym training."""

import asyncio
import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger
from uuid import uuid4

from skyrl.train.generators.base import (
    GeneratorInterface,
    GeneratorInput,
    GeneratorOutput,
    TrajectoryID,
)
from skyrl.train.generators.utils import (
    encode_messages_subset,
    get_generation_prompt_ids,
    get_response_ids_and_loss_mask_from_messages,
    get_rollout_metrics,
)
from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import (
    InferenceEngineClient,
)
from skyrl.backends.skyrl_train.inference_engines.base import ConversationType
from skyrl.train.utils.rate_limiter import create_rate_limiter
from tqdm import tqdm
from omegaconf import DictConfig

import litellm
import logging

litellm.suppress_debug_info = True
litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

MAX_NUM_RETRIES_PER_TRIAL = 2


@dataclass
class HarborAgentOutput:
    response_ids: List[int]
    reward: float
    stop_reason: str
    loss_mask: List[int]
    prompt_ids: List[int]
    rollout_logprobs: Optional[List[float]]
    trajectory_id: TrajectoryID
    summarization_count: Optional[int] = None
    num_turns: Optional[int] = None


class ThunderAgentHarborGenerator(GeneratorInterface):
    def __init__(
        self,
        generator_cfg: DictConfig,
        harbor_cfg: DictConfig,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
        max_seq_len: int,
    ):
        ie_cfg = generator_cfg.inference_engine
        self.base_url = getattr(
            inference_engine_client,
            "proxy_url",
            f"http://{ie_cfg.http_endpoint_host}:{ie_cfg.http_endpoint_port}",
        )
        thunderagent_disabled = os.getenv("SKYRL_DISABLE_THUNDERAGENT", "0") == "1"
        self._supports_program_release = (
            getattr(inference_engine_client, "proxy_url", None) is not None
            and not thunderagent_disabled
        )
        self.generator_cfg = generator_cfg
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self._harbor_trial_config_template = deepcopy(harbor_cfg)

        assert ie_cfg.served_model_name is not None, "served_model_name must be set"
        assert (
            "/" not in ie_cfg.served_model_name
        ), "served_model_name must not contain '/', Harbor expects hosted_vllm/{model_name}"
        self._harbor_trial_config_template.setdefault("agent", {})[
            "model_name"
        ] = f"hosted_vllm/{ie_cfg.served_model_name}"
        self._harbor_trial_config_template["agent"].setdefault("kwargs", {})[
            "api_base"
        ] = f"{self.base_url}/v1"

        logger.info(
            f"ThunderAgentHarborGenerator initialized. "
            f"Agent: {self._harbor_trial_config_template.get('agent', {}).get('name')}, "
            f"api_base: {self._harbor_trial_config_template.get('agent', {}).get('kwargs', {}).get('api_base')}"
        )

        custom_chat_template_path = ie_cfg.engine_init_kwargs.get("chat_template", None)
        if custom_chat_template_path:
            with open(custom_chat_template_path, "r") as f:
                self.custom_chat_template_content = f.read()
            logger.info(
                f"Custom chat template loaded from: {custom_chat_template_path}"
            )
        else:
            self.custom_chat_template_content = None

        rate_limit_config = getattr(generator_cfg, "rate_limit", None)
        self._rate_limiter = create_rate_limiter(rate_limit_config)

        self._release_endpoint = f"{self.base_url}/programs/release"
        self._release_timeout_sec = max(
            1.0, float(os.getenv("THUNDERAGENT_RELEASE_TIMEOUT_SEC", "30"))
        )
        self._release_max_attempts = max(
            1, int(os.getenv("THUNDERAGENT_RELEASE_MAX_ATTEMPTS", "4"))
        )
        self._release_retry_backoff_sec = max(
            0.0, float(os.getenv("THUNDERAGENT_RELEASE_RETRY_BACKOFF_SEC", "0.5"))
        )
        self._release_max_inflight = max(
            1, int(os.getenv("THUNDERAGENT_RELEASE_MAX_INFLIGHT", "64"))
        )
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
        self._task_circuit_breaker_enabled = self._get_bool_env(
            "HARBOR_TASK_CIRCUIT_BREAKER_ENABLED", True
        )
        self._task_circuit_breaker_threshold = self._get_int_env(
            "HARBOR_TASK_CIRCUIT_BREAKER_THRESHOLD", 2, minimum=1
        )
        self._task_hard_failure_streaks: dict[str, int] = defaultdict(int)
        self._task_circuit_breaker_open: set[str] = set()
        logger.info(
            "ThunderAgentHarborGenerator hard failure handling: "
            f"types={sorted(self._hard_verifier_failure_types)} "
            f"circuit_breaker_enabled={self._task_circuit_breaker_enabled} "
            f"threshold={self._task_circuit_breaker_threshold}"
        )

    def _ensure_release_transport(self) -> tuple[httpx.AsyncClient, asyncio.Semaphore]:
        if self._release_client is None:
            self._release_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._release_timeout_sec),
                limits=httpx.Limits(
                    max_connections=max(64, self._release_max_inflight),
                    max_keepalive_connections=min(self._release_max_inflight, 64),
                ),
            )
        if self._release_semaphore is None:
            self._release_semaphore = asyncio.Semaphore(self._release_max_inflight)
        return self._release_client, self._release_semaphore

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
            logger.warning(
                f"Invalid integer for {name}={value!r}; falling back to {default}"
            )
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

    def _record_hard_verifier_failure(
        self, task_key: str, task_label: str, exc_type: str
    ) -> bool:
        if not self._task_circuit_breaker_enabled:
            return False
        self._task_hard_failure_streaks[task_key] += 1
        failure_count = self._task_hard_failure_streaks[task_key]
        if failure_count < self._task_circuit_breaker_threshold:
            return False
        if task_key not in self._task_circuit_breaker_open:
            self._task_circuit_breaker_open.add(task_key)
            logger.warning(
                f"Opening Harbor task circuit breaker for "
                f"{task_label} after {failure_count} consecutive {exc_type} failures"
            )
        return True

    @staticmethod
    def _should_collect_rollout_details(
        sampling_params: Optional[Dict[str, Any]]
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
        extra_body["program_id"] = session_id
        return config

    async def _best_effort_release_program(
        self, program_id: Optional[str]
    ) -> None:
        if not self._supports_program_release or not program_id:
            return
        client, semaphore = self._ensure_release_transport()
        async with semaphore:
            for attempt in range(1, self._release_max_attempts + 1):
                try:
                    response = await client.post(
                        self._release_endpoint, json={"program_id": program_id}
                    )
                    if response.status_code == 404:
                        return
                    if response.status_code == 200:
                        try:
                            payload = response.json()
                        except ValueError:
                            payload = None
                        if not isinstance(payload, dict) or payload.get("released") in (
                            None,
                            True,
                            False,
                        ):
                            return
                    body = response.text[:200].replace("\n", " ")
                    logger.warning(
                        f"Program release returned status {response.status_code} "
                        f"for program_id={program_id}. body={body!r}"
                    )
                    return
                except (
                    httpx.TimeoutException,
                    httpx.NetworkError,
                    httpx.ProtocolError,
                ) as exc:
                    if attempt >= self._release_max_attempts:
                        logger.warning(
                            f"Failed to release program_id={program_id} "
                            f"after {attempt} attempts ({type(exc).__name__}: {exc!r})"
                        )
                        return
                    await asyncio.sleep(
                        self._release_retry_backoff_sec * (2 ** (attempt - 1))
                    )
                except Exception as exc:
                    logger.warning(
                        f"Failed to release program_id={program_id} "
                        f"({type(exc).__name__}: {exc!r})"
                    )
                    return

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
            direct_field_value = cls._get_maybe_mapping_value(
                agent_result, direct_field_name
            )
            if direct_field_value is not None:
                return direct_field_value
        rollout_details = cls._get_maybe_mapping_value(agent_result, "rollout_details")
        if not rollout_details:
            return None
        flattened_values: List[List[int] | List[float]] = []
        for rollout_detail in rollout_details:
            cur_rollout_field = cls._get_maybe_mapping_value(
                rollout_detail, rollout_field_name
            )
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
                [cur_message], self.tokenizer, chat_template=self.custom_chat_template_content
            )
            if cur_message["role"] == "user":
                response_ids.extend(cur_token_ids)
                loss_mask.extend([0] * len(cur_token_ids))
                if rollout_logprobs is not None:
                    rollout_logprobs.extend([0.0] * len(cur_token_ids))
                continue

            if cur_message["role"] != "assistant":
                raise ValueError(
                    f"Expected message role 'user' or 'assistant', got {cur_message['role']}"
                )
            if assistant_msg_idx >= len(assistant_completion_token_ids):
                raise ValueError(
                    f"Missing completion token ids for assistant message #{assistant_msg_idx + 1}. "
                    f"Provided {len(assistant_completion_token_ids)} completion token id lists."
                )
            if cur_token_ids[: len(generation_prompt_ids)] != generation_prompt_ids:
                raise ValueError(
                    "Assistant message tokens should start with generation prompt. "
                    f"Expected {generation_prompt_ids}, got {cur_token_ids[:len(generation_prompt_ids)]}"
                )

            generated_token_ids = assistant_completion_token_ids[assistant_msg_idx]
            if self.tokenizer.eos_token_id in cur_token_ids:
                last_eos_token_index = (
                    len(cur_token_ids) - 1 - cur_token_ids[::-1].index(self.tokenizer.eos_token_id)
                )
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
                        f"Logprobs count ({len(cur_logprobs)}) does not match completion token count "
                        f"({len(generated_token_ids)}) for assistant message #{assistant_msg_idx + 1}."
                    )
                rollout_logprobs.extend([0.0] * len(generation_prompt_ids))
                rollout_logprobs.extend(cur_logprobs)
                rollout_logprobs.extend([0.0] * len(tokens_after_eos))

            assistant_msg_idx += 1

        return response_ids, loss_mask, rollout_logprobs

    @classmethod
    def _apply_sampling_params_to_trial_config(
        cls,
        config: Dict[str, Any],
        sampling_params: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if sampling_params is None:
            return config

        agent_cfg = config.setdefault("agent", {})
        agent_kwargs = agent_cfg.setdefault("kwargs", {})
        llm_call_kwargs = deepcopy(agent_kwargs.get("llm_call_kwargs", {}))
        extra_body = deepcopy(llm_call_kwargs.get("extra_body", {}))

        temperature = sampling_params.get("temperature")
        if temperature is not None:
            agent_kwargs["temperature"] = temperature

        top_p = sampling_params.get("top_p")
        if top_p is not None:
            llm_call_kwargs["top_p"] = top_p

        stop = sampling_params.get("stop")
        if stop is not None:
            llm_call_kwargs["stop"] = stop

        max_tokens = sampling_params.get("max_tokens")
        if max_tokens is None:
            max_tokens = sampling_params.get("max_completion_tokens")
        if max_tokens is None:
            max_tokens = sampling_params.get("max_generate_length")
        if max_tokens is not None:
            llm_call_kwargs["max_tokens"] = max_tokens

        if sampling_params.get("top_k") not in (None, -1):
            extra_body["top_k"] = sampling_params["top_k"]
        if sampling_params.get("min_p") not in (None, 0, 0.0):
            extra_body["min_p"] = sampling_params["min_p"]
        if sampling_params.get("repetition_penalty") not in (None, 1, 1.0):
            extra_body["repetition_penalty"] = sampling_params["repetition_penalty"]

        if extra_body:
            llm_call_kwargs["extra_body"] = extra_body
        elif "extra_body" in llm_call_kwargs:
            llm_call_kwargs.pop("extra_body")

        if llm_call_kwargs:
            agent_kwargs["llm_call_kwargs"] = llm_call_kwargs

        requested_rollout_details = cls._should_collect_rollout_details(sampling_params)
        agent_kwargs["collect_rollout_details"] = bool(
            agent_kwargs.get("collect_rollout_details", False) or requested_rollout_details
        )
        return config

    async def generate(
        self, input_batch: GeneratorInput, disable_tqdm: bool = False
    ) -> GeneratorOutput:
        prompts = input_batch["prompts"]
        trajectory_ids = input_batch["trajectory_ids"]
        sampling_params = input_batch.get("sampling_params")

        if trajectory_ids is None:
            raise ValueError("`trajectory_ids` is required in the input batch")
        if len(prompts) != len(trajectory_ids):
            raise ValueError(
                f"Prompt count ({len(prompts)}) doesn't match "
                f"trajectory_ids count ({len(trajectory_ids)})"
            )

        all_outputs: List[HarborAgentOutput] = [None] * len(prompts)  # type: ignore[list-item]
        progress = None
        if not disable_tqdm:
            progress = tqdm(
                total=len(prompts),
                desc="Generating Trajectories",
                miniters=max(1, len(prompts) // 10),
                mininterval=5,
            )

        async def _worker(idx, prompt, trajectory_id):
            try:
                result = await self.harbor_agent_loop(
                    prompt=prompt,
                    trajectory_id=trajectory_id,
                    sampling_params=sampling_params,
                )
            except Exception as e:
                logger.warning(
                    f"Trajectory {trajectory_id} failed during Harbor postprocessing: {e}"
                )
                result = HarborAgentOutput(
                    response_ids=[0],
                    reward=0,
                    stop_reason="error",
                    loss_mask=[0],
                    prompt_ids=[0],
                    rollout_logprobs=[0.0]
                    if self._should_collect_rollout_details(sampling_params)
                    else None,
                    trajectory_id=trajectory_id,
                )
            all_outputs[idx] = result
            if progress is not None:
                progress.update(1)

        try:
            async with asyncio.TaskGroup() as tg:
                for idx, (prompt, trajectory_id) in enumerate(zip(prompts, trajectory_ids)):
                    tg.create_task(_worker(idx, prompt, trajectory_id))
        finally:
            if progress is not None:
                progress.close()
        all_outputs, rollout_metrics = self._mask_failed_instances_and_compute_metrics(
            all_outputs
        )
        rollout_logprobs = (
            [output.rollout_logprobs for output in all_outputs]
            if all_outputs and all_outputs[0].rollout_logprobs is not None
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
        all_outputs: List[HarborAgentOutput],
    ) -> tuple[List[HarborAgentOutput], dict]:
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
        successful_outputs: List[HarborAgentOutput] = []
        for output in all_outputs:
            if output.trajectory_id.instance_id in masked_instance_ids:
                output.response_ids = [0]
                output.stop_reason = "error"
                output.loss_mask = [0]
                output.prompt_ids = [0]
                output.rollout_logprobs = (
                    [0.0] if output.rollout_logprobs is not None else None
                )
                output.reward = 0
            else:
                successful_outputs.append(output)

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
            rollout_metrics["generate/avg_num_turns"] = sum(
                output.num_turns for output in successful_outputs
            ) / len(successful_outputs)
        else:
            rollout_metrics = {}

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
        sampling_params: Optional[Dict[str, Any]] = None,
    ) -> HarborAgentOutput:
        reward = None
        chat_history = None
        summarization_count = None
        num_turns = None
        successful = False
        is_context_length_error = False
        is_agent_timeout_error = False
        collect_rollout_details = self._should_collect_rollout_details(sampling_params)
        task_key = self._task_key(prompt)
        task_label = self._task_label(prompt)
        if task_key in self._task_circuit_breaker_open:
            logger.warning(
                f"Skipping Harbor trial for {task_label}: task circuit breaker already open"
            )
            return HarborAgentOutput(
                response_ids=[0],
                reward=0,
                stop_reason="error",
                loss_mask=[0],
                prompt_ids=[0],
                rollout_logprobs=[0.0] if collect_rollout_details else None,
                trajectory_id=trajectory_id,
            )

        for i in range(MAX_NUM_RETRIES_PER_TRIAL):
            prefix = f"Trajectory {trajectory_id} attempt {i+1}/{MAX_NUM_RETRIES_PER_TRIAL}"
            results = None
            trial_session_id = None
            try:
                config = deepcopy(self._harbor_trial_config_template)
                config["task"] = {"path": prompt}
                config = self._apply_sampling_params_to_trial_config(config, sampling_params)
                trial_session_id = uuid4().hex
                config = self._attach_trial_routing_ids(config, session_id=trial_session_id)
                collect_rollout_details = bool(
                    config["agent"]["kwargs"].get("collect_rollout_details", False)
                )

                from harbor.trial.trial import Trial
                from harbor.models.trial.config import TrialConfig

                trial_config = TrialConfig.model_validate(config)
                trial = Trial(trial_config)

                async with self._rate_limiter:
                    results = await trial.run()

                exc_type = (
                    results.exception_info.exception_type
                    if results.exception_info
                    else None
                )
                is_context_length_error = exc_type == "ContextLengthExceededError"
                is_agent_timeout_error = exc_type == "AgentTimeoutError"
                is_hard_verifier_failure = self._is_hard_verifier_failure(exc_type)

                if is_agent_timeout_error:
                    self._record_non_hard_outcome(task_key)
                    logger.debug(f"{prefix} hit AgentTimeoutError (no retry).")
                    break
                elif is_context_length_error:
                    self._record_non_hard_outcome(task_key)
                    logger.debug(
                        f"{prefix} hit ContextLengthExceededError, will train with reward=0."
                    )
                    reward = 0
                elif is_hard_verifier_failure:
                    breaker_open = self._record_hard_verifier_failure(
                        task_key, task_label, exc_type
                    )
                    logger.warning(
                        f"{prefix} hit hard verifier failure {exc_type} (no retry)."
                    )
                    if breaker_open:
                        logger.warning(
                            f"{prefix} stopping early because circuit breaker is open for {task_label}"
                        )
                    break
                elif not results.verifier_result:
                    self._record_non_hard_outcome(task_key)
                    logger.warning(
                        f"{prefix} failed: Exception info: {results.exception_info}."
                    )
                    continue
                else:
                    self._record_non_hard_outcome(task_key)
                    reward = results.verifier_result.rewards["reward"]

                chat_history = results.agent_result.metadata["all_messages"]
                summarization_count = results.agent_result.metadata["summarization_count"]
                num_turns = results.agent_result.metadata["n_episodes"]
                if len(chat_history) > 1 and chat_history[0]["role"] == "user":
                    successful = True
                    logger.debug(f"{prefix} successful: reward={reward}.")
                    break
                else:
                    logger.warning(
                        f"{prefix} failed: No user message in chat_history."
                    )
            except Exception as e:
                self._record_non_hard_outcome(task_key)
                logger.warning(f"{prefix} failed: Error running trial: {e}.")
                continue
            finally:
                await self._best_effort_release_program(trial_session_id)

        if not successful:
            stop_reason = "agent_timeout" if is_agent_timeout_error else "error"
            logger.warning(
                f"Trajectory {trajectory_id} failed (stop_reason={stop_reason}), will set loss mask to [0]."
            )
            return HarborAgentOutput(
                response_ids=[0],
                reward=0,
                stop_reason=stop_reason,
                loss_mask=[0],
                prompt_ids=[0],
                rollout_logprobs=[0.0] if collect_rollout_details else None,
                trajectory_id=trajectory_id,
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
        if collect_rollout_details and (
            assistant_logprobs is None or assistant_completion_token_ids is None
        ):
            raise ValueError(
                f"Harbor trial for trajectory {trajectory_id} did not return "
                "assistant logprobs/token ids despite collect_rollout_details=True."
            )
        if assistant_completion_token_ids is not None:
            response_ids, loss_mask, rollout_logprobs = (
                self._get_response_ids_and_loss_mask_from_harbor_rollout(
                    response_messages,
                    assistant_completion_token_ids=assistant_completion_token_ids,
                    assistant_logprobs=assistant_logprobs,
                )
            )
        else:
            response_ids, loss_mask, rollout_logprobs = get_response_ids_and_loss_mask_from_messages(
                response_messages,
                self.tokenizer,
                assistant_logprobs,
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

        return HarborAgentOutput(
            response_ids=response_ids,
            reward=reward,
            stop_reason=stop_reason,
            loss_mask=loss_mask,
            prompt_ids=prompt_ids,
            rollout_logprobs=rollout_logprobs,
            trajectory_id=trajectory_id,
            summarization_count=summarization_count,
            num_turns=num_turns,
        )
