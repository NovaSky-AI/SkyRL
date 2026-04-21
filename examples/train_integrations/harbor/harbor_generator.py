import asyncio
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional
from loguru import logger
from uuid import uuid4
from skyrl.train.generators.base import GeneratorInterface, GeneratorInput, GeneratorOutput, TrajectoryID
from skyrl.train.generators.utils import get_rollout_metrics, get_response_ids_and_loss_mask_from_messages
from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl.backends.skyrl_train.inference_engines.base import ConversationType
from skyrl.train.utils.rate_limiter import create_rate_limiter
from tqdm import tqdm
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


class ChatHistoryExtractor:
    """Extracts a (chat_history, summarization_count, num_turns) tuple from Harbor trial results.

    Supports two extraction strategies, tried in order:
    1. all_messages agents (terminus-2, terminus-1, terminus): metadata["all_messages"]
    2. Trajectory-based agents (mini-swe-agent, swe-agent, openhands):
       trajectory.json converted to user/assistant messages
    """

    # Agents that write trajectory.json (ATIF format) instead of metadata["all_messages"].
    # OpenHands uses condensation (off-policy) - use reject_summarization=false to allow.
    TRAJECTORY_BASED_AGENTS = frozenset(
        {"mini-swe-agent", "swe-agent", "openhands", "openhands-host"})

    @classmethod
    def extract(cls, results) -> Optional[tuple]:
        """Return (chat_history, summarization_count, num_turns) or None on failure."""
        agent_result = results.agent_result
        if agent_result is None:
            return None

        metadata = agent_result.metadata or {}
        chat_history = metadata.get("all_messages")
        if chat_history is not None:
            return chat_history, metadata.get("summarization_count", 0), metadata.get("n_episodes", 0)

        # Fallback: load from trajectory.json or completions for trajectory-based agents
        agent_name = (getattr(results.config.agent,
                      "name", None) or "").lower()
        if agent_name not in cls.TRAJECTORY_BASED_AGENTS:
            return None

        trial_path = cls._trial_path_from_uri(
            getattr(results, "trial_uri", None) or "")
        if trial_path is None:
            return None

        trajectory_path = trial_path / "agent" / "trajectory.json"
        chat_history = cls._from_atif_trajectory(trajectory_path)
        if chat_history is None:
            return None

        # Trajectory-based agents don't track summarization; use 0 for strictly appending
        return chat_history, 0, cls._count_turns(chat_history)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_turns(messages: List[dict]) -> int:
        return sum(1 for m in messages if m["role"] == "assistant")

    @staticmethod
    def _trial_path_from_uri(trial_uri: str) -> Optional[Path]:
        """Extract local filesystem path from trial_uri (e.g. file:///path/to/trial)."""
        if not trial_uri:
            return None
        try:
            parsed = urlparse(trial_uri)
            if parsed.scheme == "file" and parsed.path:
                return Path(parsed.path)
        except Exception:
            pass
        return None

    @classmethod
    def _from_atif_trajectory(cls, trajectory_path: Path) -> Optional[List[dict]]:
        """Convert ATIF trajectory JSON to user/assistant chat messages for SkyRL training.

        Handles system steps (prepended to first user message), agent observations
        (converted to user messages for alternating user/assistant pattern), and
        tool_calls (serialized into assistant content).
        """
        if not trajectory_path.exists():
            return None
        try:
            with open(trajectory_path) as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(
                f"Failed to load trajectory from {trajectory_path}: {e}")
            return None

        messages: List[dict] = []
        pending_system: List[str] = []

        for step in data.get("steps", []):
            source = step.get("source", "")
            message = step.get("message", "")
            observation = step.get("observation")

            if source == "system":
                if message:
                    pending_system.append(message)
                continue

            if source == "user":
                content = message or ""
                if pending_system:
                    content = "\n\n".join(pending_system) + "\n\n" + content
                    pending_system = []
                messages.append({"role": "user", "content": content})

            elif source == "agent":
                content = message or ""
                if step.get("tool_calls"):
                    content = content + "\n" + \
                        json.dumps({"tool_calls": step["tool_calls"]})
                if not content:
                    continue
                messages.append({"role": "assistant", "content": content})

                # Observations represent environment feedback; emit as user message
                # to maintain the alternating user/assistant pattern required for RL.
                if observation and observation.get("results"):
                    obs_parts = [r.get("content", "")
                                 for r in observation["results"] if r.get("content")]
                    if obs_parts:
                        messages.append(
                            {"role": "user", "content": "\n".join(obs_parts)})

        return messages if messages else None


@dataclass
class HarborAgentOutput:
    response_ids: List[int]
    reward: float
    stop_reason: str
    loss_mask: List[int]
    prompt_ids: List[int]
    trajectory_id: TrajectoryID
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

        logger.info(
            f"HarborGenerator initialized with Harbor config. "
            f"Agent: {self._harbor_trial_config_template.get('agent', {}).get('name')}, "
            f"Trials dir: {self._harbor_trial_config_template.get('trials_dir', 'trials')}"
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

        all_outputs: List[HarborAgentOutput] = [None] * len(prompts)  # type: ignore[list-item]
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
    ) -> HarborAgentOutput:
        """
        Run a single harbor agent.
        """
        # Run the trial to get `reward`, `chat_history`, `summarization_count`, and `num_turns`
        reward = None
        chat_history = None
        summarization_count = None
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
            return HarborAgentOutput(
                response_ids=[0],
                reward=0,
                stop_reason=stop_reason,
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
            return_dict=False,
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
