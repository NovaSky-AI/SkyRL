import asyncio
from dataclasses import dataclass
from typing import List, Optional
from loguru import logger
from uuid import uuid4
from skyrl_train.generators.base import GeneratorInterface, GeneratorInput, GeneratorOutput, TrajectoryID
from skyrl_train.generators.utils import get_rollout_metrics
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.base import ConversationType
from omegaconf import DictConfig
from pathlib import Path
from harbor.models.trial.config import TrialConfig, AgentConfig, TaskConfig, EnvironmentConfig
from harbor.models.environment_type import EnvironmentType
from harbor.models.agent.name import AgentName
from harbor.trial.trial import Trial

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
    rollout_logprobs: Optional[List[float]] = None

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

        # TerminalBench config. Parse here to ensure everything is passed in.
        self.trials_dir = terminal_bench_cfg.trials_dir
        self.agent_name = terminal_bench_cfg.agent_name
        self.max_episodes = terminal_bench_cfg.max_episodes
        self.enable_summarize = terminal_bench_cfg.get("enable_summarize", True)

        # Optional overrides for the environment
        self.override_memory_mb = terminal_bench_cfg.get("override_memory_mb")
        self.override_storage_mb = terminal_bench_cfg.get("override_storage_mb")
        self.override_cpus = terminal_bench_cfg.get("override_cpus")

        logger.info(f"TerminalBenchGenerator initialized with overrides: memory={self.override_memory_mb}, storage={self.override_storage_mb}, cpus={self.override_cpus}")

        # Read custom chat template
        custom_chat_template_path = generator_cfg.engine_init_kwargs.get("custom_chat_template_chat_completion_path", None)
        if custom_chat_template_path:
            with open(custom_chat_template_path, "r") as f:
                self.custom_chat_template_content = f.read()
            logger.info(f"TerminalBenchGenerator initialized with custom chat template read from: {custom_chat_template_path}")
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
            rollout_metrics["generate/trajectories_summarized"] = sum(1 for output in successful_outputs if output.summarization_count > 0)
            rollout_metrics["generate/trajectories_truncated"] = sum(1 for output in successful_outputs if output.stop_reason == "length")
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
        # Generate session_id for sticky routing to inference engines
        # All LLM requests in this trial will share the same session_id
        session_id = uuid4().hex

        environment_config = EnvironmentConfig(
            type=EnvironmentType.DAYTONA,
            override_cpus=self.override_cpus,
            override_memory_mb=self.override_memory_mb,
            override_storage_mb=self.override_storage_mb,
        )

        if self.agent_name == "terminus":
            trial_config = TrialConfig(
                task=TaskConfig(path=prompt),
                trials_dir=Path(self.trials_dir),
                environment=environment_config,
                agent=AgentConfig(
                    name=AgentName.TERMINUS_2.value,
                    model_name=f"hosted_vllm/{self.model_name}",
                    kwargs={
                        "api_base": f"{self.base_url}/v1",
                        "key": "fake_key",
                        "max_episodes": self.max_episodes,
                        "session_id": session_id,
                        "enable_summarize": self.enable_summarize,
                        "store_all_messages": True,
                        "collect_rollout_details": True,
                    },
                ),
            )
        elif self.agent_name == "oracle":
            trial_config = TrialConfig(
                task=TaskConfig(path=prompt),
                trials_dir=Path(self.trials_dir),
                environment=environment_config,
                agent=AgentConfig(
                    name=AgentName.ORACLE,
                    model_name=f"hosted_vllm/{self.model_name}",
                ),
            )
        else:
            raise ValueError(f"Invalid agent name: {self.agent_name}")

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
                chat_history = results.agent_result.metadata['all_messages']
                summarization_count = results.agent_result.metadata['summarization_count']
                if len(chat_history) > 1 and chat_history[0]["role"] == "user":
                    successful = True
                    logger.info(f"{prefix} successful: Results: {results.agent_result.metadata}")
                    break
                else:
                    logger.warning(f"{prefix} failed: Agent {self.agent_name} did not return a chat history with a user message. chat_history: {chat_history}\n\nResults: {results}")
            except Exception as e:
                logger.warning(f"{prefix} failed: Error running trial: {e}. Results: {results}")
                continue

        if not successful:
            # We make loss mask 0 so it does not contribute to model updates
            logger.warning(f"Trajectory {trajectory_id} failed after {MAX_NUM_RETRIES_PER_TRIAL} attempts, will set loss mask to [0].")
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
        rollout_details = getattr(results.agent_result, "rollout_details", None)

        # Extract from rollout_details tuple: ([{'prompt_token_ids': [...], 'completion_token_ids': [...], 'logprobs': [...]}],)
        try:
            details = rollout_details[0]
            if isinstance(details, list):
                details = details[0]
        except:
            logger.info(f"Unpacked details: {rollout_details[0]}")
        prompt_ids_list = details['prompt_token_ids']
        completion_ids_list = details['completion_token_ids']
        logprobs_list = details['logprobs']

        # Initial prompt is the first one
        prompt_ids = list(prompt_ids_list[0])

        # Build response_ids, loss_mask, and rollout_logprobs turn by turn
        # Structure: completion[0] + user_feedback[0] + completion[1] + user_feedback[1] + ... + completion[n-1]
        # where user_feedback[i] = prompt[i+1][len(prompt[i]) + len(completion[i]):]
        response_ids = []
        loss_mask = []
        rollout_logprobs = []

        num_turns = len(prompt_ids_list)
        for turn in range(num_turns):
            # Add completion tokens (mask = 1)
            completion = list(completion_ids_list[turn])
            logprobs = list(logprobs_list[turn])
            response_ids.extend(completion)
            loss_mask.extend([1] * len(completion))
            rollout_logprobs.extend(logprobs)

            # Add user feedback tokens after this completion (if not last turn)
            if turn < num_turns - 1:
                # user_feedback = prompt[turn+1] minus (prompt[turn] + completion[turn])
                prev_len = len(prompt_ids_list[turn]) + len(completion_ids_list[turn])
                user_tokens = list(prompt_ids_list[turn + 1][prev_len:])
                response_ids.extend(user_tokens)
                loss_mask.extend([0] * len(user_tokens))
                rollout_logprobs.extend([0.0] * len(user_tokens))

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
        if rollout_logprobs is not None:
            rollout_logprobs = rollout_logprobs[:max_response_tokens]
        return TerminalBenchAgentOutput(
            response_ids=response_ids,
            reward=reward,
            stop_reason=stop_reason,
            loss_mask=loss_mask,
            prompt_ids=prompt_ids,
            trajectory_id=trajectory_id,
            rollout_logprobs=rollout_logprobs,
            summarization_count=summarization_count,
        )
