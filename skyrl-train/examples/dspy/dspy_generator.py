import asyncio
from dataclasses import dataclass
from typing import List, Optional
from loguru import logger
from uuid import uuid4
import dspy
from skyrl_train.generators.base import GeneratorInterface, GeneratorInput, GeneratorOutput, TrajectoryID
from skyrl_train.generators.utils import get_rollout_metrics, get_response_ids_and_loss_mask_from_messages
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.base import ConversationType
from omegaconf import DictConfig
from pathlib import Path
from .trial import TrialConfig, Trial, AgentResult
from .utils import get_program, get_reward_function

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
# We have N retries for each trial, if one of the rollout (out of n_samples_per_prompt) fails
# after N attemptes, we skip this prompt altogether.
MAX_NUM_RETRIES_PER_TRIAL = 2

@dataclass
class DSPyOutput:
    response_ids: List[int]
    reward: float
    stop_reason: str
    loss_mask: List[int]
    prompt_ids: List[int]
    trajectory_id: TrajectoryID
    summarization_count: Optional[int] = None

class DSPyGenerator(GeneratorInterface):
    def __init__(
        self,
        generator_cfg: DictConfig,
        dspy_cfg: DictConfig,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
    ):
        """
        Args:
            generator_cfg: DictConfig object containing the generator configuration
            inference_engine_client: InferenceEngineClient object for interacting with the inference engines
            tokenizer: tokenizer object for encoding and decoding text
        """
        self.base_url = f"http://{generator_cfg.http_endpoint_host}:{generator_cfg.http_endpoint_port}"
        self.generator_cfg = generator_cfg
        self.dspy_cfg = dspy_cfg
        self.tokenizer = tokenizer
        self.model_name = generator_cfg.model_name

        # Create DSPy LM for the program
        # Disable DSPy cache/logs
        # dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
        dspy.disable_logging()
        
        # Create LM pointing to the inference engine endpoint
        # Add a warning if temperature is not found in sampling_params
        temperature = generator_cfg.sampling_params.get("temperature", None)
        if temperature is None:
            import warnings
            warnings.warn("Temperature not found in sampling_params, defaulting to 1.0")
            temperature = 1.0
            
        self.dspy_lm = dspy.LM(
            model=f"openai/{self.model_name}",
            api_base=f"{self.base_url}/v1",
            api_key="fake-key",
            temperature=temperature,
            model_type="chat",
            max_tokens=self.generator_cfg.sampling_params.max_generate_length,
            cache=False,
        )

        # Read custom chat template
        custom_chat_template_path = generator_cfg.engine_init_kwargs.get("custom_chat_template_chat_completion_path", None)
        if custom_chat_template_path:
            with open(custom_chat_template_path, "r") as f:
                self.custom_chat_template_content = f.read()
            logger.info(f"DSPyGenerator initialized with custom chat template read from: {custom_chat_template_path}")
        else:
            self.custom_chat_template_content = None

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        tasks = []
        for i in range(len(input_batch["prompts"])):
            tasks.append(
                self.foward(
                    prompt=input_batch["prompts"][i],
                    trajectory_id=input_batch["trajectory_ids"][i],
                )
            )

        batch_size = 10
        semaphore = asyncio.Semaphore(batch_size)

        async def run_with_limit(coro):
            async with semaphore:
                return await coro

        # Wrap each original task with a semaphore-limited task
        limited_tasks = [run_with_limit(t) for t in tasks]

        # Run all tasks (but concurrency is now limited!)
        all_outputs: List[DSPyOutput] = await asyncio.gather(*limited_tasks)

        # all_outputs: List[DSPyOutput] = await asyncio.gather(*tasks)

        # For a group of trajectories (n_samples_per_prompt trajectories for the same prompt), if one
        # of the trajectories fails, we skip the entire group. We also skip the group for rollout metric aggregation
        failed_instance_ids = set()
        num_failed_trajectories = 0  # per-trajectory, rather than per-instance
        successful_outputs: List[DSPyOutput] = []  # only for metrics purpose
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

    async def foward(
        self,
        prompt: ConversationType,
        trajectory_id: TrajectoryID,
    ) -> DSPyOutput:
        """
        Run a single terminal_bench agent.
        """
        # Generate session_id for sticky routing to inference engines
        # All LLM requests in this trial will share the same session_id
        session_id = uuid4().hex

        # "api_base": f"{self.base_url}/v1",

        
        # TODO: make each DSPy trial configurable.
        trial_config = TrialConfig(
            dspy_program=get_program(self.dspy_cfg.program), 
            example=prompt, 
            final_reward_fn=get_reward_function(self.dspy_cfg.final_reward_fn),
            local_reward_fn=get_reward_function(self.dspy_cfg.local_reward_fn),
            lm=self.dspy_lm
        )

        trial = Trial(trial_config)

        # Run the trial to get `rewards`, `chat_history`, and `summarization_count`
        prefix = f"Trajectory {trajectory_id}"
        successful = False
        reward = None
        chat_history = None
        summarization_count = None
        for i in range(MAX_NUM_RETRIES_PER_TRIAL):
            prefix = f"Trajectory {trajectory_id} attempt {i+1}/{MAX_NUM_RETRIES_PER_TRIAL}"
            results = None
            try:
                results = await trial.run()
                if results.exception_info:
                    logger.warning(f"{prefix} failed: Exception info: {results.exception_info}. Results: {results}")
                    continue
                if results.reward is None:
                    logger.warning(f"{prefix} failed: No reward returned. Results: {results}")
                    continue
                
                # Extract reward value from AgentResult
                reward = results.reward.output
                chat_history = results.traces
                
                if len(chat_history) == 3 and chat_history[2].get("content"):
                    successful = True
                    metadata = results.reward.metadata
                    logger.info(f"{prefix} successful: Results: {metadata}")
                    break
                else:
                    logger.warning(f"{prefix} failed: Did not return a chat history with an assistant message. chat_history: {chat_history}\n\nResults: {results}")
            except Exception as e:
                logger.warning(f"{prefix} failed: Error running trial: {e}. Results: {results}")
                continue

        if not successful:
            # We make loss mask 0 so it does not contribute to model updates
            logger.warning(f"Trajectory {trajectory_id} failed after {MAX_NUM_RETRIES_PER_TRIAL} attempts, will set loss mask to [0].")
            return DSPyOutput(
                response_ids=[0],
                reward=0,
                stop_reason="error",
                loss_mask=[0],
                prompt_ids=[0],
                trajectory_id=trajectory_id,
            )

        # Use the first message as the prompt. We assume to be no systems messages.
        assert chat_history[0]["role"] == "system", "The first message should be a system message"
        assert chat_history[1]["role"] == "user", "The second message should be a user message"
        prompt = [chat_history[0], chat_history[1]]
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=False,  # the message below will add it themselves
            tokenize=True,
            chat_template=self.custom_chat_template_content,
        )
        initial_prompt_length = len(prompt_ids)

        # Process response messages (everything after the first message)
        response_messages = chat_history[2:]
        assistant_logprobs = getattr(results.reward, "output_logprobs", None) if results.reward else None
        response_ids, loss_mask, rollout_logprobs = get_response_ids_and_loss_mask_from_messages(
            response_messages, self.tokenizer, assistant_logprobs, custom_chat_template=self.custom_chat_template_content
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
        return DSPyOutput(
            response_ids=response_ids,
            reward=reward,
            stop_reason=stop_reason,
            loss_mask=loss_mask,
            prompt_ids=prompt_ids,
            trajectory_id=trajectory_id,
            summarization_count=summarization_count,
        )