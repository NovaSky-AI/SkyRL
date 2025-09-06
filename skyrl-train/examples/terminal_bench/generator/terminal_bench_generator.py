import asyncio
from dataclasses import dataclass
from typing import List
from skyrl_train.generators.base import GeneratorInterface, GeneratorInput, GeneratorOutput
from skyrl_train.generators.utils import get_rollout_metrics
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.base import ConversationType
from omegaconf import DictConfig
from pathlib import Path
from sandbox.models.trial.config import TrialConfig, AgentConfig, LocalTaskConfig
from sandbox.models.task.id import LocalTaskId
from sandbox.models.agent.name import AgentName
from sandbox.trial.trial import Trial


@dataclass
class TerminalBenchAgentOutput:
    response_ids: List[int]
    reward: float
    stop_reason: str
    loss_mask: List[int]
    prompt_ids: List[int]


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

        # TerminalBench config
        self.trials_dir = terminal_bench_cfg.trials_dir
        self.agent_name = terminal_bench_cfg.agent_name
        self.sandboxes_dir = terminal_bench_cfg.sandboxes_dir
        self.max_episodes = terminal_bench_cfg.max_episodes

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        # TODO(tgriggs): Plumb the sandboxes task list here instead of using (and ignoring) empty prompts
        prompts = input_batch["prompts"]
        tasks = []
        for _ in range(len(prompts)):
            tasks.append(
                self.terminal_bench_agent_loop(
                    prompt="",
                )
            )

        all_outputs = await asyncio.gather(*tasks)

        responses = [output.response_ids for output in all_outputs]
        rewards = [output.reward for output in all_outputs]
        rollout_metrics = get_rollout_metrics(responses, rewards)

        generator_output: GeneratorOutput = {
            "prompt_token_ids": [output.prompt_ids for output in all_outputs],
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": [output.loss_mask for output in all_outputs],
            "stop_reasons": [output.stop_reason for output in all_outputs],
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": None,
        }

        return generator_output

    async def terminal_bench_agent_loop(
        self,
        prompt: ConversationType,
    ) -> TerminalBenchAgentOutput:
        """
        Run a single terminal_bench agent.
        """
        if self.agent_name == "terminus":
            trial_config = TrialConfig(
                task=LocalTaskConfig(id=LocalTaskId(path=f"{self.sandboxes_dir}/examples/tasks/hello-world")),
                trials_dir=Path(self.trials_dir),
                agent=AgentConfig(
                    name=AgentName.TERMINUS_2.value,
                    model_name=f"{self.model_name}",
                    kwargs={"api_base": f"{self.base_url}/v1", "key": "fake_key", "max_episodes": self.max_episodes},
                ),
            )
        elif self.agent_name == "oracle":
            trial_config = TrialConfig(
                task=LocalTaskConfig(id=LocalTaskId(path=f"{self.sandboxes_dir}/examples/tasks/hello-world")),
                trials_dir=Path(self.trials_dir),
                agent=AgentConfig(
                    name=AgentName.ORACLE,
                    model_name=self.model_name,
                ),
            )
        else:
            raise ValueError(f"Invalid agent name: {self.agent_name}")

        trial = Trial(trial_config)
        # Run the trial
        while True:
            results = await trial.run()
            reward = results.verifier_result.rewards
            chat_history = results.agent_result.all_messages
            if len(chat_history) > 0:
                break
            else:
                print(f"[WARNING] Agent {self.agent_name} did not return a response")

        # Use the first message as the prompt
        prompt = [chat_history[0]]
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,  # Always add generation prompt for multi-turn
            tokenize=True,
        )
        initial_prompt_length = len(prompt_ids)

        # Process response messages (everything after the first message)
        response_messages = chat_history[1:]

        response_ids = []
        loss_mask = []

        for message in response_messages:
            # Apply chat template and tokenize each message
            msg_encoding = self.tokenizer.apply_chat_template([message], add_generation_prompt=False, tokenize=True)

            # Extend response_ids with the tokens
            response_ids.extend(msg_encoding)

            # Extend loss_mask: 0s for user, 1s for assistant
            if message["role"] == "user":
                loss_mask.extend([0] * len(msg_encoding))
            else:  # assistant
                loss_mask.extend([1] * len(msg_encoding))

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
        return TerminalBenchAgentOutput(
            response_ids=response_ids,
            reward=reward,
            stop_reason=stop_reason,
            loss_mask=loss_mask,
            prompt_ids=prompt_ids,
        )
