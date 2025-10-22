import asyncio
from dataclasses import dataclass
from typing import List, Optional
from uuid import uuid4
from skyrl_train.generators.base import GeneratorInterface, GeneratorInput, GeneratorOutput
from skyrl_train.generators.utils import get_rollout_metrics, encode_messages_subset
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.base import ConversationType
from omegaconf import DictConfig
from pathlib import Path
from sandboxes.models.trial.config import TrialConfig, AgentConfig, TaskConfig, EnvironmentConfig
from sandboxes.models.environment_type import EnvironmentType
from sandboxes.models.agent.name import AgentName
from sandboxes.trial.trial import Trial


@dataclass
class TerminalBenchAgentOutput:
    response_ids: List[int]
    reward: float
    stop_reason: str
    loss_mask: List[int]
    prompt_ids: List[int]
    rollout_logprobs: Optional[List[float]]


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
        self.max_episodes = terminal_bench_cfg.max_episodes

        if self.generator_cfg.chat_template.name_or_path is not None:
            raise NotImplementedError("TerminalBenchGenerator doesn't support custom chat template")

    # async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
    #     tasks = []
    #     for prompt in input_batch["prompts"]:
    #         tasks.append(
    #             self.terminal_bench_agent_loop(
    #                 prompt=prompt,
    #             )
    #         )

    #     all_outputs = await asyncio.gather(*tasks)
        
    #     all_outputs = [output for output in all_outputs if output is not None]

    #     responses = [output.response_ids for output in all_outputs]
    #     rewards = [output.reward for output in all_outputs]
    #     rollout_metrics = get_rollout_metrics(responses, rewards)

    #     generator_output: GeneratorOutput = {
    #         "prompt_token_ids": [output.prompt_ids for output in all_outputs],
    #         "response_ids": responses,
    #         "rewards": rewards,
    #         "loss_masks": [output.loss_mask for output in all_outputs],
    #         "stop_reasons": [output.stop_reason for output in all_outputs],
    #         "rollout_metrics": rollout_metrics,
    #         "rollout_logprobs": None,
    #     }

    #     return generator_output

    async def terminal_bench_agent_loop(
        self,
        prompt: ConversationType,
    ) -> TerminalBenchAgentOutput:
        """
        Run a single terminal_bench agent.
        """
        # Generate session_id for sticky routing to inference engines
        # All LLM requests in this trial will share the same session_id
        session_id = uuid4().hex

        if self.agent_name == "terminus":
            trial_config = TrialConfig(
                task=TaskConfig(path=prompt),
                trials_dir=Path(self.trials_dir),
                environment=EnvironmentConfig(type=EnvironmentType.DOCKER),
                agent=AgentConfig(
                    name=AgentName.TERMINUS_2.value,
                    model_name=f"{self.model_name}",
                    kwargs={
                        "api_base": f"{self.base_url}/v1",
                        "key": "fake_key",
                        "session_id": session_id,
                        "max_episodes": self.max_episodes,
                    },
                ),
            )
        elif self.agent_name == "oracle":
            trial_config = TrialConfig(
                task=TaskConfig(path=prompt),
                trials_dir=Path(self.trials_dir),
                environment=EnvironmentConfig(type=EnvironmentType.DOCKER),
                agent=AgentConfig(
                    name=AgentName.ORACLE,
                    model_name=f"hosted_vllm/{self.model_name}",
                ),
            )
        else:
            raise ValueError(f"Invalid agent name: {self.agent_name}")

        trial = Trial(trial_config)
        # Run the trial
        for retry in range(3):
            try:
                results = await trial.run()
                print(f"Results: {results}")
                if not results.verifier_result:
                    print(f"[WARNING] Exception info: {results.exception_info}")
                    continue
                reward = results.verifier_result.reward
                chat_history = results.agent_result.metadata["all_messages"]
                if len(chat_history) > 0:
                    break
                else:
                    print(f"[WARNING] Agent {self.agent_name} did not return a response")
            except Exception as e:
                print(f"Error running trial: {e}")
                continue
        
        if retry == 2:
            return None

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
        rollout_logprobs = []

        # Get logprobs for assistant messages from trial results
        # Format: [[logprobs for assistant msg 1], [logprobs for assistant msg 2], ...]
        assistant_logprobs = getattr(results.agent_result, "output_logprobs", None)
        assistant_msg_idx = 0

        for message in response_messages:
            # Apply chat template and tokenize each message
            msg_encoding = encode_messages_subset([message], self.tokenizer)

            # Extend response_ids with the tokens
            response_ids.extend(msg_encoding)

            # Extend loss_mask: 0s for user, 1s for assistant
            if message["role"] == "user":
                loss_mask.extend([0] * len(msg_encoding))
                if assistant_logprobs:
                    rollout_logprobs.extend([0.0] * len(msg_encoding))
            else:  # assistant
                loss_mask.extend([1] * len(msg_encoding))
                if assistant_logprobs:
                    if assistant_msg_idx >= len(assistant_logprobs):
                        raise ValueError(
                            f"Missing logprobs for assistant message #{assistant_msg_idx + 1}. Provided {len(assistant_logprobs)} logprob lists."
                        )
                    msg_logprobs = assistant_logprobs[assistant_msg_idx]
                    if len(msg_logprobs) != len(msg_encoding):
                        raise ValueError(
                            f"Logprobs count ({len(msg_logprobs)}) does not match token count ({len(msg_encoding)}) "
                            f"for assistant message #{assistant_msg_idx + 1}."
                        )
                    rollout_logprobs.extend(msg_logprobs)
                    assistant_msg_idx += 1

        # Determine stop reason
        max_response_tokens = (
            self.generator_cfg.sampling_params.max_generate_length
            + self.generator_cfg.max_input_length
            - initial_prompt_length
        )
        print("number of reponse tokens", len(response_ids))
        # import pdb; pdb.set_trace()

        stop_reason = "complete"  # Default for trial completion
        if len(response_ids) > max_response_tokens:
            stop_reason = "length"

        # Truncate to maximum allowed length
        response_ids = response_ids[:max_response_tokens]
        loss_mask = loss_mask[:max_response_tokens]
        rollout_logprobs = rollout_logprobs[:max_response_tokens]

        return TerminalBenchAgentOutput(
            response_ids=response_ids,
            reward=reward,
            stop_reason=stop_reason,
            loss_mask=loss_mask,
            prompt_ids=prompt_ids,
            # in case sandboxes doesn't return logprobs, use None
            rollout_logprobs=rollout_logprobs if assistant_logprobs is not None else None,
        )

    

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        # For testing, skip the actual async agent loop
        max_response_tokens = (
            self.generator_cfg.sampling_params.max_generate_length
            + self.generator_cfg.max_input_length
        )

        fake_output = self.fake_generator_output(
            tokenizer=self.tokenizer,
            input_batch=input_batch,
            max_response_tokens=max_response_tokens,
        )

        return fake_output
    

    def fake_generator_output(self, tokenizer, input_batch, max_response_tokens: int, batch_size: int = None):
        """
        Create a fake generator_output dict that mimics the structure of a real one.

        Args:
            tokenizer: your tokenizer (used to get vocab size)
            input_batch: the same input_batch used in generate()
            max_response_tokens: maximum length of fake responses
            batch_size: optional override for batch size
        """
        import random
        vocab_size = len(tokenizer)
        prompts = input_batch["prompts"]
        batch_size = batch_size or len(prompts)

        def random_tokens(length):
            return [random.randint(0, vocab_size - 1) for _ in range(length)]

        fake_outputs = []
        for _ in range(batch_size):
            fake_outputs.append({
                "prompt_ids": random_tokens(32),  # arbitrary prompt length
                "response_ids": random_tokens(max_response_tokens),
                "reward": random.uniform(0, 1),
                "loss_mask": [1] * max_response_tokens,
                "stop_reason": "length",
            })

        generator_output = {
            "prompt_token_ids": [o["prompt_ids"] for o in fake_outputs],
            "response_ids": [o["response_ids"] for o in fake_outputs],
            "rewards": [o["reward"] for o in fake_outputs],
            "loss_masks": [o["loss_mask"] for o in fake_outputs],
            "stop_reasons": [o["stop_reason"] for o in fake_outputs],
            "rollout_metrics": {"avg_reward": sum(o["reward"] for o in fake_outputs) / batch_size},
            "rollout_logprobs": None,
        }

        return generator_output
