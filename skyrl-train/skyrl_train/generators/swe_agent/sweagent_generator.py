from typing import List, Dict, Any, Optional
import numpy as np

from skyrl_train.generators.base import GeneratorInterface, GeneratorInput, GeneratorOutput
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from omegaconf import DictConfig
from skyrl_train.generators.swe_agent.codeact import CodeActAgentGroup
from dataclasses import dataclass
import asyncio

# this is for the tokenizer.apply_chat_template to be able to generate assistant masks directly
# TODO (erictang000): can we make this generic for any chat template? (qwen specific right now)
chat_template = (
    "{% for message in messages %}"
    "{% if (message['role'] != 'assistant') %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% elif (message['role'] == 'assistant')%}"
    "{{'<|im_start|>' + message['role'] + '\n'}}"
    "{% generation %}"
    "{{message['content'] + '<|im_end|>' + '\n'}}"
    "{% endgeneration %}"
    "{% endif %}"
    "{% endfor %}"
)


@dataclass
class OHAgentConfig:
    max_parallel_agents: int
    max_eval_parallel_agents: int
    log_messages_dir: Optional[str] = None
    remove_think_tokens: bool = False
    qwen3_enable_thinking: bool = False


class OHAgentGenerator(GeneratorInterface):
    def __init__(
        self,
        generator_cfg: DictConfig,
        agent_cfg: OHAgentConfig,
        llm_endpoint_client: InferenceEngineClient,
        tokenizer,
        max_prompt_length: int,
    ):
        """
        Args:
            generator_cfg: DictConfig object containing the generator configuration
            llm_endpoint_client: InferenceEngineClient object for interacting with the LLM endpoints
            tokenizer: tokenizer object for encoding and decoding text
        """
        self.generator_cfg = generator_cfg
        self.llm_endpoint_client = llm_endpoint_client
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.agent_cfg = agent_cfg

    # TODO(tgriggs): `max_input_length` is currently not enforced. This will not be the responsibility of the endpoint, and will be handled by Generator or Env in a PR soon.
    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """
        Generate trajectories for the input batch.

        Returns outputs in the same order as the input batch.
        Args:
            input_batch: GeneratorInput
        Returns:
            GeneratorOutput
        """
        sampling_params = input_batch.get("sampling_params", None)

        # initialize codeact agent group
        codeact_agent_group = CodeActAgentGroup(
            batch=input_batch,
            num_trajectories=self.generator_cfg.n_samples_per_prompt,
            infer_engine=self.llm_endpoint_client,
            max_prompt_length=self.generator_cfg.max_input_length,
            max_response_length=self.generator_cfg.sampling_params.max_generate_length,
            max_starting_message_length=self.max_prompt_length,
            max_parallel_agents=self.agent_cfg.max_parallel_agents,
            max_eval_parallel_agents=self.agent_cfg.max_eval_parallel_agents,
            max_iterations=self.generator_cfg.max_turns,
            tokenizer=self.tokenizer,
            sampling_params=sampling_params,
            log_messages_dir=self.agent_cfg.log_messages_dir,
            remove_think_tokens=self.agent_cfg.remove_think_tokens,
            qwen3_enable_thinking=self.agent_cfg.qwen3_enable_thinking,
        )
        # separate thread might not be needed, just debugging hanging issue
        all_outputs, additional_info = await asyncio.to_thread(codeact_agent_group.run)

        rollout_metrics = self._rollout_metrics(all_outputs, additional_info)
        if self.generator_cfg.zero_reward_on_non_stop:
            # set reward to 0 if the stop reason is not "stop"
            rewards = self._zero_reward_if_not_stop(all_outputs["rewards"], all_outputs["stop_reasons"])
            all_outputs["rewards"] = rewards

        all_outputs["rollout_metrics"] = rollout_metrics

        generator_output: GeneratorOutput = all_outputs

        return generator_output

    def _rollout_metrics(self, all_outputs: GeneratorOutput, additional_info: Dict[str, Any]):
        responses = all_outputs["response_ids"]
        rewards = all_outputs["rewards"]

        num_tokens_arr = np.array([len(response) for response in responses])
        non_zero_rewards_arr = np.array([reward > 0.0 for reward in rewards])
        zero_rewards_arr = np.array([reward == 0.0 for reward in rewards])
        # average tokens for non zero rewards
        avg_tokens_non_zero_rewards = (
            np.mean(num_tokens_arr[non_zero_rewards_arr]) if non_zero_rewards_arr.sum() > 0 else np.zeros(1)
        )
        # average tokens for zero rewards
        avg_tokens_zero_rewards = (
            np.mean(num_tokens_arr[zero_rewards_arr]) if zero_rewards_arr.sum() > 0 else np.zeros(1)
        )
        error = additional_info["error"]
        finish = additional_info["finish"]

        reward_metrics = {}
        reward_metrics["generate/max_turn_ratio"] = sum(
            "RuntimeError: Agent reached maximum iteration in headless mode" in e for e in error if e
        ) / len(error)
        reward_metrics["generate/finish_action_ratio"] = sum(finish) / len(finish)
        reward_metrics["generate/stuck_ratio"] = sum("stuck in a loop" in e for e in error if e) / len(error)

        return {
            "generate/min_num_tokens": np.min(num_tokens_arr).item(),
            "generate/max_num_tokens": np.max(num_tokens_arr).item(),
            "generate/avg_num_tokens": np.mean(num_tokens_arr).item(),
            "generate/std_num_tokens": np.std(num_tokens_arr).item(),
            "generate/avg_tokens_non_zero_rewards": avg_tokens_non_zero_rewards.item(),
            "generate/avg_tokens_zero_rewards": avg_tokens_zero_rewards.item(),
            **reward_metrics,
        }

    def _zero_reward_if_not_stop(self, rewards: List[float], stop_reasons: List[str]):
        """Sets the reward to 0 if the stop reason is not "stop".

        This can be useful in cases where the LLM generation was truncated or aborted, but the environment still assigns non-zero reward.
        Often, we have format rewards for the LLM to follow, but in cases where the LLM didn't finish the response,
        we typically don't want to reward it. This is a general setting for all environments.
        """
        for i, stop_reason in enumerate(stop_reasons):
            if stop_reason != "stop":
                rewards[i] = 0.0
        return rewards
