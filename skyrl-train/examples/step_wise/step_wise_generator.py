"""
This file implements ``StepWiseGenerator`` for step-wise training
"""

import copy
from uuid import uuid4
import skyrl_gym
from typing import List, Dict, Any, Optional, Union, Tuple
from tqdm.asyncio import tqdm
from dataclasses import dataclass

from skyrl_train.generators.base import GeneratorInput, GeneratorOutput, TrajectoryID
from skyrl_train.generators.skyrl_gym_generator import SkyRLGymGenerator
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.base import InferenceEngineInput, ConversationType
from omegaconf import DictConfig
from skyrl_gym.envs.base_text_env import BaseTextEnvStepOutput
from skyrl_train.generators.utils import (
    apply_overlong_filtering,
    get_rollout_metrics,
)


class StepWiseGeneratorOutput(GeneratorOutput):
    trajectory_ids: List[TrajectoryID]
    is_last_step: List[bool]


@dataclass
class AgentLoopOutput:
    """Output from a single agent_loop execution."""

    response_ids: List[int]
    reward: Union[List[float], float]
    stop_reason: str
    loss_mask: List[int]
    prompt_ids: List[int]
    rollout_logprobs: Optional[List[float]]


class StepWiseGenerator(SkyRLGymGenerator):

    def __init__(
        self,
        generator_cfg: DictConfig,
        skyrl_gym_cfg: DictConfig,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
        model_name: str,
    ):
        super().__init__(generator_cfg, skyrl_gym_cfg, inference_engine_client, tokenizer, model_name)

        if self.batched:
            raise ValueError("StepWiseGenerator doesn't support `batched=True`")

        if self.custom_chat_template is not None:
            raise ValueError(
                f"StepWiseGenerator doesn't support custom chat template, got {generator_cfg.chat_template}"
            )

    async def agent_loop(
        self,
        prompt: ConversationType,
        env_class: str,
        env_extras: Dict[str, Any],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Optional[Dict[str, Any]] = None,
        trajectory_id: Optional[TrajectoryID] = None,
    ) -> List[AgentLoopOutput]:
        """
        Multi-turn generation loop that executes a single trajectory.

        Note:
            We ensure token-in-token-out generation. With one exceptions:
            - When calling Env.step() and BaseTextEnvStepOutput["postprocessed_action"] is not None.
              This will likely be deprecated soon.

        Args:
            prompt: ConversationType
            env_extras: Dict[str, Any]
            max_tokens: int
            max_input_length: int
            sampling_params: Optional[Dict[str, Any]]
        Returns:
            response_ids: List[int]
            reward: Union[float, List[float]]
            stop_reason: str
            loss_mask: List[int]
            prompt_token_ids: List[int]
            rollout_logprobs: Optional[List[float]]
        """
        # Create a new environment instance
        env_extras["max_turns"] = self.max_turns  # TODO(shu): move this to config
        env_config = self.skyrl_gym_cfg.get(env_class, DictConfig({}))
        env = skyrl_gym.make(env_class, env_config=env_config, extras=env_extras)

        session_id = (
            f"{trajectory_id.instance_id}_{trajectory_id.repetition_id}" if trajectory_id is not None else uuid4().hex
        )
        done = False

        # Need copy here since the prompt is a list of messages and we are going to modify it.
        chat_history = copy.deepcopy(prompt)

        # init() returns the first prompt to be given to the model, and optional metadata dict
        chat_history, _ = await self._run_in_executor_if_available(env.init, chat_history)

        input_ids = self.tokenizer.apply_chat_template(
            chat_history,
            add_generation_prompt=True,
            chat_template=self.custom_chat_template,
            tokenize=True,
        )

        initial_prompt_length = len(input_ids)
        loss_mask = []  # this excludes the prompt

        # Accumulate per-step rewards. Format: (reward, response_end_token_idx)
        per_step_rewards: List[Tuple[float, int]] = []
        step_id = 0

        per_step_outputs: List[AgentLoopOutput] = []
        while not done:
            # 1. Generate output
            # Token-in-token-out.
            current_prompt_length = len(input_ids)
            engine_input = InferenceEngineInput(
                prompt_token_ids=[input_ids], session_ids=[session_id], sampling_params=sampling_params
            )
            engine_output = await self.inference_engine_client.generate(engine_input)
            output = engine_output["responses"][0]
            output_ids = engine_output["response_ids"][0]
            stop_reason = engine_output["stop_reasons"][0]

            # Append eos when sampling_params.stop is not None. Does not affect 3.a as chat templates add eos_token.
            # sampling_params is not None for eval, but None for training (which uses engine.sampling_params which are from cfg)
            current_sampling_params = (
                sampling_params if sampling_params is not None else self.generator_cfg.sampling_params
            )
            stop_strs = current_sampling_params.get("stop", None)
            if (
                stop_strs is not None
                and self.generator_cfg.append_eos_token_after_stop_str_in_multi_turn
                and self.use_conversation_multi_turn
            ):
                if output.endswith(tuple(stop_strs)) and output_ids[-1] != self.tokenizer.eos_token_id:
                    output_ids.append(self.tokenizer.eos_token_id)

            # 2. Environment step
            env_step_output: BaseTextEnvStepOutput = await self._run_in_executor_if_available(env.step, output)
            new_obs = env_step_output["observations"]
            step_reward: float = env_step_output["reward"]
            done = env_step_output["done"]

            # Token-in-token-out. Follow multi-turn chat history format.
            input_ids, response_end_idx, loss_mask = self._get_next_input_ids_with_multiturn_chat_template(
                input_ids,
                output_ids,
                new_obs,
                loss_mask,
                done,
            )
            response_ids = copy.deepcopy(input_ids[current_prompt_length:])
            # response_end_idx in response_ids needs to be shifted
            response_end_idx = response_end_idx - current_prompt_length
            per_step_rewards.append((step_reward, response_end_idx))
            per_step_output = AgentLoopOutput(
                response_ids=response_ids,
                reward=step_reward,
                loss_mask=copy.deepcopy(loss_mask[current_prompt_length - initial_prompt_length :]),
                prompt_ids=copy.deepcopy(input_ids[:current_prompt_length]),
                rollout_logprobs=None,
                stop_reason=stop_reason,
            )
            assert len(per_step_output.loss_mask) == len(
                per_step_output.response_ids
            ), "loss_mask and response_ids should have the same length"
            if len(input_ids) > max_input_length:
                stop_reason = "length"
                step_id += 1
                per_step_output.stop_reason = stop_reason
                per_step_outputs.append(per_step_output)
                break

            per_step_outputs.append(per_step_output)
            step_id += 1

        for per_step_output, (reward, resp_end_idx) in zip(per_step_outputs, per_step_rewards):
            per_token_reward = [0.0] * len(per_step_output.response_ids)
            per_token_reward[resp_end_idx] = float(reward)
            # in-place update to per-token reward
            per_step_output.reward = per_token_reward

        await self._run_in_executor_if_available(env.close)

        return per_step_outputs

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """
        Generate trajectories for the input batch.

        Returns outputs in the same order as the input batch.
        Args:
            input_batch: GeneratorInput
        Returns:
            GeneratorOutput
        """
        prompts = input_batch["prompts"]
        env_classes = input_batch["env_classes"]
        env_extras = input_batch["env_extras"]
        trajectory_ids = input_batch.get("trajectory_ids", None)
        sampling_params: Optional[dict] = input_batch.get("sampling_params", None)
        max_tokens = self.generator_cfg.sampling_params.max_generate_length
        max_input_length = self.generator_cfg.max_input_length

        # Async agent loop to generate trajectories in parallel.
        tasks = []
        for i in range(len(prompts)):
            tasks.append(
                self.agent_loop(
                    prompts[i],
                    env_classes[i],
                    env_extras[i],
                    max_tokens,
                    max_input_length,
                    sampling_params=sampling_params,
                    trajectory_id=trajectory_ids[i] if trajectory_ids is not None else None,
                )
            )

        all_outputs = await tqdm.gather(
            *tasks,
            desc="Generating Trajectories",
            miniters=max(1, len(tasks) // 10),
            mininterval=5,
        )

        responses = sum([[output.response_ids for output in step_outputs] for step_outputs in all_outputs], [])
        rewards = sum([[output.reward for output in step_outputs] for step_outputs in all_outputs], [])
        stop_reasons = sum([[output.stop_reason for output in step_outputs] for step_outputs in all_outputs], [])
        loss_masks = sum([[output.loss_mask for output in step_outputs] for step_outputs in all_outputs], [])
        prompt_token_ids = sum([[output.prompt_ids for output in step_outputs] for step_outputs in all_outputs], [])

        out_trajectory_ids = []
        is_last_step = []
        for i in range(len(all_outputs)):
            step_outputs = all_outputs[i]
            for step_id in range(len(step_outputs)):
                out_trajectory_id = copy.deepcopy(trajectory_ids[i])
                out_trajectory_id.step = step_id
                out_trajectory_ids.append(out_trajectory_id)
                is_last_step.append(step_id == len(step_outputs) - 1)

        if sampling_params is not None:
            # sampling params will be a dict in the format of the inference engine backend
            # TODO: this might have to change when we support logprobs for sglang
            get_logprobs = sampling_params.get("logprobs", None) is not None
        else:
            get_logprobs = self.generator_cfg.sampling_params.logprobs is not None

        if get_logprobs:
            rollout_logprobs = [output.rollout_logprobs for output in all_outputs]
        else:
            rollout_logprobs = None

        rollout_metrics = get_rollout_metrics(responses, rewards)

        if self.generator_cfg.zero_reward_on_non_stop:
            # set reward to 0 if the stop reason is not "stop"
            rewards = self._zero_reward_if_not_stop(rewards, stop_reasons)

        if self.generator_cfg.apply_overlong_filtering:
            loss_masks = apply_overlong_filtering(loss_masks, responses, self.tokenizer.eos_token_id)

        generator_output: GeneratorOutput = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": rollout_logprobs,
            "trajectory_ids": out_trajectory_ids,
            "is_last_step": is_last_step,
        }

        return generator_output

    def _get_next_input_ids_with_multiturn_chat_template(
        self,
        input_ids: List[int],
        output_ids: List[int],
        new_obs: ConversationType,
        loss_mask: List[int],
        done: bool,
    ):
        """
        Update the loss mask and input ids given a new model response and observation, following
        token-in-token-out.

        For example (using the Qwen 2.5 chat template), a trajectory for multi-turn generation would look like:
        <|im_start|>system
        ...
        <|im_end|>
        <|im_start|>user
                            question goes here
        <|im_end|>
        <|im_start|>assistant
                            turn 1 model response goes here
                            <think>... </think>
                            ...
        <|im_end|>
        <|im_start|>user
                            turn 1 env observation goes here
                            <observation>...</observation>
        <|im_end|>
        ...

        the chat template is applied without tokenization before and after the chat history is appended to
        in order to get new token ids in the chat template format (but without re-tokenizing the entire chat history every turn)

        Args:
            chat_history: ConversationType
            chat_end_index: int
            loss_mask: List[int]
            input_ids: List[int]
            output: str
            new_obs: ConversationType
        Returns:
            chat_history: ConversationType
            chat_end_index: int
            loss_mask: List[int]
            input_ids: List[int]
        """
        # append response tokens
        input_ids += output_ids
        response_end_idx = len(input_ids) - 1
        loss_mask += [1] * len(output_ids)

        # apply chat template for observations, also generate generation prompt for next turn
        if len(new_obs) > 0:
            # For Qwen, this will generate `\n<|user|>Some observation<|im_end|>\n`. Note that the
            # first `\n` is generated since we stripped it in ``base_conversation_token_ids``.
            observation_ids = self.tokenizer.apply_chat_template(
                [*self.base_conversation, *new_obs],
                add_generation_prompt=True,
                tokenize=True,
            )[len(self.base_conversation_token_ids) :]
            input_ids += observation_ids
            loss_mask += [0] * len(observation_ids)
        else:
            if not done:
                input_ids += self.generation_prompt_ids
                loss_mask += [0] * len(self.generation_prompt_ids)
        return input_ids, response_end_idx, loss_mask
