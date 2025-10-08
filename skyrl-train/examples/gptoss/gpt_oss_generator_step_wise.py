"""
This file implements ``SkyRLGymGenerator``, an implementation of the `GeneratorInterface` that
uses SkyRL-Gym as the environment.

For details, see https://skyrl.readthedocs.io/en/latest/tutorials/skyrl_gym_generator.html
"""

import asyncio
import copy
from uuid import uuid4
import skyrl_gym
from typing import List, Dict, Any, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor
from tqdm.asyncio import tqdm
from dataclasses import dataclass

from skyrl_train.generators.base import GeneratorInterface, GeneratorInput, GeneratorOutput, TrajectoryID
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.base import InferenceEngineInput, ConversationType
from omegaconf import DictConfig
from skyrl_gym.envs.base_text_env import BaseTextEnvStepOutput
from skyrl_train.generators.utils import (
    get_custom_chat_template,
    get_generation_prompt_ids,
    apply_overlong_filtering,
    get_rollout_metrics,
)


class GPTOSSGeneratorOutput(GeneratorOutput):
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


class GPTOSSGenerator(GeneratorInterface):
    def __init__(
        self,
        generator_cfg: DictConfig,
        skyrl_gym_cfg: DictConfig,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
        model_name: str,
    ):
        """
        Args:
            generator_cfg: DictConfig object containing the generator configuration
            inference_engine_client: InferenceEngineClient object for interacting with the inference engines
            tokenizer: tokenizer object for encoding and decoding text
        """
        self.generator_cfg = generator_cfg
        self.skyrl_gym_cfg = skyrl_gym_cfg
        self.inference_engine_client = inference_engine_client
        self.tokenizer = tokenizer
        self.max_turns = generator_cfg.max_turns
        self.batched = generator_cfg.batched
        self.use_conversation_multi_turn = generator_cfg.use_conversation_multi_turn

        # optionally use custom chat template to get loss masks (i.e. for Qwen3)
        self.custom_chat_template = get_custom_chat_template(generator_cfg.chat_template)
        # get generation prompt ids for the tokenizer if needed
        self.generation_prompt_ids = get_generation_prompt_ids(tokenizer) if self.use_conversation_multi_turn else None
        if self.skyrl_gym_cfg.max_env_workers > 0:
            self.env_executor = ThreadPoolExecutor(
                max_workers=self.skyrl_gym_cfg.max_env_workers, thread_name_prefix="skyrl-gym-env-"
            )
        else:
            self.env_executor = None

        if getattr(self.generator_cfg.sampling_params, "logprobs", None) is not None and not self.generator_cfg.batched:
            raise ValueError("`sampling_params.logprobs` should be `None` if `batched` is `False`")

        # base_conversation is used when `use_conversation_multi_turn==True and custom_chat_template==None` to
        # correctly format and tokenize observations into `observation_ids`.
        # Follows https://jybsuper.github.io/posts/multiturn_tokenization/#the-breakthrough-fixed-base-approach
        self.base_conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "I am a user."},
            {"role": "assistant", "content": "I am an assistant."},
        ]
        self.base_conversation_token_ids = tokenizer.apply_chat_template(
            self.base_conversation,
            add_generation_prompt=False,
            tokenize=True,
        )
        # We remove tokens after the last EOS token so that it can be captured in `observation_ids`.
        # For details, see https://skyrl.readthedocs.io/en/latest/tutorials/skyrl_gym_generator.html#multi-turn-tokenization-and-ti-to
        if self.tokenizer.eos_token_id in self.base_conversation_token_ids:
            last_eos_token_index = (
                len(self.base_conversation_token_ids)
                - 1
                - self.base_conversation_token_ids[::-1].index(self.tokenizer.eos_token_id)
            )
            self.base_conversation_token_ids = self.base_conversation_token_ids[: last_eos_token_index + 1]
        self.think_end_token_ids: List[int] = self.tokenizer.encode("<|end|>")
        self.think_start_token_ids: List[int] = self.tokenizer.encode("<|channel|>analysis<message>")

    async def _run_in_executor_if_available(self, func, *args, **kwargs):
        if (executor := self.env_executor) is not None:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(executor, func, *args, **kwargs)
        else:
            return func(*args, **kwargs)

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
            We ensure token-in-token-out generation. With two exceptions:
            - When calling Env.step() and BaseTextEnvStepOutput["postprocessed_action"] is not None.
              This will likely be deprecated soon.
            - When custom_chat_template = True and use_conversation_multi_turn = True. We always
              re-tokenize the entire chat history every turn and at the end. This is used for cases
              like removing Qwen3 thinking tokens in non-last-round assistant message.

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

        # Instantiate chat_history and chat_end_index, which are only used if `retokenize_chat_history==True`.
        # Need copy here since the prompt is a list of messages and we are going to modify it.
        chat_history = copy.deepcopy(prompt)

        # init() returns the first prompt to be given to the model, and optional metadata dict
        chat_history, _ = await self._run_in_executor_if_available(env.init, chat_history)
        # initial_chat_history_length = len(chat_history)
        # chat_end_index = len(chat_history)
        input_ids = self.tokenizer.apply_chat_template(
            chat_history,
            add_generation_prompt=True,
            chat_template=self.custom_chat_template,
            tokenize=True,
        )

        initial_prompt_length = len(input_ids)
        loss_mask = []  # this excludes the prompt
        # rollout_logprobs = None
        # Accumulate per-step rewards. Format: (reward, response_end_token_idx)
        per_step_rewards: List[Tuple[float, Optional[int]]] = []
        step_id = 0
        # full_trajectory = {"q": input_ids}
        per_step_outputs = []
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

            # Three ways of managing input
            # b. Token-in-token-out. Follow multi-turn chat history format.
            input_ids, response_end_idx, loss_mask = self._get_next_input_ids_with_multiturn_chat_template(
                input_ids,
                output_ids,
                new_obs,
                loss_mask,
                done,
            )
            per_step_rewards.append((step_reward, response_end_idx))
            response_ids = copy.deepcopy(input_ids[current_prompt_length:])
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

        await self._run_in_executor_if_available(env.close)

        return per_step_outputs

    async def generate_batched(
        self,
        prompts: List[ConversationType],
        env_classes: List[str],
        env_extras: List[Dict[str, Any]],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Optional[Dict[str, Any]] = None,
    ) -> GeneratorOutput:
        """
        Single-turn batched generation (can use the synchronous offline engine)

        Args:
            prompts: List[ConversationType]
            env_classes: List[str]
            env_extras: List[Dict[str, Any]]
            max_tokens: int
            max_input_length: int --> Currently unused as we assume batched is used only for single-turn.
            sampling_params: Optional[Dict[str, Any]]
        Returns:
            GeneratorOutput
        """
        envs = []
        init_prompts = []
        for env_class, env_extra, prompt in zip(env_classes, env_extras, prompts):
            env_extra["max_turns"] = self.max_turns
            env_config = self.skyrl_gym_cfg.get(env_class, DictConfig({}))
            env = skyrl_gym.make(env_class, env_config=env_config, extras=env_extra)
            init_prompt, _ = await self._run_in_executor_if_available(env.init, prompt)
            init_prompts.append(init_prompt)
            envs.append(env)

        # For single-turn generation, we can use text-in-token-out, since we do not need to re-tokenize.
        engine_input = InferenceEngineInput(prompts=init_prompts, sampling_params=sampling_params)
        engine_output = await self.inference_engine_client.generate(engine_input)
        responses = engine_output["responses"]
        all_response_ids = engine_output["response_ids"]
        stop_reasons = engine_output["stop_reasons"]
        logprobs = engine_output.get("response_logprobs", None)

        truncated_responses = []
        rewards = []
        loss_masks = []
        truncated_logprobs: Optional[List[List[float]]] = [] if logprobs is not None else None

        for i, (response, response_ids, env) in enumerate(zip(responses, all_response_ids, envs)):
            # step on environment and compute reward
            env_step_output: BaseTextEnvStepOutput = await self._run_in_executor_if_available(env.step, response)
            reward = env_step_output["reward"]
            rewards.append(reward)

            if len(response_ids) > max_tokens:
                response_ids = response_ids[:max_tokens]
            loss_masks.append([1] * len(response_ids))
            truncated_responses.append(response_ids)
            if logprobs is not None:
                sample_logprobs = logprobs[i][: len(response_ids)]
                truncated_logprobs.append(sample_logprobs)

            await self._run_in_executor_if_available(env.close)

        prompt_token_ids = self.tokenizer.apply_chat_template(prompts, add_generation_prompt=True, tokenize=True)
        responses = truncated_responses
        rollout_metrics = get_rollout_metrics(responses, rewards)

        if self.generator_cfg.apply_overlong_filtering:
            loss_masks = apply_overlong_filtering(loss_masks, responses, self.tokenizer.eos_token_id)

        generator_output: GeneratorOutput = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": truncated_logprobs,
        }

        return generator_output

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

        if self.batched:
            return await self.generate_batched(
                prompts, env_classes, env_extras, max_tokens, max_input_length, sampling_params
            )

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

    def _zero_reward_if_not_stop(self, rewards: List[float], stop_reasons: List[str]):
        """Sets the reward to 0 if the stop reason is not "stop".

        This can be useful in cases where the LLM generation was truncated or aborted, but the environment still assigns non-zero reward.
        Often, we have format rewards for the LLM to follow, but in cases where the LLM didn't finish the response,
        we typically don't want to reward it. This is a general setting for all environments.
        """
        for i, stop_reason in enumerate(stop_reasons):
            if stop_reason != "stop":
                if isinstance(rewards[i], list):
                    rewards[i] = [0.0] * len(rewards[i])
                else:
                    rewards[i] = 0.0
        return rewards

    # ----------------------------------------------------------------------------
    # Three methods of managing chat history and input ids in `agent_loop()`
    # ----------------------------------------------------------------------------
    def _get_next_input_ids_by_retokenizing_chat_history(
        self,
        chat_history: ConversationType,
        chat_end_index: int,
        output: str,
        new_obs: ConversationType,
    ):
        """
        Update the chat history and input ids given a new model response and observation by retokenizing
        the entire chat history. Hence token-in-token-out is not followed.

        loss_mask is not maintained because we get it at the end of the trajectory with
        `response_encodings["assistant_masks"]`.

        Returns:
            chat_history: The updated chat history.
            chat_end_index: The updated chat end index.
            input_ids: The new input IDs after tokenizing the chat history.
        """
        # assert self.use_conversation_multi_turn and self.custom_chat_template
        # remove eos token from end of output if it exists, since it will be reapplied by the chat template
        if output.endswith(self.tokenizer.eos_token):
            output = output[: -len(self.tokenizer.eos_token)]

        # Add assistant response to chat history
        chat_history += [{"role": "assistant", "content": output}]
        chat_end_index += 1

        # Add observations to chat history
        if len(new_obs) > 0:
            chat_history += new_obs
            chat_end_index += len(new_obs)

        # re-apply whole chat template so length check is correct
        input_ids = self.tokenizer.apply_chat_template(
            chat_history[:chat_end_index],
            chat_template=self.custom_chat_template,
            add_generation_prompt=False,
            tokenize=True,
        )
        return chat_history, chat_end_index, input_ids

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
        # assert self.use_conversation_multi_turn and not self.custom_chat_template

        # 1. Get think and response tokens
        # think_ids, answer_ids = get_think_and_answer_ids(output_ids, think_start_token_ids=self.think_start_token_ids, think_end_token_ids=self.think_end_token_ids)
        # full_trajectory[f"t{step_id}"], full_trajectory[f"r{step_id}"] = think_ids, answer_ids
        # custom chat template - only append answer ids
        input_ids += output_ids
        response_end_idx = len(input_ids) - 1
        loss_mask += [1] * len(output_ids)

        # 2. apply chat template for observations, also generate generation prompt for next turn
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
            # full_trajectory[f"o{step_id}"] = observation_ids
        else:
            if not done:
                input_ids += self.generation_prompt_ids
                loss_mask += [0] * len(self.generation_prompt_ids)
        return input_ids, response_end_idx, loss_mask
