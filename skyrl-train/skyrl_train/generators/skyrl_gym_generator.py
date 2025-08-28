"""
This file implements ``SkyRLGymGenerator``, an implementation of the `GeneratorInterface` that
uses SkyRL-Gym as the environment.

For details, see https://skyrl.readthedocs.io/en/latest/tutorials/skyrl_gym_generator.html
"""

import asyncio
import copy
from uuid import uuid4
import skyrl_gym
from typing import List, Dict, Any, Optional
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm.asyncio import tqdm
from dataclasses import dataclass

from skyrl_train.generators.base import GeneratorInterface, GeneratorInput, GeneratorOutput
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.base import InferenceEngineInput, ConversationType
from omegaconf import DictConfig
from skyrl_gym.envs.base_text_env import BaseTextEnvStepOutput
from skyrl_train.generators.utils import get_custom_chat_template, get_generation_prompt_ids, apply_overlong_filtering


@dataclass
class AgentLoopOutput:
    """Output from a single agent_loop execution."""

    response_ids: List[int]
    reward: float
    stop_reason: str
    loss_mask: List[int]
    prompt_ids: List[int]
    rollout_logprobs: Optional[List[float]]


class SkyRLGymGenerator(GeneratorInterface):
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
        self.custom_chat_template = get_custom_chat_template(model_name)
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

    async def agent_loop(
        self,
        prompt: ConversationType,
        env_class: str,
        env_extras: List[Dict[str, Any]],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Optional[Dict[str, Any]] = None,
    ) -> AgentLoopOutput:
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
            env_extras: List[Dict[str, Any]]
            max_tokens: int
            max_input_length: int
            sampling_params: Optional[Dict[str, Any]]
        Returns:
            response_ids: List[int]
            reward: float
            stop_reason: str
            loss_mask: List[int]
            prompt_token_ids: List[int]
            rollout_logprobs: Optional[List[float]]
        """
        retokenize_chat_history = self.use_conversation_multi_turn and self.custom_chat_template

        # Create a new environment instance
        env_extras["max_turns"] = self.max_turns  # TODO(shu): move this to config
        env_config = self.skyrl_gym_cfg.get(env_class, DictConfig({}))
        env = skyrl_gym.make(env_class, env_config=env_config, extras=env_extras)

        trajectory_id = uuid4().hex
        done = False

        # Instantiate chat_history and chat_end_index, which are only used if `retokenize_chat_history==True`.
        # Need copy here since the prompt is a list of messages and we are going to modify it.
        chat_history = copy.deepcopy(prompt)

        # init() returns the first prompt to be given to the model, and optional metadata dict
        chat_history, _ = env.init(chat_history)
        initial_chat_history_length = len(chat_history)
        chat_end_index = len(chat_history)
        input_ids = self.tokenizer.apply_chat_template(
            chat_history,
            # If retokenize_chat_history==True, avoid including the generation prompt in both the
            # prompt_ids and response_ids due to how `response_encodings["input_ids"]` works.
            add_generation_prompt=not retokenize_chat_history,
            tokenize=True,
        )

        initial_prompt_length = len(input_ids)
        loss_mask = []  # this excludes the prompt
        rollout_logprobs = None

        while not done:
            # 1. Generate output
            if retokenize_chat_history:
                engine_input = InferenceEngineInput(
                    prompts=[chat_history], trajectory_ids=[trajectory_id], sampling_params=sampling_params
                )
            else:
                # Token-in-token-out.
                engine_input = InferenceEngineInput(
                    prompt_token_ids=[input_ids], trajectory_ids=[trajectory_id], sampling_params=sampling_params
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
            if self.env_executor is not None:
                loop = asyncio.get_running_loop()
                env_step_output: BaseTextEnvStepOutput = await loop.run_in_executor(self.env_executor, env.step, output)
            else:
                env_step_output: BaseTextEnvStepOutput = env.step(output)
            new_obs = env_step_output["observations"]
            reward = env_step_output["reward"]
            done = env_step_output["done"]

            if env_step_output.get("postprocessed_action", None) is not None:
                # TODO(Charlie): come back to this, we should deprecate postprocessed action
                print(
                    "WARNING: postprocessed action may violate token-in-token-out. Ideally you "
                    "post-process it in the token space rather than string space. "
                    "A better solution coming soon."
                )
                output = env_step_output["postprocessed_action"]
                output_ids = self.tokenizer.encode(output, add_special_tokens=False)

            # 3. Update states: input ids, loss_mask, chat_history, etc.
            # Three ways of managing input
            if retokenize_chat_history:
                # a. We always re-tokenize the entire chat history every turn and at the end.
                chat_history, chat_end_index, input_ids = self._get_next_input_ids_by_retokenizing_chat_history(
                    chat_history, chat_end_index, output, new_obs
                )
            elif self.use_conversation_multi_turn:
                # b. Token-in-token-out. Follow multi-turn chat history format.
                input_ids, loss_mask = self._get_next_input_ids_with_multiturn_chat_template(
                    input_ids, loss_mask, output_ids, new_obs, done
                )
            else:
                # c. Token-in-token-out. All steps/observations are appended to a single assistant message.
                loss_mask, input_ids, rollout_logprobs = self._get_next_input_ids_with_single_turn_chat_template(
                    output_ids, new_obs, loss_mask, input_ids, rollout_logprobs
                )

            if len(input_ids) > max_input_length:
                stop_reason = "length"
                break

        env.close()

        prompt_ids = input_ids[:initial_prompt_length]
        if retokenize_chat_history:
            response_encodings = self.tokenizer.apply_chat_template(
                chat_history[initial_chat_history_length:],
                chat_template=self.custom_chat_template,
                add_generation_prompt=False,
                return_dict=True,
                return_assistant_tokens_mask=True,
                tokenize=True,
            )
            loss_mask = response_encodings["assistant_masks"]
            response_ids = response_encodings["input_ids"]
        else:
            response_ids = input_ids[initial_prompt_length:]
        assert len(loss_mask) == len(response_ids), "loss_mask and response_ids should have the same length"

        if not self.use_conversation_multi_turn:
            # we might need to add the eos token to the response ids
            if response_ids[-1] != self.tokenizer.eos_token_id:
                response_ids.append(self.tokenizer.eos_token_id)
                loss_mask.append(1)

        # need to truncate loss mask correctly for responses that go to max length
        if self.max_turns > 1:
            # max total resp length = max tokens (max length of final turn generation) + max_input_length (max input for any generation turn) - len(original prompt)
            max_response_tokens = max_tokens + max_input_length - initial_prompt_length
        else:
            max_response_tokens = max_tokens

        if len(response_ids) > max_response_tokens:
            stop_reason = "length"
        response_ids = response_ids[:max_response_tokens]
        loss_mask = loss_mask[:max_response_tokens]

        return AgentLoopOutput(
            response_ids=response_ids,
            reward=reward,
            stop_reason=stop_reason,
            loss_mask=loss_mask,
            prompt_ids=prompt_ids,
            rollout_logprobs=rollout_logprobs,
        )

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
            init_prompt, _ = env.init(prompt)
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
            env_step_output: BaseTextEnvStepOutput = env.step(response)
            reward = env_step_output["reward"]
            rewards.append(reward)

            if len(response_ids) > max_tokens:
                response_ids = response_ids[:max_tokens]
            loss_masks.append([1] * len(response_ids))
            truncated_responses.append(response_ids)
            if logprobs is not None:
                sample_logprobs = logprobs[i][: len(response_ids)]
                truncated_logprobs.append(sample_logprobs)

            env.close()

        prompt_token_ids = self.tokenizer.apply_chat_template(prompts, add_generation_prompt=True, tokenize=True)
        responses = truncated_responses
        rollout_metrics = self._rollout_metrics(responses, rewards)

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
                )
            )

        all_outputs = await tqdm.gather(
            *tasks,
            desc="Generating Trajectories",
            miniters=max(1, len(tasks) // 10),
            mininterval=5,
        )

        responses = [output.response_ids for output in all_outputs]
        rewards = [output.reward for output in all_outputs]
        stop_reasons = [output.stop_reason for output in all_outputs]
        loss_masks = [output.loss_mask for output in all_outputs]
        prompt_token_ids = [output.prompt_ids for output in all_outputs]

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

        rollout_metrics = self._rollout_metrics(responses, rewards)

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
        }

        return generator_output

    def _rollout_metrics(self, responses: List[List[int]], rewards: List[float]):
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

        return {
            "generate/min_num_tokens": np.min(num_tokens_arr).item(),
            "generate/max_num_tokens": np.max(num_tokens_arr).item(),
            "generate/avg_num_tokens": np.mean(num_tokens_arr).item(),
            "generate/std_num_tokens": np.std(num_tokens_arr).item(),
            "generate/avg_tokens_non_zero_rewards": avg_tokens_non_zero_rewards.item(),
            "generate/avg_tokens_zero_rewards": avg_tokens_zero_rewards.item(),
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
        assert self.use_conversation_multi_turn and self.custom_chat_template
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
            chat_history[:chat_end_index], add_generation_prompt=False, tokenize=True
        )
        return chat_history, chat_end_index, input_ids

    def _get_next_input_ids_with_multiturn_chat_template(
        self,
        input_ids: List[int],
        loss_mask: List[int],
        output_ids: List[int],
        new_obs: ConversationType,
        done: bool,
    ):
        """
        Update the loss mask and input ids given a new model response and observation, following
        token-in-token-out.

        This function is used if `use_conversation_multi_turn` is True. It assumes that the input to the LLM is formatted as a list of messages, with observations
        stored in user messages.

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
        assert self.use_conversation_multi_turn and not self.custom_chat_template

        # 1. Directly append generated output
        input_ids += output_ids
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
        else:
            if not done:
                input_ids += self.generation_prompt_ids
                loss_mask += [0] * len(self.generation_prompt_ids)

        return input_ids, loss_mask

    def _get_next_input_ids_with_single_turn_chat_template(
        self,
        output_ids: List[int],
        new_obs: ConversationType,
        loss_mask: List[int],
        input_ids: List[int],
        logprobs: Optional[List[float]],
    ):
        """
        Update the loss mask and input ids given a new model response and observation, following
        token-in-token-out.

        This function is used if `use_conversation_multi_turn` is False. It assumes that the input to the LLM is a list of token ids
        and that the multi-turn conversation happens in a single assistant message.

        For example (using the Qwen 2.5 chat template), a trajectory for single-turn generation would look like:
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

                            turn 1 env observation goes here
                            <observation>...</observation>

                            turn 2 model response goes here:
                            <think>... </think>
                            ...
        Args:
            output_ids: List[int]
            new_obs: ConversationType
            loss_mask: List[int]
            input_ids: List[int]
        Returns:
            loss_mask: List[int]
            input_ids: List[int]
            logprobs: Optional[List[float]]
        """
        # just update raw tokens and loss mask
        new_resp_tokens = output_ids.copy()
        if new_resp_tokens[-1] == self.tokenizer.eos_token_id:
            # remove the eos token since we are continuing the current assistant message
            new_resp_tokens = new_resp_tokens[:-1]
        loss_mask += [1] * len(new_resp_tokens)
        input_ids += new_resp_tokens

        if len(new_obs) > 0:
            for obs in new_obs:
                obs_tokens = self.tokenizer.encode(obs["content"], add_special_tokens=False)
                loss_mask += [0] * len(obs_tokens)
                # logprobs for observation tokens doesn't matter since they will be masked out during loss computation
                if logprobs:
                    logprobs += [1] * len(obs_tokens)
                input_ids += obs_tokens

        return loss_mask, input_ids, logprobs
