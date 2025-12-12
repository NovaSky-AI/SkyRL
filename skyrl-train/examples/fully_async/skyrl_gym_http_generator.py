import aiohttp
from typing import Union
import copy
from uuid import uuid4
from skyrl_train.generators.skyrl_gym_generator import (
    SkyRLGymGenerator,
    TrajectoryOutput,
    StepWiseOutput,
    AgentLoopState,
    TurnOutput,
)
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from omegaconf import DictConfig
from typing import List, Dict, Any, Optional, Tuple
from skyrl_gym.envs.base_text_env import BaseTextEnvStepOutput
from skyrl_train.inference_engines.base import ConversationType
from skyrl_train.generators.base import TrajectoryID
import skyrl_gym


class SkyRLGymHTTPGenerator(SkyRLGymGenerator):
    def __init__(
        self,
        generator_cfg: DictConfig,
        skyrl_gym_cfg: DictConfig,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
        model_name: str,
    ):
        super().__init__(generator_cfg, skyrl_gym_cfg, inference_engine_client, tokenizer, model_name)
        self.model_name = model_name

        self.enable_http_endpoint = generator_cfg.enable_http_endpoint
        self.http_endpoint_host = generator_cfg.http_endpoint_host
        self.http_endpoint_port = generator_cfg.http_endpoint_port

        assert self.enable_http_endpoint, "HTTP endpoint must be enabled for SkyRLGymHTTPGenerator"
        assert (
            self.use_conversation_multi_turn
        ), "HTTP endpoint in SkyRLGymGenerator does not support use_conversation_multi_turn being False."
        assert self.custom_chat_template is not None, "SkyRLGymHTTPGenerator requires a custom chat template."
        # Store the base URL for direct HTTP requests
        self.base_url = f"http://{self.http_endpoint_host}:{self.http_endpoint_port}"

    async def agent_loop(
        self,
        prompt: ConversationType,
        env_class: str,
        env_extras: Dict[str, Any],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Optional[Dict[str, Any]] = None,
        trajectory_id: Optional[TrajectoryID] = None,
    ) -> Union[TrajectoryOutput, StepWiseOutput]:
        """
        Multi-turn generation loop that executes a single trajectory.

        This overrides the parent class's agent_loop to use the HTTP endpoint for generation,
        without respecting token-in-token-out generation. We largely follow the retokenize_chat_history
        codepath as a result.

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
        assert (
            not self.generator_cfg.step_wise_training
        ), "`step_wise_training` is not supported with `SkyRLGymHTTPGenerator`"

        # Create a new environment instance
        env_extras["max_turns"] = self.max_turns  # TODO(shu): move this to config
        env_config = self.skyrl_gym_cfg.get(env_class, DictConfig({}))
        env = skyrl_gym.make(env_class, env_config=env_config, extras=env_extras)

        session_id = (
            f"{trajectory_id.instance_id}_{trajectory_id.repetition_id}" if trajectory_id is not None else uuid4().hex
        )

        # Instantiate chat_history, which is used for retokenize_chat_history codepath.
        # Need copy here since the prompt is a list of messages and we are going to modify it.
        chat_history = copy.deepcopy(prompt)

        # init() returns the first prompt to be given to the model, and optional metadata dict
        chat_history, _ = await self._run_in_executor_if_available(env.init, chat_history)
        initial_chat_history_length = len(chat_history)
        initial_input_ids = self.tokenizer.apply_chat_template(
            chat_history,
            # If retokenize_chat_history==True, avoid including the generation prompt in both the
            # prompt_ids and response_ids due to how `response_encodings["input_ids"]` works.
            add_generation_prompt=False,
            chat_template=self.custom_chat_template,
            tokenize=True,
            **self.generator_cfg.chat_template_kwargs,
        )

        initial_prompt_length = len(initial_input_ids)
        # Accumulate per-step rewards. Format: (reward, response_end_token_idx)
        per_step_rewards: List[Tuple[float, Optional[int]]] = []

        # Initialize agent loop state
        agent_loop_state = AgentLoopState(
            chat_history=chat_history,
            input_ids=initial_input_ids,
            loss_mask=None,
            rollout_logprobs=None,
            response_end_idx=None,
            done=False,
        )

        stop_reason = None

        while not agent_loop_state.done:

            if len(agent_loop_state.input_ids) > max_input_length:
                stop_reason = "length"
                break

            # 1. Generate output by sending an HTTP request to the endpoint
            conn = aiohttp.TCPConnector(limit=0, limit_per_host=0)  # 0 = no limit; without conn, has 100
            async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=None)) as session:
                headers = {"Content-Type": "application/json"}
                messages = [{"role": m["role"], "content": m["content"]} for m in agent_loop_state.chat_history]
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "session_id": session_id,
                    **(sampling_params or {}),
                }
                async with session.post(f"{self.base_url}/v1/chat/completions", json=payload, headers=headers) as resp:
                    output_json = await resp.json()
            # Parse responses
            output = output_json["choices"][0]["message"]["content"]
            stop_reason = output_json["choices"][0]["finish_reason"]
            # Note: HTTP endpoint doesn't return output_ids or logprobs, so we set them to empty
            output_ids = []
            response_logprobs = None

            # 2. Environment step
            env_step_output: BaseTextEnvStepOutput = await self._run_in_executor_if_available(env.step, output)
            new_obs = env_step_output["observations"]
            step_reward: float = env_step_output["reward"]
            agent_loop_state.done = env_step_output["done"]
            assert (
                env_step_output.get("postprocessed_action", None) is None
            ), "postprocessed action is not supported for SkyRLGymHTTPGenerator"

            # Create turn output
            turn_output = TurnOutput(
                output=output,
                output_ids=output_ids,
                output_logprobs=response_logprobs,
                new_obs=new_obs,
                reward=step_reward,
            )

            # 3. Update states: input ids, loss_mask, chat_history, etc.
            # We always re-tokenize the entire chat history every turn and at the end.
            agent_loop_state = self._update_agent_state_by_retokenizing_chat_history(agent_loop_state, turn_output)
            # TODO(tgriggs): Support turn-level rewards for multi-turn chat template
            per_step_rewards.append((step_reward, agent_loop_state.response_end_idx))

        # Get environment-specific metrics after the episode is done
        env_metrics = env.get_metrics()
        # Close the environment
        await self._run_in_executor_if_available(env.close)

        prompt_ids = agent_loop_state.input_ids[:initial_prompt_length]
        response_encodings = self.tokenizer.apply_chat_template(
            agent_loop_state.chat_history[initial_chat_history_length:],
            chat_template=self.custom_chat_template,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
            tokenize=True,
            **self.generator_cfg.chat_template_kwargs,
        )
        loss_mask = response_encodings["assistant_masks"]
        response_ids = response_encodings["input_ids"]

        assert len(loss_mask) == len(response_ids), "loss_mask and response_ids should have the same length"

        # Build reward output
        # TODO(Charlie): Currently, the possible response truncation will not affect the reward
        # in the if branch, but some final rewards may be lost in the else branch. Fix this
        # when we support turn-level rewards for the `retokenize_chat_history` codepath.
        reward_out = self._build_per_token_rewards(per_step_rewards, response_ids, appended_eos_token=False)

        return TrajectoryOutput(
            response_ids=response_ids,
            reward=reward_out,
            stop_reason=stop_reason,
            loss_mask=loss_mask,
            prompt_ids=prompt_ids,
            rollout_logprobs=agent_loop_state.rollout_logprobs,
            env_metrics=env_metrics,
        )
