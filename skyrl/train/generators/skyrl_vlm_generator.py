"""
SkyRLVLMGymGenerator: VLM (vision-language model) multi-turn RL generator.
"""

import copy
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, TypedDict
from uuid import uuid4

from loguru import logger

import skyrl_gym
from skyrl.backends.renderer import decode_mm_kwargs
from skyrl.backends.skyrl_train.inference_engines.base import (
    ConversationType,
    InferenceEngineInput,
    MultiModalFeatures,
)
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)
from skyrl.train.config import GeneratorConfig, SkyRLGymConfig
from skyrl.train.generators.base import GeneratorOutput, TrajectoryID
from skyrl.train.generators.skyrl_gym_generator import (
    SkyRLGymGenerator,
    TrajectoryOutput,
)


class RenderedConversation(TypedDict):
    prompt_ids: list[int]
    features: Optional[MultiModalFeatures]


class SkyRLVLMGymGenerator(SkyRLGymGenerator):
    """VLM generator that handles multi-modal (text + image) observations."""

    def __init__(
        self,
        generator_cfg: GeneratorConfig,
        skyrl_gym_cfg: SkyRLGymConfig,
        inference_engine_client: RemoteInferenceClient,
        tokenizer,
    ):
        super().__init__(generator_cfg, skyrl_gym_cfg, inference_engine_client, tokenizer)
        logger.info("Initialized SkyRLVLMGymGenerator (VLM multi-modal generator)")

    def _validate_cfg(self, generator_cfg: GeneratorConfig):
        if generator_cfg.batched:
            raise ValueError("SkyRLVLMGymGenerator does not support batched generation. Set `batched=False`.")
        if generator_cfg.step_wise_trajectories:
            raise ValueError("SkyRLVLMGymGenerator does not support step-wise trajectories.")
        if not generator_cfg.use_conversation_multi_turn:
            raise ValueError(
                "SkyRLVLMGymGenerator requires `use_conversation_multi_turn=True` "
                "because multi-modal observations must be in separate user messages."
            )

    async def _render_conversation(self, conversation: ConversationType) -> RenderedConversation:
        rendered = await self.inference_engine_client.render_chat_completion(
            {"json": {"model": self.inference_engine_client.model_name, "messages": conversation}}
        )
        return RenderedConversation(prompt_ids=rendered["token_ids"], features=rendered.get("features", None))

    async def agent_loop(
        self,
        prompt: ConversationType,
        env_class: str,
        env_extras: Dict[str, Any],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Optional[Dict[str, Any]] = None,
        trajectory_id: Optional[TrajectoryID] = None,
    ) -> TrajectoryOutput:
        """Multi-turn VLM generation loop for a single trajectory.
        The conversation is treated as the source of truth and re-tokenized each step.
        Generated tokens keep their original logprobs; observation tokens
        are obtained by slicing the re-render and masked out (loss_mask=0).
        """
        # ── Setup ──────────────────────────────────────────────────────
        env_extras["max_turns"] = self.max_turns
        env_config = getattr(self.skyrl_gym_cfg, env_class, dict())
        env = skyrl_gym.make(env_class, env_config=env_config, extras=env_extras)

        session_id = (
            f"{trajectory_id.instance_id}_{trajectory_id.repetition_id}" if trajectory_id is not None else uuid4().hex
        )

        conversation = copy.deepcopy(prompt)
        conversation, _ = await self._run_in_executor_if_available(env.init, conversation)

        # Render initial conversation → prompt_ids
        # latest_features always points to the most recent render's features
        # (each render covers the full conversation, so later renders supersede earlier ones)
        initial_render = await self._render_conversation(conversation)
        prompt_ids = initial_render["prompt_ids"]
        latest_features = initial_render["features"]

        current_sampling_params: dict = (
            sampling_params if sampling_params is not None else asdict(self.generator_cfg.sampling_params)
        )
        get_logprobs = self.generator_cfg.sampling_params.logprobs is not None
        stop_strs = current_sampling_params.get("stop", None)

        # ── Accumulators ───────────────────────────────────────────────
        response_ids: List[int] = []
        loss_mask: List[int] = []
        rollout_logprobs: Optional[List[float]] = [] if get_logprobs else None
        per_step_rewards: List[Tuple[float, int]] = []
        stop_reason = "stop"
        done = False

        # ── Main loop ─────────────────────────────────────────────────
        # To avoid a second render call per turn, we defer obs-token
        # extraction: after appending obs we just record the slice offset,
        # then compute the actual obs tokens from the *next* turn's render
        # (which produces identical token_ids since the conversation hasn't
        # changed in between).
        # NOTE: This only works for standard tokenizers which preserve sequence
        # extension, meaning that each successful assistant produces token
        # sequences which are a prefix of subsequent turns. In general, this
        # will not hold for thinking models, e.g., Qwen3-Thinking.
        pending_obs_offset: Optional[int] = None

        while not done:
            # 1. Render full conversation for this turn's generation input
            rendered_conversation = await self._render_conversation(conversation)
            input_ids = rendered_conversation["prompt_ids"]
            latest_features = rendered_conversation["features"]

            # 1b. Flush pending obs tokens from the previous turn
            if pending_obs_offset is not None:
                obs_tokens = input_ids[pending_obs_offset:]
                response_ids.extend(obs_tokens)
                loss_mask.extend([0] * len(obs_tokens))
                if rollout_logprobs is not None:
                    rollout_logprobs.extend([0.0] * len(obs_tokens))
                pending_obs_offset = None

            if len(input_ids) > max_input_length:
                stop_reason = "length"
                break

            # 2. Generate
            engine_input = InferenceEngineInput(
                prompt_token_ids=[input_ids],
                session_ids=[session_id],
                sampling_params=current_sampling_params,
                mm_features=[latest_features] if latest_features is not None else None,
            )
            engine_output = await self.inference_engine_client.generate(engine_input)

            gen_text = engine_output["responses"][0]
            gen_ids = engine_output["response_ids"][0]
            stop_reason = engine_output["stop_reasons"][0]
            gen_logprobs = engine_output["response_logprobs"][0] if engine_output.get("response_logprobs") else None

            # 2b. Append eos when sampling_params.stop is not None
            added_eos = False
            if stop_strs is not None and self.generator_cfg.append_eos_token_after_stop_str_in_multi_turn:
                if gen_text.endswith(tuple(stop_strs)) and gen_ids[-1] != self.tokenizer.eos_token_id:
                    gen_ids.append(self.tokenizer.eos_token_id)
                    if gen_logprobs is not None:
                        gen_logprobs.append(0.0)
                    added_eos = True

            # 3. Environment step
            env_step_output = await self._run_in_executor_if_available(env.step, gen_text)
            new_obs = env_step_output["observations"]
            step_reward: float = env_step_output["reward"]
            done = env_step_output["done"]

            # 4. Append assistant message to conversation
            conversation.append({"role": "assistant", "content": gen_text})

            # 5. Track generated tokens (loss_mask=1, except appended eos which is masked out)
            response_ids.extend(gen_ids)
            if added_eos:
                loss_mask.extend([1] * (len(gen_ids) - 1) + [0])
            else:
                loss_mask.extend([1] * len(gen_ids))
            if rollout_logprobs is not None:
                rollout_logprobs.extend(gen_logprobs if gen_logprobs else [0.0] * len(gen_ids))

            if gen_ids:
                per_step_rewards.append((step_reward, len(response_ids) - 1))

            # 6. If episode continues, defer obs token extraction to next render
            if not done:
                conversation.extend(new_obs)
                pending_obs_offset = len(input_ids) + len(gen_ids)

        # ── Build per-token rewards ───────────────────────────────────
        per_token_reward: List[float] = [0.0] * len(response_ids)
        for reward, idx in per_step_rewards:
            per_token_reward[idx] = float(reward)

        # ── Deserialize vision tensors from the most recent render ────
        mm_kwargs = decode_mm_kwargs((latest_features or {}).get("kwargs_data"))
        pixel_values = mm_kwargs["pixel_values"]
        image_grid_thw = mm_kwargs["image_grid_thw"]

        # ── Cleanup ───────────────────────────────────────────────────
        env_metrics = env.get_metrics()
        await self._run_in_executor_if_available(env.close)

        return TrajectoryOutput(
            response_ids=response_ids,
            reward=per_token_reward,
            stop_reason=stop_reason,
            loss_mask=loss_mask,
            prompt_ids=prompt_ids,
            rollout_logprobs=rollout_logprobs,
            env_metrics=env_metrics,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

    async def generate_batched(self, *args, **kwargs) -> GeneratorOutput:
        raise NotImplementedError(
            "SkyRLVLMGymGenerator does not support batched generation. "
            "Use the default async agent_loop path instead."
        )
