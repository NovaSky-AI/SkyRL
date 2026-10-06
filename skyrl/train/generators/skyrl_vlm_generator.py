"""
SkyRLVLMGymGenerator: VLM (vision-language model) multi-turn RL generator.
"""

import copy
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import torch
from loguru import logger

import skyrl_gym
from skyrl.backends.renderer import decode_mm_kwargs
from skyrl.backends.skyrl_train.inference_servers.base import (
    ConversationType,
    InferenceEngineInput,
    MultiModalFeatures,
)
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)
from skyrl.train.config import GeneratorConfig, SkyRLGymConfig
from skyrl.train.generators.base import (
    TRAINING_PHASE_TRAIN,
    GeneratorOutput,
    TrainingPhase,
    TrajectoryID,
)
from skyrl.train.generators.skyrl_gym_generator import (
    SkyRLGymGenerator,
    TrajectoryOutput,
)
from skyrl.train.generators.vlm_chat_renderer import (
    VLLMChatRenderer,
    append_mm_features,
    shift_mm_features,
    truncate_mm_features,
)


class SkyRLVLMGymGenerator(SkyRLGymGenerator):
    """VLM generator that handles multi-modal (text + image) observations.

    The prompt and each observation are rendered through the inference server's
    ``/v1/chat/completions/render``. The rollout is token-in-token-out: an observation is rendered
    after a fixed base conversation and only its suffix is appended, with image placeholder offsets
    shifted into the trajectory. Earlier turns are never re-rendered.
    """

    def __init__(
        self,
        generator_cfg: GeneratorConfig,
        skyrl_gym_cfg: SkyRLGymConfig,
        inference_engine_client: RemoteInferenceClient,
        tokenizer,
        policy_model_name: Optional[str] = None,
    ):
        super().__init__(generator_cfg, skyrl_gym_cfg, inference_engine_client, tokenizer, policy_model_name)
        self.renderer = VLLMChatRenderer(
            client=inference_engine_client,
            base_conversation=self.base_conversation,
            eos_token_id=self.tokenizer.eos_token_id,
            chat_template_kwargs=self.generator_cfg.chat_template_kwargs,
            model_name=getattr(inference_engine_client, "model_name", None),
        )
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
        if self.custom_chat_template is not None:
            raise ValueError(
                "SkyRLVLMGymGenerator does not support a custom chat template, got "
                f"{generator_cfg.chat_template}. The inference server applies the model's chat template."
            )
        super()._validate_cfg(generator_cfg)

    async def agent_loop(
        self,
        prompt: ConversationType,
        env_class: str,
        env_extras: Dict[str, Any],
        max_tokens: int,
        max_input_length: int,
        sampling_params: Optional[Dict[str, Any]] = None,
        trajectory_id: Optional[TrajectoryID] = None,
        cache_salt: Optional[str] = None,
        training_phase: TrainingPhase = TRAINING_PHASE_TRAIN,
    ) -> TrajectoryOutput:
        """Multi-turn VLM generation loop for a single trajectory.

        Generated tokens are appended as returned by the engine (loss_mask=1, except an appended
        eos), and each observation's rendered tokens after them (loss_mask=0). Image features
        accumulate per trajectory and are sent with every generate call. The final observation is
        dropped from the response, along with its images.

        The per-trajectory ``session_id`` is released via ``finish_session`` on
        completion, error, or cancellation so session-aware routing policies can
        free the replica capacity held by the trajectory.
        """
        agent_loop_start_time = time.monotonic()
        time_splits = {"llm": 0.0, "env": 0.0}
        session_id = (
            f"{trajectory_id.instance_id}_{trajectory_id.repetition_id}" if trajectory_id is not None else uuid4().hex
        )
        try:
            # ── Setup ──────────────────────────────────────────────────────
            env_extras["max_turns"] = self.max_turns
            env_extras = self._setup_env_extras(env_class, env_extras, sampling_params, trajectory_id)
            env_config = getattr(self.skyrl_gym_cfg, env_class, dict())
            env = skyrl_gym.make(env_class, env_config=env_config, extras=env_extras)

            conversation = copy.deepcopy(prompt)
            conversation, _ = await self._run_in_executor_if_available(env.init, conversation)

            rendered_prompt = await self.renderer.render_prompt(conversation)
            input_ids: List[int] = list(rendered_prompt.token_ids)
            # Placeholder offsets index into ``input_ids``.
            mm_features: Optional[MultiModalFeatures] = rendered_prompt.features
            prompt_length = len(input_ids)

            # As in SkyRLGymGenerator, `sampling_params` is passed to the engine as is (None means the
            # engine's defaults, which come from the config), so read stop/logprobs from the config.
            current_sampling_params: dict = (
                sampling_params if sampling_params is not None else asdict(self.generator_cfg.sampling_params)
            )
            stop_strs = current_sampling_params.get("stop", None)
            get_logprobs = current_sampling_params.get("logprobs", None) is not None

            # ── Accumulators (over ``input_ids[prompt_length:]``) ──────────
            loss_mask: List[int] = []
            rollout_logprobs: Optional[List[float]] = [] if get_logprobs else None
            # (reward, index into the response of the turn's last generated token)
            per_step_rewards: List[Tuple[float, int]] = []
            response_length = 0
            stop_reason = "stop"
            done = False
            # (number of messages through the observation, its token ids, add_generation_prompt), read only
            # by the re-render check.
            rendered_observations: List[Tuple[int, List[int], bool]] = []

            # ── Main loop ─────────────────────────────────────────────────
            while not done:
                if len(input_ids) > max_input_length:
                    stop_reason = "length"
                    break

                # 1. Generate
                engine_input = InferenceEngineInput(
                    prompt_token_ids=[input_ids],
                    session_ids=[session_id],
                    sampling_params=sampling_params,
                    mm_features=[mm_features] if mm_features is not None else None,
                    cache_salt=cache_salt,
                )
                llm_call_start_time = time.monotonic()
                engine_output = await self.inference_engine_client.generate(engine_input, model=self.policy_model_name)
                time_splits["llm"] += time.monotonic() - llm_call_start_time

                gen_text = engine_output["responses"][0]
                gen_ids = list(engine_output["response_ids"][0])
                stop_reason = engine_output["stop_reasons"][0]
                gen_logprobs = engine_output["response_logprobs"][0] if engine_output.get("response_logprobs") else None

                # 1b. Append eos when sampling_params.stop is not None
                added_eos = False
                if stop_strs is not None and self.generator_cfg.append_eos_token_after_stop_str_in_multi_turn:
                    if gen_text.endswith(tuple(stop_strs)) and gen_ids[-1] != self.tokenizer.eos_token_id:
                        gen_ids.append(self.tokenizer.eos_token_id)
                        if gen_logprobs is not None:
                            gen_logprobs.append(0.0)
                        added_eos = True

                # 2. Environment step
                env_step_start_time = time.monotonic()
                env_step_output = await self._run_in_executor_if_available(env.step, gen_text)
                time_splits["env"] += time.monotonic() - env_step_start_time
                new_obs = env_step_output["observations"]
                step_reward: float = env_step_output["reward"]
                done = env_step_output["done"]
                # Only read by the re-render check; the rollout never re-renders the conversation.
                conversation.append({"role": "assistant", "content": gen_text})
                conversation.extend(new_obs)

                # 3. Render the observation after a fixed base and append its tokens and images.
                obs = await self.renderer.render_observation(new_obs, done)
                if obs.token_ids:
                    rendered_observations.append((len(conversation), obs.token_ids, not done))
                obs_start = len(input_ids) + len(gen_ids)
                if obs.features is not None:
                    mm_features = append_mm_features(mm_features, shift_mm_features(obs.features, obs_start))
                input_ids += gen_ids + obs.token_ids
                loss_mask += ([1] * (len(gen_ids) - 1) + [0] if added_eos else [1] * len(gen_ids)) + [0] * len(
                    obs.token_ids
                )
                if rollout_logprobs is not None:
                    rollout_logprobs += (gen_logprobs if gen_logprobs else [0.0] * len(gen_ids)) + [0.0] * len(
                        obs.token_ids
                    )
                # The response ends after the last generated tokens; the final observation is dropped.
                response_length = obs_start - prompt_length
                if gen_ids:
                    per_step_rewards.append((step_reward, response_length - 1))

            if self.generator_cfg.vision_language_rerender_check:
                await self._log_rerender_mismatch(conversation, rendered_observations)

            # ── Build outputs ─────────────────────────────────────────────
            prompt_ids = input_ids[:prompt_length]
            response_ids = input_ids[prompt_length : prompt_length + response_length]
            loss_mask = loss_mask[:response_length]
            if rollout_logprobs is not None:
                rollout_logprobs = rollout_logprobs[:response_length]
            reward_out = self._build_per_token_rewards(per_step_rewards, response_ids, appended_eos_token=False)
            pixel_values, image_grid_thw = self._decode_vision_features(mm_features, prompt_length + response_length)

            # ── Cleanup ───────────────────────────────────────────────────
            env_metrics = env.get_metrics()
            await self._run_in_executor_if_available(env.close)

            agent_loop_output = TrajectoryOutput(
                response_ids=response_ids,
                reward=reward_out,
                stop_reason=stop_reason,
                loss_mask=loss_mask,
                prompt_ids=prompt_ids,
                rollout_logprobs=rollout_logprobs,
                env_metrics=env_metrics,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
            agent_loop_output = self._post_process_agent_loop_output(agent_loop_output, env_extras, trajectory_id)
            agent_loop_output.e2e_time = time.monotonic() - agent_loop_start_time
            agent_loop_output.time_splits = time_splits
            return agent_loop_output

        finally:
            await self.inference_engine_client.finish_session(session_id)

    async def _log_rerender_mismatch(
        self, conversation: ConversationType, rendered_observations: List[Tuple[int, List[int], bool]]
    ) -> None:
        """Log a warning for each observation whose appended tokens differ from a re-render in context.

        Each observation was rendered after the fixed base conversation. Here the conversation up to
        and including the observation is re-rendered, and its last tokens must equal the appended
        ones. Assistant turns are not compared: the appended generated tokens are not expected to
        match a re-render of the generated text. Chat templates rewrite assistant turns (Qwen3.5
        drops the reasoning of earlier turns and inserts an empty think block when the text has no
        ``</think>``), and the generated text may end with the eos string or be truncated.
        """
        for num_messages, obs_ids, add_generation_prompt in rendered_observations:
            rendered = await self.renderer.render_prompt(
                conversation[:num_messages], add_generation_prompt=add_generation_prompt
            )
            tail = rendered.token_ids[-len(obs_ids) :]
            if tail == obs_ids:
                continue
            first_diff = next((i for i, (a, b) in enumerate(zip(tail, obs_ids)) if a != b), len(tail))
            window = slice(max(0, first_diff - 8), first_diff + 8)
            logger.warning(
                f"Observation ending at message {num_messages} renders differently in the full conversation "
                f"than after the fixed base, at observation token {first_diff}. "
                f"Re-rendered: {self.tokenizer.decode(tail[window])!r}, "
                f"token-in-token-out: {self.tokenizer.decode(obs_ids[window])!r}"
            )

    @staticmethod
    def _decode_vision_features(
        mm_features: Optional[MultiModalFeatures], num_tokens: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Decode the image tensors for the images whose placeholders lie in the first ``num_tokens``.

        Images in the dropped final observation are left out, so the tensors match the placeholder
        tokens in ``prompt_ids + response_ids``. They are decoded from the same serialized items that
        were sent to the inference engine. Returns ``(None, None)`` without images.
        """
        kept = truncate_mm_features(mm_features, num_tokens)
        if kept is None:
            return None, None
        mm_kwargs = decode_mm_kwargs(kept.get("kwargs_data"))
        return mm_kwargs["pixel_values"], mm_kwargs["image_grid_thw"]

    async def generate_batched(self, *args, **kwargs) -> GeneratorOutput:
        raise NotImplementedError(
            "SkyRLVLMGymGenerator does not support batched generation. "
            "Use the default async agent_loop path instead."
        )
