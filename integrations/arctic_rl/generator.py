"""ArcticGenerator — implements SkyRL's GeneratorInterface for Arctic RL.

Routes rollout generation to the Arctic RL inference engine via
``ArcticRLClient.generate()``.  After generation, each completion is
scored by the corresponding skyrl-gym environment (e.g. GSM8K) so that
the reward signal is available for GRPO training.
"""

import asyncio
import concurrent.futures
import logging
import os
from typing import Any, Dict, List, Optional

import skyrl_gym
from skyrl.train.generators.base import GeneratorInput, GeneratorInterface, GeneratorOutput

logger = logging.getLogger(__name__)

# 8 matches verl's agent_loop concurrency in the xid2pl9f reference run.
_DEFAULT_SCORING_WORKERS = int(os.environ.get("ARCTIC_RL_SCORING_WORKERS", "8"))


def _score_one(payload: Dict[str, Any]) -> float:
    """Picklable env.init/step/close used by the scoring ProcessPoolExecutor."""
    import skyrl_gym as _sg

    env = _sg.make(
        payload["env_class"],
        env_config=payload["env_config"],
        extras=payload["extras"],
    )
    try:
        env.init(payload["prompt"])
        step_out = env.step(payload["text"])
        return float(step_out["reward"])
    finally:
        env.close()


class ArcticGenerator(GeneratorInterface):

    def __init__(
        self,
        arctic_client,
        tokenizer,
        sampling_params: Optional[Any] = None,
        skyrl_gym_cfg: Optional[Any] = None,
    ):
        self.arctic_client = arctic_client
        self.tokenizer = tokenizer
        self.skyrl_gym_cfg = skyrl_gym_cfg
        self.default_sampling_params = {
            "temperature": 1.0,
            "max_tokens": 4096,
            "top_p": 1.0,
        }
        # Process pool (not threads) so per-worker BIRD sqlite handles and
        # other reward-state globals stay isolated.
        self._scoring_pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=_DEFAULT_SCORING_WORKERS,
        )

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        prompts = input_batch["prompts"]
        sampling_params = input_batch.get("sampling_params") or self.default_sampling_params
        env_classes: List[str] = input_batch.get("env_classes", [])
        env_extras: List[Dict[str, Any]] = input_batch.get("env_extras", [])

        prompt_texts, prompt_token_ids_list = [], []
        for prompt in prompts:
            text = (
                self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
                if isinstance(prompt, list)
                else str(prompt)
            )
            prompt_texts.append(text)
            prompt_token_ids_list.append(self.tokenizer.encode(text, add_special_tokens=False))

        # arctic_platform.rl client.generate is async; await directly.
        raw_outputs = await self.arctic_client.generate(
            prompts=prompt_texts,
            sampling_params=sampling_params,
        )

        # Build scoring payloads serially (tokenize/decode is cheap), then
        # fan reward computation out to the pool. Two passes keep alignment
        # with raw_outputs trivial.
        response_ids: List[List[int]] = []
        loss_masks: List[List[int]] = []
        stop_reasons: List[str] = []
        scoring_inputs: List[Optional[Dict[str, Any]]] = []

        for i, output in enumerate(raw_outputs):
            token_ids = output.get("token_ids", [])
            text = output.get("text", "")
            if not token_ids and text:
                token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not text and token_ids:
                text = self.tokenizer.decode(token_ids, skip_special_tokens=True)

            response_ids.append(token_ids)
            loss_masks.append([1] * len(token_ids))
            stop_reasons.append("completed" if output.get("finish_reason") == "stop" else "length")

            env_class = env_classes[i] if i < len(env_classes) else None
            if not env_class:
                if i == 0:
                    logger.warning(
                        "ArcticGenerator: no env_classes for sample %d (len=%d)",
                        i, len(env_classes),
                    )
                scoring_inputs.append(None)
                continue

            env_config = (
                getattr(self.skyrl_gym_cfg, env_class, dict()) if self.skyrl_gym_cfg else dict()
            )
            scoring_inputs.append(
                {
                    "env_class": env_class,
                    "env_config": env_config,
                    "extras": env_extras[i] if i < len(env_extras) else {},
                    "prompt": prompts[i],
                    "text": text,
                }
            )

        loop = asyncio.get_running_loop()
        futures = [
            (
                loop.run_in_executor(self._scoring_pool, _score_one, payload)
                if payload is not None
                else None
            )
            for payload in scoring_inputs
        ]
        rewards: List[float] = []
        for i, fut in enumerate(futures):
            if fut is None:
                rewards.append(0.0)
                continue
            try:
                rewards.append(await fut)
            except Exception as e:
                if i == 0:
                    logger.warning(
                        "ArcticGenerator reward scoring failed: %s", e, exc_info=True
                    )
                rewards.append(0.0)

        return GeneratorOutput(
            prompt_token_ids=prompt_token_ids_list,
            response_ids=response_ids,
            rewards=rewards,
            loss_masks=loss_masks,
            stop_reasons=stop_reasons,
            rollout_metrics=None,
            rollout_logprobs=None,
            trajectory_ids=input_batch.get("trajectory_ids"),
            rollout_expert_indices=None,
            is_last_step=None,
        )
