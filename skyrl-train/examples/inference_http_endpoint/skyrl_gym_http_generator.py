import asyncio
import aiohttp
from skyrl_train.generators.skyrl_gym_generator import SkyRLGymGenerator
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from omegaconf import DictConfig
from typing import List, Dict, Any, Optional
from skyrl_gym.envs.base_text_env import BaseTextEnvStepOutput
from skyrl_train.inference_engines.base import InferenceEngineInput, ConversationType, InferenceEngineOutput
from skyrl_train.generators.utils import apply_overlong_filtering
from skyrl_train.generators.base import GeneratorOutput
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

        if self.enable_http_endpoint:
            assert (
                self.use_conversation_multi_turn
            ), "HTTP endpoint in SkyRLGymGenerator does not support use_conversation_multi_turn being False."
            # Store the base URL for direct HTTP requests
            self.base_url = f"http://{self.http_endpoint_host}:{self.http_endpoint_port}"
        else:
            self.base_url = None

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
        Exactly the same as SkyRLGymGenerator.generate_batched, but uses the HTTP endpoint.
        We also have to re-encode the string responses to token ids, meaning it is not token-in-token-out.
        """
        envs = []
        init_prompts = []
        trajectory_ids = []  # for load balancing
        counter = 0
        for env_class, env_extra, prompt in zip(env_classes, env_extras, prompts):
            env_extra["max_turns"] = self.max_turns
            env_config = self.skyrl_gym_cfg.get(env_class, DictConfig({}))
            env = skyrl_gym.make(env_class, env_config=env_config, extras=env_extra)
            init_prompt, _ = env.init(prompt)
            init_prompts.append(init_prompt)
            envs.append(env)
            trajectory_ids.append(counter)
            counter += 1

        # For single-turn generation, we can use text-in-token-out, since we do not need to re-tokenize.
        engine_input = InferenceEngineInput(
            prompts=init_prompts, trajectory_ids=trajectory_ids, sampling_params=sampling_params
        )

        # The only line different from SkyRLGymGenerator.generate_batched
        engine_output = await _generate_with_http_endpoint(self.base_url, self.model_name, engine_input)

        responses = engine_output["responses"]
        stop_reasons = engine_output["stop_reasons"]
        logprobs = engine_output.get("response_logprobs", None)

        truncated_responses = []
        rewards = []
        loss_masks = []
        truncated_logprobs: Optional[List[List[float]]] = [] if logprobs is not None else None

        for i, (response, env) in enumerate(zip(responses, envs)):
            # step on environment and compute reward
            env_step_output: BaseTextEnvStepOutput = env.step(response)
            reward = env_step_output["reward"]
            rewards.append(reward)

            # The other difference from SkyRLGymGenerator.generate_batched, /chat/completions response
            # does not include token outputs, but only the string response.
            response_ids = self.tokenizer.encode(response, add_special_tokens=False)

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


async def _generate_with_http_endpoint(
    base_url: str,
    model_name: str,
    input_batch: InferenceEngineInput,
) -> InferenceEngineOutput:
    """
    Generate responses using direct ClientSession.post calls with the InferenceEngineClient HTTP endpoint.

    Equivalent to running `self.inference_engine_client.generate()`, but with the HTTP endpoint.
    """
    prompts = input_batch.get("prompts")
    trajectory_ids = input_batch.get("trajectory_ids")
    sampling_params = input_batch.get("sampling_params")
    if trajectory_ids is not None:
        assert len(prompts) == len(trajectory_ids), "prompts and trajectory_ids must have the same length"

    # Use aiohttp session for direct HTTP requests
    conn = aiohttp.TCPConnector(limit=0, limit_per_host=0)  # 0 = no limit; without conn, has 100
    async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=None)) as session:
        headers = {"Content-Type": "application/json"}
        output_tasks = []

        for i, prompt in enumerate(prompts):
            trajectory_id = trajectory_ids[i] if trajectory_ids is not None else None
            payload = {
                "model": model_name,
                "messages": [{"role": m["role"], "content": m["content"]} for m in prompt],
                "trajectory_id": trajectory_id,
                **(sampling_params or {}),
            }
            output_tasks.append(session.post(f"{base_url}/v1/chat/completions", json=payload, headers=headers))

        # Execute all requests concurrently
        responses = await asyncio.gather(*output_tasks)

        # Parse responses
        results = []
        finish_reasons = []

        for response in responses:
            response_json = await response.json()
            choice = response_json["choices"][0]
            results.append(choice["message"]["content"])
            finish_reasons.append(choice["finish_reason"])

    inference_engine_output: InferenceEngineOutput = {
        "responses": results,
        "stop_reasons": finish_reasons,
        "response_ids": None,
    }

    return inference_engine_output
