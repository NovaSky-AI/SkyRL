from typing import Any, List, TypedDict
from skyrl_agent.integrations.base import (
    AsyncInferBackend,
    GeneratorOutput,
    GeneratorInput,
    register_backend,
    BackendSpec,
)
import omegaconf
from loguru import logger
import os
import aiohttp


class OpenAIBackendConfig(TypedDict):
    model_name: str
    api_url: str


class OpenAIBackend(AsyncInferBackend):
    def __init__(self, infer_engine: Any, cfg: OpenAIBackendConfig):
        assert os.environ.get("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is not set"
        self.model_name = cfg["model_name"]
        self.api_url = cfg["api_url"]
        try:
            self.model_max_len = cfg["model_max_len"]
        except omegaconf.errors.ConfigKeyError:
            logger.info("`model_max_len` not provided, using `max_position_embeddings` from the model config")
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(self.model_name)
            self.model_max_len = config.max_position_embeddings
            logger.info(f"Using `model_max_len` {self.model_max_len}")

    async def async_generate_prompts(self, prompts: str, sampling_params: dict, **kwargs) -> str:
        # NOTE: In some agents like OpenHands, the generate calls are from a different thread, so the session needs to be created in the same thread
        # TODO: support long lived session depending on the task
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:

            headers = {"Content-Type": "application/json"}
            if not isinstance(sampling_params, dict):
                sampling_params = dict(sampling_params)
            payload = sampling_params.copy()
            payload["model"] = self.model_name

            # TODO: user template might be applied twice here. need to double check.
            payload["messages"] = {"role": "user", "content": prompts}
            output = await session.post(f"{self.api_url}/v1/chat/completions", json=payload, headers=headers)
            output = await output.json()

            try:
                return output["choices"][0]["message"]["content"]
            except Exception as e:
                logger.info(f"Errored out while extracting first response from output {output} with exception {str(e)}")

    async def async_generate_ids(self, input_ids: List[int], sampling_params: dict, **kwargs) -> str:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
            headers = {"Content-Type": "application/json"}
            if not isinstance(sampling_params, dict):
                sampling_params = dict(sampling_params)
            payload = sampling_params.copy()
            payload["model"] = self.model_name
            payload["max_tokens"] = self.model_max_len - len(input_ids) - 1
            print(f"max tokens: {payload['max_tokens']}")

            payload["prompt"] = input_ids
            output = await session.post(f"{self.api_url}/v1/completions", json=payload, headers=headers)
            output = await output.json()

        return output["choices"][0]["text"], output["choices"][0]["finish_reason"]


class OpenAIGeneratorOutput(GeneratorOutput):
    def __init__(self, result: Any):
        self.result = result


class OpenAIGeneratorInput(GeneratorInput):
    def __init__(self, input_batch: Any):
        self.input_batch = input_batch


register_backend(
    "openai_server",
    BackendSpec(
        infer_backend_cls=OpenAIBackend,
        generator_output_cls=OpenAIGeneratorOutput,
        generator_input_cls=OpenAIGeneratorInput,
    ),
)
