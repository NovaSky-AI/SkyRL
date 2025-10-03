from typing import Any, List
from ..base import AsyncInferBackend, GeneratorOutput, GeneratorInput


class SkyRLBackend(AsyncInferBackend):
    def __init__(self, infer_engine, cfg: Any = None):
        self.client = infer_engine

    async def async_generate_prompts(self, prompts: Any, sampling_params: Any, **kwargs) -> List[str]:
        input_obj = {
            "prompts": [prompts],
            "session_ids": [kwargs.get("request_id", None)],
            "sampling_params": sampling_params,
        }
        output = await self.client.generate(input_obj)
        return output["responses"][0], output["stop_reasons"][0]

    async def async_generate_ids(self, input_ids: List[int], sampling_params: Any, **kwargs) -> List[str]:
        input_obj = {
            "prompt_token_ids": [input_ids],
            "session_ids": [kwargs.get("request_id", None)],
            "sampling_params": sampling_params,
        }
        output = await self.client.generate(input_obj)
        # todo(@csy) probably need to be finish_reason
        # https://github.com/vllm-project/vllm/blob/a0f8a7964694a6077689b242b5eca95de392d4bb/vllm/v1/engine/__init__.py#L22
        return output["responses"][0], output["stop_reasons"][0]


class SkyRLGeneratorOutput(GeneratorOutput):
    def __init__(self, result: Any):
        from skyrl_train.generators.utils import get_rollout_metrics

        # Add more skyrl-specific rollout metrics.
        assert "rollout_metrics" in result, "rollout_metrics should be in the result"
        skyrl_rollout_metrics = get_rollout_metrics(result["response_ids"], result["rewards"])
        result["rollout_metrics"].update(skyrl_rollout_metrics)
        self.result = result


class SkyRLGeneratorInput(GeneratorInput):
    def __init__(self, input_batch: Any):
        self.input_batch = input_batch["env_extras"]
