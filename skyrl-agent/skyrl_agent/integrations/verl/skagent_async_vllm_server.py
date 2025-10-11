from typing import Any, Tuple
from typing import Optional

import ray
from vllm import SamplingParams
from typing import Dict

try:
    from omegaconf import DictConfig, ListConfig, OmegaConf
except Exception:
    DictConfig = None  # type: ignore
    ListConfig = None  # type: ignore
    OmegaConf = None  # type: ignore

from vllm.inputs import TokensPrompt
from vllm.outputs import RequestOutput
from verl.workers.rollout.vllm_rollout.vllm_async_server import AsyncvLLMServerRegular


@ray.remote(num_cpus=1)
class SkyAgentAsyncvLLMServer(AsyncvLLMServerRegular):
    async def generate(
        self, prompt_ids: list[int], sampling_params: dict[str, Any], request_id: str
    ) -> Tuple[str, str]:
        # max_tokens = self.max_model_len - len(prompt_ids)
        # sampling_params.pop("max_tokens", None)
        # sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_params)
        # Defensive sanitize of sampling params for vLLM
        sp: Dict[str, Any] = dict(sampling_params) if sampling_params is not None else {}
        # Ensure max_tokens exists and is an int within model limit
        if "max_tokens" not in sp or sp["max_tokens"] is None:
            max_tokens = self.max_model_len - len(prompt_ids)
            sp["max_tokens"] = int(max_tokens)
        else:
            try:
                sp["max_tokens"] = int(sp["max_tokens"])
            except Exception:
                sp["max_tokens"] = int(self.max_model_len - len(prompt_ids))
        # vLLM expects stop to be a list; rely on caller to provide correct shape
        sampling_params = SamplingParams(**sp)
        prompt = TokensPrompt(prompt_token_ids=prompt_ids)
        generator = self.engine.generate(prompt=prompt, sampling_params=sampling_params, request_id=request_id)

        # Get final response
        final_res: Optional[RequestOutput] = None
        async for output in generator:
            final_res = output
        assert final_res is not None

        response_str = final_res.outputs[0].text
        stop_reason = final_res.outputs[0].finish_reason

        return response_str, stop_reason
