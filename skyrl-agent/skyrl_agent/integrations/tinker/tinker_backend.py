from typing import Any, List
from ..base import AsyncInferBackend, GeneratorOutput, GeneratorInput


class TinkerBackend(AsyncInferBackend):
    def __init__(self, infer_engine, tokenizer: Any = None, cfg: Any = None):
        self.client = infer_engine
        self.tokenizer = tokenizer

    async def async_generate_prompts(self, prompts: Any, sampling_params: Any, **kwargs) -> List[str]:
        raise NotImplementedError

    async def async_generate_ids(self, input_ids: List[int], sampling_params: Any, **kwargs) -> List[str]:
        from tinker.types import ModelInput, SamplingParams

        prompt_ids = ModelInput.from_ints(tokens=input_ids)
        tinker_sampling_params = SamplingParams(**sampling_params)
        output = await self.client.sample_async(
            prompt=prompt_ids, num_samples=1, sampling_params=tinker_sampling_params
        )
        output_tokens = output.sequences[0].tokens
        # get raw content and skip special tokens like <im_end>
        message = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
        # check if finish_reason is stop
        # currently check if last token is tokenizer.eos_token
        if output_tokens[-1] == self.tokenizer.eos_token_id:
            finish_reason = "stop"
        else:
            finish_reason = "length"

        meta_info = {
            "output_tokens": output_tokens,
            "finish_reason": finish_reason,
            "logprobs": output.sequences[0].logprobs,
        }

        return message, meta_info


class TinkerGeneratorOutput(GeneratorOutput):
    def __init__(self, result: Any):
        self.result = result


class TinkerGeneratorInput(GeneratorInput):
    def __init__(self, input_batch: Any):
        self.input_batch = input_batch
