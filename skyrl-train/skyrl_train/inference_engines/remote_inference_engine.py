import aiohttp
from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
    InferenceEngineInput,
    InferenceEngineOutput,
    NamedWeightsUpdateRequest,
)
from typing import List, Optional, Dict, Any
import json
from transformers import PreTrainedTokenizerBase


class RemoteInferenceEngine(InferenceEngineInterface):
    """
    Lightweight client to call into an OpenAI-compatible server over HTTP with a customizable backend.
    """

    def __init__(
        self,
        url: str,
        model_name: str,
        engine_backend: str,
        tokenizer: PreTrainedTokenizerBase,
        tp_size: Optional[int] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the InferenceEngine."""
        self.url = f"http://{url}"
        self.model_name = model_name
        self.engine_backend = engine_backend
        self.tp_size = tp_size
        self.sampling_params = sampling_params if sampling_params is not None else {}
        self.tokenizer = tokenizer

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        # 1. Prepare inputs
        prompts = input_batch.get("prompts")
        prompt_token_ids: Optional[List[List[int]]] = input_batch.get("prompt_token_ids")
        request_sampling_params = input_batch.get("sampling_params")

        # For token-in-token-out, convert prompts to token ids if needed
        if (prompts is None and prompt_token_ids is None) or (prompts is not None and prompt_token_ids is not None):
            raise ValueError("Either `prompts` or `prompt_token_ids` must be provided, but not both.")
        if prompt_token_ids is None:
            prompt_token_ids = self.tokenizer.apply_chat_template(
                prompts,
                add_generation_prompt=True,
                add_special_tokens=False,
                return_dict=True,
                tokenize=True,
            )["input_ids"]

        sampling_params = request_sampling_params if request_sampling_params is not None else self.sampling_params
        if "n" in sampling_params and sampling_params["n"] > 1:
            raise ValueError(
                "n is not supported yet for remote inference engines. "
                "You can set `config.generator.n_samples_per_prompt` instead."
            )

        # 2. Send a batched request to the server
        response = None
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
            headers = {"Content-Type": "application/json"}
            payload = {}
            request_url = ""
            if self.engine_backend == "vllm":
                # vLLM does not support /generate, use /completions instead. It supports batch generation.
                payload = sampling_params.copy()
                payload["model"] = self.model_name
                payload["prompt"] = prompt_token_ids
                request_url = f"{self.url}/v1/completions"
            elif self.engine_backend == "sglang":
                # SGLang supports /generate, works exactly like its Python `async_generate()` method
                # and can do batch generation.
                payload = {
                    "input_ids": prompt_token_ids,
                    "sampling_params": sampling_params,
                }
                request_url = f"{self.url}/generate"
            else:
                raise ValueError(f"Invalid engine backend: {self.engine_backend}")
            async with session.post(request_url, json=payload, headers=headers) as resp:
                response = await resp.json()

        # 3. Parse outputs
        outputs = []
        output_ids = []
        finish_reasons = []

        if self.engine_backend == "vllm":
            for i, choice in enumerate(response.get("choices", [])):
                # Since n=1, index i represents the output for `prompt[i]`
                assert choice["index"] == i, "Expect the choices to be ordered by index."
                text = choice["text"]
                outputs.append(text)
                finish_reasons.append(choice["finish_reason"])
                # TODO(Charlie): this is not token-in-token-out because vLLM does not support
                # returning token IDs via HTTP requests. Fix after this vLLM PR is merged:
                # https://github.com/vllm-project/vllm/pull/22587
                output_ids.append(self.tokenizer.encode(text, add_special_tokens=False))
        elif self.engine_backend == "sglang":
            # since prompt_token_ids is a list of lists, response is a list of dicts
            for output in response:
                cur_output_ids = output["output_ids"]
                output_ids.append(cur_output_ids)
                # SGLang only returns tokens not text when skip_tokenizer_init is True, so
                # we manually decode it.
                outputs.append(self.tokenizer.decode(cur_output_ids, skip_special_tokens=True))
                finish_reasons.append(output["meta_info"]["finish_reason"]["type"])
        else:
            raise ValueError(f"Invalid engine backend: {self.engine_backend}")

        return InferenceEngineOutput(
            responses=outputs, stop_reasons=finish_reasons, response_ids=output_ids, response_logprobs=None
        )

    async def wake_up(self, *args: Any, **kwargs: Any):
        async with aiohttp.ClientSession() as session:
            resp = await session.post(f"{self.url}/wake_up", json={"tags": kwargs.get("tags", 1)})
            return await resp.json()

    async def sleep(self, *args: Any, **kwargs: Any):
        async with aiohttp.ClientSession() as session:
            resp = await session.post(f"{self.url}/sleep", json={"level": kwargs.get("level", 1)})
            return await resp.json()

    async def init_weight_update_communicator(
        self, master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing: bool = False
    ):
        """
        Initialize the distributed process group for syncing weights.
        """

        path = "/init_weights_update_group" if self.engine_backend == "sglang" else "/init_weight_update_communicator"
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.url}{path}",
                json={
                    "master_address": master_addr,
                    "master_port": master_port,
                    "rank_offset": rank_offset,
                    "world_size": world_size,
                    "group_name": group_name,
                    "backend": backend,
                    "override_existing": override_existing,
                },
            ) as response:
                return await response.json()

    async def update_named_weights(self, request: NamedWeightsUpdateRequest):
        if "names" not in request:
            raise ValueError(f"Expected update weight request with 'names' entry, got keys: {request.keys()}")

        assert (
            len(request["names"]) == 1
        ), f"Remote inference engines support only requests with a single named weight at a time , got request with {len(request['names'])} entries"

        if request.get("extras") and "ipc_handles" in request["extras"][0]:
            raise ValueError(
                "Remote inference engines do not support CUDA IPC weight updates. Only local engines support IPC."
            )
        if self.engine_backend == "vllm":
            weight_update_method = "update_weights"
        elif self.engine_backend == "sglang":
            weight_update_method = "update_weights_from_distributed"
        else:
            raise ValueError(f"Invalid engine backend: {self.engine_backend}")

        async with aiohttp.ClientSession() as session:
            name = request["names"][0]
            dtype = request["dtypes"][0]
            shape = request["shapes"][0]

            resp = await session.post(
                f"{self.url}/{weight_update_method}",
                json={
                    "name": name,
                    "dtype": dtype,
                    "shape": shape,
                },
            )
            return await resp.json()

    # TODO(tgriggs): Come up with a (more) elegant way to handle text or json responses, and test it and handle errors.
    async def reset_prefix_cache(self):
        if self.engine_backend == "vllm":
            reset_prefix_cache_method = "reset_prefix_cache"
        elif self.engine_backend == "sglang":
            reset_prefix_cache_method = "flush_cache"
        else:
            raise ValueError(f"Invalid engine backend: {self.engine_backend}")

        async with aiohttp.ClientSession() as session:
            resp = await session.post(f"{self.url}/{reset_prefix_cache_method}")
            text = await resp.text()

        # First try to parse it as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # If invalid JSON, return raw text plus status
            return {
                "status": resp.status,
                "body": text,
            }

    async def teardown(self):
        await self._destroy_weights_update_group()

    async def _destroy_weights_update_group(self):
        async with aiohttp.ClientSession() as session:
            resp = await session.post(f"{self.url}/destroy_weights_update_group")
            return await resp.json()


def create_remote_inference_engines(
    urls: List[str],
    model_name: str,
    engine_backend: str,
    tokenizer: PreTrainedTokenizerBase,
    tensor_parallel_size: Optional[int] = None,
    sampling_params: Optional[Dict[str, Any]] = None,
):
    return [
        RemoteInferenceEngine(
            url=url,
            model_name=model_name,
            tokenizer=tokenizer,
            engine_backend=engine_backend,
            tp_size=tensor_parallel_size,
            sampling_params=sampling_params,
        )
        for url in urls
    ]
