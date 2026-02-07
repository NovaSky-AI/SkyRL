import aiohttp
from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
    InferenceEngineInput,
    InferenceEngineOutput,
)
from skyrl_train.weight_sync import WeightLoader, WeightUpdateRequest, BroadcastWeightUpdateRequest
from typing import List, Optional, Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from skyrl_train.weight_sync.transfer_strategy import WeightSyncInitInfo
import json
from transformers import PreTrainedTokenizerBase


class RemoteWeightLoader(WeightLoader):
    """Loads weights into remote inference engine via HTTP.

    This loader coordinates weight updates with remote inference servers
    (vLLM or SGLang) via their HTTP APIs.
    """

    def __init__(self, url: str, engine_backend: str) -> None:
        """Initialize the loader.

        Args:
            url: Base URL of the remote inference server.
            engine_backend: Backend type ("vllm" or "sglang").
        """
        self._url = url
        self._engine_backend = engine_backend

    async def init_communicator(self, init_info: "WeightSyncInitInfo") -> Dict[str, Any]:
        """Initialize the distributed process group for syncing weights.

        Args:
            init_info: WeightSyncInitInfo from the sender.

        Returns:
            Response from the remote server.

        Raises:
            ValueError: If init_info strategy is not BroadcastTransferStrategy (remote only supports broadcast).
        """
        from skyrl_train.weight_sync import BroadcastTransferStrategy

        if init_info.strategy_type() is not BroadcastTransferStrategy:
            raise ValueError(
                f"Remote inference engines only support BroadcastTransferStrategy, got: {init_info.strategy_type().__name__}"
            )

        async with aiohttp.ClientSession() as session:
            if self._engine_backend == "sglang":
                # SGLang native API - only send fields it expects
                async with session.post(
                    f"{self._url}/init_weights_update_group",
                    json={
                        "master_address": init_info.master_addr,
                        "master_port": init_info.master_port,
                        "rank_offset": init_info.rank_offset,
                        "world_size": init_info.world_size,
                        "group_name": init_info.group_name,
                        "backend": init_info.backend,
                    },
                ) as response:
                    return await response.json()
            elif self._engine_backend == "vllm":
                # vLLM - our custom API, pass entire init_info
                from dataclasses import asdict

                async with session.post(
                    f"{self._url}/init_weight_update_communicator",
                    json=asdict(init_info),
                ) as response:
                    return await response.json()
            else:
                raise ValueError(f"Invalid engine backend: {self._engine_backend}")

    async def load_weights(self, request: WeightUpdateRequest) -> Dict[str, Any]:
        """Load weights via HTTP to the remote inference server.

        Remote engines only support broadcast weight updates (no IPC).
        Each request should contain a single weight to update.

        Args:
            request: Weight update request.

        Returns:
            Response from the remote server.
        """
        async with aiohttp.ClientSession() as session:
            if self._engine_backend == "sglang":
                # SGLang native API expects singular name, dtype, shape
                resp = await session.post(
                    f"{self._url}/update_weights_from_distributed",
                    json={
                        "name": request.names[0],
                        "dtype": request.dtypes[0],
                        "shape": request.shapes[0],
                    },
                )
                return await resp.json()
            elif self._engine_backend == "vllm":
                # vLLM - our custom API, pass entire request
                from dataclasses import asdict

                resp = await session.post(
                    f"{self._url}/update_weights",
                    json=asdict(request),
                )
                return await resp.json()
            else:
                raise ValueError(f"Invalid engine backend: {self._engine_backend}")

    async def destroy_group(self) -> Dict[str, Any]:
        """Destroy the weights update group.

        Returns:
            Response from the remote server.
        """
        async with aiohttp.ClientSession() as session:
            resp = await session.post(f"{self._url}/destroy_weights_update_group")
            return await resp.json()


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
        pp_size: Optional[int] = None,
        dp_size: Optional[int] = None,
        ep_size: Optional[int] = None,
    ):
        """Initialize the InferenceEngine."""
        self.url = f"http://{url}"
        self.model_name = model_name
        self.engine_backend = engine_backend
        self._tp_size = tp_size
        self._pp_size = pp_size
        self._dp_size = dp_size
        self._ep_size = ep_size
        self.tokenizer = tokenizer

        # Create weight loader for coordinating weight updates
        self._weight_loader = RemoteWeightLoader(self.url, engine_backend)

    def tp_size(self) -> int:
        return self._tp_size

    def pp_size(self) -> int:
        return self._pp_size

    def dp_size(self) -> int:
        return self._dp_size

    def ep_size(self) -> int:
        return self._ep_size

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        # 1. Prepare inputs
        prompts = input_batch.get("prompts")
        prompt_token_ids: Optional[List[List[int]]] = input_batch.get("prompt_token_ids")
        request_sampling_params = input_batch.get("sampling_params")

        assert (
            prompts is None and prompt_token_ids is not None
        ), "RemoteInferenceEngine only accepts `prompt_token_ids`, not `prompts`."

        sampling_params = request_sampling_params if request_sampling_params is not None else {}
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

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        body = request_payload.get("json", {})
        # NOTE(Charlie): cannot reuse payload["headers"] since we are posting a new request.
        # Otherwise will lead to json decode error.
        headers = {"Content-Type": "application/json"}
        response = None
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
            request_url = f"{self.url}/v1/chat/completions"
            async with session.post(request_url, json=body, headers=headers) as resp:
                response = await resp.json()

        return response

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        body = request_payload.get("json", {})
        headers = {"Content-Type": "application/json"}
        response = None
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
            request_url = f"{self.url}/v1/completions"
            async with session.post(request_url, json=body, headers=headers) as resp:
                response = await resp.json()

        return response

    async def wake_up(self, *args: Any, **kwargs: Any):
        async with aiohttp.ClientSession() as session:
            resp = await session.post(f"{self.url}/wake_up", json={"tags": kwargs.get("tags", 1)})
            return await resp.json()

    async def sleep(self, *args: Any, **kwargs: Any):
        async with aiohttp.ClientSession() as session:
            # TODO(Charlie): this is vLLM's API, not SGLang (which uses tags). Fix when need to
            # support sleeping with remote engines.
            resp = await session.post(f"{self.url}/sleep", json={"level": kwargs.get("level", 1)})
            return await resp.json()

    async def init_weight_update_communicator(self, init_info: "WeightSyncInitInfo"):
        """Initialize the distributed process group for syncing weights.

        Args:
            init_info: WeightSyncInitInfo from the sender.

        Note: Remote engines only support broadcast strategy.
        """
        return await self._weight_loader.init_communicator(init_info)

    async def update_named_weights(self, request: WeightUpdateRequest):
        if not isinstance(request, BroadcastWeightUpdateRequest):
            raise ValueError(
                "Remote inference engines do not support CUDA IPC weight updates. Only local engines support IPC."
            )

        assert (
            len(request) == 1
        ), f"Remote inference engines support only requests with a single named weight at a time, got request with {len(request)} entries"

        return await self._weight_loader.load_weights(request)

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
        await self._weight_loader.destroy_group()

    async def abort_generation(self) -> None:
        raise NotImplementedError("Abort generation is not supported for remote inference engines.")


def create_remote_inference_engines(
    urls: List[str],
    model_name: str,
    engine_backend: str,
    tokenizer: PreTrainedTokenizerBase,
    tensor_parallel_size: Optional[int] = None,
    pipeline_parallel_size: Optional[int] = None,
    data_parallel_size: Optional[int] = None,
    expert_parallel_size: Optional[int] = None,
):
    return [
        RemoteInferenceEngine(
            url=url,
            model_name=model_name,
            tokenizer=tokenizer,
            engine_backend=engine_backend,
            tp_size=tensor_parallel_size,
            pp_size=pipeline_parallel_size,
            dp_size=data_parallel_size,
            ep_size=expert_parallel_size,
        )
        for url in urls
    ]
