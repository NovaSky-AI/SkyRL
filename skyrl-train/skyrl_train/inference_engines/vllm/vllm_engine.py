import os
from typing import List, Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from skyrl_train.weight_sync.transfer_strategy import WeightSyncInitInfo
from dataclasses import dataclass
from http import HTTPStatus
import ray
import torch
import asyncio
import vllm
from types import SimpleNamespace
from vllm import SamplingParams
from vllm.inputs import TokensPrompt

# vLLM API compatibility: nightly (0.14+) restructured module paths
# Try new paths first, fall back to old paths for stable releases
try:
    # vLLM nightly (0.14+) - restructured into subdirectories
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion
    from vllm.entrypoints.openai.models.serving import BaseModelPath, OpenAIServingModels
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
        ChatCompletionResponse,
    )
    from vllm.entrypoints.openai.completion.protocol import (
        CompletionRequest,
        CompletionResponse,
    )
    from vllm.entrypoints.openai.engine.protocol import ErrorResponse
except ImportError:
    # vLLM stable (<=0.13) - flat module structure
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
    from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
    from vllm.entrypoints.openai.protocol import (
        ChatCompletionRequest,
        ChatCompletionResponse,
        ErrorResponse,
        CompletionRequest,
        CompletionResponse,
    )

from vllm.lora.request import LoRARequest
from uuid import uuid4
import warnings
from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
    InferenceEngineInput,
    InferenceEngineOutput,
)
from skyrl_train.weight_sync import WeightLoader, WeightUpdateRequest
from skyrl_train.inference_engines.vllm.utils import pop_openai_kwargs
from loguru import logger
import time
from packaging import version


@dataclass
class Logprob:
    logprob: float
    rank: int
    token_id: str


def setup_envvars_for_vllm(kwargs, bundle_indices):
    noset_visible_devices = kwargs.pop("noset_visible_devices")
    if kwargs.get("distributed_executor_backend") == "ray":
        # a hack to make the script work.
        # stop ray from manipulating *_VISIBLE_DEVICES
        # at the top-level when the distributed_executor_backend is ray.
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ.pop("ROCR_VISIBLE_DEVICES", None)
        os.environ.pop("HIP_VISIBLE_DEVICES", None)
    elif noset_visible_devices:
        # We need to set CUDA_VISIBLE_DEVICES to the ray assigned GPU
        # when the distributed_executor_backend is not rayargs and
        # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])

    num_gpus = kwargs.pop("num_gpus")
    if bundle_indices is not None:
        os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(num_gpus)
        os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
        logger.info(f"creating LLM with bundle_indices={bundle_indices}")


class WorkerWrap:
    def test_rpc(self, *args, **kwargs):
        """Test RPC call to worker"""
        return args, kwargs

    def init_weight_update_communicator(self, init_info: bytes):
        """Init weight update communicator from init info.

        Args:
            init_info: Pickled bytes of WeightSyncInitInfo from the sender.
        """
        import pickle

        assert torch.distributed.is_initialized(), "default torch process group must be initialized"

        # Unpickle init_info to restore the original object type
        assert isinstance(init_info, bytes), f"Expected bytes, got {type(init_info).__name__}"
        init_info = pickle.loads(init_info)

        strategy_cls = init_info.strategy_type()

        if hasattr(self, "_weight_receiver") and self._weight_receiver is not None:
            # TODO(haochen): we should get rid of this flag and override existing receiver.
            if init_info.override_existing_receiver:
                self._weight_receiver.teardown()
                self._weight_receiver = None
            else:
                warnings.warn(
                    "Detected an existing weight receiver. "
                    "For overriding, use `generator.override_existing_update_group=enable`"
                )
                return

        self._weight_receiver = strategy_cls.create_receiver(init_info)

    def load_weights(self, request: bytes) -> None:
        """Load weights using the receiver.

        This method is called via collective_rpc from VLLMWeightLoader.

        Args:
            request: Pickled bytes of WeightUpdateRequest.
        """
        import pickle

        # Unpickle request to restore the original object type
        assert isinstance(request, bytes), f"Expected bytes, got {type(request).__name__}"
        request = pickle.loads(request)

        weight_list = []
        for name, tensor in self._weight_receiver.receive_weights(request):
            weight_list.append((name, tensor))

        self.model_runner.model.load_weights(weights=weight_list)

        for weight in weight_list:
            del weight

    # TODO (sumanthrh): Add destroy process group RPC as a atexit handler to Trainer code.
    def teardown_weight_receiver(self):
        if not hasattr(self, "_weight_receiver") or self._weight_receiver is None:
            warnings.warn("No weight receiver to teardown")
            return
        self._weight_receiver.teardown()


class BaseVLLMInferenceEngine(InferenceEngineInterface):
    """Base class containing shared logic between sync and async VLLM engines."""

    def __init__(self, *args, bundle_indices: list = None, **kwargs):
        setup_envvars_for_vllm(kwargs, bundle_indices)
        vllm_v1_disable_multiproc = kwargs.pop("vllm_v1_disable_multiproc", False)
        if vllm_v1_disable_multiproc or vllm.__version__ == "0.8.2":
            # https://github.com/vllm-project/vllm/blob/effc5d24fae10b29996256eb7a88668ff7941aed/examples/offline_inference/reproduciblity.py#L11
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        # Store common attributes
        self._tp_size = kwargs.get("tensor_parallel_size", 1)
        self._pp_size = kwargs.get("pipeline_parallel_size", 1)
        self._dp_size = kwargs.get("data_parallel_size", 1)
        self._is_lora = kwargs.get("enable_lora", False)

        # Let subclass create the appropriate engine
        self.llm = self._create_engine(*args, **kwargs)

        # Weight loader is created by subclass after engine initialization
        self._weight_loader = None

    def tp_size(self):
        return self._tp_size

    def pp_size(self):
        return self._pp_size

    def dp_size(self):
        return self._dp_size

    def _create_engine(self, *args, **kwargs):
        """Abstract method for subclasses to implement engine creation."""
        raise NotImplementedError("Subclasses must implement _create_engine")

    def _preprocess_prompts(self, input_batch: InferenceEngineInput):
        """Common prompt preprocessing logic."""
        prompts = input_batch.get("prompts")
        prompt_token_ids = input_batch.get("prompt_token_ids")
        request_sampling_params = input_batch.get("sampling_params")

        assert (
            prompts is None and prompt_token_ids is not None
        ), "VLLMInferenceEngine only accepts `prompt_token_ids`, not `prompts`."

        sampling_params = (
            SamplingParams(**request_sampling_params) if request_sampling_params is not None else SamplingParams()
        )

        return prompt_token_ids, sampling_params

    def _postprocess_outputs(self, outputs):
        """Common output processing logic."""
        responses: List[str] = []
        stop_reasons: List[str] = []
        response_ids: List[List[int]] = []
        response_logprobs: Optional[List[List[float]]] = []

        for output in outputs:
            # TODO(tgriggs): Support n>1 sampling.
            assert (
                len(output.outputs) == 1
            ), "Each prompt should have only one responses. n>1 sampling is supported by copying prompts."
            resp = output.outputs[0]
            responses.append(resp.text)
            stop_reasons.append(resp.finish_reason)
            response_ids.append(resp.token_ids)
            _logprobs = None
            if resp.logprobs:
                _logprobs = []
                for i, token_logprobs in enumerate(resp.logprobs):
                    token_logprobs: Dict[str, Logprob]
                    token_id = resp.token_ids[i]
                    logprob = token_logprobs[token_id].logprob
                    _logprobs.append(logprob)
                    del token_logprobs
            response_logprobs.append(_logprobs)

        if len(response_logprobs) and response_logprobs[0] is None:
            response_logprobs = None  # hack: assume uniform sampling params

        return InferenceEngineOutput(
            responses=responses,
            stop_reasons=stop_reasons,
            response_ids=response_ids,
            response_logprobs=response_logprobs,
        )

    def _get_engine(self):
        """Get the underlying engine for RPC calls."""
        return self.llm.engine if hasattr(self.llm, "engine") else self.llm

    def reset_prefix_cache(self):
        """Reset the prefix cache. Subclasses override for async version."""
        return self.llm.llm_engine.reset_prefix_cache()

    async def abort_generation(self) -> None:
        raise NotImplementedError("Abort generation is only supported for AsyncVLLMInferenceEngine.")


class VLLMInferenceEngine(BaseVLLMInferenceEngine):
    """Synchronous VLLM engine."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._weight_loader = VLLMWeightLoader(self.llm, is_async=False)

    def _create_engine(self, *args, **kwargs):
        # Pipeline parallelism requires AsyncLLMEngine
        if kwargs.get("pipeline_parallel_size", 1) > 1:
            raise ValueError(
                "Pipeline parallelism is only supported with AsyncVLLMInferenceEngine. "
                "Please set `generator.async_engine=true` in your config."
            )
        return vllm.LLM(*args, **kwargs)

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        prompt_token_ids, sampling_params = self._preprocess_prompts(input_batch)

        # Check if LoRA is enabled and create LoRA requests
        lora_requests = None
        if self._is_lora:
            lora_int_ids = list(self.llm.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                batch_size = len(prompt_token_ids)
                # dummy_lora_path for placeholder (actual loading done in add_lora())
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/dummy_lora_path")
                ] * batch_size

        outputs = await asyncio.to_thread(
            self.llm.generate,
            prompts=[TokensPrompt(prompt_token_ids=r) for r in prompt_token_ids],
            sampling_params=sampling_params,
            lora_request=lora_requests,
        )

        return self._postprocess_outputs(outputs)

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Only supported in AsyncVLLMInferenceEngine."""
        raise NotImplementedError("`chat_completion` is only supported in AsyncVLLMInferenceEngine.")

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Only supported in AsyncVLLMInferenceEngine."""
        raise NotImplementedError("`completion` is only supported in AsyncVLLMInferenceEngine.")

    async def wake_up(self, *args: Any, **kwargs: Any):
        await asyncio.to_thread(self.llm.wake_up, tags=kwargs.get("tags", None))

    async def sleep(self, *args: Any, **kwargs: Any):
        engine = self._get_engine().llm_engine
        output_processor = engine.output_processor
        if output_processor.has_unfinished_requests():
            logger.warning(
                "Calling sleep() with unfinished requests in vLLM engine. This is unexpected since all "
                "generation should be done before sleep() is called. Check for potential failures or "
                "dangling requests in your Generator/Env. Aborting all unfinished requests."
            )
            unfinished_request_ids = list(output_processor.request_states.keys())
            await asyncio.to_thread(engine.abort_request, unfinished_request_ids)

        level = 1 if self._is_lora else kwargs.get("level", 2)
        await asyncio.to_thread(self.llm.sleep, level=level)

    async def init_weight_update_communicator(self, init_info: "WeightSyncInitInfo"):
        import pickle

        engine = self._get_engine()
        # Pickle the init_info to preserve type through collective_rpc
        pickled_init_info = pickle.dumps(init_info)
        return await asyncio.to_thread(
            engine.collective_rpc,
            "init_weight_update_communicator",
            args=(pickled_init_info,),
        )

    async def _load_lora_from_disk(self, lora_path: str):
        """Load LoRA adapters from disk using vLLM's native add_lora method."""
        lora_id = int(time.time_ns() % 0x7FFFFFFF)
        lora_request = LoRARequest(lora_name=f"{lora_id}", lora_int_id=lora_id, lora_path=lora_path)
        result = self.llm.llm_engine.add_lora(lora_request)
        return result

    async def update_named_weights(self, request: WeightUpdateRequest):
        from skyrl_train.weight_sync import LoraLoadRequest

        # Handle LoRA disk loading request
        if isinstance(request, LoraLoadRequest):
            return await self._load_lora_from_disk(request.lora_path)

        if not len(request):
            raise ValueError("Weight update request must not be empty")

        # Use the weight loader to coordinate weight transfer
        return await self._weight_loader.load_weights(request)

    async def teardown(self):
        await self._teardown_weight_receiver()

    async def reset_prefix_cache(self):
        return await asyncio.to_thread(self.llm.llm_engine.reset_prefix_cache)

    async def _teardown_weight_receiver(self):
        engine = self._get_engine()
        return await asyncio.to_thread(engine.collective_rpc, "teardown_weight_receiver")


class AsyncVLLMInferenceEngine(BaseVLLMInferenceEngine):
    """Asynchronous VLLM engine."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._weight_loader = VLLMWeightLoader(self.llm, is_async=True)

    def _create_engine(self, *args, **kwargs):
        openai_kwargs = pop_openai_kwargs(kwargs)
        # TODO (erictang000): potentially enable log requests for a debugging mode
        if version.parse(vllm.__version__) >= version.parse("0.10.0"):
            engine_args = vllm.AsyncEngineArgs(enable_log_requests=False, **kwargs)
        else:
            engine_args = vllm.AsyncEngineArgs(disable_log_requests=True, **kwargs)
        engine = vllm.AsyncLLMEngine.from_engine_args(engine_args)

        # Adapted from https://github.com/volcengine/verl/blob/e90f18c40aa639cd25092b78a5ff7e2d2508c088/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L327
        model_config = engine.model_config
        model_path = kwargs.get("model")
        # TODO(Charlie): add a config similar to vllm's `served_model_name`. See https://github.com/NovaSky-AI/SkyRL/pull/238#discussion_r2326561295
        model_name = model_path

        base_model_paths = [BaseModelPath(name=model_name, model_path=model_path)]

        # vllm >= 0.11.2 removed model_config from OpenAI serving APIs
        is_new_api = version.parse(vllm.__version__) >= version.parse("0.11.2")
        legacy_kwargs = {}
        if is_new_api:
            models = OpenAIServingModels(engine, base_model_paths)
        else:
            models = OpenAIServingModels(engine, model_config, base_model_paths)
            legacy_kwargs["model_config"] = model_config

        # TODO(Charlie): revisit kwargs `enable_auto_tools` and `tool_parser` when we need to
        # support OAI-style tool calling; and `request_logger` for better debugging.
        self.openai_serving_chat = OpenAIServingChat(
            engine_client=engine,
            models=models,
            response_role="assistant",
            request_logger=None,
            chat_template=openai_kwargs.pop("chat_template", None),  # used to template /chat/completions requests
            chat_template_content_format="auto",
            **legacy_kwargs,
            **openai_kwargs,
        )

        # TODO(Charlie): revisit kwargs `return_tokens_as_token_ids`,
        # `enable_prompt_tokens_details`, `enable_force_include_usage`.
        self.openai_serving_completion = OpenAIServingCompletion(
            engine_client=engine,
            models=models,
            request_logger=None,
            **legacy_kwargs,
        )
        return engine

    async def _load_lora_from_disk(self, lora_path: str):
        """Load LoRA adapters from disk using vLLM's native add_lora method."""
        lora_id = int(time.time_ns() % 0x7FFFFFFF)
        lora_request = LoRARequest(lora_name=f"{lora_id}", lora_int_id=lora_id, lora_path=lora_path)
        result = await self.llm.add_lora(lora_request)
        return result

    async def _collect_outputs(self, prompt_token_ids, request_id: str, sampling_params: SamplingParams):
        """Collect outputs for a single prompt."""
        # Check if LoRA is enabled and create LoRA request
        final_output = None
        lora_request = None

        if self._is_lora:
            lora_int_ids = list(await self.llm.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                # dummy_lora_path for placeholder (actual loading done in add_lora())
                lora_request = LoRARequest(
                    lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/dummy_lora_path"
                )

        async for request_output in self.llm.generate(
            prompt=TokensPrompt(prompt_token_ids=prompt_token_ids),
            sampling_params=sampling_params,
            request_id=request_id,
            lora_request=lora_request,
        ):
            final_output = request_output

        return final_output

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        """Generate responses using vLLM's async engine."""
        prompt_token_ids, sampling_params = self._preprocess_prompts(input_batch)

        tasks = []
        for prompt in prompt_token_ids:
            # Schedule the collection of outputs for each prompt.
            # Avoid duplicate request_ids
            request_id = str(uuid4().hex)
            task = asyncio.create_task(self._collect_outputs(prompt, request_id, sampling_params))
            tasks.append(task)
        outputs = await asyncio.gather(*tasks)

        return self._postprocess_outputs(outputs)

    async def wake_up(self, *args: Any, **kwargs: Any):
        await self.llm.wake_up(tags=kwargs.get("tags", None))

    async def sleep(self, *args: Any, **kwargs: Any):
        engine = self._get_engine()
        output_processor = engine.output_processor
        # make sure that the engine is alive
        engine.engine_core.ensure_alive()
        if output_processor.has_unfinished_requests():
            logger.warning(
                "Calling sleep() with unfinished requests in vLLM engine. This is unexpected since all "
                "generation should be done before sleep() is called. Check for potential failures or "
                "dangling requests in your Generator/Env. Aborting all unfinished requests."
            )
            unfinished_request_ids = list(output_processor.request_states.keys())
            await engine.abort(unfinished_request_ids)

        # TODO(team): remove once vllm fixes this
        # otherwise waking it up will output gibberish: https://github.com/vllm-project/vllm/issues/17103
        await self.reset_prefix_cache()
        level = 1 if self._is_lora else kwargs.get("level", 2)
        await self.llm.sleep(level=level)

    async def init_weight_update_communicator(self, init_info: "WeightSyncInitInfo"):
        import pickle

        engine = self._get_engine()
        # Pickle the init_info to preserve type through collective_rpc
        pickled_init_info = pickle.dumps(init_info)
        return await engine.collective_rpc(
            "init_weight_update_communicator",
            args=(pickled_init_info,),
        )

    async def update_named_weights(self, request: WeightUpdateRequest):
        from skyrl_train.weight_sync import LoraLoadRequest

        # Check for LoRA disk loading request
        if isinstance(request, LoraLoadRequest):
            return await self._load_lora_from_disk(request.lora_path)

        if not len(request):
            raise ValueError("Weight update request must not be empty")

        # Use the weight loader to coordinate weight transfer
        return await self._weight_loader.load_weights(request)

    async def teardown(self):
        await self._teardown_weight_receiver()

    async def reset_prefix_cache(self):
        engine = self._get_engine()
        await engine.reset_prefix_cache()

    async def _teardown_weight_receiver(self):
        engine = self._get_engine()
        return await engine.collective_rpc("teardown_weight_receiver")

    # ----------------------------------------
    # Methods for handling OpenAI API requests
    # ----------------------------------------

    async def _handle_openai_request(self, request_payload: Dict[str, Any], endpoint: str) -> Dict[str, Any]:
        """Handle OpenAI API request."""
        assert endpoint in ["/chat/completions", "/completions"]

        body = request_payload.get("json", {})
        headers = request_payload.get("headers", {})

        # 1. Build request
        try:
            if endpoint == "/chat/completions":
                request = ChatCompletionRequest(**body)
            else:
                request = CompletionRequest(**body)
            assert request.stream is False, "Streaming is not supported in SkyRL yet, please set stream to False."
        except Exception as e:
            if version.parse(vllm.__version__) >= version.parse("0.10.0"):
                try:
                    from vllm.entrypoints.openai.engine.protocol import ErrorInfo
                except ImportError:
                    from vllm.entrypoints.openai.protocol import ErrorInfo

                return ErrorResponse(
                    error=ErrorInfo(
                        message=str(e),
                        type=HTTPStatus.BAD_REQUEST.phrase,
                        code=HTTPStatus.BAD_REQUEST.value,
                    ),
                ).model_dump()
            else:
                return ErrorResponse(
                    message=str(e),
                    type=HTTPStatus.BAD_REQUEST.phrase,
                    code=HTTPStatus.BAD_REQUEST.value,
                ).model_dump()

        # 2. Call vllm engine
        try:
            # Create a minimal request-like object with attributes used by vLLM
            minimal_request = _MinimalRequest(headers)
            if endpoint == "/chat/completions":
                generator = await self.openai_serving_chat.create_chat_completion(request, minimal_request)
                assert isinstance(generator, (ChatCompletionResponse, ErrorResponse))
            else:
                generator = await self.openai_serving_completion.create_completion(request, minimal_request)
                assert isinstance(generator, (CompletionResponse, ErrorResponse))
            return generator.model_dump()

        except Exception as e:
            # Handle it here so we can surface the error from a ray worker.
            if version.parse(vllm.__version__) >= version.parse("0.10.0"):
                try:
                    from vllm.entrypoints.openai.engine.protocol import ErrorInfo
                except ImportError:
                    from vllm.entrypoints.openai.protocol import ErrorInfo

                return ErrorResponse(
                    error=ErrorInfo(
                        message=str(e),
                        type=HTTPStatus.INTERNAL_SERVER_ERROR.phrase,
                        code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                    ),
                ).model_dump()
            else:
                return ErrorResponse(
                    message=str(e),
                    type=HTTPStatus.INTERNAL_SERVER_ERROR.phrase,
                    code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
                ).model_dump()

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """OpenAI-compatible HTTP endpoint for handling `/chat/completions` in Python vLLM engine.

        Accepts a JSON-serializable payload: {"json": <request-body>, "headers": <headers-dict>}.
        Constructs a minimal request-like object for vLLM's openai_serving_chat.
        Returns a plain dict, either a ChatCompletionResponse or an ErrorResponse, both defined
        in vllm.entrypoints.openai.protocol.
        """
        return await self._handle_openai_request(request_payload, endpoint="/chat/completions")

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """OpenAI-compatible HTTP endpoint for handling `/completions` in Python vLLM engine.

        Accepts a JSON-serializable payload: {"json": <request-body>, "headers": <headers-dict>}.
        Constructs a minimal request-like object for vLLM's openai_serving_completion.
        Returns a plain dict, either a CompletionResponse or an ErrorResponse, both defined
        in vllm.entrypoints.openai.protocol.
        """
        return await self._handle_openai_request(request_payload, endpoint="/completions")

    async def abort_generation(self) -> None:
        """
        Abort all running and waiting requests, which make the ongoing requests return the
        already-generated tokens with a stop_reason of "abort".
        """
        engine = self._get_engine()
        # Collect all request IDs currently tracked by the scheduler/output processor
        unfinished_request_ids = list(engine.output_processor.request_states.keys())
        if unfinished_request_ids:
            await engine.abort(unfinished_request_ids)
        await engine.reset_prefix_cache()  # avoid KV-cache pollution
        logger.info(f"abort_generation() finished, aborted {len(unfinished_request_ids)} requests")


class _MinimalRequest:
    """
    Minimal request-like object for vLLM's openai_serving_chat and openai_serving_completion.

    We cannot use the original user Request object because it cannot be serialized and hence
    cannot be a ray method argument. Instead we take the original request's headers and
    reconstruct an instance of _MinimalRequest to mimic the FastAPI Request object.

    The fields depend on what vLLM accesses internally.
    """

    def __init__(self, headers):
        self.headers = headers  # Expect a mapping with .get support
        self.state = SimpleNamespace()  # vLLM sets raw_request.state.request_metadata


class VLLMWeightLoader(WeightLoader):
    """Loads weights into vLLM engine, managing RPC coordination.

    This loader encapsulates the collective_rpc calls to workers.
    Workers create the appropriate receiver locally for the actual weight transfer.
    """

    def __init__(self, engine: Any, is_async: bool = False) -> None:
        """Initialize the loader.

        Args:
            engine: The vLLM engine (LLM or AsyncLLMEngine).
            is_async: Whether this is for AsyncVLLMInferenceEngine.
        """
        self._engine = engine.engine if hasattr(engine, "engine") else engine
        self._is_async = is_async

    async def load_weights(self, request: WeightUpdateRequest) -> None:
        """Load weights by coordinating RPC to workers.

        Sends the request to workers via collective_rpc. Workers create
        the receiver locally and use it to receive and load weights.

        Args:
            request: Weight update request.
        """
        import pickle

        # Pickle the request to preserve type through collective_rpc
        pickled_request = pickle.dumps(request)

        if self._is_async:
            await self._engine.collective_rpc(
                "load_weights",
                args=(pickled_request,),
            )
        else:
            await asyncio.to_thread(
                self._engine.collective_rpc,
                "load_weights",
                args=(pickled_request,),
            )


VLLMRayActor = ray.remote(VLLMInferenceEngine)
AsyncVLLMRayActor = ray.remote(AsyncVLLMInferenceEngine)
