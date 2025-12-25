"""SGLang inference engine implementation."""

import torch
import os
from typing import List, Optional, Dict, Any
import ray
from loguru import logger
import multiprocessing as mp

import sglang.srt.entrypoints.engine
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.utils import (
    assert_pkg_version,
    is_cuda,
    maybe_set_triton_cache_manager,
    set_prometheus_multiproc_dir,
    set_ulimit,
    MultiprocessingSerializer,
)
from sglang.srt.managers.tokenizer_manager import (
    UpdateWeightsFromTensorReqInput,
    UpdateWeightsFromDistributedReqInput,
    InitWeightsUpdateGroupReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
)
from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
    InferenceEngineInput,
    InferenceEngineOutput,
)
from skyrl_train.weight_sync import (
    WeightLoader,
    WeightUpdateRequest,
    CudaIpcWeightUpdateRequest,
    BroadcastWeightUpdateRequest,
)


# Patch SGLang's _set_envs_and_config to avoid signal handler issues in Ray actors
# Based on VERL's solution: https://github.com/sgl-project/sglang/issues/6723
# https://github.com/volcengine/verl/blob/v0.4.1/verl/workers/rollout/sglang_rollout/sglang_rollout.py#L85
def _patched_set_envs_and_config(server_args):
    """Patched version of SGLang's _set_envs_and_config that removes signal handler registration."""
    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = str(int(getattr(server_args, "enable_nccl_nvls", False)))
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"

    # Set prometheus env vars
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()

    # Set ulimit
    set_ulimit()

    # Fix triton bugs
    if server_args.tp_size * server_args.dp_size > 1:
        # FIXME: remove this after https://github.com/triton-lang/triton/pull/4295 is used as a dependency.
        maybe_set_triton_cache_manager()

    # Check flashinfer version
    if server_args.attention_backend == "flashinfer":
        assert_pkg_version(
            "flashinfer_python",
            "0.2.5",
            "Please uninstall the old version and reinstall the latest version by following the instructions at https://docs.flashinfer.ai/installation.html.",
        )
    if is_cuda():
        assert_pkg_version(
            "sgl-kernel",
            "0.1.1",
            "Please reinstall the latest version with `pip install sgl-kernel --force-reinstall`",
        )

    # Set mp start method
    mp.set_start_method("spawn", force=True)

    # We do NOT register signal handlers here to avoid Ray actor issues
    # Original SGLang code had: signal.signal(signal.SIGCHLD, sigchld_handler)
    # But this fails in Ray actors since signal handlers only work in main thread


# Apply the patch
sglang.srt.entrypoints.engine._set_envs_and_config = _patched_set_envs_and_config


# TODO(charlie): duplicate of setup_envvars_for_vllm, is it needed?
def setup_envvars_for_sglang(kwargs, bundle_indices):
    distributed_executor_backend = kwargs.pop("distributed_executor_backend", None)
    noset_visible_devices = kwargs.pop("noset_visible_devices", None)
    if distributed_executor_backend == "ray":
        # a hack to make the script work.
        # stop ray from manipulating *_VISIBLE_DEVICES
        # at the top-level when the distributed_executor_backend is ray.
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ.pop("ROCR_VISIBLE_DEVICES", None)
        os.environ.pop("HIP_VISIBLE_DEVICES", None)
        pass
    elif noset_visible_devices:
        # We need to set CUDA_VISIBLE_DEVICES to the ray assigned GPU
        # when the distributed_executor_backend is not rayargs and
        # RAY_EXPERIMENTAL_NOSET_*_VISIBLE_DEVICES is set.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])


def sglang_custom_weight_loader(model, named_tensors):
    """Custom weight loader for SGLang that handles CUDA IPC.

    This function is called by SGLang's model runner to load weights.
    It reconstructs tensors from CudaIpcWeightUpdateRequest using CUDA IPC handles.

    Note: Broadcast path is not handled here.
    Because unlike vLLM where we control WorkerWrap and can store the
    process group there, SGLang's custom_weight_loader only receives (model, tensors).
    The process group (_model_update_group) is stored in model_runner, which is not
    accessible from the model object. We also cannot create the group lazily inside
    custom_weight_loader because torch.distributed group creation requires coordination
    (all processes must join at the same time), and by the time custom_weight_loader
    is called, the training side has already completed its init. Therefore, broadcast
    uses SGLang's native update_weights_from_distributed API which has internal access
    to the process group.
    """
    from skyrl_train.weight_sync import CudaIpcWeightUpdateRequest, CudaIpcWeightTransferReceiver

    # Extract tensor name and data
    name, tensor = named_tensors[0]
    if name != "ipc_request":
        raise ValueError(f"Expected tensor name 'ipc_request', got: {name}")

    # Deserialize request from tensor
    request = CudaIpcWeightUpdateRequest.deserialize(tensor.cpu().numpy().tobytes())

    # Get model info and create receiver
    model_dtype = next(model.parameters()).dtype
    receiver = CudaIpcWeightTransferReceiver(model_dtype)

    # Receive weights via IPC
    weights_to_load = list(receiver.receive_weights(request))
    model.load_weights(weights_to_load)


CUSTOM_WEIGHT_LOADER_PATH = "skyrl_train.inference_engines.sglang.sglang_engine.sglang_custom_weight_loader"


class SGLangWeightLoader(WeightLoader):
    """Loads weights into SGLang engine, managing weight transfer coordination.

    This loader encapsulates the SGLang-specific weight loading logic for both
    IPC and broadcast transfer paths:
    - IPC: Uses update_weights_from_tensor with our custom_weight_loader
    - Broadcast: Uses SGLang's native update_weights_from_distributed API
    """

    def __init__(self, engine: Any, tp_size: int) -> None:
        """Initialize the loader.

        Args:
            engine: The SGLang engine.
            tp_size: Tensor parallel size.
        """
        self._engine = engine
        self._tp_size = tp_size

    async def init_communicator(self, init_info) -> None:
        """Initialize the process group for broadcast weight sync.

        This is only needed for the broadcast path. IPC path does not require
        a process group since it uses CUDA IPC handles directly.

        Args:
            init_info: WeightSyncInitInfo from the sender.
        """
        from skyrl_train.weight_sync import BroadcastTransferStrategy

        if init_info.strategy_type() is BroadcastTransferStrategy:
            obj = InitWeightsUpdateGroupReqInput(
                master_address=init_info.master_addr,
                master_port=init_info.master_port,
                rank_offset=init_info.rank_offset,
                world_size=init_info.world_size,
                group_name=init_info.group_name,
                backend=init_info.backend,
            )
            # NOTE(charlie): Call the async method on tokenizer_manager directly to avoid event loop
            # conflicts. Same underlying implementation: https://github.com/sgl-project/sglang/blob/v0.4.8.post1/python/sglang/srt/model_executor/model_runner.py#L689
            success, message = await self._engine.tokenizer_manager.init_weights_update_group(obj, None)
            if not success:
                raise RuntimeError(f"Failed to initialize weight update group: {message}")

    async def load_weights(self, request: WeightUpdateRequest) -> None:
        """Load weights by coordinating with SGLang's weight update APIs.

        Args:
            request: Weight update request.
        """
        if isinstance(request, CudaIpcWeightUpdateRequest):
            await self._load_via_ipc(request)
        elif isinstance(request, BroadcastWeightUpdateRequest):
            await self._load_via_broadcast(request)
        else:
            raise TypeError(f"Unknown request type: {type(request).__name__}")

    async def _load_via_ipc(self, request: CudaIpcWeightUpdateRequest) -> None:
        """Load weights via CUDA IPC using custom weight loader.

        Uses SGLangWeightTransferReceiver internally to receive weights
        from IPC handles.
        """
        tensor_array = torch.frombuffer(bytearray(request.serialize()), dtype=torch.uint8)

        # Use SGLang's API to update weights with our custom loader
        request_tensor = [("ipc_request", tensor_array)]
        obj = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[
                MultiprocessingSerializer.serialize(request_tensor) for _ in range(self._tp_size)
            ],
            load_format=CUSTOM_WEIGHT_LOADER_PATH,
            flush_cache=False,  # TODO(charlie): flush cache on last weight update?
        )

        success, message = await self._engine.tokenizer_manager.update_weights_from_tensor(obj, None)
        if not success:
            raise RuntimeError(f"IPC weight update failed: {message}")

    async def _load_via_broadcast(self, request: BroadcastWeightUpdateRequest) -> None:
        """Load weights via torch.distributed broadcast.

        Uses SGLang's native update_weights_from_distributed API which internally
        uses the process group created during init_weights_update_group.
        """
        assert len(request) == 1, f"Broadcast only supports a single weight at a time, got {len(request)} entries"

        obj = UpdateWeightsFromDistributedReqInput(
            name=request.names[0], dtype=request.dtypes[0], shape=request.shapes[0]
        )

        success, message = await self._engine.tokenizer_manager.update_weights_from_distributed(obj, None)
        if not success:
            raise RuntimeError(f"Broadcast weight update failed: {message}")


class SGLangInferenceEngine(InferenceEngineInterface):
    """SGLang inference engine that implements InferenceEngineInterface."""

    def __init__(self, *args, bundle_indices: Optional[List[int]] = None, **kwargs):
        setup_envvars_for_sglang(kwargs, bundle_indices)

        # Store common attributes
        self._tp_size = kwargs.get("tp_size", 1)
        if self._tp_size > 1:
            raise ValueError(
                "As of now, we don't support tensor parallel inference engine with SGLang. "
                "Please set `inference_engine_tensor_parallel_size` to 1."
            )
        self.tokenizer = kwargs.pop("tokenizer", None)

        # Unused kwargs
        _ = kwargs.pop("num_gpus", 1)

        # Add custom weight loader
        kwargs["custom_weight_loader"] = CUSTOM_WEIGHT_LOADER_PATH

        # Always use token-in-token-out SGLang engine
        # NOTE(Charlie): unlike vLLM, SGLang cannot do token-in-token-out and
        # token-in-text-out in the same engine config.
        kwargs["skip_tokenizer_init"] = True

        # Create the SGLang engine (signal handler issue is now fixed by patching)
        self.engine = Engine(**kwargs)
        logger.info(f"Created SGLang engine with kwargs: {kwargs}")

        # Create weight loader for coordinating weight updates
        self._weight_loader = SGLangWeightLoader(self.engine, self._tp_size)

    def tp_size(self):
        return self._tp_size

    def pp_size(self):
        # Pipeline parallelism not supported for SGLang
        return 1

    def dp_size(self):
        # TODO(tgriggs): EP/DP not yet supported for SGLang
        return 1

    def _preprocess_prompts(self, input_batch: InferenceEngineInput):
        """Preprocess prompts for SGLang generation."""
        prompts = input_batch.get("prompts")
        prompt_token_ids = input_batch.get("prompt_token_ids")
        request_sampling_params = input_batch.get("sampling_params")

        assert (
            prompts is None and prompt_token_ids is not None
        ), "SGLangInferenceEngine only accepts `prompt_token_ids`, not `prompts`."

        # Use request sampling params if provided.
        sampling_params = request_sampling_params if request_sampling_params is not None else {}

        return prompt_token_ids, sampling_params

    def _postprocess_outputs(self, outputs):
        """Process SGLang outputs to match expected format."""
        responses: List[str] = []
        stop_reasons: List[str] = []
        response_ids: List[List[int]] = []

        for output in outputs:
            response_ids.append(output["output_ids"])
            responses.append(self.tokenizer.decode(output["output_ids"], skip_special_tokens=True))
            stop_reasons.append(output["meta_info"]["finish_reason"]["type"])

        return InferenceEngineOutput(
            responses=responses,
            response_ids=response_ids,
            stop_reasons=stop_reasons,
            response_logprobs=None,
        )

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        """Generate responses using SGLang engine."""
        token_ids_prompts, sampling_params = self._preprocess_prompts(input_batch)
        outputs = await self.engine.async_generate(input_ids=token_ids_prompts, sampling_params=sampling_params)
        return self._postprocess_outputs(outputs)

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        # TODO(charlie): implement this in the future
        raise NotImplementedError()

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        # TODO(charlie): implement this in the future
        raise NotImplementedError()

    async def init_weight_update_communicator(self, init_info):
        """Initialize weight update communicator for SGLang.

        Args:
            init_info: WeightSyncInitInfo from the sender.
        """
        return await self._weight_loader.init_communicator(init_info)

    async def update_named_weights(self, request: WeightUpdateRequest) -> None:
        """Update named weights in SGLang engine."""
        await self._weight_loader.load_weights(request)

    async def wake_up(self, tags: Optional[List[str]] = None):
        """Wake up the engine. For multi-stage waking up, pass in `"weight"` or `"kv_cache"` to tags."""
        obj = ResumeMemoryOccupationReqInput(tags=tags)
        # Call the underlying async method for the same reason as in `init_weight_update_communicator`
        await self.engine.tokenizer_manager.resume_memory_occupation(obj, None)
        logger.info(
            f"From SGLang engine -- Free GPU memory after wake up with tags {tags if tags is not None else 'None'}: "
            + f"{torch.cuda.mem_get_info()[0] / 1024**2:.1f} MB"
        )

    async def sleep(self, tags: Optional[List[str]] = None):
        """Put engine to sleep."""
        obj = ReleaseMemoryOccupationReqInput(tags=tags)
        # Call the underlying async method for the same reason as in `init_weight_update_communicator`
        await self.engine.tokenizer_manager.release_memory_occupation(obj, None)
        logger.info(
            f"From SGLang engine -- Free GPU memory after sleep with tags {tags if tags is not None else 'None'}: "
            + f"{torch.cuda.mem_get_info()[0] / 1024**2:.1f} MB"
        )

    async def teardown(self):
        """Shutdown the SGLang engine."""
        self.engine.shutdown()

    async def reset_prefix_cache(self):
        """Reset prefix cache in SGLang engine."""
        # Call the underlying async method for the same reason as in `init_weight_update_communicator`
        return await self.engine.tokenizer_manager.flush_cache()

    async def abort_generation(self) -> None:
        raise NotImplementedError("Abort generation is not supported for SGLang inference engines.")


SGLangRayActor = ray.remote(SGLangInferenceEngine)
