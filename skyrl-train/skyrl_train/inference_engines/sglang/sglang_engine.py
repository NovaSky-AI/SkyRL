import os
from typing import List, Optional
import ray
import multiprocessing as mp

import sglang.srt.entrypoints.engine
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.utils import (
    assert_pkg_version,
    is_cuda,
    maybe_set_triton_cache_manager,
    set_prometheus_multiproc_dir,
    set_ulimit,
)
from sglang.srt.managers.tokenizer_manager import (
    UpdateWeightsFromDistributedReqInput,
    InitWeightsUpdateGroupReqInput,
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
)
from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
    InferenceEngineInput,
    InferenceEngineOutput,
    NamedWeightUpdateRequest,
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

        # Extract sampling params
        sampling_params_dict = kwargs.pop("sampling_params", None)
        self.sampling_params = sampling_params_dict or {}

        # Unused kwargs
        _ = kwargs.pop("num_gpus", 1)

        # Create the SGLang engine (signal handler issue is now fixed by patching)
        self.engine = Engine(**kwargs)
        print(f"Created SGLang engine with kwargs: {kwargs}")

    def tp_size(self):
        """Return the tensor parallel size."""
        return self._tp_size

    def _preprocess_prompts(self, input_batch: InferenceEngineInput):
        """Preprocess prompts for SGLang generation."""
        prompts = input_batch.get("prompts")
        prompt_token_ids = input_batch.get("prompt_token_ids")
        request_sampling_params = input_batch.get("sampling_params")

        if (prompts is None and prompt_token_ids is None) or (prompts is not None and prompt_token_ids is not None):
            raise ValueError("Either `prompts` or `prompt_token_ids` must be provided, but not both.")

        # Use request sampling params if provided, otherwise use defaults
        sampling_params = request_sampling_params if request_sampling_params is not None else self.sampling_params

        if prompt_token_ids is None:
            prompt_token_ids = self.tokenizer.apply_chat_template(
                prompts,
                add_generation_prompt=True,
                add_special_tokens=False,
                return_dict=True,
                tokenize=True,
            )["input_ids"]

        return prompt_token_ids, sampling_params

    def _postprocess_outputs(self, outputs):
        """Process SGLang outputs to match expected format."""
        responses: List[str] = []
        stop_reasons: List[str] = []

        if not isinstance(outputs, list):
            outputs = [outputs]

        for output in outputs:
            responses.append(output["text"])
            stop_reasons.append(output["meta_info"]["finish_reason"]["type"])

        return InferenceEngineOutput(
            responses=responses,
            stop_reasons=stop_reasons,
        )

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        """Generate responses using SGLang engine."""
        token_ids_prompts, sampling_params = self._preprocess_prompts(input_batch)

        # Generate using SGLang's async method
        # Otherwise using `await asyncio.to_thread(self.engine.generate)` will cause issues
        # as SGLang's `generate()` method calls `loop = asyncio.get_event_loop()`, raising error
        # `RuntimeError: There is no current event loop in thread 'ThreadPoolExecutor-0_0'.`
        outputs = await self.engine.async_generate(input_ids=token_ids_prompts, sampling_params=sampling_params)

        return self._postprocess_outputs(outputs)

    async def init_weight_update_communicator(
        self, master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing: bool = False
    ):
        """Initialize weight update communicator for SGLang."""
        obj = InitWeightsUpdateGroupReqInput(
            master_address=master_addr,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=world_size,
            group_name=group_name,
            backend=backend,
        )

        # NOTE(charlie): Call the async method on tokenizer_manager directly to avoid event loop
        # conflicts since `sgl.Engine.init_weights_update_group` is sync, yet uses
        # `asyncio.get_event_loop()` which prevents us from using `asyncio.to_thread`. We cannot
        # call the sync method directly either because it runs into `RuntimeError: this event loop
        # is already running.`
        # Same underlying implementation: https://github.com/sgl-project/sglang/blob/v0.4.8.post1/python/sglang/srt/model_executor/model_runner.py#L689
        success, message = await self.engine.tokenizer_manager.init_weights_update_group(obj, None)
        return success, message

    async def update_named_weight(self, request: NamedWeightUpdateRequest):
        """Update named weights in SGLang engine."""
        if request.get("extras") and "ipc_handles" in request["extras"]:
            raise ValueError(
                "SGLang local engines do not support CUDA IPC weight updates yet. " + "Only vLLM does at the moment."
            )
        obj = UpdateWeightsFromDistributedReqInput(name=request["name"], dtype=request["dtype"], shape=request["shape"])

        # Call the underlying async method for the same reason as in `init_weight_update_communicator`
        success, message = await self.engine.tokenizer_manager.update_weights_from_distributed(obj, None)
        return success, message

    async def wake_up(self, tags: Optional[List[str]] = None):
        """Wake up the engine. For multi-stage waking up, pass in `"weight"` or `"kv_cache"` to tags."""
        if tags is None:
            obj = ResumeMemoryOccupationReqInput()
        else:
            obj = ResumeMemoryOccupationReqInput(tags=tags)
        # Call the underlying async method for the same reason as in `init_weight_update_communicator`
        await self.engine.tokenizer_manager.resume_memory_occupation(obj, None)

    async def sleep(self, tags: Optional[List[str]] = None):
        """Put engine to sleep."""
        if tags is None:
            obj = ReleaseMemoryOccupationReqInput()
        else:
            obj = ReleaseMemoryOccupationReqInput(tags=tags)
        # Call the underlying async method for the same reason as in `init_weight_update_communicator`
        await self.engine.tokenizer_manager.release_memory_occupation(obj, None)

    async def teardown(self):
        """Shutdown the SGLang engine."""
        self.engine.shutdown()

    async def reset_prefix_cache(self):
        """Reset prefix cache in SGLang engine."""
        # Call the underlying async method for the same reason as in `init_weight_update_communicator`
        return await self.engine.tokenizer_manager.flush_cache()


SGLangRayActor = ray.remote(SGLangInferenceEngine)
