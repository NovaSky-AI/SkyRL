import ray
from packaging import version
from ray.actor import ActorHandle
from typing import Any, List, Dict
from ray.util.placement_group import PlacementGroupSchedulingStrategy, placement_group

from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
    InferenceEngineInput,
    InferenceEngineOutput,
    NamedWeightsUpdateRequest,
)
from skyrl_train.inference_engines.utils import get_rendezvous_addr_port


class RayWrappedInferenceEngine(InferenceEngineInterface):
    """
    A thin wrapper around a Ray ActorHandle to another InferenceEngineInterface.
    This class implements the InferenceEngineInterface by delegating calls to the remote actor.
    """

    def __init__(self, inference_engine_actor: ActorHandle):
        self.inference_engine_actor = inference_engine_actor

    def tp_size(self):
        return ray.get(self.inference_engine_actor.tp_size.remote())

    def dp_size(self):
        return ray.get(self.inference_engine_actor.dp_size.remote())

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        return await self.inference_engine_actor.generate.remote(input_batch=input_batch)

    async def wake_up(self, *args: Any, **kwargs: Any):
        return await self.inference_engine_actor.wake_up.remote(*args, **kwargs)

    async def sleep(self, *args: Any, **kwargs: Any):
        return await self.inference_engine_actor.sleep.remote(*args, **kwargs)

    async def init_weight_update_communicator(
        self, master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing: bool = False
    ):
        return await self.inference_engine_actor.init_weight_update_communicator.remote(
            master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing
        )

    async def update_named_weights(self, request: NamedWeightsUpdateRequest):
        return await self.inference_engine_actor.update_named_weights.remote(request)

    async def teardown(self):
        return await self.inference_engine_actor.teardown.remote()

    async def reset_prefix_cache(self):
        return await self.inference_engine_actor.reset_prefix_cache.remote()

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self.inference_engine_actor.chat_completion.remote(request_payload)

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self.inference_engine_actor.completion.remote(request_payload)


def create_ray_wrapped_inference_engines(
    num_inference_engines: int,
    tensor_parallel_size: int,
    model_dtype: str,
    pretrain: str,
    seed: int,
    vllm_v1_disable_multiproc: bool,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    expert_parallel_size: int = 1,
    data_parallel_size: int = 1,
    shared_pg=None,
    gpu_memory_utilization=None,
    inference_engine_enable_sleep=False,
    async_engine=False,
    max_num_batched_tokens=8192,
    max_num_seqs=1024,
    tokenizer=None,
    backend="vllm",
    sleep_level=2,  # we only set to 1 for unit tests that do not explicitly sync weights
    engine_init_kwargs: Dict[str, Any] = {},
) -> List[InferenceEngineInterface]:
    """
    Create a list of RayWrappedInferenceEngine instances wrapping Ray actor handles to InferenceEngineInterface instances.
    """
    from skyrl_train.utils import ray_noset_visible_devices, get_all_env_variables, get_ray_pg_ready_with_timeout
    from skyrl_train.utils.constants import SKYRL_RAY_PG_TIMEOUT_IN_S

    if backend == "vllm":
        import vllm
        from skyrl_train.inference_engines.vllm.vllm_engine import VLLMRayActor, AsyncVLLMRayActor

        # if a dev version is being used, skip the version check
        if "dev" not in vllm.__version__:
            assert version.parse(vllm.__version__) >= version.parse("0.8.3"), "SkyRL-Train only supports vLLM >= 0.8.3"
    elif backend == "sglang":
        # We import SGLang later to avoid importing vllm. See `get_sglang_engine` for more.
        pass
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    inference_engine_actors = []
    noset_visible_devices = ray_noset_visible_devices(ray.get(get_all_env_variables.remote()))
    # NOTE: we use the ray backend for tensor parallel size > 1 to explicitly manage resource allocation
    # TODO: we should be able to support mp backend by allocating resources at engine level
    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "ray"
    data_parallel_backend = "mp"
    use_hybrid_engine = shared_pg is not None
    num_gpus_per_actor = int(tensor_parallel_size == 1)

    if use_hybrid_engine and tensor_parallel_size == 1:
        # Every worker will use 0.2 GPU, so that we can schedule
        # inference and training workers on the same GPUs.
        num_gpus_per_actor = 0.2

    per_engine_gpu_count = tensor_parallel_size * data_parallel_size
    if not use_hybrid_engine:
        # Create a big placement group to ensure that all inference engines are packed
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_inference_engines * per_engine_gpu_count)]
        shared_pg = placement_group(bundles, strategy="PACK")
        get_ray_pg_ready_with_timeout(shared_pg, timeout=SKYRL_RAY_PG_TIMEOUT_IN_S)

    for i in range(num_inference_engines):
        base_pg_index = i * per_engine_gpu_count

        # Get DP group rendezvous (addr, port) on the same node as DP rank 0 for this engine.
        data_parallel_address, data_parallel_rpc_port = get_rendezvous_addr_port(shared_pg, base_pg_index)

        if backend == "vllm":
            if async_engine:
                actor_class = AsyncVLLMRayActor
            else:
                actor_class = VLLMRayActor

            # Launch one actor per DP rank
            for dp_rank in range(data_parallel_size):

                # Contiguous TP slice reserved for a single DP rank.
                base_dp_pg_index = base_pg_index + dp_rank * tensor_parallel_size
                dp_rank_bundles = (
                    list(range(base_dp_pg_index, base_dp_pg_index + tensor_parallel_size))
                    if tensor_parallel_size > 1
                    else None
                )
                dp_rank_sched = PlacementGroupSchedulingStrategy(
                    placement_group=shared_pg,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=base_dp_pg_index,
                )

                dp_kwargs = (
                    {
                        "data_parallel_backend": data_parallel_backend,
                        "data_parallel_size": data_parallel_size,
                        "data_parallel_rank": dp_rank,
                        "data_parallel_address": data_parallel_address,
                        "data_parallel_rpc_port": data_parallel_rpc_port,
                    }
                    if data_parallel_size > 1
                    else {}
                )

                engine = actor_class.options(
                    num_cpus=num_gpus_per_actor,
                    num_gpus=num_gpus_per_actor,
                    scheduling_strategy=dp_rank_sched,
                ).remote(
                    model=pretrain,
                    enforce_eager=enforce_eager,
                    worker_extension_cls="skyrl_train.inference_engines.vllm.vllm_engine.WorkerWrap",
                    tensor_parallel_size=tensor_parallel_size,
                    enable_expert_parallel=expert_parallel_size > 1,
                    distributed_executor_backend=distributed_executor_backend,
                    seed=seed + i * data_parallel_size + dp_rank,
                    enable_prefix_caching=enable_prefix_caching,
                    dtype=model_dtype,
                    trust_remote_code=True,
                    vllm_v1_disable_multiproc=vllm_v1_disable_multiproc,
                    gpu_memory_utilization=gpu_memory_utilization,
                    bundle_indices=dp_rank_bundles,
                    num_gpus=0.2 if use_hybrid_engine else 1,
                    enable_sleep_mode=inference_engine_enable_sleep,
                    noset_visible_devices=noset_visible_devices,
                    max_num_batched_tokens=max_num_batched_tokens,
                    max_num_seqs=max_num_seqs,
                    max_logprobs=1,  # only need chosen-token logprobs
                    **dp_kwargs,
                    **engine_init_kwargs,
                )
                inference_engine_actors.append(engine)
        elif backend == "sglang":
            # NOTE: there is no async / sync engine distinction in SGLang

            bundle_indices = None
            if per_engine_gpu_count > 1:
                bundle_indices = list(range(i * per_engine_gpu_count, (i + 1) * per_engine_gpu_count))

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=shared_pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=i * per_engine_gpu_count,
            )

            # NOTE(Charlie): We need `torch.cuda.is_available()` to be True to import SGLang. Otherwise, it requires
            # importing vllm. See https://github.com/sgl-project/sglang/blob/v0.4.8.post1/python/sglang/srt/layers/quantization/utils.py#L11-L17
            # Similar comment: https://github.com/volcengine/verl/blob/9cc307767b0c787e8f5ef581dac929f7bde044ef/verl/workers/fsdp_workers.py#L520-L527
            @ray.remote
            def get_sglang_engine():
                # A workaround to avoid importing vllm is to give this task a GPU.
                import os

                before_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                from skyrl_train.inference_engines.sglang.sglang_engine import SGLangRayActor

                os.environ["CUDA_VISIBLE_DEVICES"] = before_cuda_visible_devices

                actor_class = SGLangRayActor
                engine = actor_class.options(
                    num_cpus=num_gpus_per_actor,
                    num_gpus=num_gpus_per_actor,
                    scheduling_strategy=scheduling_strategy,
                ).remote(
                    model_path=pretrain,
                    tp_size=tensor_parallel_size,
                    mem_fraction_static=gpu_memory_utilization,
                    random_seed=seed + i,
                    disable_radix_cache=not enable_prefix_caching,
                    dtype=model_dtype,
                    trust_remote_code=True,
                    max_prefill_tokens=max_num_batched_tokens,
                    max_running_requests=max_num_seqs,
                    # Borrowed from veRL's SGLang rollout
                    mm_attention_backend="fa3",
                    attention_backend="fa3",
                    enable_memory_saver=inference_engine_enable_sleep,
                    # Will be popped before instantiating sgl.Engine
                    distributed_executor_backend=distributed_executor_backend,
                    noset_visible_devices=noset_visible_devices,
                    bundle_indices=bundle_indices,
                    num_gpus=0.2 if use_hybrid_engine else 1,
                    tokenizer=tokenizer,
                    **engine_init_kwargs,
                )
                return engine

            engine = ray.get(get_sglang_engine.remote())

            inference_engine_actors.append(engine)

    engines = [RayWrappedInferenceEngine(actor_handle) for actor_handle in inference_engine_actors]

    if inference_engine_enable_sleep:
        if backend == "vllm":
            sleep_refs = [engine.inference_engine_actor.sleep.remote(level=sleep_level) for engine in engines]
        elif backend == "sglang":
            # NOTE(Charlie): we always need to sync weights after waking up: https://github.com/sgl-project/sglang/issues/7939
            assert sleep_level == 2, "SGLang always discards weights, so sleep_level is not applicable."
            sleep_refs = [engine.inference_engine_actor.sleep.remote() for engine in engines]
        ray.get(sleep_refs)

    return engines
