from typing import List
import ray
import vllm
from skyrl_train.inference_engines.vllm.vllm_engine import VLLMInferenceEngine
from skyrl_train.inference_engines.ray_wrapped_inference_engine import RayWrappedInferenceEngine
from ray.util.placement_group import PlacementGroupSchedulingStrategy, placement_group

from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
)


class FlashRLVLLMInferenceEngine(VLLMInferenceEngine):

    def _create_engine(self, *args, **kwargs):
        # apply flashrl's patch just before init
        from vllm.model_executor.layers.patch import apply_patch as apply_flashrl_patch

        apply_flashrl_patch()

        llm = vllm.LLM(*args, **kwargs)
        return llm


VLLMRayActor = ray.remote(FlashRLVLLMInferenceEngine)


def create_ray_wrapped_inference_engines_flashrl(
    num_inference_engines: int,
    tensor_parallel_size: int,
    model_dtype: str,
    pretrain: str,
    seed: int,
    vllm_v1_disable_multiproc: bool,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    shared_pg=None,
    gpu_memory_utilization=None,
    inference_engine_enable_sleep=False,
    async_engine=False,
    max_num_batched_tokens=8192,
    max_num_seqs=1024,
    tokenizer=None,
    backend="vllm",
) -> List[InferenceEngineInterface]:
    """
    Create a list of RayWrappedInferenceEngine instances wrapping Ray actor handles to InferenceEngineInterface instances.
    """
    from skyrl_train.utils import ray_noset_visible_devices, get_all_env_variables, get_ray_pg_ready_with_timeout
    from skyrl_train.utils.constants import SKYRL_RAY_PG_TIMEOUT_IN_S

    assert not async_engine, "`async_engine` is not supported for FlashRL"

    if backend != "vllm":
        raise ValueError(f"Unsupported FlashRL backend: {backend}")

    inference_engine_actors = []
    noset_visible_devices = ray_noset_visible_devices(ray.get(get_all_env_variables.remote()))
    # NOTE: we use the ray backend for tensor parallel size > 1 to explicitly manage resource allocation
    # TODO: we should be able to support mp backend by allocating resources at engine level
    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "ray"
    use_hybrid_engine = shared_pg is not None
    num_gpus = int(tensor_parallel_size == 1)
    if use_hybrid_engine and tensor_parallel_size == 1:
        # every worker will use 0.2 GPU, so that we can schedule
        # 2 instances on the same GPUs.
        num_gpus = 0.2

    if not use_hybrid_engine:
        # Create a big placement group to ensure that all inference engines are packed
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_inference_engines * tensor_parallel_size)]
        shared_pg = placement_group(bundles, strategy="PACK")
        get_ray_pg_ready_with_timeout(shared_pg, timeout=SKYRL_RAY_PG_TIMEOUT_IN_S)

    for i in range(num_inference_engines):
        bundle_indices = None
        if tensor_parallel_size > 1:
            bundle_indices = list(range(i * tensor_parallel_size, (i + 1) * tensor_parallel_size))

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=shared_pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=i * tensor_parallel_size,
        )

        if backend == "vllm":

            engine = VLLMRayActor.options(
                num_cpus=num_gpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                model=pretrain,
                enforce_eager=enforce_eager,
                worker_extension_cls="skyrl_train.inference_engines.vllm.vllm_engine.WorkerWrap",
                tensor_parallel_size=tensor_parallel_size,
                seed=seed + i,
                distributed_executor_backend=distributed_executor_backend,
                enable_prefix_caching=enable_prefix_caching,
                dtype=model_dtype,
                trust_remote_code=True,
                vllm_v1_disable_multiproc=vllm_v1_disable_multiproc,
                gpu_memory_utilization=gpu_memory_utilization,
                bundle_indices=bundle_indices,
                num_gpus=0.2 if use_hybrid_engine else 1,
                enable_sleep_mode=inference_engine_enable_sleep,
                noset_visible_devices=noset_visible_devices,
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_seqs=max_num_seqs,
                tokenizer=tokenizer,
                # only need the logprobs for the chosen token if any
                max_logprobs=1,
            )

        inference_engine_actors.append(engine)

    engines = [RayWrappedInferenceEngine(actor_handle) for actor_handle in inference_engine_actors]

    if inference_engine_enable_sleep:
        sleep_refs = [engine.inference_engine_actor.sleep.remote() for engine in engines]
        ray.get(sleep_refs)

    return engines
