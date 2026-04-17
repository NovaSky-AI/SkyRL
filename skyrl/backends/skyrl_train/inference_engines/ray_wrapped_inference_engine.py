from typing import TYPE_CHECKING, Any, Dict, List

import ray
from packaging import version
from ray.actor import ActorHandle

if TYPE_CHECKING:
    from skyrl.backends.skyrl_train.weight_sync.transfer_strategy import (
        WeightSyncInitInfo,
    )
from ray.util.placement_group import PlacementGroupSchedulingStrategy, placement_group

from skyrl.backends.skyrl_train.inference_engines.base import (
    InferenceEngineInput,
    InferenceEngineInterface,
    InferenceEngineOutput,
)
from skyrl.backends.skyrl_train.inference_engines.utils import get_rendezvous_addr_port
from skyrl.backends.skyrl_train.weight_sync import WeightUpdateRequest


class RayWrappedInferenceEngine(InferenceEngineInterface):
    """
    A thin wrapper around a Ray ActorHandle to another InferenceEngineInterface.
    This class implements the InferenceEngineInterface by delegating calls to the remote actor.
    """

    def __init__(self, inference_engine_actor: ActorHandle):
        self.inference_engine_actor = inference_engine_actor

    def tp_size(self):
        return ray.get(self.inference_engine_actor.tp_size.remote())

    def pp_size(self):
        return ray.get(self.inference_engine_actor.pp_size.remote())

    def dp_size(self):
        return ray.get(self.inference_engine_actor.dp_size.remote())

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        return await self.inference_engine_actor.generate.remote(input_batch=input_batch)

    async def sample(
        self,
        prompt_token_ids: List[int],
        num_samples: int,
        sampling_params: Dict[str, Any],
    ) -> InferenceEngineOutput:
        return await self.inference_engine_actor.sample.remote(
            prompt_token_ids=prompt_token_ids,
            num_samples=num_samples,
            sampling_params=sampling_params,
        )

    async def wake_up(self, *args: Any, **kwargs: Any):
        return await self.inference_engine_actor.wake_up.remote(*args, **kwargs)

    async def sleep(self, *args: Any, **kwargs: Any):
        return await self.inference_engine_actor.sleep.remote(*args, **kwargs)

    async def init_weight_update_communicator(self, init_info: "WeightSyncInitInfo"):
        return await self.inference_engine_actor.init_weight_update_communicator.remote(init_info)

    async def update_named_weights(self, request: WeightUpdateRequest):
        return await self.inference_engine_actor.update_named_weights.remote(request)

    async def teardown(self):
        return await self.inference_engine_actor.teardown.remote()

    async def reset_prefix_cache(self):
        return await self.inference_engine_actor.reset_prefix_cache.remote()

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self.inference_engine_actor.chat_completion.remote(request_payload)

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self.inference_engine_actor.completion.remote(request_payload)

    async def pause_generation(self) -> None:
        return await self.inference_engine_actor.pause_generation.remote()

    async def resume_generation(self) -> None:
        return await self.inference_engine_actor.resume_generation.remote()


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
    pipeline_parallel_size: int = 1,
    data_parallel_size: int = 1,
    shared_pg=None,
    gpu_memory_utilization=None,
    inference_engine_enable_sleep=False,
    async_engine=False,
    max_num_batched_tokens=8192,
    max_num_seqs=1024,
    tokenizer=None,
    backend="vllm",
    sleep_level=2,  # we only set to 1 for unit tests that do not explicitly sync weights or for LoRA
    enable_lora=False,
    max_lora_rank=64,
    max_loras=1,
    fully_sharded_loras=False,
    language_model_only=False,
    engine_init_kwargs: Dict[str, Any] = {},
    rope_scaling: Dict[str, Any] = {},
    rope_theta: float | None = None,
    enable_ray_prometheus_stats: bool = False,
    enable_return_routed_experts: bool = False,
    served_model_name: str | None = None,
    distributed_executor_backend: str = "ray",
) -> List[InferenceEngineInterface]:
    """
    Create a list of RayWrappedInferenceEngine instances wrapping Ray actor handles to InferenceEngineInterface
    instances.

    Args:
        shared_pg: A single placement group for colocated training, or None.
        distributed_executor_backend: vLLM distributed executor backend.
            "ray" spawns TP/PP workers as Ray tasks.
            "mp" spawns workers as local processes with CUDA_VISIBLE_DEVICES.
    """
    from skyrl.env_vars import SKYRL_RAY_PG_TIMEOUT_IN_S
    from skyrl.train.utils.utils import (
        ResolvedPlacementGroup,
        get_all_env_variables,
        get_ray_pg_ready_with_timeout,
        ray_noset_visible_devices,
    )

    if backend == "vllm":
        import vllm

        from skyrl.backends.skyrl_train.inference_engines.vllm.vllm_engine import (
            AsyncVLLMRayActor,
            VLLMRayActor,
        )

        if "dev" not in vllm.__version__:
            assert version.parse(vllm.__version__) >= version.parse("0.18.0"), "SkyRL-Train requires vLLM >= 0.18.0"
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    inference_engine_actors = []
    noset_visible_devices = ray_noset_visible_devices(ray.get(get_all_env_variables.remote()))

    resolved_executor_backend = (
        "uni" if (tensor_parallel_size == 1 and pipeline_parallel_size == 1) else distributed_executor_backend
    )
    use_mp_backend = resolved_executor_backend == "mp"

    data_parallel_backend = "mp"
    use_hybrid_engine = shared_pg is not None
    per_engine_gpu_count = tensor_parallel_size * pipeline_parallel_size * data_parallel_size

    num_gpus_per_actor = int(tensor_parallel_size == 1 and pipeline_parallel_size == 1)
    if use_hybrid_engine and tensor_parallel_size == 1 and pipeline_parallel_size == 1:
        num_gpus_per_actor = 0.2

    # Both mp and ray backends use a single shared PG with per-GPU bundles.
    if not use_hybrid_engine:
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_inference_engines * per_engine_gpu_count)]
        raw_pg = placement_group(bundles, strategy="PACK")
        get_ray_pg_ready_with_timeout(raw_pg, timeout=SKYRL_RAY_PG_TIMEOUT_IN_S)
        shared_pg = ResolvedPlacementGroup(raw_pg)

    assert isinstance(
        shared_pg, ResolvedPlacementGroup
    ), f"shared_pg must be a `ResolvedPlacementGroup` got {type(shared_pg)}."

    # Use reordered bundle indices to ensure GPU-aware ordering.
    # Ray placement groups don't guarantee bundle order, so bundles on the same node
    # may not have consecutive indices. The reordered indices map logical positions
    # to physical bundle indices sorted by (node_id, gpu_id).
    reordered = shared_pg.reordered_bundle_indices
    raw_pg = shared_pg.pg

    # Pre-compute GPU IDs per (engine, dp_rank) so we can set
    # CUDA_VISIBLE_DEVICES for the mp-spawned workers to see only the
    # TP*PP GPUs allocated to that DP rank.
    # Since reordered indices are sorted by (node_id, gpu_id), the physical
    # GPU IDs are directly available from shared_pg.bundle_gpu_ids.
    engine_gpu_ids_map = {}
    if use_mp_backend:
        all_gpu_ids = shared_pg.bundle_gpu_ids
        tp_pp_size = tensor_parallel_size * pipeline_parallel_size
        for engine_idx in range(num_inference_engines):
            for dp_rank in range(data_parallel_size):
                logical_base = engine_idx * per_engine_gpu_count + dp_rank * tp_pp_size
                engine_gpu_ids_map[(engine_idx, dp_rank)] = [all_gpu_ids[logical_base + k] for k in range(tp_pp_size)]

    for i in range(num_inference_engines):
        logical_base = i * per_engine_gpu_count
        base_pg_index = reordered[logical_base]

        # Get DP group rendezvous (addr, port) on the same node as DP rank 0 for this engine.
        data_parallel_address, data_parallel_rpc_port = get_rendezvous_addr_port(raw_pg, base_pg_index)

        if backend == "vllm":
            if async_engine:
                actor_class = AsyncVLLMRayActor
            else:
                actor_class = VLLMRayActor

            lora_kwargs = {
                "enable_lora": enable_lora,
                "max_lora_rank": max_lora_rank,
                "max_loras": max_loras,
                "fully_sharded_loras": fully_sharded_loras,
            }

            rope_engine_kwargs = {}
            if rope_scaling:
                rope_engine_kwargs["rope_scaling"] = rope_scaling
                if "max_model_len" not in engine_init_kwargs:
                    rope_factor = rope_scaling.get("factor", None)
                    rope_max_pos = rope_scaling.get("original_max_position_embeddings", None)
                    assert rope_factor is not None, "Please provide rope scaling `factor` to compute model max length"
                    assert (
                        rope_max_pos is not None
                    ), "Please provide rope `original_max_position_embeddings` to compute model max length"
                    rope_engine_kwargs["max_model_len"] = int(rope_factor * rope_max_pos)
            if rope_theta is not None:
                rope_engine_kwargs["rope_theta"] = rope_theta

            other_kwargs = {}

            # served_model_name allows using a different model name for HTTP endpoint validation
            # than the actual model path. See generator.served_model_name in ppo_base_config.yaml.
            if served_model_name is not None:
                other_kwargs["served_model_name"] = served_model_name

            # Launch one actor per DP rank
            for dp_rank in range(data_parallel_size):

                # Contiguous TP*PP slice reserved for a single DP rank.
                tp_pp_size = tensor_parallel_size * pipeline_parallel_size
                logical_dp_base = logical_base + dp_rank * tp_pp_size
                base_dp_pg_index = reordered[logical_dp_base]

                if use_mp_backend:
                    dp_rank_bundles = None
                    mp_gpu_ids = engine_gpu_ids_map.get((i, dp_rank))
                    mp_gpu_ids_str = ",".join(str(g) for g in mp_gpu_ids) if mp_gpu_ids is not None else None
                else:
                    dp_rank_bundles = (
                        [reordered[logical_dp_base + k] for k in range(tp_pp_size)] if tp_pp_size > 1 else None
                    )
                    mp_gpu_ids_str = None

                dp_rank_sched = PlacementGroupSchedulingStrategy(
                    placement_group=raw_pg,
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

                mp_kwargs = {}
                if mp_gpu_ids_str is not None:
                    mp_kwargs["mp_cuda_visible_devices"] = mp_gpu_ids_str

                engine = actor_class.options(
                    num_cpus=num_gpus_per_actor,
                    num_gpus=num_gpus_per_actor,
                    scheduling_strategy=dp_rank_sched,
                ).remote(
                    model=pretrain,
                    enforce_eager=enforce_eager,
                    language_model_only=language_model_only,
                    worker_extension_cls="skyrl.backends.skyrl_train.inference_engines.vllm.vllm_engine.WorkerWrap",
                    tensor_parallel_size=tensor_parallel_size,
                    pipeline_parallel_size=pipeline_parallel_size,
                    enable_expert_parallel=expert_parallel_size > 1,
                    distributed_executor_backend=resolved_executor_backend,
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
                    enable_ray_prometheus_stats=enable_ray_prometheus_stats,
                    enable_return_routed_experts=enable_return_routed_experts,
                    **dp_kwargs,
                    **engine_init_kwargs,
                    **lora_kwargs,
                    **rope_engine_kwargs,
                    **other_kwargs,
                    **mp_kwargs,
                )
                inference_engine_actors.append(engine)

    engines = [RayWrappedInferenceEngine(actor_handle) for actor_handle in inference_engine_actors]

    if inference_engine_enable_sleep:
        # NOTE(shu): set to 1 for LoRA
        sleep_level = 1 if enable_lora else sleep_level
        sleep_refs = [engine.inference_engine_actor.sleep.remote(level=sleep_level) for engine in engines]
        ray.get(sleep_refs)

    return engines
