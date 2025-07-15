import os
import time

import ray
import torch
from loguru import logger
from omegaconf.dictconfig import DictConfig
from ray.util.placement_group import PlacementGroup


class Timer:
    def __init__(self, message, update_dict=None):
        self.message = message
        self.update_dict = update_dict

    def __enter__(self):
        self.start_time = time.time()
        logger.opt(depth=1).info(f"Started: '{self.message}'")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.opt(depth=1).info(f"Finished: '{self.message}', time cost: {time.time() - self.start_time:.2f}s")
        if self.update_dict is not None:
            self.update_dict[self.message] = time.time() - self.start_time

    async def __aenter__(self):
        self.start_time = time.time()
        logger.opt(depth=1).info(f"Started: '{self.message}'")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        logger.opt(depth=1).info(f"Finished: '{self.message}', time cost: {time.time() - self.start_time:.2f}s")
        if self.update_dict is not None:
            self.update_dict[self.message] = time.time() - self.start_time


def validate_batch_sizes(cfg: DictConfig):
    """
    Validate configured batch sizes.

    Explanation of how batching operates:
    1. Each prompt in train_batch_size creates `n_samples_per_prompt` total samples.
    2. During training, these samples are split across data parallel (DP) workers, making the effective per-GPU batch size: `train_batch_size * n_samples_per_prompt / dp_size`.
    3. Mini batches are similarly normalized to per-gpu mini batches with size: `mini_batch_size * n_samples_per_prompt / dp_size`.
    4. Per-gpu train batch size must be divisble by per-gpu mini batch size, otherwise the last mini batch will be incomplete.
    5. Per-gpu mini batch size must be divisible by per-gpu micro batch size, otherwise the last micro batch will be incomplete.
    """
    assert cfg.trainer.train_batch_size >= cfg.trainer.policy_mini_batch_size
    assert cfg.trainer.policy_mini_batch_size > 0, "policy_mini_batch_size must be greater than 0"
    if cfg.trainer.critic.model.path is not None:
        assert cfg.trainer.train_batch_size >= cfg.trainer.critic_mini_batch_size
        assert cfg.trainer.critic_mini_batch_size > 0, "critic_mini_batch_size must be greater than 0"
    assert cfg.trainer.micro_train_batch_size_per_gpu > 0, "micro_train_batch_size_per_gpu must be greater than 0"
    assert cfg.trainer.micro_forward_batch_size_per_gpu > 0, "micro_forward_batch_size_per_gpu must be greater than 0"

    # Validate policy mini batch size
    policy_world_size = cfg.trainer.placement.policy_num_nodes * cfg.trainer.placement.policy_num_gpus_per_node
    policy_dp_size = policy_world_size // cfg.trainer.policy.sequence_parallel_size
    assert (
        cfg.trainer.train_batch_size % cfg.trainer.policy_mini_batch_size == 0
    ), f"train_batch_size {cfg.trainer.train_batch_size} should be divisible by policy_mini_batch_size {cfg.trainer.policy_mini_batch_size}"
    policy_mini_batch_size_per_gpu = (
        cfg.trainer.policy_mini_batch_size * cfg.generator.n_samples_per_prompt // policy_dp_size
    )
    assert policy_mini_batch_size_per_gpu > 0, (
        f"Invalid policy_mini_batch_size_per_gpu: {policy_mini_batch_size_per_gpu}. "
        f"mini_batch_size={cfg.trainer.policy_mini_batch_size}, "
        f"n_samples_per_prompt={cfg.generator.n_samples_per_prompt}, "
        f"dp_size={policy_dp_size}"
    )
    assert (
        policy_mini_batch_size_per_gpu % cfg.trainer.micro_train_batch_size_per_gpu == 0
    ), f"normalized policy_mini_batch_size_per_gpu {policy_mini_batch_size_per_gpu} should be divisible by micro_train_batch_size_per_gpu {cfg.trainer.micro_train_batch_size_per_gpu}"
    assert (
        policy_mini_batch_size_per_gpu // cfg.trainer.micro_train_batch_size_per_gpu > 0
    ), f"normalized policy_mini_batch_size_per_gpu {policy_mini_batch_size_per_gpu} should be larger than micro_train_batch_size_per_gpu {cfg.trainer.micro_train_batch_size_per_gpu}"
    policy_train_batch_size_per_gpu = (
        cfg.trainer.train_batch_size * cfg.generator.n_samples_per_prompt // policy_dp_size
    )

    # `train_batch_size_per_gpu` should be divisible by `policy_mini_batch_size_per_gpu`
    assert (
        policy_train_batch_size_per_gpu % policy_mini_batch_size_per_gpu == 0
    ), f"normalized policy_train_batch_size_per_gpu (train_batch_size * n_samples_per_prompt // policy_dp_size) {policy_train_batch_size_per_gpu} should be divisible by policy_mini_batch_size_per_gpu (policy_mini_batch_size * n_samples_per_prompt // policy_dp_size) {policy_mini_batch_size_per_gpu}"

    # Validate critic mini batch size
    critic_world_size = cfg.trainer.placement.critic_num_nodes * cfg.trainer.placement.critic_num_gpus_per_node
    critic_dp_size = critic_world_size // cfg.trainer.critic.sequence_parallel_size

    if cfg.trainer.critic.model.path is not None:
        assert (
            cfg.trainer.train_batch_size % cfg.trainer.critic_mini_batch_size == 0
        ), f"train_batch_size {cfg.trainer.train_batch_size} should be divisible by critic_mini_batch_size {cfg.trainer.critic_mini_batch_size}"
        critic_mini_batch_size_per_gpu = (
            cfg.trainer.critic_mini_batch_size * cfg.generator.n_samples_per_prompt // critic_dp_size
        )
        assert critic_mini_batch_size_per_gpu > 0, (
            f"Invalid critic_mini_batch_size_per_gpu: {critic_mini_batch_size_per_gpu}. "
            f"mini_batch_size={cfg.trainer.critic_mini_batch_size}, "
            f"n_samples_per_prompt={cfg.generator.n_samples_per_prompt}, "
            f"dp_size={critic_dp_size}"
        )
        assert (
            critic_mini_batch_size_per_gpu % cfg.trainer.micro_train_batch_size_per_gpu == 0
        ), f"normalized critic_mini_batch_size_per_gpu {critic_mini_batch_size_per_gpu} should be divisible by micro_train_batch_size_per_gpu {cfg.trainer.micro_train_batch_size_per_gpu}"
        assert (
            critic_mini_batch_size_per_gpu // cfg.trainer.micro_train_batch_size_per_gpu > 0
        ), f"normalized critic_mini_batch_size_per_gpu {critic_mini_batch_size_per_gpu} should be larger than micro_train_batch_size_per_gpu {cfg.trainer.micro_train_batch_size_per_gpu}"
        critic_train_batch_size_per_gpu = (
            cfg.trainer.train_batch_size * cfg.generator.n_samples_per_prompt // critic_dp_size
        )
        assert (
            critic_train_batch_size_per_gpu % critic_mini_batch_size_per_gpu == 0
        ), f"normalized critic_train_batch_size_per_gpu (train_batch_size * n_samples_per_prompt // critic_dp_size) {critic_train_batch_size_per_gpu} should be divisible by critic_mini_batch_size_per_gpu (critic_mini_batch_size * n_samples_per_prompt // critic_dp_size) {critic_mini_batch_size_per_gpu}"


def validate_cfg(cfg: DictConfig):
    if cfg.generator.max_turns == 1:
        assert (
            cfg.generator.max_input_length == cfg.trainer.max_prompt_length
        ), "generator.max_input_length should be set equal to trainer.max_prompt_length for single-turn generation"
    else:
        assert (
            cfg.generator.max_input_length >= cfg.trainer.max_prompt_length
        ), "generator.max_input_length should be set greater than or equal to trainer.max_prompt_length for multi-turn generation"

    if not cfg.generator.run_engines_locally:
        assert cfg.generator.num_inference_engines == len(
            cfg.generator.remote_inference_engine_urls
        ), "num_inference_engines should be equal to the number of remote_inference_engine_urls"

    if not cfg.generator.async_engine:
        assert (
            cfg.generator.batched
        ), "if we are using the offline engine, we need to put generator in batched mode for faster generation"
    if cfg.generator.backend == "sglang" and cfg.generator.run_engines_locally:
        raise ValueError("SGLang backend currently does not support local engines")

    assert (
        cfg.trainer.sequence_parallel_backend == "ulysses"
    ), f"only ulysses is supported as of now, got {cfg.trainer.sequence_parallel_backend}"

    assert cfg.trainer.algorithm.advantage_estimator in (
        "gae",
        "grpo",
    ), f"invalid advantage estimator: {cfg.trainer.algorithm.advantage_estimator}"
    # if advantage estimator is GAE, then critic path should be provided
    if cfg.trainer.algorithm.advantage_estimator == "gae":
        assert (
            cfg.trainer.critic.model.path
        ), "`trainer.critic.model.path` should be provided for PPO training, got `None`"

    assert not (
        cfg.trainer.algorithm.use_kl_in_reward and cfg.trainer.algorithm.use_kl_loss
    ), "use_kl_in_reward and use_kl_loss should be mutually exclusive"

    if cfg.trainer.strategy in ("fsdp", "fsdp2"):
        assert not (
            cfg.trainer.policy.fsdp_config.cpu_offload and cfg.trainer.strategy == "fsdp"
        ), "fwd pass cpu offloading is not supported for FSDP1 policy worker, use FSDP2 instead"
        assert not (
            cfg.trainer.critic.fsdp_config.cpu_offload and cfg.trainer.strategy == "fsdp"
        ), "fwd pass cpu offloading is not supported for FSDP1 critic worker, use FSDP2 instead"

    if cfg.trainer.strategy == "deepspeed":
        assert (
            cfg.trainer.policy.deepspeed_config.zero_optimization.stage == 3
        ), "only deepspeed stage 3 is currently supported!"

    validate_batch_sizes(cfg)

    # tracker
    if cfg.trainer.logger == "wandb":
        assert os.environ.get("WANDB_API_KEY"), "`WANDB_API_KEY` is required for `wandb` logger"

    if cfg.trainer.max_ckpts_to_keep == 0:
        raise ValueError(
            "`max_ckpts_to_keep` must be greater than 0 to keep the last N checkpoints or negative to keep all checkpoints"
        )

    # resolve override_existing_update_group
    if cfg.generator.override_existing_update_group == "auto":
        if cfg.generator.backend == "vllm" and not cfg.generator.run_engines_locally:
            # remote engines can be launched separately so we `enable` by default
            cfg.generator.override_existing_update_group = "enable"
        else:
            # for local engines or sglang, we disable
            cfg.generator.override_existing_update_group = "disable"

    assert cfg.trainer.algorithm.ppo_loss_type in (
        "regular",
        "dual_clip",
    ), f"invalid ppo_loss_type: {cfg.trainer.algorithm.ppo_loss_type}. Must be one of `['regular', 'dual_clip']`"

    assert cfg.trainer.algorithm.loss_reduction in (
        "token_mean",
        "sequence_mean",
    ), f"invalid loss_reduction: {cfg.trainer.algorithm.loss_reduction}. Must be one of `['token_mean', 'sequence_mean']`"

    if cfg.trainer.strategy == "deepspeed" and not (
        cfg.trainer.policy.optimizer_config.offload_after_step
        and cfg.trainer.critic.optimizer_config.offload_after_step
    ):
        raise ValueError(
            "`offload_after_step=False` is not supported for DeepSpeed, please set `offload_after_step` to `true` for both policy and critic"
        )

    if cfg.generator.backend == "sglang" and not cfg.generator.use_conversation_multi_turn:
        raise NotImplementedError("`use_conversation_multi_turn=False` is not supported for SGLang backend")


@ray.remote
def get_all_env_variables():
    import os

    return os.environ


def ray_noset_visible_devices(env_vars=os.environ):
    # Refer to
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/nvidia_gpu.py#L95-L96
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/amd_gpu.py#L102-L103
    # https://github.com/ray-project/ray/blob/3b9e729f6a669ffd85190f901f5e262af79771b0/python/ray/_private/accelerators/amd_gpu.py#L114-L115
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/npu.py#L94-L95
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/hpu.py#L116-L117
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/neuron.py#L108-L109
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/tpu.py#L171-L172
    # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/intel_gpu.py#L97-L98
    NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
        "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
        "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
        "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
    ]
    return any(env_vars.get(env_var) for env_var in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST)


def get_physical_gpu_id():
    import torch

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return str(props.uuid)


def initialize_ray(cfg: DictConfig):
    # TODO(sumanthrh): introduce a debug mode and add debugging flags like `CUDA_LAUNCH_BLOCKING` here
    env_vars = {}
    if cfg.generator.backend == "vllm":
        # NOTE (sumanthrh): In vllm >= 0.9.0, we need to explicitly allow for serialization via pickle for collective RPCs.
        # During weight transfer, we use IPC handles, which contains a `function` object and requires pickling.
        env_vars["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        if not os.environ.get("VLLM_USE_V1", False):
            logger.info(
                "`VLLM_USE_V1` is not specified, setting `VLLM_USE_V1` to 1. To override, set `VLLM_USE_V1` explicitly"
            )
            env_vars["VLLM_USE_V1"] = "1"
            env_vars["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    # TODO: this can be removed if we standardize on env files.
    # But it's helpful for a quickstart
    if os.environ.get("WANDB_API_KEY"):
        logger.info("Exporting wandb api key to ray runtime env")
        env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

    if os.environ.get("SKYRL_LD_LIBRARY_PATH_EXPORT"):
        # export `LD_LIBRARY_PATH` to ray runtime env.
        # For some reason the `LD_LIBRARY_PATH` is not exported to the worker with .env file.
        logger.info(f"Exporting `LD_LIBRARY_PATH` to ray runtime env: {os.environ['LD_LIBRARY_PATH']}")
        env_vars["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
    ray.init(runtime_env={"env_vars": env_vars})


def get_ray_pg_ready_with_timeout(pg: PlacementGroup, timeout: int = 60):
    try:
        ray.get(pg.ready(), timeout=timeout)
    except Exception as e:
        # Extract resource demands from the placement group
        bundles = pg.bundle_specs
        total_gpus = sum(bundle.get("GPU", 0) for bundle in bundles)
        total_cpus = sum(bundle.get("CPU", 0) for bundle in bundles)

        raise RuntimeError(
            f"Failed to create placement group with {len(bundles)} bundles "
            f"(requiring {total_gpus} GPUs, {total_cpus} CPUs total) in {timeout} seconds. "
            f"This might indicate insufficient GPU resources.\n"
            f"Error: {e}"
        )


# NOTE (sumanthrh): For SGLang, the string representations here should also match those used by (and supported by) SGLang.
# This is because we do not control the update weight implementation with SGLang backend.
# With VLLM, we use a custom Worker extension to have a custom update weight implementation.
def torch_dtype_to_str(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bfloat16"
    elif dtype == torch.float16:
        return "float16"
    elif dtype == torch.float32:
        return "float32"
    else:
        return str(dtype)


def str_to_torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "bfloat16":
        return torch.bfloat16
    elif dtype == "float16":
        return torch.float16
    elif dtype == "float32":
        return torch.float32
    else:
        return torch.dtype(dtype)


def format_gib(mem_bytes: int) -> str:
    return f"{mem_bytes / (1024 ** 3):.2f} GiB"


def print_mem(tag: str, mem: dict):
    print(
        f"{tag} - Allocated: {format_gib(mem['allocated'])}, "
        f"Reserved: {format_gib(mem['reserved'])}, "
        f"Free: {format_gib(mem['free'])}, "
        f"Total: {format_gib(mem['total'])}"
    )
