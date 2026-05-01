import asyncio
import copy
import importlib
import os
import subprocess
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import ray
import requests
import torch
from loguru import logger
from ray.util.placement_group import placement_group
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from skyrl.backends.skyrl_train.distributed.dispatch import (
    concatenate_outputs_after_mesh_dispatch,
)
from skyrl.backends.skyrl_train.inference_engines.base import InferenceEngineInput
from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import (
    InferenceEngineClient,
)
from skyrl.backends.skyrl_train.inference_engines.ray_wrapped_inference_engine import (
    create_ray_wrapped_inference_engines,
)
from skyrl.backends.skyrl_train.inference_engines.remote_inference_engine import (
    create_remote_inference_engines,
)
from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    RemoteInferenceClient,
)
from skyrl.backends.skyrl_train.inference_servers.server_group import ServerGroup
from skyrl.backends.skyrl_train.inference_servers.setup import create_inference_servers
from skyrl.backends.skyrl_train.inference_servers.utils import (
    _uses_lora_weight_sync,
    build_vllm_cli_args,
)
from skyrl.backends.skyrl_train.inference_servers.vllm_router import VLLMRouter
from skyrl.backends.skyrl_train.training_batch import (
    TensorBatch,
    TrainingInputBatch,
    TrainingOutputBatch,
)
from skyrl.backends.skyrl_train.workers.worker import PPORayActorGroup
from skyrl.env_vars import _SKYRL_USE_NEW_INFERENCE, SKYRL_PYTHONPATH_EXPORT
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.dataset import PromptDataset
from skyrl.train.dataset.replay_buffer import Experience
from skyrl.train.generators.base import ConversationType, GeneratorInput, TrajectoryID
from skyrl.train.utils import get_ray_pg_ready_with_timeout
from skyrl.train.utils.utils import (
    ResolvedPlacementGroup,
    initialize_ray,
    peer_access_supported,
    print_mem,
)
from skyrl.utils.tok import get_tokenizer

TEST_DATA_PATH = os.path.expanduser("~/data/gsm8k/validation.parquet")


def get_test_actor_config() -> SkyRLTrainConfig:
    """Get base config with test-specific overrides."""
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = "Qwen/Qwen2.5-0.5B-Instruct"
    cfg.trainer.logger = "console"
    return cfg


def get_rank_0_memory(actor_group, message: str):
    mem = ray.get(actor_group.async_run_ray_method("pass_through", "get_cuda_memory"))[0]
    print_mem(message, mem)
    return mem["allocated"]


def make_dummy_tensorbatch(seq_len=10, num_actions=4) -> TensorBatch:
    B, T = 2, seq_len
    data = TensorBatch(
        sequences=torch.ones(B, T, dtype=int, device="cpu"),
        attention_mask=torch.ones(B, T, dtype=int, device="cpu"),
    )
    data.metadata = {"response_length": num_actions}
    return data


def make_dummy_training_batch(batch_size=2, seq_len=10, num_actions=4, action_lengths=None) -> TrainingInputBatch:
    """Create a dummy TrainingInputBatch.

    Args:
        action_lengths: Optional list of per-sample valid action lengths.
            If provided, loss_mask and response_mask will be right-padded per
            sample (1s then 0s). Length must equal batch_size. Each value must
            be <= num_actions.
    """

    torch.manual_seed(42)

    loss_mask = torch.ones((batch_size, num_actions), dtype=int, device="cpu")
    response_mask = torch.ones((batch_size, num_actions), dtype=int, device="cpu")
    if action_lengths is not None:
        assert len(action_lengths) == batch_size
        for i, valid_len in enumerate(action_lengths):
            loss_mask[i, valid_len:] = 0
            response_mask[i, valid_len:] = 0

    # Add all the required fields for training
    data = TrainingInputBatch(
        {
            "sequences": torch.randint(0, 100, (batch_size, seq_len), device="cpu"),
            "attention_mask": torch.ones((batch_size, seq_len), dtype=int, device="cpu"),
            "action_log_probs": 0.4 * torch.ones((batch_size, num_actions), device="cpu"),
            "base_action_log_probs": 0.3 * torch.ones((batch_size, num_actions), device="cpu"),
            "values": 0.5 * torch.ones((batch_size, num_actions), device="cpu"),
            "returns": 0.5 * torch.ones((batch_size, num_actions), device="cpu"),
            "advantages": 0.6 * torch.ones((batch_size, num_actions), device="cpu"),
            "loss_mask": loss_mask,
            "response_mask": response_mask,
            "rollout_logprobs": 0.2 * torch.ones((batch_size, num_actions), device="cpu"),
        }
    )
    data.metadata = {"response_length": num_actions}
    return data


def make_dummy_experience(seq_len=10, num_actions=4) -> Experience:
    torch.manual_seed(42)
    B, T = 2, seq_len
    num_actions = num_actions

    return Experience(
        sequences=torch.randint(0, 100, (B, T), device="cpu"),
        action_log_probs=0.4 * torch.ones((B, num_actions), device="cpu"),
        base_action_log_probs=0.3 * torch.ones((B, num_actions), device="cpu"),
        rollout_logprobs=0.2 * torch.ones((B, num_actions), device="cpu"),
        values=0.5 * torch.ones((B, num_actions), device="cpu"),
        returns=0.5 * torch.ones((B, num_actions), device="cpu"),
        advantages=0.6 * torch.ones((B, num_actions), device="cpu"),
        attention_mask=torch.ones((B, T), dtype=int, device="cpu"),
        loss_mask=torch.ones((B, num_actions), dtype=int, device="cpu"),
        action_mask=torch.ones((B, num_actions), dtype=int, device="cpu"),
        num_actions=num_actions,
        info={},
    )


def import_worker(strategy: str, worker_type: str):
    if strategy in ("fsdp", "fsdp2"):
        module_path = "skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker"
    elif strategy == "megatron":
        module_path = "skyrl.backends.skyrl_train.workers.megatron.megatron_worker"
    else:
        raise ValueError(f"Unknown strategy type for {worker_type}: {strategy}")

    module = importlib.import_module(module_path)
    return getattr(module, f"{worker_type.capitalize()}Worker")


def init_worker_with_type(
    worker_type: str, shared_pg=None, colocate_all=False, num_gpus_per_node=1, num_nodes=1, cfg=None
) -> PPORayActorGroup:
    if cfg is None:
        cfg = get_test_actor_config()

    if shared_pg is not None:
        pg = ResolvedPlacementGroup(shared_pg)
        num_gpus_per_actor = 0.2
    else:
        bundles = [{"GPU": num_gpus_per_node, "CPU": num_gpus_per_node} for _ in range(num_nodes)]
        raw_pg = placement_group(bundles, strategy="PACK")
        get_ray_pg_ready_with_timeout(raw_pg, timeout=30)
        pg = ResolvedPlacementGroup(raw_pg)
        num_gpus_per_actor = 0.75

    worker_cls = import_worker(cfg.trainer.strategy, worker_type)
    model = PPORayActorGroup(
        cfg.trainer,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        ray_actor_type=worker_cls,
        pg=pg,
        num_gpus_per_actor=num_gpus_per_actor,
        colocate_all=colocate_all,
        sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
        record_memory=cfg.trainer.policy.record_memory,
    )
    # we use policy model path for all tests (regardless of actor type)
    ray.get(model.async_init_model(cfg.trainer.policy.model.path))
    return model


class Timer:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.opt(depth=1).info(f"{self.message}, time cost: {time.time() - self.start_time:.2f}s")


def get_available_gpus():
    """Get list of available GPU IDs from CUDA_VISIBLE_DEVICES or all available GPUs"""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        # Parse CUDA_VISIBLE_DEVICES (can be comma-separated list)
        gpu_ids = [int(x.strip()) for x in cuda_visible.split(",") if x.strip().isdigit()]
        return gpu_ids
    else:
        # If not set, warn user but proceed with all GPUs
        try:
            import torch

            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_ids = list(range(gpu_count))
                print(f"CUDA_VISIBLE_DEVICES not set. Using all {gpu_count} GPUs: {gpu_ids}")
                print("This might conflict with other processes. Consider setting CUDA_VISIBLE_DEVICES explicitly.")
                return gpu_ids
            else:
                return []
        except Exception as e:
            print(f"Error getting available GPUs: {e}")
            return []


def wait_for_server(url: str, health_path: str, timeout: int = 60, interval: float = 1.0):
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Server at {url} did not come online within {timeout} seconds")
        try:
            response = requests.get(f"http://{url}/{health_path}")
            if response.ok:
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(interval)


def levenshtein(s1, s2):
    m, n = len(s1), len(s2)
    # Initialize matrix of zeros
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    # Initialize first column and first row of the matrix
    for i in range(m + 1):
        dp[i][0] = i  # Deletion from s1 to empty string
    for j in range(n + 1):
        dp[0][j] = j  # Insertion to s1 from empty string
    # Compute the Levenshtein distance matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1  # No cost if characters match
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + cost,  # Substitution
            )
    return dp[m][n]


def are_responses_similar(responses_a: List[str], responses_b: List[str], tolerance: float = 0.01) -> float:
    if len(responses_a) != len(responses_b):
        return False

    total_length = 0
    total_diff = 0

    for s1, s2 in zip(responses_a, responses_b):
        max_len = max(len(s1), len(s2))
        total_length += max_len
        diff = levenshtein(s1, s2)
        total_diff += diff

    difference = float(total_diff / total_length)
    return difference <= tolerance


def get_test_prompts(model: str, num_samples: int = 20) -> List[ConversationType]:
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    # Ensure pad_token is set correctly
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _ensure_chat_template(tokenizer)

    dataset = PromptDataset(
        datasets=[TEST_DATA_PATH],
        tokenizer=tokenizer,
        max_prompt_length=512,
    )

    # Extract the actual prompts from the dataset
    prompts = []
    for i in range(min(num_samples, len(dataset))):
        prompt_data, _, _, _ = dataset[i]  # dataset returns (messages, env_class, extra, uid)
        prompts.append(prompt_data)

    return prompts


def _ensure_chat_template(tokenizer):
    """Set a minimal chat template if the tokenizer doesn't ship with one."""
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}{{ message['content'] + '\n' }}"
            "{% elif message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\n' }}"
            "{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + '\n' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
        )


def get_test_generator_input(
    model: str,
    num_prompts: int = 20,
    n_samples_per_prompt: int = 1,
    max_prompt_length: int = 512,
    data_path: str = TEST_DATA_PATH,
    env_class: str = "gsm8k",
):
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    # Ensure pad_token is set correctly
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    _ensure_chat_template(tokenizer)

    dataset = PromptDataset(
        datasets=[data_path],
        tokenizer=tokenizer,
        max_prompt_length=max_prompt_length,
    )

    prompts = []
    env_extras = []
    for i in range(min(num_prompts, len(dataset))):
        prompt_data, _, env_extra, _ = dataset[i]  # dataset returns (messages, env_class, extra, uid)
        prompts.extend([prompt_data] * n_samples_per_prompt)
        env_extras.extend([env_extra] * n_samples_per_prompt)

    env_classes = [env_class] * len(prompts)

    input_batch: GeneratorInput = {
        "prompts": prompts,
        "env_classes": env_classes,
        "env_extras": env_extras,
        "trajectory_ids": [TrajectoryID(instance_id=f"{i}", repetition_id=0) for i in range(len(prompts))],
    }

    return input_batch


def get_model_logits_from_actor(actor_group: PPORayActorGroup, input_sequences, attention_mask):
    """Helper function to get model logits for comparison"""

    seq_len = input_sequences.shape[1]
    num_actions_val = seq_len - 5  # Leave some tokens for response

    data = TrainingInputBatch(
        {
            "sequences": input_sequences,
            "attention_mask": attention_mask,
        }
    )
    data.metadata = {"response_length": num_actions_val}

    results_refs = actor_group.async_run_ray_method("mesh", "forward", data)
    results = ray.get(results_refs)
    ret_databatch: TrainingOutputBatch = concatenate_outputs_after_mesh_dispatch(actor_group.actor_infos, results)
    logits = ret_databatch["output"]

    return logits


@lru_cache(5)
def log_once(msg):
    logger.info(msg)
    return None


def ray_init_for_tests():
    env_vars = {}
    if not peer_access_supported(max_num_gpus_per_node=4):
        log_once("Disabling NCCL P2P for test environment")
        env_vars = {"NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1"}
    # TODO (erictang000): refactor this to use the same prepare_runtime_environment function as in utils.py for tests
    # to remove duplicate code
    if SKYRL_PYTHONPATH_EXPORT:
        env_vars["PYTHONPATH"] = os.environ.get("PYTHONPATH")
    env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    env_vars["NVTE_FUSED_ATTN"] = "0"
    env_vars["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH")
    env_vars["_SKYRL_USE_NEW_INFERENCE"] = "1" if _SKYRL_USE_NEW_INFERENCE else "0"
    ray.init(runtime_env={"env_vars": env_vars})


async def run_inference(client, prompts, sampling_params, tokenizer=None):
    engine_input = InferenceEngineInput(prompts=prompts, sampling_params=sampling_params)
    if isinstance(client, RemoteInferenceClient):
        # convert to prompt token ids for RemoteInferenceClient
        if tokenizer is None:
            from skyrl.utils.tok import get_tokenizer

            tokenizer = get_tokenizer(client.model_name)
            _ensure_chat_template(tokenizer)
        prompt_token_ids = tokenizer.apply_chat_template(
            prompts,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )
        engine_input = InferenceEngineInput(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)
    return await client.generate(engine_input)


@dataclass
class InferenceEngineState:
    """Manages inference engine lifecycle with clean resource cleanup."""

    client: Union[InferenceEngineClient, RemoteInferenceClient]
    pg: Optional[Any]  # placement group
    router: Optional[VLLMRouter]
    server_groups: Optional[List[ServerGroup]] = None
    prefill_server_groups: Optional[List[ServerGroup]] = None
    decode_server_groups: Optional[List[ServerGroup]] = None

    def __post_init__(self):
        # internal attribute to track if the inference engines need a wake_up()
        # call before generation
        self._needs_wake_up = False
        self._cleanup_pg = False

    def _close_common(self):
        """Shutdown router, server_group, and Ray actors (sync resources).

        For local engines (InferenceEngineClient wrapping RayWrappedInferenceEngines),
        kills the underlying Ray actors so their torch.distributed TCPStore sockets
        are released promptly, preventing port conflicts between tests.
        """
        if self.router is not None:
            self.router.shutdown()
        # Handle shutdown for prefill and decode server groups separately
        group_lists = (
            [self.server_groups]
            if not self.prefill_server_groups
            else [self.prefill_server_groups, self.decode_server_groups]
        )
        for group_list in group_lists:
            if group_list is not None:
                for group in group_list:
                    group.shutdown()
                if self._cleanup_pg:
                    if len(group_list):
                        # TODO (sumanthrh): This is a bit hacky, this assumes pg is the same
                        # for groups in the group list - which is true for creation in
                        # `create_inference_servers`
                        # we should have a better way for cleaning up pg state
                        group = group_list[0]
                        try:
                            ray.util.remove_placement_group(group._get_placement_group())
                        except Exception as e:
                            logger.info(f"Encountered error at pg cleanup: {e}")

        if isinstance(self.client, InferenceEngineClient):
            for engine in self.client.engines:
                if hasattr(engine, "inference_engine_actor"):
                    ray.kill(engine.inference_engine_actor)
            self.client.engines.clear()

    def close(self):
        """Sync close. Use from sync tests, fixtures, and finally blocks."""
        self._close_common()
        if isinstance(self.client, RemoteInferenceClient):
            asyncio.run(self.client.aclose())

    async def aclose(self):
        """Async close. Use from async tests and finally blocks."""
        self._close_common()
        if isinstance(self.client, RemoteInferenceClient):
            await self.client.aclose()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    async def __aenter__(self):
        if self._needs_wake_up:
            await self.client.wake_up()
            self._needs_wake_up = False
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()
        return False

    @classmethod
    def create(
        cls,
        cfg: SkyRLTrainConfig,
        # optional overrides
        model: Optional[str] = None,
        use_local: Optional[bool] = None,
        async_engine: Optional[bool] = None,
        tp_size: Optional[int] = None,
        colocate_all: Optional[bool] = None,
        backend: Optional[str] = None,
        gpu_memory_utilization: Optional[float] = None,
        num_inference_engines: Optional[int] = None,
        sleep_level: int = 2,  # use level 1 in unit tests that do not explicitly sync weights or for LoRA
        enable_lora: bool = False,
        active_lora_name: Optional[str] = None,
        max_num_seqs: Optional[int] = None,
        engine_init_kwargs: Optional[Dict[str, Any]] = None,
        use_new_inference_servers: Optional[bool] = None,
        distributed_executor_backend: Optional[str] = None,
        expert_parallel_size: Optional[int] = None,
        enable_pd: bool = False,
        num_prefill: int = 0,
        language_model_only: Optional[bool] = None,
    ) -> "InferenceEngineState":
        """
        Instantiates inference engines in SkyRL with the provided configuration and overrides

        if `use_new_inference_servers` is not None, it will be used in favour of the `_SKYRL_USE_NEW_INFERENCE` environment variable.
        """
        # create a cfg copy and apply overrides
        cfg = copy.deepcopy(cfg)
        ie_cfg = cfg.generator.inference_engine
        if model is not None:
            cfg.trainer.policy.model.path = model
        if backend is not None:
            ie_cfg.backend = backend
        if use_local is not None:
            ie_cfg.run_engines_locally = use_local
        if async_engine is not None:
            ie_cfg.async_engine = async_engine
        if tp_size is not None:
            ie_cfg.tensor_parallel_size = tp_size
        if colocate_all is not None:
            cfg.trainer.placement.colocate_all = colocate_all
        if gpu_memory_utilization is not None:
            ie_cfg.gpu_memory_utilization = gpu_memory_utilization
        if num_inference_engines is not None:
            ie_cfg.num_engines = num_inference_engines
        if max_num_seqs is not None:
            ie_cfg.max_num_seqs = max_num_seqs
        if engine_init_kwargs is not None:
            ie_cfg.engine_init_kwargs = engine_init_kwargs
        if distributed_executor_backend is not None:
            ie_cfg.distributed_executor_backend = distributed_executor_backend
        if expert_parallel_size is not None:
            ie_cfg.expert_parallel_size = expert_parallel_size
        if enable_pd:
            ie_cfg.enable_pd = True
            ie_cfg.num_prefill = num_prefill
        if language_model_only is not None:
            ie_cfg.language_model_only = language_model_only

        assert ie_cfg.run_engines_locally, "This test does not yet support remote engines."

        if not ray.is_initialized():
            initialize_ray(cfg)
        if cfg.trainer.placement.colocate_all:
            per_engine_gpu_count = (
                ie_cfg.tensor_parallel_size * ie_cfg.pipeline_parallel_size * ie_cfg.data_parallel_size
            )
            total_gpu_slots = ie_cfg.num_engines * per_engine_gpu_count
            raw_pg = placement_group(
                [{"GPU": 1, "CPU": 1}] * total_gpu_slots,
                strategy="PACK",
            )
            get_ray_pg_ready_with_timeout(raw_pg, timeout=60)
            shared_pg = ResolvedPlacementGroup(raw_pg)
            sleep = True
        else:
            shared_pg, sleep = None, False

        # Extract served_model_name from config if set
        served_model_name = ie_cfg.served_model_name

        tokenizer = get_tokenizer(cfg.trainer.policy.model.path)

        # Return both router and server group if created to keep references alive
        router = None
        needs_wake_up = False
        server_groups = None
        prefill_server_groups = None
        decode_server_groups = None
        if use_new_inference_servers or (use_new_inference_servers is None and _SKYRL_USE_NEW_INFERENCE):
            # NOTE: In the case of the new inference backend, server is up by default, so we don't need
            # any special handling for sleep
            cli_args = build_vllm_cli_args(cfg)
            if enable_lora:
                cli_args.enable_lora = True
            if cli_args.enable_lora and active_lora_name is None:
                active_lora_name = "skyrl-lora"

            setup = create_inference_servers(
                ie_cfg,
                cli_args,
                log_path=cfg.trainer.log_path,
                placement_group=shared_pg if cfg.trainer.placement.colocate_all else None,
            )
            router = setup.router
            server_groups = setup.server_groups
            prefill_server_groups = setup.prefill_server_groups
            decode_server_groups = setup.decode_server_groups
            proxy_url = setup.proxy_url
            server_urls = setup.server_urls

            client = RemoteInferenceClient(
                proxy_url=proxy_url,
                server_urls=server_urls,
                model_name=served_model_name if served_model_name else cfg.trainer.policy.model.path,
                enable_return_routed_experts=ie_cfg.enable_return_routed_experts,
                active_lora_name=active_lora_name,
                uses_lora_weight_sync=_uses_lora_weight_sync(cfg),
                data_parallel_size=ie_cfg.data_parallel_size,
                tokenizer=get_tokenizer(cfg.trainer.policy.model.path),
            )
        else:
            eps = create_ray_wrapped_inference_engines(
                num_inference_engines=ie_cfg.num_engines,
                tensor_parallel_size=ie_cfg.tensor_parallel_size,
                expert_parallel_size=ie_cfg.expert_parallel_size,
                model_dtype="bfloat16",
                pretrain=cfg.trainer.policy.model.path,
                seed=42,
                vllm_v1_disable_multiproc=True,
                data_parallel_size=ie_cfg.data_parallel_size,
                enable_prefix_caching=ie_cfg.enable_prefix_caching,
                enforce_eager=ie_cfg.enforce_eager,
                shared_pg=shared_pg,
                gpu_memory_utilization=ie_cfg.gpu_memory_utilization,
                inference_engine_enable_sleep=sleep,
                async_engine=ie_cfg.async_engine,
                max_num_batched_tokens=8192,
                max_num_seqs=ie_cfg.max_num_seqs,
                tokenizer=tokenizer,
                backend=ie_cfg.backend,
                sleep_level=sleep_level,
                enable_lora=enable_lora,
                engine_init_kwargs=ie_cfg.engine_init_kwargs,
                enable_return_routed_experts=ie_cfg.enable_return_routed_experts,
                served_model_name=served_model_name,
                distributed_executor_backend=ie_cfg.distributed_executor_backend,
            )
            client = InferenceEngineClient(
                eps, tokenizer, cfg.trainer.policy.model.path, cfg.trainer.policy.model.lora, ie_cfg
            )
            if sleep:
                # NOTE: this is a hacky fix to allow creation from both sync and async contexts
                # TODO: simplify this when old inference path is removed and unify on async context
                try:
                    asyncio.get_running_loop()
                    # Inside an async context (e.g. pytest-asyncio) - defer wake_up to __aenter__
                    needs_wake_up = True
                except RuntimeError:
                    asyncio.run(client.wake_up())
                    needs_wake_up = False
            else:
                needs_wake_up = False
        state = cls(
            client=client,
            pg=raw_pg if shared_pg else None,
            router=router,
            server_groups=server_groups,
            prefill_server_groups=prefill_server_groups,
            decode_server_groups=decode_server_groups,
        )
        state._needs_wake_up = needs_wake_up
        state._cleanup_pg = not shared_pg
        return state


def init_remote_inference_servers(
    tp_size: int,
    backend: str,
    tokenizer: PreTrainedTokenizerBase,
    config: SkyRLTrainConfig,
    model: str,
) -> Tuple[InferenceEngineClient, subprocess.Popen]:
    available_gpus = get_available_gpus()
    assert (
        len(available_gpus) >= tp_size
    ), f"Not enough GPUs available. Need {tp_size}, but only {len(available_gpus)} available: {available_gpus}"

    selected_gpus = available_gpus[:tp_size]
    gpu_ids_str = ",".join(map(str, selected_gpus))
    print(f"Using GPUs {gpu_ids_str} for vLLM server (tensor_parallel_size={tp_size})")

    def get_free_port():
        import socket

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    engine_port = get_free_port()

    # Launch vLLM server using subprocess
    if backend == "vllm":
        remote_server_command = [
            "uv",
            "run",
            "--isolated",
            "--extra",
            "fsdp",
            "-m",
            "skyrl.backends.skyrl_train.inference_engines.vllm.vllm_server",
            "--model",
            model,
            "--enforce-eager",
            "--gpu-memory-utilization",
            "0.8",
            "--tensor-parallel-size",
            str(tp_size),
            # TODO (erictang000): for 0.13+ vllm, the MP backend runs into issues with CUDA_VISIBLE_DEVICES
            # when we refactor the inference backend to use remote inference engines as a default, revisit this
            "--distributed-executor-backend",
            "ray",
            "--dtype",
            "bfloat16",
            "--host",
            "127.0.0.1",
            "--port",
            str(engine_port),
            "--worker-extension-cls",
            "skyrl.backends.skyrl_train.inference_engines.vllm.vllm_engine.WorkerWrap",
        ]
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    # Set CUDA_VISIBLE_DEVICES environment variable for the subprocess
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids_str

    # Start the vLLM server process
    server_process = subprocess.Popen(remote_server_command, env=env)
    try:
        wait_for_server(url=f"localhost:{engine_port}", health_path="health", timeout=400)
    except TimeoutError as e:
        print(f"Received timeout error while waiting for server: {e}")
        server_process.terminate()
        server_process.wait()
        raise

    print(f"Server at localhost:{engine_port} is online")

    engines = create_remote_inference_engines(
        urls=[f"localhost:{engine_port}"],
        model_name=model,
        tokenizer=tokenizer,
        engine_backend=backend,
        tensor_parallel_size=tp_size,
        data_parallel_size=1,
        expert_parallel_size=1,
    )

    client = InferenceEngineClient(
        engines,
        tokenizer,
        config.trainer.policy.model.path,
        config.trainer.policy.model.lora,
        config.generator.inference_engine,
    )
    return client, server_process
