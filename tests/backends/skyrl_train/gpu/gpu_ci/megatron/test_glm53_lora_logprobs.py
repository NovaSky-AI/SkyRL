"""GLM 5.3 LoRA parity on a multi-node B300 Ray cluster.

Run with:
SKYRL_GLM53_SHARED_DIR=/shared/skyrl-tests \
uv run --isolated --extra dev --extra megatron -- \
pytest -s -m b300 tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_glm53_lora_logprobs.py

For a two-GPU Qwen dry run, also set:
SKYRL_GLM53_MODEL=Qwen/Qwen2.5-1.5B-Instruct

For a prestarted multi-node Ray cluster, also set:
SKYRL_GLM53_RAY_ADDRESS=auto
"""

import hashlib
import os
import shutil
import uuid
from pathlib import Path

import pytest
import ray
import torch

from skyrl.backends.skyrl_train.distributed.dispatch import (
    WorkerOutput,
    loss_fn_outputs_to_tensor,
)
from skyrl.backends.skyrl_train.inference_servers.base import InferenceEngineInput
from skyrl.backends.skyrl_train.inference_servers.engine_utils import (
    get_sampling_params_for_backend,
)
from skyrl.backends.skyrl_train.inference_servers.utils import resolve_policy_model_name
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.workers.megatron import (
    megatron_worker as _megatron_worker_mod,
)
from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
    MegatronPolicyWorkerBase,
)
from skyrl.train.config import SamplingParams, SkyRLLoraConfig, SkyRLTrainConfig
from skyrl.train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl.train.utils.utils import validate_cfg
from skyrl.utils.tok import get_tokenizer
from tests.backends.skyrl_train.gpu.gpu_ci.conftest import ray_init
from tests.backends.skyrl_train.gpu.utils import (
    InferenceEngineState,
    Timer,
    init_worker_with_type,
)

MODEL = "zai-org/GLM-5.3"
SMALL_DRY_RUN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
POLICY_GPUS = 8
INFERENCE_TP = 8
MAX_GENERATE_LENGTH = 128
MEGATRON_MEAN_DIFF_THRESHOLD = 5e-2
VLLM_MEAN_DIFF_THRESHOLD = 3e-1
LORA_NOISE_SEED = 42
LORA_NOISE_STD = 1e-3
MIN_UPDATED_LOGPROB_DIFF = 1e-5

TEST_PROMPTS = [
    "What is 2 + 3? Answer with only the number.",
    "Name the largest planet in our solar system. Answer briefly.",
    "Complete the sequence: 1, 1, 2, 3, 5, __.",
    "Write one short sentence explaining why ice floats on water.",
]

MEGATRON_LORA_TARGET_MODULES = [
    "linear_q_down_proj",
    "linear_q_up_proj",
    "linear_kv_down_proj",
    "linear_kv_up_proj",
    "linear_proj",
    "linear_fc1",
    "linear_fc2",
]

VLLM_LORA_TARGET_MODULES = [
    "fused_qkv_a_proj",
    "q_a_proj",
    "q_b_proj",
    "q_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "o_proj",
    "gate_up_proj",
    "down_proj",
    "experts",
]


class _PerturbableMegatronPolicyWorker(MegatronPolicyWorkerBase):
    def add_lora_noise(self, seed: int, std: float) -> dict[str, float | int]:
        from megatron.core.utils import unwrap_model

        updated_parameters = 0
        updated_elements = 0
        delta_norm = 0.0
        assert self._is_lora

        with torch.no_grad():
            for chunk_index, chunk in enumerate(self.actor_module):
                model = unwrap_model(chunk)
                for name, parameter in model.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    digest = hashlib.sha256(f"{chunk_index}:{name}".encode()).digest()
                    parameter_seed = seed + int.from_bytes(digest[:8], "little")
                    generator = torch.Generator(device=parameter.device)
                    generator.manual_seed(parameter_seed % (2**63 - 1))
                    noise = torch.randn(
                        parameter.shape,
                        dtype=parameter.dtype,
                        device=parameter.device,
                        generator=generator,
                    )
                    noise.mul_(std)
                    parameter.add_(noise)

                    updated_parameters += 1
                    updated_elements += parameter.numel()
                    delta_norm += torch.linalg.vector_norm(
                        noise, dtype=torch.float32
                    ).item()

        assert updated_parameters > 0, "noise update found no trainable LoRA parameters"
        return {
            "updated_parameters": updated_parameters,
            "updated_elements": updated_elements,
            "delta_norm": delta_norm,
        }


_PerturbablePolicyWorker = ray.remote(num_gpus=1)(_PerturbableMegatronPolicyWorker)


@pytest.fixture
def glm53_ray_init_fixture():
    with ray_init(
        extra_env_vars={"NVTE_FUSED_ATTN": "1"},
        address=os.environ.get("SKYRL_GLM53_RAY_ADDRESS"),
    ):
        yield


def _get_test_topology(model: str) -> tuple[int, int]:
    if model == SMALL_DRY_RUN_MODEL:
        return 1, 1
    return POLICY_GPUS, INFERENCE_TP


def _get_glm53_lora_config(model: str, lora_sync_path: str) -> SkyRLTrainConfig:
    policy_gpus, inference_tp = _get_test_topology(model)
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.logger = "console"
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.loss_reduction = "sequence_mean"
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = policy_gpus
    cfg.trainer.policy.model.path = model
    cfg.trainer.policy.language_model_only = True
    cfg.trainer.ref.language_model_only = True
    cfg.trainer.policy.inference_only_init = True
    cfg.trainer.policy.model.lora = SkyRLLoraConfig(
        rank=32,
        alpha=32,
        dropout=0.0,
        lora_sync_path=lora_sync_path,
        target_modules=(
            "all-linear"
            if model == SMALL_DRY_RUN_MODEL
            else MEGATRON_LORA_TARGET_MODULES
        ),
        max_loras=1,
    )

    megatron = cfg.trainer.policy.megatron_config
    megatron.tensor_model_parallel_size = policy_gpus
    megatron.pipeline_model_parallel_size = 1
    megatron.context_parallel_size = 1
    megatron.expert_model_parallel_size = 1 if model == SMALL_DRY_RUN_MODEL else 8
    megatron.expert_tensor_parallel_size = 1
    megatron.moe_token_dispatcher_type = "alltoall"
    megatron.moe_router_load_balancing_type = "none"
    megatron.moe_grouped_gemm = True
    megatron.moe_router_score_function = "sigmoid"
    megatron.lora_config.merge_lora = False
    megatron.ddp_config.average_in_collective = False
    megatron.transformer_config_kwargs = {
        "dsa_kernel_backend": "tilelang",
        "qk_pos_emb_head_dim": 64,
        "dsa_indexer_topk_freq": 4,
        "dsa_indexer_skip_topk_offset": 3,
        "dsa_indexer_rope_interleaved": True,
        "dsa_indexer_rotate_activation": False,
        "dsa_indexer_k_norm_epsilon": 1e-6,
        "mtp_num_layers": 0,
        "mtp_use_repeated_layer": False,
        "calculate_per_token_loss": True,
        "gradient_accumulation_fusion": False,
        "sequence_parallel": True,
        "recompute_granularity": "full",
        "recompute_method": "uniform",
        "recompute_num_layers": 1,
        "recompute_modules": [],
    }
    if model == SMALL_DRY_RUN_MODEL:
        megatron.transformer_config_kwargs = {
            "calculate_per_token_loss": True,
            "gradient_accumulation_fusion": False,
            "sequence_parallel": False,
        }

    cfg.trainer.flash_attn = False
    cfg.trainer.remove_microbatch_padding = False
    cfg.trainer.fused_lm_head_logprob = True
    max_sequence_length = 1024 if model == SMALL_DRY_RUN_MODEL else 32768
    cfg.trainer.max_tokens_per_microbatch = max_sequence_length
    cfg.trainer.logprobs_chunk_size = min(8192, max_sequence_length)
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.micro_train_batch_size_per_gpu = 1

    inference = cfg.generator.inference_engine
    inference.backend = "vllm"
    inference.run_engines_locally = True
    inference.language_model_only = True
    inference.num_engines = 1
    inference.tensor_parallel_size = inference_tp
    inference.pipeline_parallel_size = 1
    inference.data_parallel_size = 1
    inference.distributed_executor_backend = "ray"
    inference.weight_sync_backend = "nccl"
    inference.enforce_eager = False
    inference.gpu_memory_utilization = 0.8
    inference.max_num_seqs = 8
    inference.max_num_batched_tokens = max_sequence_length
    inference.enable_prefix_caching = False
    inference.enable_chunked_prefill = True
    inference.engine_init_kwargs = {
        "max_model_len": 32768,
        "kv_cache_dtype": "fp8",
        "disable_custom_all_reduce": True,
        "linear_backend": "triton",
        "moe_backend": "triton",
        "lora_target_modules": VLLM_LORA_TARGET_MODULES,
        "trust_remote_code": True,
    }
    if model == SMALL_DRY_RUN_MODEL:
        inference.enforce_eager = True
        inference.engine_init_kwargs = {
            "max_model_len": max_sequence_length,
            "disable_custom_all_reduce": True,
            "trust_remote_code": True,
        }

    cfg.generator.sampling_params = SamplingParams(
        max_generate_length=MAX_GENERATE_LENGTH,
        logprobs=1,
        temperature=0.0,
    )
    cfg.generator.batched = False
    cfg.generator.max_turns = 1
    validate_cfg(cfg)
    return cfg


def _get_prompt_token_ids(tokenizer) -> list[list[int]]:
    conversations = [[{"role": "user", "content": prompt}] for prompt in TEST_PROMPTS]
    return tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=False,
    )


async def _generate(client, tokenizer, model: str | None = None):
    prompt_token_ids = _get_prompt_token_ids(tokenizer)
    sampling_params = get_sampling_params_for_backend(
        "vllm",
        SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_generate_length=MAX_GENERATE_LENGTH,
            min_p=0.0,
            logprobs=1,
        ),
    )

    with Timer("generate_with_vllm"):
        output = await client.generate(
            InferenceEngineInput(
                prompt_token_ids=prompt_token_ids, sampling_params=sampling_params
            ),
            model=model,
        )

    responses = output["response_ids"]
    rollout_logprobs = output["response_logprobs"]
    assert rollout_logprobs is not None
    rewards = [[0.0] * len(response) for response in responses]
    loss_masks = [[1] * len(response) for response in responses]

    sequences, attention_mask, response_mask, rewards_t, loss_mask_t, logprobs_t, _ = (
        convert_prompts_responses_to_batch_tensors(
            pad_token_id=tokenizer.pad_token_id,
            prompts=prompt_token_ids,
            responses=responses,
            rewards=rewards,
            loss_masks=loss_masks,
            logprobs=rollout_logprobs,
        )
    )
    assert logprobs_t is not None
    num_actions = response_mask.shape[1]
    batch_size = sequences.shape[0]
    training_input = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "rewards": rewards_t,
            "loss_mask": loss_mask_t,
            "rollout_logprobs": logprobs_t,
            "rollout_expert_indices": None,
            "action_log_probs": torch.zeros(
                (batch_size, num_actions), dtype=torch.float32
            ),
            "base_action_log_probs": torch.zeros(
                (batch_size, num_actions), dtype=torch.float32
            ),
            "advantages": torch.zeros((batch_size, num_actions), dtype=torch.float32),
        }
    )
    training_input.metadata = {"response_length": num_actions}
    return responses, response_mask, logprobs_t, training_input


def _get_megatron_logprobs(policy, training_input):
    refs = policy.async_run_ray_method("mesh", "forward", data=training_input)
    results = ray.get(refs)
    output = WorkerOutput.cat(policy.actor_infos, results)
    return loss_fn_outputs_to_tensor(output.loss_fn_outputs, key="logprobs")


def _assert_logprobs_match(label, expected, actual, response_mask, threshold):
    mask = response_mask.bool()
    expected_valid = expected[mask]
    actual_valid = actual[mask]
    difference = (expected_valid - actual_valid).abs()
    mean_difference = difference.mean().item()
    print(
        f"{label}: tokens={difference.numel()}, mean_diff={mean_difference:.6f}, "
        f"max_diff={difference.max().item():.6f}"
    )
    assert torch.isfinite(difference).all()
    assert mean_difference < threshold, (
        f"{label} mean diff {mean_difference:.6f} exceeds {threshold}"
    )


def _assert_logprobs_changed(before, after, response_mask):
    difference = (before[response_mask.bool()] - after[response_mask.bool()]).abs()
    mean_difference = difference.mean().item()
    print(
        f"dummy LoRA update effect: tokens={difference.numel()}, "
        f"mean_diff={mean_difference:.6f}, max_diff={difference.max().item():.6f}"
    )
    assert torch.isfinite(difference).all()
    assert mean_difference > MIN_UPDATED_LOGPROB_DIFF, (
        f"dummy LoRA update changed mean logprob by only {mean_difference:.6f}"
    )


def _init_perturbable_policy(cfg, client, policy_gpus):
    original_policy_worker = _megatron_worker_mod.PolicyWorker
    _megatron_worker_mod.PolicyWorker = _PerturbablePolicyWorker
    try:
        policy = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_nodes=1,
            num_gpus_per_node=policy_gpus,
            cfg=cfg,
        )
    finally:
        _megatron_worker_mod.PolicyWorker = original_policy_worker

    ray.get(
        policy.async_run_ray_method(
            "pass_through",
            "init_weight_sync_state",
            client,
            cfg.generator.inference_engine,
        )
    )
    return policy


@pytest.mark.asyncio
@pytest.mark.megatron
@pytest.mark.b300
async def test_glm53_lora_init_and_dummy_update_match_vllm(glm53_ray_init_fixture):
    model = os.environ.get("SKYRL_GLM53_MODEL", MODEL)
    policy_gpus, inference_tp = _get_test_topology(model)
    shared_dir = Path(os.environ["SKYRL_GLM53_SHARED_DIR"])
    lora_sync_path = shared_dir / f"glm53-lora-parity-{uuid.uuid4().hex}"
    lora_sync_path.mkdir(parents=True)

    assert ray.cluster_resources().get("GPU", 0) >= policy_gpus + inference_tp, (
        f"LoRA parity requires {policy_gpus + inference_tp} GPUs in the connected Ray cluster"
    )

    cfg = _get_glm53_lora_config(model, str(lora_sync_path))
    tokenizer = get_tokenizer(model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        async with InferenceEngineState.create(
            cfg=cfg,
            model=model,
            use_local=True,
            colocate_all=False,
            backend="vllm",
            enable_lora=True,
        ) as engines:
            client = engines.client
            adapter_loaded = False
            try:
                base_responses, base_mask, base_logprobs, base_input = await _generate(
                    client, tokenizer, model
                )
                policy = _init_perturbable_policy(cfg, client, policy_gpus)

                initial_megatron_logprobs = _get_megatron_logprobs(policy, base_input)
                _assert_logprobs_match(
                    "base vLLM vs initialized Megatron LoRA",
                    base_logprobs,
                    initial_megatron_logprobs,
                    base_mask,
                    MEGATRON_MEAN_DIFF_THRESHOLD,
                )

                with Timer("publish_initialized_lora"):
                    ray.get(
                        policy.async_run_ray_method(
                            "pass_through",
                            "broadcast_to_inference_engines",
                            client,
                            cfg.generator.inference_engine,
                        )
                    )
                adapter_loaded = True
                await client.reset_prefix_cache()

                adapter_name = resolve_policy_model_name(cfg)
                lora_responses, lora_mask, lora_logprobs, lora_input = await _generate(
                    client, tokenizer, adapter_name
                )
                assert base_responses == lora_responses
                assert torch.equal(base_mask, lora_mask)
                _assert_logprobs_match(
                    "base vLLM vs initialized vLLM LoRA",
                    base_logprobs,
                    lora_logprobs,
                    base_mask,
                    VLLM_MEAN_DIFF_THRESHOLD,
                )

                lora_megatron_logprobs = _get_megatron_logprobs(policy, lora_input)
                _assert_logprobs_match(
                    "initialized vLLM LoRA vs Megatron LoRA",
                    lora_logprobs,
                    lora_megatron_logprobs,
                    lora_mask,
                    MEGATRON_MEAN_DIFF_THRESHOLD,
                )

                with Timer("apply_dummy_lora_update"):
                    update_receipts = ray.get(
                        policy.async_run_ray_method(
                            "pass_through",
                            "add_lora_noise",
                            LORA_NOISE_SEED,
                            LORA_NOISE_STD,
                        )
                    )
                for receipt in update_receipts:
                    assert receipt["updated_parameters"] > 0
                    assert receipt["updated_elements"] > 0
                    assert receipt["delta_norm"] > 0

                updated_logprobs = _get_megatron_logprobs(policy, lora_input)
                _assert_logprobs_changed(
                    lora_megatron_logprobs, updated_logprobs, lora_mask
                )

                await client.unload_lora_adapter(adapter_name)
                adapter_loaded = False
                with Timer("publish_dummy_updated_lora"):
                    ray.get(
                        policy.async_run_ray_method(
                            "pass_through",
                            "broadcast_to_inference_engines",
                            client,
                            cfg.generator.inference_engine,
                        )
                    )
                adapter_loaded = True
                await client.reset_prefix_cache()

                _, updated_mask, updated_vllm_logprobs, updated_input = await _generate(
                    client, tokenizer, adapter_name
                )
                updated_megatron_logprobs = _get_megatron_logprobs(
                    policy, updated_input
                )
                _assert_logprobs_match(
                    "dummy-updated vLLM LoRA vs Megatron LoRA",
                    updated_vllm_logprobs,
                    updated_megatron_logprobs,
                    updated_mask,
                    MEGATRON_MEAN_DIFF_THRESHOLD,
                )
            finally:
                if adapter_loaded:
                    await client.unload_lora_adapter(resolve_policy_model_name(cfg))
    finally:
        shutil.rmtree(lora_sync_path)
