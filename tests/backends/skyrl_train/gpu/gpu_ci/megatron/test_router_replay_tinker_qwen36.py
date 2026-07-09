"""GPU test: Rollout Routing Replay (R3) through the Tinker/Megatron path on Qwen3.6-35B-A3B.

R3 records the MoE expert selections vLLM makes during rollout and replays them in
the Megatron training forward, so training activates the same experts inference did.
This removes the routing-driven component of the train-vs-rollout logprob mismatch
that destabilizes RL on MoE models.

R3 is enabled purely by a launch flag (``moe_enable_routing_replay``); the SkyRL-Train
backend caches the captured routing server-side keyed by the full sampled sequence and
replays it on forward_backward, with no client-facing Tinker fields. This test drives
that exact path: it captures routing during sampling, stores it via the backend's cache
helper, rebuilds the training tensor via a cache lookup keyed by the training sequence,
and asserts R3 lowers the mean |vLLM logprob - Megatron logprob| versus replay disabled.

Runs on 1 node of 8xH200. Run with:
  NVTE_FLASH_ATTN=0 uv run --isolated --extra dev --extra megatron --extra tinker -- \
    pytest -s tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_router_replay_tinker_qwen36.py
"""

from collections import OrderedDict
from types import SimpleNamespace

import pytest
import ray
import torch
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.distributed.dispatch import (
    WorkerOutput,
    loss_fn_outputs_to_tensor,
)
from skyrl.backends.skyrl_train.inference_servers.engine_utils import (
    get_sampling_params_for_backend,
)
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train_backend import (
    SkyRLTrainBackend,
    _build_rollout_expert_indices,
    _routing_cache_key,
)
from skyrl.train.config import SamplingParams, SkyRLTrainConfig
from skyrl.train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl.train.generators.base import GeneratorInput
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator
from skyrl.train.utils.utils import validate_cfg
from tests.backends.skyrl_train.gpu.utils import (
    InferenceEngineState,
    Timer,
    get_test_generator_input,
    init_worker_with_type,
)

MODEL_NAME = "Qwen/Qwen3.6-35B-A3B"
NUM_PROMPTS = 8
N_SAMPLES_PER_PROMPT = 2
MAX_GENERATE_LENGTH = 256


def get_test_actor_config() -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.policy.model.path = MODEL_NAME
    # Qwen3.6 (qwen3_5_moe) is a hybrid GDN model: language_model_only routes it
    # to the native GPTModel + GDN thd packing path, which supports sample packing.
    cfg.trainer.policy.language_model_only = True
    cfg.generator.inference_engine.language_model_only = True
    cfg.trainer.remove_microbatch_padding = True
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.placement.policy_num_gpus_per_node = 8
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 2
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.context_parallel_size = 1
    cfg.trainer.policy.megatron_config.expert_model_parallel_size = 8
    cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = 1
    cfg.generator.inference_engine.tensor_parallel_size = 8
    cfg.generator.inference_engine.enable_return_routed_experts = True
    # validate_cfg ties capture to replay; set both so the sampling config is valid.
    cfg.trainer.policy.megatron_config.moe_enable_routing_replay = True
    cfg.generator.inference_engine.gpu_memory_utilization = 0.6
    cfg.generator.inference_engine.distributed_executor_backend = "mp"
    # See https://github.com/vllm-project/vllm/issues/36921 for the GDN prefill backend.
    cfg.generator.inference_engine.engine_init_kwargs = {"gdn_prefill_backend": "triton"}
    # No ref model in this forward-only test; disable KL so validate_cfg doesn't
    # require ref.language_model_only.
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_kl_in_reward = False
    cfg.trainer.logger = "console"
    validate_cfg(cfg)
    return cfg


def _cache_backend():
    """Stand-in exposing just the cache attributes the R3 store helper touches."""
    return SimpleNamespace(_routed_experts_cache=OrderedDict(), _routed_experts_cache_cap=1 << 20)


@pytest.mark.megatron
def test_r3_reduces_train_rollout_mismatch_via_tinker_path(ray_init_fixture):
    """R3 (server-side routing cache + Megatron replay) lowers train-vs-rollout logprob mismatch."""
    try:
        cfg = get_test_actor_config()
        cfg.generator.sampling_params = SamplingParams(
            max_generate_length=MAX_GENERATE_LENGTH,
            logprobs=1,
            temperature=1.0,
        )
        cfg.generator.batched = False
        cfg.generator.max_turns = 1

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

        with InferenceEngineState.create(
            cfg=cfg,
            model=MODEL_NAME,
            use_local=True,
            colocate_all=True,
            backend="vllm",
            sleep_level=1,
            gpu_memory_utilization=0.6,
        ) as engines:
            import asyncio

            client, pg = engines.client, engines.pg
            asyncio.run(client.wake_up())

            generator = SkyRLGymGenerator(
                generator_cfg=cfg.generator,
                skyrl_gym_cfg=cfg.environment.skyrl_gym,
                inference_engine_client=client,
                tokenizer=tokenizer,
            )

            input_batch: GeneratorInput = get_test_generator_input(
                model=MODEL_NAME,
                num_prompts=NUM_PROMPTS,
                n_samples_per_prompt=N_SAMPLES_PER_PROMPT,
                max_prompt_length=512,
                env_class="gsm8k",
            )
            input_batch["sampling_params"] = get_sampling_params_for_backend(
                "vllm",
                SamplingParams(
                    temperature=1.0,
                    top_p=1.0,
                    top_k=-1,
                    max_generate_length=MAX_GENERATE_LENGTH,
                    min_p=0.0,
                    logprobs=1,
                ),
            )

            with Timer("generate_with_routing_capture"):
                generator_output = asyncio.run(generator.generate(input_batch))

            indices = generator_output["rollout_expert_indices"]
            responses = generator_output["response_ids"]
            prompt_token_ids = generator_output["prompt_token_ids"]
            assert indices is not None, "rollout_expert_indices is None; vLLM routing capture failed for Qwen3.6"
            assert len(indices) == len(responses)
            asyncio.run(client.sleep())

        # Exercise the real server-side cache: store each sample's routing keyed by
        # (model_id, digest of the full sampled sequence), exactly as the backend
        # does in _aggregate_sample_results.
        model_id = "r3_test_model"
        cache_backend = _cache_backend()
        for prompt_ids, response, sample_routing in zip(prompt_token_ids, responses, indices):
            if sample_routing:
                SkyRLTrainBackend._store_routed_experts(cache_backend, model_id, prompt_ids, response, sample_routing)

        rewards = generator_output["rewards"]
        if rewards and not isinstance(rewards[0], list):
            rewards = [[r] * len(resp) for r, resp in zip(rewards, responses)]

        # Reference tensors (sequences/masks/rollout logprobs) via the native helper.
        sequences, attention_mask, response_mask, rewards_t, loss_mask_t, logprobs_t, _ = (
            convert_prompts_responses_to_batch_tensors(
                tokenizer=tokenizer,
                prompts=prompt_token_ids,
                responses=responses,
                rewards=rewards,
                loss_masks=generator_output["loss_masks"],
                logprobs=generator_output.get("rollout_logprobs"),
            )
        )
        assert logprobs_t is not None

        # forward_backward reconstructs the sequence as prompt + response; look routing
        # up from the cache by that key and build the training tensor.
        max_seq_len = sequences.shape[1]
        full_sequences = [list(p) + list(r) for p, r in zip(prompt_token_ids, responses)]
        per_sample = [
            cache_backend._routed_experts_cache.get(_routing_cache_key(model_id, fs)) for fs in full_sequences
        ]
        assert any(r is not None for r in per_sample), "no cached routing matched the training sequences"
        rii_tensor = _build_rollout_expert_indices(full_sequences, per_sample, max_seq_len)
        assert rii_tensor is not None

        num_actions = response_mask.shape[1]
        batch_size = sequences.shape[0]

        def build_training_input(with_replay: bool) -> TrainingInputBatch:
            batch = {
                "sequences": sequences,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "rewards": rewards_t,
                "loss_mask": loss_mask_t,
                "rollout_logprobs": logprobs_t,
                "action_log_probs": torch.zeros((batch_size, num_actions), dtype=torch.float32),
                "base_action_log_probs": torch.zeros((batch_size, num_actions), dtype=torch.float32),
                "advantages": torch.zeros((batch_size, num_actions), dtype=torch.float32),
                "action_mask": response_mask.to(dtype=torch.int64),
            }
            if with_replay:
                batch["rollout_expert_indices"] = rii_tensor
            ti = TrainingInputBatch(batch)
            ti.metadata = {"response_length": num_actions}
            return ti

        def run_megatron_forward(enable_replay: bool) -> torch.Tensor:
            cfg.trainer.policy.megatron_config.moe_enable_routing_replay = enable_replay
            actor_group = init_worker_with_type("policy", shared_pg=pg, colocate_all=True, num_gpus_per_node=8, cfg=cfg)
            training_input = build_training_input(with_replay=enable_replay)
            refs = actor_group.async_run_ray_method("mesh", "forward", data=training_input)
            output = WorkerOutput.cat(actor_group.actor_infos, ray.get(refs))
            logprobs = loss_fn_outputs_to_tensor(output.loss_fn_outputs, key="logprobs")
            for actor in actor_group._actor_handlers:
                ray.kill(actor)
            return logprobs

        r3_logprobs = run_megatron_forward(enable_replay=True)
        no_r3_logprobs = run_megatron_forward(enable_replay=False)

        mask = response_mask.bool()
        vllm_valid = logprobs_t[mask]
        r3_diff = (vllm_valid - r3_logprobs[mask]).abs()
        no_r3_diff = (vllm_valid - no_r3_logprobs[mask]).abs()

        print(f"vLLM logprobs     - mean: {vllm_valid.mean().item():.6f}")
        print(f"With replay    - |logprob diff| mean: {r3_diff.mean().item():.6f}, max: {r3_diff.max().item():.6f}")
        print(
            f"Without replay - |logprob diff| mean: {no_r3_diff.mean().item():.6f}, max: {no_r3_diff.max().item():.6f}"
        )

        assert r3_diff.mean().item() < no_r3_diff.mean().item(), (
            "Router replay through the Tinker path should reduce train-vs-rollout logprob "
            f"mismatch, but with_replay={r3_diff.mean().item():.6f} >= "
            f"without_replay={no_r3_diff.mean().item():.6f}"
        )
    finally:
        ray.shutdown()
