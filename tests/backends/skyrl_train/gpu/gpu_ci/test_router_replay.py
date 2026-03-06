"""
Run with:
uv run --isolated --extra dev --extra megatron -- pytest -s tests/backends/skyrl_train/gpu/gpu_ci/test_router_replay.py
"""

import ray
import pytest
import asyncio
import torch
from transformers import AutoTokenizer
from tests.backends.skyrl_train.gpu.utils import (
    InferenceEngineState,
    get_test_generator_input,
    Timer,
    init_worker_with_type,
)
from skyrl.backends.skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch
from skyrl.train.utils.utils import validate_cfg
from skyrl.train.config import SkyRLTrainConfig, SamplingParams
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator
from skyrl.train.generators.base import GeneratorInput
from skyrl.backends.skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl.train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.utils.replay_utils import _split_replay_indices

MOE_MODEL_NAME = "/home/ray/moonlight16b"
# MOE_MODEL_NAME = "Qwen/Qwen3-30B-A3B"
REPLAY_NUM_LAYERS = 2
NUM_PROMPTS = 10
N_SAMPLES_PER_PROMPT = 5


def get_test_actor_config(model_name=MOE_MODEL_NAME) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = model_name
    cfg.trainer.micro_forward_batch_size_per_gpu = 2
    cfg.trainer.micro_train_batch_size_per_gpu = 2
    cfg.trainer.use_sample_packing = False
    cfg.trainer.logger = "console"
    if "moonlight" in model_name:
        # flash attn not supported for moonlight16b
        cfg.trainer.policy.megatron_config.moe_token_dispatcher_type = "alltoall"
        cfg.trainer.policy.megatron_config.moe_router_load_balancing_type = "seq_aux_loss"
        cfg.trainer.policy.megatron_config.transformer_config_kwargs["moe_aux_loss_coeff"] = 0
        cfg.trainer.policy.megatron_config.moe_router_score_function = "sigmoid"
        cfg.trainer.policy.megatron_config.moe_router_enable_expert_bias = True
        cfg.trainer.policy.megatron_config.transformer_config_kwargs["moe_router_bias_update_rate"] = 0
        cfg.trainer.policy.megatron_config.transformer_config_kwargs["moe_router_dtype"] = "fp32"
        cfg.trainer.policy.megatron_config.transformer_config_kwargs["moe_router_topk"] = 6
        cfg.trainer.policy.megatron_config.transformer_config_kwargs["moe_router_pre_softmax"] = True
        cfg.trainer.policy.megatron_config.transformer_config_kwargs["moe_router_group_topk"] = 1
        cfg.trainer.policy.megatron_config.transformer_config_kwargs["moe_router_num_groups"] = 1
        cfg.trainer.flash_attn = False
    validate_cfg(cfg)
    return cfg


@pytest.mark.megatron
def test_megatron_router_replay(ray_init_fixture):
    """
    Test that SkyRLGymGenerator returns rollout_inference_indices
    for MoE models with enable_return_routed_experts=True.
    """
    try:
        cfg = get_test_actor_config(model_name=MOE_MODEL_NAME)
        cfg.trainer.strategy = "megatron"
        cfg.generator.inference_engine.enable_return_routed_experts = True
        cfg.generator.inference_engine.tensor_parallel_size = 2
        cfg.generator.sampling_params = SamplingParams(
            max_generate_length=128,
            logprobs=1,
            temperature=1.0,
        )
        cfg.generator.batched = False
        cfg.generator.max_turns = 1

        num_prompts = 1

        tokenizer = AutoTokenizer.from_pretrained(MOE_MODEL_NAME, trust_remote_code=True)

        with InferenceEngineState.create(
            cfg=cfg,
            model=MOE_MODEL_NAME,
            use_local=True,
            backend="vllm",
            sleep_level=1,
            gpu_memory_utilization=0.9,
            max_num_seqs=1,
        ) as engines:
            client = engines.client

            asyncio.run(client.wake_up())

            generator = SkyRLGymGenerator(
                generator_cfg=cfg.generator,
                skyrl_gym_cfg=cfg.environment.skyrl_gym,
                inference_engine_client=client,
                tokenizer=tokenizer,
            )

            input_batch: GeneratorInput = get_test_generator_input(
                model=MOE_MODEL_NAME,
                num_prompts=num_prompts,
                n_samples_per_prompt=1,
                max_prompt_length=512,
                env_class="gsm8k",
            )
            input_batch["sampling_params"] = get_sampling_params_for_backend(
                "vllm",
                SamplingParams(
                    temperature=1.0,
                    top_p=1.0,
                    top_k=-1,
                    max_generate_length=128,
                    min_p=0.0,
                    logprobs=1,
                ),
            )

            with Timer("generate_with_router_replay"):
                generator_output = asyncio.run(generator.generate(input_batch))

            # --- Basic output checks ---
            assert (
                "rollout_inference_indices" in generator_output
            ), "rollout_inference_indices missing from GeneratorOutput"
            indices = generator_output["rollout_inference_indices"]
            assert (
                indices is not None
            ), "rollout_inference_indices should not be None when enable_return_routed_experts=True"

            responses = generator_output["response_ids"]
            assert len(indices) == len(
                responses
            ), f"Batch size mismatch: {len(indices)} indices vs {len(responses)} responses"

        rewards = generator_output["rewards"]
        if rewards and not isinstance(rewards[0], list):
            rewards = [[r] * len(resp) for r, resp in zip(rewards, responses)]
        (sequences, attention_mask, response_mask, rewards_t, loss_mask_t, logprobs_t, rii_tensor) = (
            convert_prompts_responses_to_batch_tensors(
                tokenizer=tokenizer,
                prompts=generator_output["prompt_token_ids"],
                responses=responses,
                rewards=rewards,
                loss_masks=generator_output["loss_masks"],
                logprobs=generator_output.get("rollout_logprobs"),
                rollout_inference_indices=indices,
            )
        )

        assert rii_tensor is not None
        rii_tensor = rii_tensor[:, :, :REPLAY_NUM_LAYERS, :]

        num_actions = response_mask.shape[1]
        batch_size = sequences.shape[0]
        training_input = TrainingInputBatch(
            {
                "sequences": sequences,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "rewards": rewards_t,
                "loss_mask": loss_mask_t,
                "rollout_logprobs": (
                    logprobs_t
                    if logprobs_t is not None
                    else torch.zeros((batch_size, num_actions), dtype=torch.float32)
                ),
                "rollout_inference_indices": rii_tensor,
                "action_log_probs": torch.zeros((batch_size, num_actions), dtype=torch.float32),
                "base_action_log_probs": torch.zeros((batch_size, num_actions), dtype=torch.float32),
                "advantages": torch.zeros((batch_size, num_actions), dtype=torch.float32),
                "action_mask": response_mask.to(dtype=torch.int64),
            }
        )
        training_input.metadata = {"response_length": num_actions}

        cfg.trainer.policy.megatron_config.transformer_config_kwargs = {
            "num_layers": REPLAY_NUM_LAYERS,
            "moe_enable_routing_replay": True,
        }
        cfg.trainer.placement.policy_num_gpus_per_node = 2
        cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 2
        cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
        cfg.trainer.policy.megatron_config.context_parallel_size = 1
        cfg.trainer.policy.megatron_config.expert_model_parallel_size = 1
        cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = 1
        cfg.trainer.micro_forward_batch_size_per_gpu = 1
        cfg.trainer.micro_train_batch_size_per_gpu = 1

        actor_group = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=2,
            cfg=cfg,
        )

        expected_per_layer = _split_replay_indices(rii_tensor.to(torch.long))

        state = ray.get(
            actor_group.async_run_ray_method(
                "pass_through",
                "debug_setup_router_replay_state",
                data=training_input,
            )
        )[0]

        assert state is not None, "Worker returned None state"
        assert (
            "REPLAY_FORWARD" in state["action"]
        ), f"RouterReplay action should be REPLAY_FORWARD, got: {state['action']}"
        assert state["num_instances"] == len(expected_per_layer), (
            f"Expected {len(expected_per_layer)} replay instances (one per layer), " f"got {state['num_instances']}"
        )
        for layer_idx, (got, expected) in enumerate(zip(state["target_indices"], expected_per_layer)):
            assert torch.equal(
                got.to(torch.long), expected.to(torch.long)
            ), f"Layer {layer_idx}: Megatron target indices differ from vLLM indices"
        print(
            f"PASSED: vLLM routing indices ({rii_tensor.shape}) correctly "
            f"loaded into {state['num_instances']} Megatron RouterReplay instances"
        )

    finally:
        ray.shutdown()


@pytest.mark.megatron
def test_logprobs(ray_init_fixture):
    """
    Check that logprob diff is lower when using router replay. Requires full 8xH100 setup to do full forward pass.
    """
    try:
        cfg = get_test_actor_config(model_name=MOE_MODEL_NAME)
        cfg.trainer.strategy = "megatron"
        cfg.generator.inference_engine.enable_return_routed_experts = True
        cfg.generator.inference_engine.tensor_parallel_size = 8
        cfg.generator.sampling_params = SamplingParams(
            max_generate_length=128,
            logprobs=1,
            temperature=1.0,
        )
        cfg.generator.batched = False
        cfg.generator.max_turns = 1

        tokenizer = AutoTokenizer.from_pretrained(MOE_MODEL_NAME, trust_remote_code=True)

        with InferenceEngineState.create(
            cfg=cfg,
            model=MOE_MODEL_NAME,
            use_local=True,
            colocate_all=True,
            backend="vllm",
            sleep_level=1,
            gpu_memory_utilization=0.9,
        ) as engines:
            client, pg = engines.client, engines.pg
            asyncio.run(client.wake_up())

            generator = SkyRLGymGenerator(
                generator_cfg=cfg.generator,
                skyrl_gym_cfg=cfg.environment.skyrl_gym,
                inference_engine_client=client,
                tokenizer=tokenizer,
            )

            input_batch: GeneratorInput = get_test_generator_input(
                model=MOE_MODEL_NAME,
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
                    max_generate_length=128,
                    min_p=0.0,
                    logprobs=1,
                ),
            )

            with Timer("generate_with_router_replay"):
                generator_output = asyncio.run(generator.generate(input_batch))

            indices = generator_output["rollout_inference_indices"]
            responses = generator_output["response_ids"]
            assert (
                indices is not None
            ), "rollout_inference_indices should not be None when enable_return_routed_experts=True"
            assert len(indices) == len(
                responses
            ), f"Batch size mismatch: {len(indices)} indices vs {len(responses)} responses"
            asyncio.run(client.sleep())

        rewards = generator_output["rewards"]
        if rewards and not isinstance(rewards[0], list):
            rewards = [[r] * len(resp) for r, resp in zip(rewards, responses)]
        (sequences, attention_mask, response_mask, rewards_t, loss_mask_t, logprobs_t, rii_tensor) = (
            convert_prompts_responses_to_batch_tensors(
                tokenizer=tokenizer,
                prompts=generator_output["prompt_token_ids"],
                responses=responses,
                rewards=rewards,
                loss_masks=generator_output["loss_masks"],
                logprobs=generator_output.get("rollout_logprobs"),
                rollout_inference_indices=indices,
            )
        )

        assert rii_tensor is not None
        num_actions = response_mask.shape[1]
        batch_size = sequences.shape[0]
        training_input = TrainingInputBatch(
            {
                "sequences": sequences,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "rewards": rewards_t,
                "loss_mask": loss_mask_t,
                "rollout_logprobs": (
                    logprobs_t
                    if logprobs_t is not None
                    else torch.zeros((batch_size, num_actions), dtype=torch.float32)
                ),
                "rollout_inference_indices": rii_tensor,
                "action_log_probs": torch.zeros((batch_size, num_actions), dtype=torch.float32),
                "base_action_log_probs": torch.zeros((batch_size, num_actions), dtype=torch.float32),
                "advantages": torch.zeros((batch_size, num_actions), dtype=torch.float32),
                "action_mask": response_mask.to(dtype=torch.int64),
            }
        )
        training_input.metadata = {"response_length": num_actions}

        cfg.trainer.placement.policy_num_gpus_per_node = 8
        cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 4
        cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
        cfg.trainer.policy.megatron_config.context_parallel_size = 1
        cfg.trainer.policy.megatron_config.expert_model_parallel_size = 8
        cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = 1
        cfg.trainer.micro_forward_batch_size_per_gpu = 1
        cfg.trainer.micro_train_batch_size_per_gpu = 1

        import os

        os.environ["SKYRL_DEBUG_LOGITS"] = "1"

        def run_megatron_forward(enable_replay: bool, debug: bool = False) -> torch.Tensor:
            cfg.trainer.policy.megatron_config.transformer_config_kwargs = {
                "moe_enable_routing_replay": enable_replay,
            }
            actor_group = init_worker_with_type(
                "policy",
                shared_pg=pg,
                colocate_all=True,
                num_gpus_per_node=8,
                cfg=cfg,
            )

            if debug:
                diag = ray.get(actor_group.async_run_ray_method("pass_through", "debug_model_config"))[0]
                print(f"\n=== Model Config (replay={enable_replay}) ===")
                for k, v in diag.items():
                    if k != "weight_stats":
                        print(f"  {k}: {v}")
                print(f"  weight_stats ({len(diag['weight_stats'])} params):")
                for name, stats in sorted(diag["weight_stats"].items()):
                    print(
                        f"    {name}: shape={stats['shape']}, mean={stats['mean']:.6f}, std={stats['std']:.6f}, norm={stats['norm']:.2f}"
                    )

            refs = actor_group.async_run_ray_method("mesh", "forward", data=training_input)
            results = ray.get(refs)
            outputs = concatenate_outputs_after_mesh_dispatch(actor_group.actor_infos, results)["output"]

            for actor in actor_group._actor_handlers:
                ray.kill(actor)
            return outputs

        r3_logprobs = run_megatron_forward(enable_replay=True, debug=True)
        no_r3_logprobs = run_megatron_forward(enable_replay=False)

        r3_diff = (logprobs_t - r3_logprobs).abs()
        no_r3_diff = (logprobs_t - no_r3_logprobs).abs()
        print(f"vLLM logprobs     - mean: {logprobs_t.mean().item():.6f}, std: {logprobs_t.std().item():.6f}")
        print(f"Megatron (replay) - mean: {r3_logprobs.mean().item():.6f}, std: {r3_logprobs.std().item():.6f}")
        print(f"Megatron (no rep) - mean: {no_r3_logprobs.mean().item():.6f}, std: {no_r3_logprobs.std().item():.6f}")
        print(f"With replay    - logprob diff mean: {r3_diff.mean().item():.6f}, std: {r3_diff.std().item():.6f}")
        print(f"Without replay - logprob diff mean: {no_r3_diff.mean().item():.6f}, std: {no_r3_diff.std().item():.6f}")

        assert r3_diff.mean().item() < no_r3_diff.mean().item(), (
            f"Router replay should reduce logprob diff vs rollout, "
            f"but with_replay={r3_diff.mean().item():.6f} >= without_replay={no_r3_diff.mean().item():.6f}"
        )
    finally:
        ray.shutdown()


@pytest.mark.megatron
def test_forward_backward(ray_init_fixture):
    """
    Check that forward_backward produces similar losses with and without
    router replay (same weights, so routing decisions should nearly match).
    Requires full 8xH100 setup.
    """
    try:
        cfg = get_test_actor_config(model_name=MOE_MODEL_NAME)
        cfg.trainer.strategy = "megatron"
        cfg.generator.inference_engine.enable_return_routed_experts = True
        cfg.generator.inference_engine.tensor_parallel_size = 8
        cfg.generator.sampling_params = SamplingParams(
            max_generate_length=128,
            logprobs=1,
            temperature=1.0,
        )
        cfg.generator.batched = False
        cfg.generator.max_turns = 1

        tokenizer = AutoTokenizer.from_pretrained(MOE_MODEL_NAME, trust_remote_code=True)

        with InferenceEngineState.create(
            cfg=cfg,
            model=MOE_MODEL_NAME,
            use_local=True,
            colocate_all=True,
            backend="vllm",
            sleep_level=1,
            gpu_memory_utilization=0.9,
        ) as engines:
            client, pg = engines.client, engines.pg
            asyncio.run(client.wake_up())

            generator = SkyRLGymGenerator(
                generator_cfg=cfg.generator,
                skyrl_gym_cfg=cfg.environment.skyrl_gym,
                inference_engine_client=client,
                tokenizer=tokenizer,
            )

            input_batch: GeneratorInput = get_test_generator_input(
                model=MOE_MODEL_NAME,
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
                    max_generate_length=128,
                    min_p=0.0,
                    logprobs=1,
                ),
            )

            with Timer("generate_with_router_replay"):
                generator_output = asyncio.run(generator.generate(input_batch))

            indices = generator_output["rollout_inference_indices"]
            responses = generator_output["response_ids"]
            assert (
                indices is not None
            ), "rollout_inference_indices should not be None when enable_return_routed_experts=True"
            assert len(indices) == len(
                responses
            ), f"Batch size mismatch: {len(indices)} indices vs {len(responses)} responses"
            asyncio.run(client.sleep())

        rewards = generator_output["rewards"]
        if rewards and not isinstance(rewards[0], list):
            rewards = [[r] * len(resp) for r, resp in zip(rewards, responses)]
        (sequences, attention_mask, response_mask, rewards_t, loss_mask_t, logprobs_t, rii_tensor) = (
            convert_prompts_responses_to_batch_tensors(
                tokenizer=tokenizer,
                prompts=generator_output["prompt_token_ids"],
                responses=responses,
                rewards=rewards,
                loss_masks=generator_output["loss_masks"],
                logprobs=generator_output.get("rollout_logprobs"),
                rollout_inference_indices=indices,
            )
        )

        assert rii_tensor is not None
        num_actions = response_mask.shape[1]
        batch_size = sequences.shape[0]
        training_input = TrainingInputBatch(
            {
                "sequences": sequences,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "rewards": rewards_t,
                "loss_mask": loss_mask_t,
                "rollout_logprobs": (
                    logprobs_t
                    if logprobs_t is not None
                    else torch.zeros((batch_size, num_actions), dtype=torch.float32)
                ),
                "rollout_inference_indices": rii_tensor,
                "action_log_probs": torch.zeros((batch_size, num_actions), dtype=torch.float32),
                "base_action_log_probs": torch.zeros((batch_size, num_actions), dtype=torch.float32),
                "advantages": torch.zeros((batch_size, num_actions), dtype=torch.float32),
                "action_mask": response_mask.to(dtype=torch.int64),
            }
        )
        training_input.metadata = {"response_length": num_actions}

        cfg.trainer.placement.policy_num_gpus_per_node = 8
        cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 4
        cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
        cfg.trainer.policy.megatron_config.context_parallel_size = 1
        cfg.trainer.policy.megatron_config.expert_model_parallel_size = 8
        cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = 1
        cfg.trainer.micro_forward_batch_size_per_gpu = 1
        cfg.trainer.micro_train_batch_size_per_gpu = 1

        def run_megatron_forward_backward(enable_replay: bool) -> dict:
            cfg.trainer.policy.megatron_config.transformer_config_kwargs = {
                "moe_enable_routing_replay": enable_replay,
            }
            actor_group = init_worker_with_type(
                "policy",
                shared_pg=pg,
                colocate_all=True,
                num_gpus_per_node=8,
                cfg=cfg,
            )
            results = ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", data=training_input))
            for actor in actor_group._actor_handlers:
                ray.kill(actor)
            return results[0]

        metrics_replay = run_megatron_forward_backward(enable_replay=True)
        metrics_no_replay = run_megatron_forward_backward(enable_replay=False)

        loss_replay = metrics_replay["policy_loss"]
        loss_no_replay = metrics_no_replay["policy_loss"]
        print(f"With replay    - loss: {loss_replay:.6f}")
        print(f"Without replay - loss: {loss_no_replay:.6f}")
        print(f"With replay metrics: {metrics_replay}")
        print(f"Without replay metrics: {metrics_no_replay}")

        diff = abs(loss_replay - loss_no_replay)
        threshold = 0.5
        print(f"Loss diff: {diff:.6f} (threshold: {threshold})")
        assert diff < threshold, (
            f"Losses with/without replay should be similar (same weights), "
            f"but diff={diff:.6f} >= threshold={threshold}"
        )
    finally:
        ray.shutdown()
