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
from skyrl.train.utils.utils import validate_cfg
from skyrl.train.config import (
    SkyRLTrainConfig,
    SamplingParams,
)
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator
from skyrl.train.generators.base import GeneratorInput
from skyrl.backends.skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl.train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.distributed.dispatch import concatenate_outputs_after_mesh_dispatch

MOE_MODEL_NAME = "Qwen/Qwen3-30B-A3B"


def get_test_actor_config(model_name=MOE_MODEL_NAME) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = model_name
    cfg.trainer.micro_forward_batch_size_per_gpu = 2
    cfg.trainer.micro_train_batch_size_per_gpu = 2
    cfg.trainer.use_sample_packing = False
    cfg.trainer.logger = "console"

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
            max_generate_length=16,
            logprobs=1,
            temperature=1.0,
        )
        cfg.generator.batched = False
        cfg.generator.max_turns = 1
        cfg.generator.use_conversation_multi_turn = True
        cfg.generator.apply_overlong_filtering = False
        cfg.generator.zero_reward_on_non_stop = False

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
                    max_generate_length=16,
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

            # --- Shape & value validation per sample ---
            for i, (sample_indices, sample_response) in enumerate(zip(indices, responses)):
                response_len = len(sample_response)
                assert (
                    len(sample_indices) == response_len
                ), f"Sample {i}: indices length {len(sample_indices)} != response length {response_len}"

                if response_len == 0:
                    continue

                # Each token position should have [layer_num, topk] structure
                layer_num = len(sample_indices[0])
                assert layer_num > 0, f"Sample {i}: expected > 0 MoE layers, got {layer_num}"

                topk = len(sample_indices[0][0])
                assert topk > 0, f"Sample {i}: expected topk > 0, got {topk}"

                for t, token_indices in enumerate(sample_indices):
                    assert (
                        len(token_indices) == layer_num
                    ), f"Sample {i}, token {t}: expected {layer_num} layers, got {len(token_indices)}"
                    for l_idx, layer_indices in enumerate(token_indices):
                        assert (
                            len(layer_indices) == topk
                        ), f"Sample {i}, token {t}, layer {l_idx}: expected topk={topk}, got {len(layer_indices)}"
                        for k, expert_id in enumerate(layer_indices):
                            assert isinstance(expert_id, int), (
                                f"Sample {i}, token {t}, layer {l_idx}, k {k}: "
                                f"expected int expert id, got {type(expert_id)}"
                            )
                            assert expert_id >= 0, (
                                f"Sample {i}, token {t}, layer {l_idx}, k {k}: "
                                f"expected non-negative expert id, got {expert_id}"
                            )
            from skyrl.backends.skyrl_train.utils.replay_utils import _split_replay_indices
            replay_tensor = torch.tensor(indices, dtype=torch.long)
            per_layer_replay = _split_replay_indices(replay_tensor)
            reconstructed = torch.stack(per_layer_replay, dim=2)
            assert torch.equal(
                replay_tensor, reconstructed
            ), "Replay index translation changed values between vLLM and Megatron layout"

            prompt_ids = generator_output["prompt_token_ids"]
            rollout_logprobs = generator_output.get("rollout_logprobs", None)
            loss_masks = generator_output["loss_masks"]
            rewards = generator_output["rewards"]
            if rewards and not isinstance(rewards[0], list):
                rewards = [[reward] * len(response) for reward, response in zip(rewards, responses)]

            (
                sequences_tensor,
                attention_masks_tensor,
                response_masks_tensor,
                rewards_tensor,
                loss_masks_tensor,
                rollout_logprobs_tensor,
                rollout_inference_indices_tensor,
            ) = convert_prompts_responses_to_batch_tensors(
                tokenizer=tokenizer,
                prompts=prompt_ids,
                responses=responses,
                rewards=rewards,
                loss_masks=loss_masks,
                logprobs=rollout_logprobs,
                rollout_inference_indices=indices,
            )

            assert rollout_inference_indices_tensor is not None
            assert rollout_inference_indices_tensor.shape[0] == len(responses)
            assert rollout_inference_indices_tensor.shape[1] == response_masks_tensor.shape[1]

            training_input = TrainingInputBatch(
                {
                    "sequences": sequences_tensor,
                    "attention_mask": attention_masks_tensor,
                    "response_mask": response_masks_tensor,
                    "rewards": rewards_tensor,
                    "loss_mask": loss_masks_tensor,
                    "rollout_logprobs": rollout_logprobs_tensor,
                    "rollout_inference_indices": rollout_inference_indices_tensor,
                }
            )
            training_input.metadata = {"response_length": response_masks_tensor.shape[1]}
            assert training_input["rollout_inference_indices"] is not None

            cfg.trainer.policy.megatron_config.transformer_config_kwargs = {}
            cfg.trainer.policy.megatron_config.transformer_config_kwargs["num_layers"] = 2
            cfg.trainer.policy.megatron_config.transformer_config_kwargs["moe_enable_routing_replay"] = True
            cfg.trainer.placement.policy_num_gpus_per_node = 2
            cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 2
            cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
            cfg.trainer.policy.megatron_config.context_parallel_size = 1
            cfg.trainer.policy.megatron_config.expert_model_parallel_size = 1
            cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = 1
            cfg.trainer.micro_forward_batch_size_per_gpu = 1
            cfg.trainer.micro_train_batch_size_per_gpu = 1

            num_actions = response_masks_tensor.shape[1]
            batch_size = sequences_tensor.shape[0]
            if training_input.get("rollout_logprobs") is None:
                training_input["rollout_logprobs"] = torch.zeros((batch_size, num_actions), dtype=torch.float32)
            training_input["action_log_probs"] = torch.zeros((batch_size, num_actions), dtype=torch.float32)
            training_input["base_action_log_probs"] = torch.zeros((batch_size, num_actions), dtype=torch.float32)
            training_input["advantages"] = torch.zeros((batch_size, num_actions), dtype=torch.float32)
            training_input["action_mask"] = response_masks_tensor.to(dtype=torch.int64)

            actor_group = init_worker_with_type(
                "policy",
                shared_pg=None,
                colocate_all=False,
                num_gpus_per_node=2,
                cfg=cfg,
            )

            forward_refs = actor_group.async_run_ray_method("mesh", "forward", data=training_input)
            all_rank_forward_outputs = ray.get(forward_refs)
            forward_output = concatenate_outputs_after_mesh_dispatch(actor_group.actor_infos, all_rank_forward_outputs)[
                "output"
            ]
            expected_per_layer = _split_replay_indices(training_input["rollout_inference_indices"].to(torch.long))
            forward_state = ray.get(actor_group.async_run_ray_method("pass_through", "get_last_router_replay_state"))[0]
            
            assert forward_state is not None
            assert len(forward_state["global_indices"]) == len(expected_per_layer)
            for got, expected in zip(forward_state["global_indices"], expected_per_layer):
                assert torch.equal(got.to(torch.long), expected.to(torch.long))
            assert forward_state["replay_backward_total_entries"] > 0

            training_input.metadata["global_step"] = 0
            fb_results = ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", training_input))
            assert isinstance(fb_results[0], dict)
            assert "policy_loss" in fb_results[0]
            fb_state = ray.get(actor_group.async_run_ray_method("pass_through", "get_last_router_replay_state"))[0]
            assert fb_state is not None
            assert len(fb_state["global_indices"]) == len(expected_per_layer)
            ray.get(actor_group.async_run_ray_method("pass_through", "optim_step"))

            print("Router replay test passed:")
            print(f"  Batch size: {len(indices)}")
            print(f"  Response lengths: {[len(r) for r in responses]}")
            if indices and indices[0]:
                print(f"  Layers: {len(indices[0][0])}, TopK: {len(indices[0][0][0])}")
            print(f"  Handoff replay tensor shape: {tuple(rollout_inference_indices_tensor.shape)}")
            print(f"  Megatron forward shape: {tuple(forward_output.shape)}")

    finally:
        ray.shutdown()
