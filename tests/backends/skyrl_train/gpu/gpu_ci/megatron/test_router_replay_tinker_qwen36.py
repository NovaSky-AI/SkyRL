"""GPU test: Rollout Routing Replay (R3) through the Tinker sample path on Qwen3.6-35B-A3B.

R3 records the MoE expert selections vLLM makes during rollout and replays them in
the Megatron training forward, so training activates the same experts inference did.
This removes the routing-driven component of the train-vs-rollout logprob mismatch
that destabilizes RL on MoE models.

R3 is enabled purely by a launch flag (``moe_enable_routing_replay``); routing is
stashed on the vLLM servers keyed by the full sampled sequence, and forward_backward
pulls it back by digest, with no client-facing Tinker fields. This test drives that
data path against real vLLM servers: it samples through ``RemoteInferenceClient.sample``
with ``stash_routed_experts`` (``/skyrl/v1/completions`` stashes each choice's routing
and strips it from the response), fetches it back with the ``/skyrl/v1/routed_experts/fetch``
fan-out keyed by each training sequence's digest, packs it with the backend's
``_build_rollout_expert_indices``, and asserts replay lowers the mean
|vLLM logprob - Megatron logprob| versus the same worker without replay.

Runs on 1 node of 8 GPUs (two vLLM engines, so the fetch fans out across servers). Run with:
  uv run --isolated --extra dev --extra megatron --extra tinker -- \
    pytest -s tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_router_replay_tinker_qwen36.py
"""

import asyncio

import pytest
import ray
import torch
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.distributed.dispatch import (
    WorkerOutput,
    loss_fn_outputs_to_tensor,
)
from skyrl.backends.skyrl_train.inference_servers.routed_experts_stash import (
    sequence_digest,
)
from skyrl.backends.skyrl_train.inference_servers.utils import (
    resolve_policy_model_name,
)
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.utils.routed_experts import (
    compact_routed_expert_indices,
)
from skyrl.backends.skyrl_train_backend import _build_rollout_expert_indices
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl.train.utils.utils import validate_cfg
from tests.backends.skyrl_train.gpu.gpu_ci.conftest import ray_init
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
NUM_ENGINES = 2


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
    cfg.generator.inference_engine.num_engines = NUM_ENGINES
    cfg.generator.inference_engine.tensor_parallel_size = 8 // NUM_ENGINES
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


@pytest.mark.asyncio
@pytest.mark.megatron
async def test_r3_reduces_train_rollout_mismatch_via_tinker_path():
    """R3 (server-side routing stash + Megatron replay) lowers train-vs-rollout logprob mismatch."""
    with ray_init():
        cfg = get_test_actor_config()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model_name = resolve_policy_model_name(cfg)

        generator_input = get_test_generator_input(
            model=MODEL_NAME,
            num_prompts=NUM_PROMPTS,
            n_samples_per_prompt=1,
            max_prompt_length=512,
            env_class="gsm8k",
        )
        prompt_ids = [
            tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=False)
            for messages in generator_input["prompts"]
        ]

        async with InferenceEngineState.create(
            cfg=cfg,
            model=MODEL_NAME,
            use_local=True,
            colocate_all=True,
            backend="vllm",
            sleep_level=1,
            gpu_memory_utilization=0.6,
        ) as engines:
            client, pg = engines.client, engines.pg
            await client.wake_up()

            async def sample(seed: int, ids: list[int]) -> dict:
                return await client.sample(
                    {
                        "json": {
                            "model": model_name,
                            "prompt": {"chunks": [{"type": "encoded_text", "tokens": ids}]},
                            "num_samples": N_SAMPLES_PER_PROMPT,
                            "sampling_params": {
                                "temperature": 1.0,
                                "max_tokens": MAX_GENERATE_LENGTH,
                                "seed": seed,
                                "top_k": -1,
                                "top_p": 1.0,
                            },
                            "stash_routed_experts": True,
                        }
                    }
                )

            with Timer("sample_with_routing_stash"):
                outputs = await asyncio.gather(*[sample(seed, ids) for seed, ids in enumerate(prompt_ids)])

            prompts, responses, rollout_logprobs = [], [], []
            for ids, output in zip(prompt_ids, outputs):
                for sequence in output["sequences"]:
                    assert "routed_experts" not in sequence, "sample responses must not carry routing"
                    prompts.append(ids)
                    responses.append(list(sequence["tokens"]))
                    rollout_logprobs.append(list(sequence["logprobs"]))

            # forward_backward reconstructs each sample as prompt + response and
            # addresses the stash by its digest; the fetch fans out to every server.
            full_sequences = [p + r for p, r in zip(prompts, responses)]
            digest_hexes = [sequence_digest(seq).hex() for seq in full_sequences]
            with Timer("fetch_stashed_routing"):
                fetched = await client.fetch_routed_experts(model_name, digest_hexes)
            assert set(fetched) == set(digest_hexes), "every sampled sequence must be found in the servers' stash"
            per_sample_routing = [compact_routed_expert_indices(fetched[h]) for h in digest_hexes]
            for seq, routing in zip(full_sequences, per_sample_routing):
                # vLLM routes every forwarded token: the prompt plus all but the last sampled token.
                assert routing.shape[0] == len(seq) - 1, (routing.shape, len(seq))

            await client.sleep()

            sequences, attention_mask, response_mask, rewards_t, loss_mask_t, logprobs_t, _, _ = (
                convert_prompts_responses_to_batch_tensors(
                    pad_token_id=tokenizer.pad_token_id,
                    prompts=prompts,
                    responses=responses,
                    rewards=[[0.0] * len(r) for r in responses],
                    loss_masks=[[1] * len(r) for r in responses],
                    logprobs=rollout_logprobs,
                )
            )
            rollout_expert_indices, router_padding_mask = _build_rollout_expert_indices(
                full_sequences, per_sample_routing, attention_mask
            )

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
                    "rollout_expert_indices": rollout_expert_indices,
                    "router_padding_mask": router_padding_mask,
                    "action_log_probs": torch.zeros((batch_size, num_actions), dtype=torch.float32),
                    "base_action_log_probs": torch.zeros((batch_size, num_actions), dtype=torch.float32),
                    "advantages": torch.zeros((batch_size, num_actions), dtype=torch.float32),
                }
            )
            training_input.metadata = {"response_length": num_actions}
            no_replay_input = training_input.select(
                [k for k in training_input if k not in ("rollout_expert_indices", "router_padding_mask")]
            )

            # One worker scores both batches: it skips replay for a batch without
            # routes, so the comparison runs on identical weights.
            policy = init_worker_with_type("policy", shared_pg=pg, colocate_all=True, num_gpus_per_node=8, cfg=cfg)

            def run_megatron_forward(data: TrainingInputBatch) -> torch.Tensor:
                results = ray.get(policy.async_run_ray_method("mesh", "forward", data=data))
                output = WorkerOutput.cat(policy.actor_infos, results)
                return loss_fn_outputs_to_tensor(output.loss_fn_outputs, key="logprobs")

            r3_logprobs = run_megatron_forward(training_input)
            no_r3_logprobs = run_megatron_forward(no_replay_input)

            for actor in policy._actor_handlers:
                ray.kill(actor)

        mask = response_mask.bool()
        vllm_valid = logprobs_t[mask]
        r3_diff = (vllm_valid - r3_logprobs[mask]).abs()
        no_r3_diff = (vllm_valid - no_r3_logprobs[mask]).abs()

        print(f"vLLM logprobs  - mean: {vllm_valid.mean().item():.6f} over {vllm_valid.numel()} tokens")
        print(f"With replay    - |logprob diff| mean: {r3_diff.mean().item():.6f}, max: {r3_diff.max().item():.6f}")
        print(
            f"Without replay - |logprob diff| mean: {no_r3_diff.mean().item():.6f}, max: {no_r3_diff.max().item():.6f}"
        )

        assert r3_diff.mean().item() < no_r3_diff.mean().item(), (
            "Router replay through the Tinker path should reduce train-vs-rollout logprob "
            f"mismatch, but with_replay={r3_diff.mean().item():.6f} >= "
            f"without_replay={no_r3_diff.mean().item():.6f}"
        )
