"""
Run with:
uv run --isolated --extra dev --extra megatron pytest -s tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_router_replay.py
"""

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
from skyrl.backends.skyrl_train.inference_servers.utils import (
    _uses_lora_weight_sync,
    resolve_policy_model_name,
)
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.utils.packed_tensor import PackedTensor
from skyrl.train.config import SamplingParams, SkyRLTrainConfig
from skyrl.train.dataset.preprocess import (
    convert_prompts_responses_to_batch_tensors,
    make_router_padding_mask,
)
from skyrl.train.generators.base import GeneratorInput
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator
from skyrl.train.utils.utils import validate_cfg
from tests.backends.skyrl_train.gpu.gpu_ci.conftest import ray_init
from tests.backends.skyrl_train.gpu.utils import (
    InferenceEngineState,
    Timer,
    get_test_generator_input,
    init_worker_with_type,
)

MOE_MODEL_NAME = "moonshotai/Moonlight-16B-A3B-Instruct"
NUM_PROMPTS = 10
N_SAMPLES_PER_PROMPT = 4
MAX_GENERATE_LENGTH = 128
# Moonlight 16B: 27 MoE layers, top_k=6, 64 routed experts.
MOONLIGHT_NUM_LAYERS = 27
MOONLIGHT_TOPK = 6
MOONLIGHT_NUM_EXPERTS = 64


def _packed_moonlight_routes(attention_mask: torch.Tensor) -> PackedTensor:
    """Moonlight-shaped routes packed to each trajectory's real tokens."""
    route_offsets = torch.arange(MOONLIGHT_TOPK, dtype=torch.int32)
    segments = []
    for real_tokens in attention_mask.sum(dim=1).tolist():
        route_start = torch.randint(
            0,
            MOONLIGHT_NUM_EXPERTS,
            (real_tokens, MOONLIGHT_NUM_LAYERS, 1),
            dtype=torch.int32,
        )
        segments.append((route_start + route_offsets) % MOONLIGHT_NUM_EXPERTS)
    return PackedTensor.from_segments(segments)


def _extra_env_vars_for_model(model_name: str) -> dict[str, str] | None:
    """Per-model Ray env-var overrides for the router-replay tests.

    The gpu_ci conftest globally sets ``NVTE_FUSED_ATTN=0``; MLA models such as
    Moonlight need cuDNN fused attention for the Megatron forward with sample
    packing, so re-enable it here.

    ``VLLM_USE_FLASHINFER_MOE_FP16=0`` forces vLLM's unquantized MoE oracle off
    the FlashInfer TRTLLM block-layout kernel. That kernel is auto-selected on
    Blackwell (sm100) and asserts ``K % blockK == 0`` (which Moonlight's sharded
    intermediate size violates), crashing engine-core init. Disabling it falls
    back to the Triton MoE kernel -- the same path H100 (sm90) already uses, so
    this is a no-op there.
    """
    if "moonlight" in model_name.lower() or "glm-4" in model_name.lower():
        return {"NVTE_FUSED_ATTN": "1", "VLLM_USE_FLASHINFER_MOE_FP16": "0"}
    return None


def get_test_actor_config(model_name=MOE_MODEL_NAME) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = model_name
    cfg.trainer.micro_forward_batch_size_per_gpu = 2
    cfg.trainer.micro_train_batch_size_per_gpu = 2
    cfg.trainer.remove_microbatch_padding = True
    cfg.generator.inference_engine.distributed_executor_backend = "mp"
    # turn on optimizer offload for Moonlight 16B on 4xH100
    cfg.trainer.policy.megatron_config.optimizer_config_kwargs = {
        "overlap_cpu_optimizer_d2h_h2d": True,
        "use_precision_aware_optimizer": True,
        "optimizer_cpu_offload": True,
        "optimizer_offload_fraction": 1.0,
    }
    # flash attn + mla works without sample packing, logprobs are crazy/wrong
    # but flash-attn correctly throws error with sample packing
    # we should add an assert that if you set remove_microbatch_padding=False flash attn can accidentally be used
    # and that we enable nvte fused attn for moonlight models with remove_microbatch_padding=True
    # need to enable nvte fused attn for router replay tests when using moonlight models with remove_microbatch_padding=True
    cfg.trainer.logger = "console"
    if "Moonlight" in model_name:
        if cfg.trainer.policy.megatron_config.transformer_config_kwargs is None:
            cfg.trainer.policy.megatron_config.transformer_config_kwargs = {}

        cfg.trainer.flash_attn = False
    validate_cfg(cfg)
    return cfg


def build_training_input_from_text_samples(
    tokenizer: AutoTokenizer, prompt_response_pairs: list[tuple[str, str]]
) -> TrainingInputBatch:
    prompts = []
    responses = []
    rewards = []
    loss_masks = []

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    for prompt_text, response_text in prompt_response_pairs:
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        response_ids = tokenizer.encode(response_text, add_special_tokens=False)
        if tokenizer.eos_token_id is not None and (not response_ids or response_ids[-1] != tokenizer.eos_token_id):
            response_ids.append(tokenizer.eos_token_id)

        prompts.append(prompt_ids)
        responses.append(response_ids)
        rewards.append([0.0] * len(response_ids))
        loss_masks.append([1] * len(response_ids))

    sequences, attention_mask, response_mask, rewards_t, loss_mask_t, _, _, _, _ = (
        convert_prompts_responses_to_batch_tensors(
            pad_token_id=tokenizer.pad_token_id,
            prompts=prompts,
            responses=responses,
            rewards=rewards,
            loss_masks=loss_masks,
        )
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
            "rollout_logprobs": torch.zeros((batch_size, num_actions), dtype=torch.float32),
            "action_log_probs": torch.zeros((batch_size, num_actions), dtype=torch.float32),
            "base_action_log_probs": torch.zeros((batch_size, num_actions), dtype=torch.float32),
            "advantages": torch.zeros((batch_size, num_actions), dtype=torch.float32),
        }
    )
    training_input.metadata = {"response_length": num_actions}
    return training_input


def _configure_lora_adapter(cfg: SkyRLTrainConfig) -> None:
    """Serve the policy as a LoRA adapter (Megatron ``merge_lora=False``) so vLLM runs with ``enable_lora``.

    Moonlight is MLA, so the attention LoRA target is linear_proj only; linear_fc1/fc2 put
    LoRA on the routed experts, exercising vLLM's fused-MoE LoRA path under route capture.
    """
    lora = cfg.trainer.policy.model.lora
    lora.rank = 32
    lora.alpha = 32
    lora.target_modules = ["linear_proj", "linear_fc1", "linear_fc2"]
    cfg.trainer.policy.megatron_config.lora_config.merge_lora = False
    # lora_B is zero-initialized; a LoRA-scale LR makes the few steps below move it.
    cfg.trainer.policy.optimizer_config.lr = 1e-4


async def _sync_policy_to_engines(policy, client, cfg: SkyRLTrainConfig) -> None:
    """Colocated weight sync: engines asleep, policy on GPU -> engines awake, policy offloaded."""
    policy.offload_to_cpu(offload_optimizer=True, offload_model=False)
    await client.wake_up(tags=["weights"])
    ray.get(
        policy.async_run_ray_method(
            "pass_through", "broadcast_to_inference_engines", client, cfg.generator.inference_engine
        )
    )
    policy.offload_to_cpu(offload_optimizer=False, offload_model=True)
    await client.wake_up(tags=["kv_cache"])
    await client.reset_prefix_cache()


async def _train_sync_and_check_lora_adapter(policy, client, cfg: SkyRLTrainConfig, tokenizer) -> None:
    """Move the adapter off its zero init, sync it, and check vLLM applies it on /skyrl/v1/generate."""
    # No routes in the batch, so this trains on native routing even with replay enabled on the worker.
    train_input = build_training_input_from_text_samples(
        tokenizer,
        [(f"What is {i} + {i}?", f"The answer is {i + i}.") for i in range(8)],
    )
    gen = torch.Generator().manual_seed(0)
    train_input["advantages"] = torch.randn(train_input["advantages"].shape, generator=gen)
    for _ in range(3):
        ray.get(policy.async_run_ray_method("mesh", "forward_backward", data=train_input))
        ray.get(policy.async_run_ray_method("pass_through", "optim_step"))

    await _sync_policy_to_engines(policy, client, cfg)

    # Greedy rollouts under the adapter name must diverge from the base model on the same prompts.
    probe_prompts = [
        tokenizer.encode(f"Question: what is {i} times {i + 3}? Answer:", add_special_tokens=False) for i in range(8)
    ]
    greedy = get_sampling_params_for_backend(
        "vllm", SamplingParams(temperature=0.0, max_generate_length=32, logprobs=1)
    )
    base_out = await client.generate(
        {"prompt_token_ids": probe_prompts, "sampling_params": greedy}, model=client.model_name
    )
    adapter_out = await client.generate(
        {"prompt_token_ids": probe_prompts, "sampling_params": greedy}, model=resolve_policy_model_name(cfg)
    )
    assert base_out["rollout_expert_indices"] is not None
    assert adapter_out["rollout_expert_indices"] is not None
    base_lp = torch.tensor([lp[0] for lp in base_out["response_logprobs"]])
    adapter_lp = torch.tensor([lp[0] for lp in adapter_out["response_logprobs"]])
    outputs_differ = base_out["response_ids"] != adapter_out["response_ids"]
    first_lp_diff = (base_lp - adapter_lp).abs().max().item()
    print(f"adapter vs base: outputs_differ={outputs_differ}, max first-token logprob diff={first_lp_diff:.6f}")
    assert (
        outputs_differ or first_lp_diff > 1e-3
    ), "adapter rollouts match the base model -- /skyrl/v1/generate is not applying the LoRA adapter"


@pytest.mark.asyncio
@pytest.mark.h100
@pytest.mark.parametrize(
    "tp,pp,cp,ep,etp,extra_tf_kwargs,use_lora",
    [
        pytest.param(2, 2, 1, 2, 1, {"num_layers_in_first_pipeline_stage": 13}, False, id="tp2_pp2_ep2"),
        pytest.param(2, 2, 1, 2, 1, {"num_layers_in_first_pipeline_stage": 13}, True, id="tp2_pp2_ep2_lora"),
    ],
)
async def test_logprobs(tp, pp, cp, ep, etp, extra_tf_kwargs, use_lora):
    """
    Check that logprob diff is lower when using router replay. Runs on 4xH100.

    With ``use_lora`` the policy is served as a LoRA adapter (Megatron ``merge_lora=False``),
    so rollouts go through ``/skyrl/v1/generate`` with ``model=<adapter>``. The adapter is
    trained a few steps and synced first, and must visibly change rollouts vs the base model.

    Both scoring passes run on one worker: it skips replay for a batch that carries no routes,
    so dropping them gives native routing with identical weights (including the adapter).
    """
    with ray_init(extra_env_vars=_extra_env_vars_for_model(MOE_MODEL_NAME)):
        cfg = get_test_actor_config(model_name=MOE_MODEL_NAME)
        cfg.trainer.strategy = "megatron"
        cfg.generator.inference_engine.enable_return_routed_experts = True
        cfg.generator.inference_engine.tensor_parallel_size = 4
        cfg.generator.inference_engine.num_engines = 1
        cfg.generator.sampling_params = SamplingParams(
            max_generate_length=MAX_GENERATE_LENGTH,
            logprobs=1,
            temperature=1.0,
        )
        cfg.generator.batched = False
        cfg.generator.max_turns = 1

        cfg.trainer.placement.policy_num_gpus_per_node = 4
        if extra_tf_kwargs is not None:
            cfg.trainer.policy.megatron_config.transformer_config_kwargs.update(extra_tf_kwargs)
        cfg.trainer.policy.megatron_config.tensor_model_parallel_size = tp
        cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = pp
        cfg.trainer.policy.megatron_config.context_parallel_size = cp
        cfg.trainer.policy.megatron_config.expert_model_parallel_size = ep
        cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = etp
        cfg.trainer.micro_forward_batch_size_per_gpu = 2
        cfg.trainer.micro_train_batch_size_per_gpu = 2
        cfg.trainer.policy.megatron_config.moe_enable_routing_replay = True
        if use_lora:
            _configure_lora_adapter(cfg)
        validate_cfg(cfg)
        assert _uses_lora_weight_sync(cfg) == use_lora

        tokenizer = AutoTokenizer.from_pretrained(MOE_MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        async with InferenceEngineState.create(
            cfg=cfg,
            model=MOE_MODEL_NAME,
            use_local=True,
            colocate_all=True,
            backend="vllm",
            sleep_level=1,
            gpu_memory_utilization=0.8 if use_lora else 0.9,
        ) as engines:
            client, pg = engines.client, engines.pg

            policy = None
            if use_lora:
                # The adapter only exists on vLLM after a sync, so build the policy first.
                await client.sleep()
                policy = init_worker_with_type(
                    "policy",
                    shared_pg=pg,
                    colocate_all=True,
                    num_gpus_per_node=4,
                    cfg=cfg,
                )
                ray.get(
                    policy.async_run_ray_method(
                        "pass_through", "init_weight_sync_state", client, cfg.generator.inference_engine
                    )
                )
                await _train_sync_and_check_lora_adapter(policy, client, cfg, tokenizer)
            else:
                await client.wake_up()

            generator = SkyRLGymGenerator(
                generator_cfg=cfg.generator,
                skyrl_gym_cfg=cfg.environment.skyrl_gym,
                inference_engine_client=client,
                tokenizer=tokenizer,
                policy_model_name=resolve_policy_model_name(cfg) if use_lora else None,
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
                    max_generate_length=MAX_GENERATE_LENGTH,
                    min_p=0.0,
                    logprobs=1,
                ),
            )

            with Timer("generate_with_router_replay"):
                generator_output = await generator.generate(input_batch)

            indices = generator_output["rollout_expert_indices"]
            responses = generator_output["response_ids"]
            assert (
                indices is not None
            ), "rollout_expert_indices should not be None when enable_return_routed_experts=True"
            assert len(indices) == len(
                responses
            ), f"Batch size mismatch: {len(indices)} indices vs {len(responses)} responses"
            await client.sleep()

            rewards = generator_output["rewards"]
            if rewards and not isinstance(rewards[0], list):
                rewards = [[r] * len(resp) for r, resp in zip(rewards, responses)]
            sequences, attention_mask, response_mask, rewards_t, loss_mask_t, logprobs_t, rii_tensor, _, _ = (
                convert_prompts_responses_to_batch_tensors(
                    pad_token_id=tokenizer.pad_token_id,
                    prompts=generator_output["prompt_token_ids"],
                    responses=responses,
                    rewards=rewards,
                    loss_masks=generator_output["loss_masks"],
                    logprobs=generator_output.get("rollout_logprobs"),
                    rollout_expert_indices=indices,
                )
            )

            assert rii_tensor is not None and logprobs_t is not None
            router_padding_mask = make_router_padding_mask(attention_mask, [len(sample) for sample in indices])
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
                    "rollout_expert_indices": rii_tensor,
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

            if policy is None:
                policy = init_worker_with_type(
                    "policy",
                    shared_pg=pg,
                    colocate_all=True,
                    num_gpus_per_node=4,
                    cfg=cfg,
                )
            else:
                policy.backload_to_gpu(backload_optimizer=False, backload_model=True)

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
        r3_valid = r3_logprobs[mask]
        no_r3_valid = no_r3_logprobs[mask]

        r3_diff = (vllm_valid - r3_valid).abs()
        no_r3_diff = (vllm_valid - no_r3_valid).abs()
        print(f"vLLM logprobs     - mean: {vllm_valid.mean().item():.6f}, std: {vllm_valid.std().item():.6f}")
        print(f"Megatron (replay) - mean: {r3_valid.mean().item():.6f}, std: {r3_valid.std().item():.6f}")
        print(f"Megatron (no rep) - mean: {no_r3_valid.mean().item():.6f}, std: {no_r3_valid.std().item():.6f}")
        print(f"With replay    - logprob diff mean: {r3_diff.mean().item():.6f}, std: {r3_diff.std().item():.6f}")
        print(f"Without replay - logprob diff mean: {no_r3_diff.mean().item():.6f}, std: {no_r3_diff.std().item():.6f}")

        assert r3_diff.mean().item() < no_r3_diff.mean().item(), (
            f"Router replay should reduce logprob diff vs rollout, "
            f"but with_replay={r3_diff.mean().item():.6f} >= without_replay={no_r3_diff.mean().item():.6f}"
        )


@pytest.mark.h100
@pytest.mark.parametrize(
    "tp,pp,cp,ep,etp,extra_tf_kwargs",
    [
        pytest.param(2, 2, 1, 2, 1, {"num_layers_in_first_pipeline_stage": 13}, id="tp2_pp2_ep2"),
    ],
)
def test_forward_backward(tp, pp, cp, ep, etp, extra_tf_kwargs):
    """
    Check that forward_backward with router replay completes without error.
    Uses dummy expert routing indices (no vLLM engine needed). Runs on 4xH100.
    Non-zero advantages / action_log_probs verify the loss is actually computed.
    """
    with ray_init(extra_env_vars=_extra_env_vars_for_model(MOE_MODEL_NAME)):
        cfg = get_test_actor_config(model_name=MOE_MODEL_NAME)
        cfg.trainer.strategy = "megatron"

        tokenizer = AutoTokenizer.from_pretrained(MOE_MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        num_samples = NUM_PROMPTS * N_SAMPLES_PER_PROMPT
        prompts = []
        responses = []
        rewards = []
        loss_masks = []
        for i in range(num_samples):
            prompt_ids = tokenizer.encode(f"What is {i} + {i}?", add_special_tokens=False)
            response_ids = tokenizer.encode(f"The answer is {i + i}.", add_special_tokens=False)
            if tokenizer.eos_token_id is not None and (not response_ids or response_ids[-1] != tokenizer.eos_token_id):
                response_ids.append(tokenizer.eos_token_id)
            prompts.append(prompt_ids)
            responses.append(response_ids)
            rewards.append([1.0] * len(response_ids))
            loss_masks.append([1] * len(response_ids))

        sequences, attention_mask, response_mask, rewards_t, loss_mask_t, _, _, _, _ = (
            convert_prompts_responses_to_batch_tensors(
                pad_token_id=tokenizer.pad_token_id,
                prompts=prompts,
                responses=responses,
                rewards=rewards,
                loss_masks=loss_masks,
            )
        )

        batch_size = sequences.shape[0]
        num_actions = response_mask.shape[1]

        rollout_expert_indices = _packed_moonlight_routes(attention_mask)

        gen = torch.Generator().manual_seed(42)
        training_input = TrainingInputBatch(
            {
                "sequences": sequences,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "rewards": rewards_t,
                "loss_mask": loss_mask_t,
                "rollout_logprobs": -torch.rand((batch_size, num_actions), generator=gen) * 2.0,
                "rollout_expert_indices": rollout_expert_indices,
                "router_padding_mask": ~attention_mask.bool(),
                "action_log_probs": -torch.rand((batch_size, num_actions), generator=gen) * 2.0,
                "base_action_log_probs": -torch.rand((batch_size, num_actions), generator=gen) * 2.0,
                "advantages": torch.randn((batch_size, num_actions), generator=gen),
            }
        )
        training_input.metadata = {"response_length": num_actions}

        cfg.trainer.placement.policy_num_gpus_per_node = 4
        if extra_tf_kwargs is not None:
            cfg.trainer.policy.megatron_config.transformer_config_kwargs.update(extra_tf_kwargs)
        cfg.trainer.policy.megatron_config.tensor_model_parallel_size = tp
        cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = pp
        cfg.trainer.policy.megatron_config.context_parallel_size = cp
        cfg.trainer.policy.megatron_config.expert_model_parallel_size = ep
        cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = etp
        cfg.trainer.micro_forward_batch_size_per_gpu = 2
        cfg.trainer.micro_train_batch_size_per_gpu = 2
        cfg.trainer.policy.megatron_config.moe_enable_routing_replay = True

        actor_group = init_worker_with_type(
            "policy",
            num_gpus_per_node=4,
            cfg=cfg,
        )

        ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", data=training_input))
        ray.get(actor_group.async_run_ray_method("pass_through", "optim_step"))
        results = ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", data=training_input))

        # `forward_backward` returns a WorkerOutput; metrics are already
        # all-reduced across DP ranks, so any rank's dict is representative.
        loss = results[0].metrics["policy_loss"]
        print(f"Router replay forward_backward - loss: {loss:.6f}")
        assert loss is not None and not torch.isnan(torch.tensor(loss)), "Loss should be valid (not NaN)"
        assert loss != 0.0, "Loss should be non-zero with non-zero advantages"

        for actor in actor_group._actor_handlers:
            ray.kill(actor)


@pytest.mark.h100
@pytest.mark.parametrize(
    "tp,pp,cp,ep,etp,extra_tf_kwargs",
    [
        pytest.param(2, 2, 1, 2, 1, {"num_layers_in_first_pipeline_stage": 13}, id="tp2_pp2_ep2"),
    ],
)
def test_forward_backward_variable_length_full_recompute(tp, pp, cp, ep, etp, extra_tf_kwargs):
    """Replayed routes must stay paired with their own microbatch.

    Each forward microbatch appends its expert routes to a FIFO that
    activation-checkpoint recomputation drains once during backward. If a
    microbatch queues its routes more than once (or not at all), backward
    replays a *different* microbatch's routes. Megatron's MoE all-to-all then
    computes split sizes from a token count that doesn't match the tensors in
    flight and the collective fails.

    Two conditions are needed to expose that, and both are set here:

    * ``recompute_granularity="full"`` so backward actually recomputes the
      MoE layers and consumes the queue (the default only recomputes
      ``core_attn``, which never replays routes).
    * Sequence lengths that differ *between* microbatches, so a mispaired
      replay changes the token count rather than silently reusing a
      same-shaped tensor. Uniform lengths can mask the bug entirely.

    Prompts are built with deliberately spread lengths and ``micro_*=2`` over
    8 samples, giving 4 microbatches whose padded widths differ.
    """
    with ray_init(extra_env_vars=_extra_env_vars_for_model(MOE_MODEL_NAME)):
        cfg = get_test_actor_config(model_name=MOE_MODEL_NAME)
        cfg.trainer.strategy = "megatron"

        tokenizer = AutoTokenizer.from_pretrained(MOE_MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Lengths chosen so consecutive microbatches (pairs, micro_bs=2) have
        # different maxima: a stale replay is then shape-visible, not benign.
        filler_words = [1, 6, 2, 14, 3, 25, 4, 40]
        prompts, responses, rewards, loss_masks = [], [], [], []
        for i, filler in enumerate(filler_words):
            prompt_ids = tokenizer.encode(
                "Question: " + ("token " * filler) + f"what is {i} plus {i}?",
                add_special_tokens=False,
            )
            response_ids = tokenizer.encode(
                ("because " * filler) + f"the answer is {i + i}.",
                add_special_tokens=False,
            )
            if tokenizer.eos_token_id is not None and (not response_ids or response_ids[-1] != tokenizer.eos_token_id):
                response_ids.append(tokenizer.eos_token_id)
            prompts.append(prompt_ids)
            responses.append(response_ids)
            rewards.append([1.0] * len(response_ids))
            loss_masks.append([1] * len(response_ids))

        sequences, attention_mask, response_mask, rewards_t, loss_mask_t, _, _, _, _ = (
            convert_prompts_responses_to_batch_tensors(
                pad_token_id=tokenizer.pad_token_id,
                prompts=prompts,
                responses=responses,
                rewards=rewards,
                loss_masks=loss_masks,
            )
        )

        # Guard the premise: if padding collapsed the spread, the test would
        # pass for the wrong reason.
        real_token_counts = attention_mask.sum(dim=-1)
        assert (
            real_token_counts.unique().numel() > 1
        ), f"variable-length premise broken: every sample has {real_token_counts[0].item()} real tokens"

        batch_size = sequences.shape[0]
        num_actions = response_mask.shape[1]

        rollout_expert_indices = _packed_moonlight_routes(attention_mask)

        gen = torch.Generator().manual_seed(42)
        training_input = TrainingInputBatch(
            {
                "sequences": sequences,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "rewards": rewards_t,
                "loss_mask": loss_mask_t,
                "rollout_logprobs": -torch.rand((batch_size, num_actions), generator=gen) * 2.0,
                "rollout_expert_indices": rollout_expert_indices,
                "router_padding_mask": ~attention_mask.bool(),
                "action_log_probs": -torch.rand((batch_size, num_actions), generator=gen) * 2.0,
                "base_action_log_probs": -torch.rand((batch_size, num_actions), generator=gen) * 2.0,
                "advantages": torch.randn((batch_size, num_actions), generator=gen),
                "action_mask": response_mask.to(dtype=torch.int64),
            }
        )
        training_input.metadata = {"response_length": num_actions}

        cfg.trainer.placement.policy_num_gpus_per_node = 4
        if extra_tf_kwargs is not None:
            cfg.trainer.policy.megatron_config.transformer_config_kwargs.update(extra_tf_kwargs)
        # Recompute whole layers so backward re-runs the MoE routers and drains
        # the replay queue; ``core_attn`` alone never replays routes.
        cfg.trainer.policy.megatron_config.transformer_config_kwargs.update(
            {
                "recompute_granularity": "full",
                "recompute_method": "uniform",
                "recompute_num_layers": 1,
            }
        )
        cfg.trainer.policy.megatron_config.transformer_config_kwargs.pop("recompute_modules", None)
        cfg.trainer.policy.megatron_config.tensor_model_parallel_size = tp
        cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = pp
        cfg.trainer.policy.megatron_config.context_parallel_size = cp
        cfg.trainer.policy.megatron_config.expert_model_parallel_size = ep
        cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = etp
        # 8 samples / 2 per microbatch = 4 microbatches of differing widths.
        cfg.trainer.micro_forward_batch_size_per_gpu = 2
        cfg.trainer.micro_train_batch_size_per_gpu = 2
        cfg.trainer.policy.megatron_config.moe_enable_routing_replay = True

        actor_group = init_worker_with_type(
            "policy",
            num_gpus_per_node=4,
            cfg=cfg,
        )

        # Two steps: the first leaves any surplus queue entry behind, so a
        # mispairing shows up on the second even if the first survives.
        ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", data=training_input))
        ray.get(actor_group.async_run_ray_method("pass_through", "optim_step"))
        results = ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", data=training_input))

        loss = results[0].metrics["policy_loss"]
        print(f"Variable-length replay forward_backward - loss: {loss:.6f}")
        assert loss is not None and not torch.isnan(torch.tensor(loss)), "Loss should be valid (not NaN)"
        assert loss != 0.0, "Loss should be non-zero with non-zero advantages"

        for actor in actor_group._actor_handlers:
            ray.kill(actor)
