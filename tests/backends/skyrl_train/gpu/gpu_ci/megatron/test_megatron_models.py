"""
Run with:
uv run --isolated --extra dev --extra megatron -- pytest -s tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_megatron_models.py
"""

import pytest
import ray
import torch
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.distributed.dispatch import (
    WorkerOutput,
    loss_fn_outputs_to_tensor,
)
from skyrl.backends.skyrl_train.inference_engines.utils import (
    get_sampling_params_for_backend,
)
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.train.config import SamplingParams, SkyRLTrainConfig
from skyrl.train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl.train.generators.base import GeneratorInput
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator
from skyrl.train.utils.utils import validate_cfg
from tests.backends.skyrl_train.gpu.gpu_ci.conftest import ray_init
from tests.backends.skyrl_train.gpu.utils import (
    InferenceEngineState,
    Timer,
    _ensure_chat_template,
    get_test_generator_input,
    init_worker_with_type,
)

NUM_PROMPTS = 10
N_SAMPLES_PER_PROMPT = 8
MAX_GENERATE_LENGTH = 128


# vLLM's Triton MLA decode kernel (the only MLA backend on sm < 9.0) fails
# to compile for glm-4's MLA shape; FLASH_ATTN_MLA / FLASHMLA need Hopper.
_skip_mla_on_pre_hopper = pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 9,
    reason="no working MLA backend for glm-4 on pre-Hopper GPUs",
)


def get_test_actor_config(model_name) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = model_name
    cfg.trainer.micro_forward_batch_size_per_gpu = 2
    cfg.trainer.micro_train_batch_size_per_gpu = 2
    cfg.trainer.remove_microbatch_padding = True
    cfg.generator.inference_engine.distributed_executor_backend = "ray"
    # flash attn + mla works without sample packing, logprobs are crazy/wrong
    # but flash-attn correctly throws error with sample packing
    # we should add an assert that if you set remove_microbatch_padding=False flash attn can accidentally be used
    # and that we enable nvte fused attn for moonlight models with remove_microbatch_padding=True
    # need to enable nvte fused attn for router replay tests when using moonlight models with remove_microbatch_padding=True
    cfg.trainer.logger = "console"
    is_mla_model = "moonlight" in model_name.lower() or "glm-4" in model_name.lower()
    if is_mla_model:
        if cfg.trainer.policy.megatron_config.transformer_config_kwargs is None:
            cfg.trainer.policy.megatron_config.transformer_config_kwargs = {}

        cfg.trainer.flash_attn = False

        # cuDNN fused attention does not support THD (sample packing) layout on
        # pre-Hopper GPUs (sm < 90), FA2 doesn't support MLA, and FA3 is
        # Hopper-only, so there is no viable TE attention backend for
        # MLA + sample_packing on Ada/Ampere.  Fall back to BSHD.
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 9:
            cfg.trainer.remove_microbatch_padding = False
    if "qwen3.5" in model_name.lower():
        # Qwen3.5 hybrid GDN checkpoints report a ...ForConditionalGeneration arch
        # and auto-dispatch to the VL bridge -> Qwen3VLModel, which self-packs and
        # double-packs against SkyRL's sample packing (corrupting the GDN
        # cu_seqlens). language_model_only routes them to the native GPTModel + GDN
        # thd path instead, which supports packed sequences directly.
        cfg.trainer.remove_microbatch_padding = True
        cfg.trainer.policy.language_model_only = True
        cfg.trainer.ref.language_model_only = True
        # validate_cfg requires policy/ref/generator language_model_only to agree.
        cfg.generator.inference_engine.language_model_only = True
    # Large MoE models: Megatron's DistributedOptimizer eagerly materializes
    # the fp32 master + AdamW state on GPU at init (~6x model size), which
    # OOMs on 4xH100 before forward ever runs. These tests only forward +
    # weight-sync, so skip optimizer construction entirely.
    is_large_moe = (
        ("qwen3.5-35b" in model_name.lower() and "tiny" not in model_name.lower())
        or ("nemotron-3-nano" in model_name.lower())
        or ("nemotron-3-ultra" in model_name.lower())
    )
    if is_large_moe:
        cfg.trainer.policy.inference_only_init = True
    if "nemotron-3-ultra" in model_name.lower():
        # Nemotron-Ultra ships MTP layers (num_nextn_predict_layers=1). Megatron-Bridge's
        # Mamba provider builds an `mtp_hybrid_override_pattern` from that, and its
        # finalize() does `[pattern] * mtp_num_layers`. SkyRL disables MTP for
        # training by nulling mtp_num_layers, but for the Mamba provider that leaves
        # the pattern set -> `[pattern] * None` -> TypeError. Clear both up front
        # (transformer_config_kwargs are applied right before provider.finalize()).
        if cfg.trainer.policy.megatron_config.transformer_config_kwargs is None:
            cfg.trainer.policy.megatron_config.transformer_config_kwargs = {}
        cfg.trainer.policy.megatron_config.transformer_config_kwargs.update(
            {"mtp_num_layers": 0, "mtp_hybrid_override_pattern": None}
        )
    validate_cfg(cfg)
    return cfg


def _extra_env_vars_for_model(model_name: str) -> dict[str, str] | None:
    # MLA models need cuDNN fused attention (the conftest globally sets
    # NVTE_FUSED_ATTN=0; re-enable it here so the fused backend is available).
    if "moonlight" in model_name.lower() or "glm-4" in model_name.lower():
        return {"NVTE_FUSED_ATTN": "1"}
    return None


def _engine_overrides_for_model(model_name: str) -> dict:
    """Per-model overrides for vLLM engine init."""
    overrides = {"engine_init_kwargs": {}, "gpu_memory_utilization": 0.9}
    if "Nemotron-3-Nano" in model_name:
        overrides["engine_init_kwargs"]["max_model_len"] = 4096
        # Megatron policy init also needs room alongside vLLM on the same
        # GPU, so lower vLLM's pool footprint.
        overrides["gpu_memory_utilization"] = 0.5
    if "Nemotron-3-Ultra" in model_name:
        # 550B sharded 16-way (vLLM TP8 x PP2) is ~69 GB of weights per GPU, so
        # the KV pool needs gmu well above 0.5 just to leave cache room. vLLM
        # and the Megatron policy alternate on-GPU via sleep/wake (sleep_level=2),
        # so vLLM can claim most of the H200 while loading. Cap context to 4096.
        # Starting point -- tune alongside the parallelism.
        overrides["engine_init_kwargs"]["max_model_len"] = 4096
        overrides["gpu_memory_utilization"] = 0.85
    # Large MoE: Megatron policy init also needs room alongside vLLM on the
    # same GPU, so lower vLLM's pool footprint.
    if "qwen3.5-35b" in model_name.lower() and "tiny" not in model_name.lower():
        overrides["gpu_memory_utilization"] = 0.5
    return overrides


async def generate_with_vllm(generator, client, model_name, tokenizer, return_training_input=False):
    input_batch: GeneratorInput = get_test_generator_input(
        model=model_name,
        num_prompts=NUM_PROMPTS,
        n_samples_per_prompt=N_SAMPLES_PER_PROMPT,
        max_prompt_length=512,
        env_class="gsm8k",
    )
    input_batch["sampling_params"] = get_sampling_params_for_backend(
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
        generator_output = await generator.generate(input_batch)

    responses = generator_output["response_ids"]

    rewards = generator_output["rewards"]
    if rewards and not isinstance(rewards[0], list):
        rewards = [[r] * len(resp) for r, resp in zip(rewards, responses)]

    sequences, attention_mask, response_mask, rewards_t, loss_mask_t, logprobs_t, _ = (
        convert_prompts_responses_to_batch_tensors(
            tokenizer=tokenizer,
            prompts=generator_output["prompt_token_ids"],
            responses=responses,
            rewards=rewards,
            loss_masks=generator_output["loss_masks"],
            logprobs=generator_output.get("rollout_logprobs"),
        )
    )
    if return_training_input:
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
                "rollout_expert_indices": None,
                "action_log_probs": torch.zeros((batch_size, num_actions), dtype=torch.float32),
                "base_action_log_probs": torch.zeros((batch_size, num_actions), dtype=torch.float32),
                "advantages": torch.zeros((batch_size, num_actions), dtype=torch.float32),
                "action_mask": response_mask.to(dtype=torch.int64),
            }
        )
        training_input.metadata = {"response_length": num_actions}
        return (response_mask, logprobs_t), training_input
    else:
        return (response_mask, logprobs_t)


async def construct_training_input_from_generator_output(generator_output, tokenizer):
    return convert_prompts_responses_to_batch_tensors(
        tokenizer=tokenizer,
        prompts=generator_output["prompt_token_ids"],
        responses=generator_output["response_ids"],
        rewards=generator_output["rewards"],
        loss_masks=generator_output["loss_masks"],
    )


@pytest.mark.asyncio
@pytest.mark.megatron_models
@pytest.mark.parametrize(
    "tp,pp,cp,ep,etp,inference_tp,inference_pp,num_gpus,num_nodes,model_name,vllm_threshold,megatron_threshold",
    [
        pytest.param(2, 1, 1, 2, 1, 2, 1, 4, 1, "eatang/qwen3-moe-tiny-random", 1e-1, 2e-1, id="qwen3-moe_tp2_ep2"),
        pytest.param(1, 2, 2, 1, None, 2, 1, 4, 1, "eatang/qwen3-moe-tiny-random", 1e-1, 2e-1, id="qwen3-moe_pp2_cp2"),
        pytest.param(
            2,
            1,
            1,
            2,
            1,
            2,
            1,
            4,
            1,
            "eatang/glm-4.7-flash-tiny-random",
            1e-1,
            2e-2,
            id="glm-4.7-flash_tp2_ep2",
            marks=_skip_mla_on_pre_hopper,
        ),
        pytest.param(
            2,
            1,
            1,
            2,
            1,
            4,
            1,
            4,
            1,
            "eatang/qwen3.5-moe-tiny-random",
            1e-1,
            2e-1,
            id="qwen3.5-moe_tp2_ep2",
            marks=pytest.mark.skip(reason="running into correctness issues for tiny qwen3.5"),
        ),
        # Qwen3.5-0.8B (dense hybrid GDN, real weights) via language_model_only ->
        # native GPTModel + GDN thd packing path. TP=2 across 2 GPUs, sample
        # packing on. Real weights, so logprobs should match vLLM tightly.
        pytest.param(
            2,
            1,
            1,
            1,
            None,
            2,
            1,
            2,
            1,
            "Qwen/Qwen3.5-0.8B",
            1e-1,
            5e-2,
            id="qwen3.5-0.8b-dense_tp2",
        ),
        # Nemotron-3-Nano (30B MoE, bf16) on 4xH100-80G. Mesh: TP=4 EP=4
        # ETP=1 -> DP=1. vLLM TP=4 across the same 4 GPUs (colocated).
        # TP=1 OOMed in the EP alltoall because dense layers were replicated
        # on every GPU; TP=4 shards them 4-way and matches the qwen3.5-35b
        # layout below. AdamW optimizer is skipped entirely via is_large_moe
        # in get_test_actor_config (forward-only test), and vLLM gmu is
        # lowered to 0.5 so the policy shard + vLLM pool fit on each H100.
        pytest.param(
            4,
            1,
            1,
            4,
            1,
            4,
            1,
            4,
            1,
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            5e-1,
            5e-2,
            id="nemotron3-nano_tp4_ep4_h100",
            marks=pytest.mark.h100,
        ),
        # Qwen3.5-35B-A3B (~35B MoE, ~3B activated) on 4xH100-80G. Mesh:
        # TP=4 EP=4 ETP=1 -> DP=1. vLLM TP=4 across the same 4 GPUs
        # (colocated). Thresholds mirror the GLM-4.7-Flash entry; tune as
        # we find what the actual logprob diffs look like.
        pytest.param(
            4,
            1,
            1,
            4,
            1,
            4,
            1,
            4,
            1,
            "Qwen/Qwen3.5-35B-A3B",
            3e-1,
            5e-2,
            id="qwen3.5-35b-a3b_h100_tp4_ep4",
            marks=pytest.mark.h100,
        ),
        # Nemotron-3-Ultra (550B MoE, ~55B activated, bf16) on 8 nodes x 8xH200
        # = 64 GPUs. Megatron mesh: TP=8 PP=2 CP=1 EP=16 ETP=1 -> DP=4
        # (8*2 = 16 GPUs per model replica, 64/16 = 4 DP, ~69 GiB weights/GPU).
        # vLLM: TP=8 (intra-node, NVLink) x PP=4 (across 4 nodes, EFA) = 32
        # GPUs/engine, num_engines = 64/32 = 2 (colocated over the same 64 GPUs).
        # vLLM TP must divide NemotronH's Mamba n_groups (=8), so TP=16 is invalid;
        # the cross-node parallelism comes from PP instead. vLLM PP=4 (not 2) shards
        # vLLM weights ~34 GiB/GPU so that during the colocated Megatron->vLLM weight
        # broadcast the policy shard (~69 GiB) + woken vLLM weights (~34 GiB) fit
        # alongside the broadcast buffers on the 141 GiB H200 (PP=2 -> ~69+69 OOMs).
        # Needs VLLM_USE_RAY_V2_EXECUTOR_BACKEND=1 (compiled-DAG shm channel crashes
        # the raylet cross-node) and the ray distributed executor backend.
        pytest.param(
            8,
            2,
            1,
            16,
            1,
            8,
            4,
            64,
            8,
            "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
            3e-1,
            # 550B hybrid-MoE diverges more between vLLM and Megatron than the
            # smaller MoE cases (observed diff mean ~0.059, driven by a minority
            # of high-divergence tokens: Megatron logprob std ~0.445 vs vLLM ~0.178).
            8e-2,
            id="nemotron3-ultra_tp8pp4_mega_tp8pp2ep16_8node",
            marks=pytest.mark.h100,
        ),
        # Same as above but EP=32 (full expert sharding, ETP=1 -> EDP=1). This is the
        # parallelism the full-FT training recipe uses (EP=16 OOMs there once the
        # optimizer is added). Diagnostic: training at EP=32 produced GARBAGE vLLM
        # generations from step 1, while EP=16 (above) passes -> this isolates whether
        # the Megatron->vLLM expert weight gather is correct at EP=32. A blown-up
        # post-sync diff (vs pre-sync OK) would pin the bug to the EP=32 sync gather.
        pytest.param(
            8,
            2,
            1,
            32,
            1,
            8,
            4,
            64,
            8,
            "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
            3e-1,
            8e-2,
            id="nemotron3-ultra_tp8pp4_mega_tp8pp2ep32_8node",
            marks=pytest.mark.h100,
        ),
        # Same as the validated EP=16 case but Megatron PP=4 (vs 2). DISCRIMINATOR: full-FT
        # training at EP=16/PP=4 produced GARBAGE vLLM generations, while this inference-only
        # path at EP=16/PP=2 passes (0.298). PP=4 is the only parallelism change from the
        # validated case, so if this PASSES, PP=4 sync is fine and the training garbage comes
        # from the full-FT optimizer init; if it FAILS, PP=4 itself breaks the weight-sync gather.
        pytest.param(
            8,
            4,
            1,
            16,
            1,
            8,
            4,
            64,
            8,
            "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
            3e-1,
            8e-2,
            id="nemotron3-ultra_tp8pp4_mega_tp8pp4ep16_8node",
            marks=pytest.mark.h100,
        ),
    ],
)
async def test_logprobs_matching_roundtrip(
    tp, pp, cp, ep, etp, inference_tp, inference_pp, num_gpus, num_nodes, model_name, vllm_threshold, megatron_threshold
):
    """
    Check that logprob diff matches acrosss vllm and megatron.
    """
    assert num_gpus % num_nodes == 0, f"num_gpus ({num_gpus}) must be divisible by num_nodes ({num_nodes})"
    num_gpus_per_node = num_gpus // num_nodes
    with ray_init(extra_env_vars=_extra_env_vars_for_model(model_name)):
        cfg = get_test_actor_config(model_name=model_name)
        cfg.trainer.strategy = "megatron"
        cfg.generator.inference_engine.tensor_parallel_size = inference_tp
        cfg.generator.inference_engine.pipeline_parallel_size = inference_pp
        cfg.generator.inference_engine.num_engines = num_gpus // (inference_tp * inference_pp)
        # Colocated weight sync keeps the Megatron policy shard and the woken vLLM
        # weights on the same GPUs simultaneously. For large models this is tight;
        # the expandable_segments allocator reclaims fragmentation so the weight
        # broadcast's working buffers fit.
        cfg.generator.inference_engine.use_expandable_segments = True
        cfg.generator.sampling_params = SamplingParams(
            max_generate_length=MAX_GENERATE_LENGTH,
            logprobs=1,
            temperature=0.0,
        )
        cfg.generator.batched = False
        cfg.generator.max_turns = 1

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        # Some checkpoints (e.g. NemotronH-Ultra) ship no chat template; install a
        # minimal one so the generator can format prompts. No-op if one exists.
        # vLLM and Megatron both see the same formatted input, so the logprob
        # comparison stays valid.
        _ensure_chat_template(tokenizer)

        engine_overrides = _engine_overrides_for_model(model_name)
        async with InferenceEngineState.create(
            cfg=cfg,
            model=model_name,
            use_local=True,
            colocate_all=True,
            backend="vllm",
            sleep_level=2,  # full sleep — this test explicitly syncs weights
            gpu_memory_utilization=engine_overrides["gpu_memory_utilization"],
            engine_init_kwargs=engine_overrides["engine_init_kwargs"],
        ) as engines:
            client, pg = engines.client, engines.pg
            await client.wake_up()

            generator = SkyRLGymGenerator(
                generator_cfg=cfg.generator,
                skyrl_gym_cfg=cfg.environment.skyrl_gym,
                inference_engine_client=client,
                tokenizer=tokenizer,
            )

            (response_mask, logprobs_t), training_input = await generate_with_vllm(
                generator, client, model_name, tokenizer, return_training_input=True
            )
            await client.sleep()
            cfg.trainer.placement.policy_num_nodes = num_nodes
            cfg.trainer.placement.policy_num_gpus_per_node = num_gpus_per_node
            cfg.trainer.policy.megatron_config.tensor_model_parallel_size = tp
            cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = pp
            cfg.trainer.policy.megatron_config.context_parallel_size = cp
            cfg.trainer.policy.megatron_config.expert_model_parallel_size = ep
            cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = etp
            cfg.trainer.micro_forward_batch_size_per_gpu = 2
            cfg.trainer.micro_train_batch_size_per_gpu = 2

            policy = init_worker_with_type(
                "policy",
                shared_pg=pg,
                colocate_all=True,
                num_gpus_per_node=num_gpus_per_node,
                num_nodes=num_nodes,
                cfg=cfg,
            )
            ray.get(
                policy.async_run_ray_method(
                    "pass_through", "init_weight_sync_state", client, cfg.generator.inference_engine
                )
            )

            refs = policy.async_run_ray_method("mesh", "forward", data=training_input)
            results = ray.get(refs)
            policy_output = WorkerOutput.cat(policy.actor_infos, results)
            logprobs_megatron = loss_fn_outputs_to_tensor(policy_output.loss_fn_outputs, key="logprobs")

            mask = response_mask.bool()

            vllm_valid = logprobs_t[mask]
            logprobs_megatron_valid = logprobs_megatron[mask]

            logprobs_diff = (vllm_valid - logprobs_megatron_valid).abs()
            print(f"vLLM logprobs     - mean: {vllm_valid.mean().item():.6f}, std: {vllm_valid.std().item():.6f}")
            print(
                f"Megatron - mean: {logprobs_megatron_valid.mean().item():.6f}, std: {logprobs_megatron_valid.std().item():.6f}"
            )
            print(f"logprob diff mean: {logprobs_diff.mean().item():.6f}, std: {logprobs_diff.std().item():.6f}")

            assert (
                logprobs_diff.mean().item() < megatron_threshold
            ), f"Logprob diff should be less than {megatron_threshold}, but is {logprobs_diff.mean().item():.6f}"

            # sync weights
            policy.offload_to_cpu(offload_optimizer=True, offload_model=False)
            await client.wake_up(tags=["weights"])
            with Timer("sync_weights"):
                ray.get(
                    policy.async_run_ray_method(
                        "pass_through", "broadcast_to_inference_engines", client, cfg.generator.inference_engine
                    )
                )
            policy.offload_to_cpu(offload_optimizer=False, offload_model=True)
            await client.wake_up(tags=["kv_cache"])

            response_mask_2, logprobs_t_2 = await generate_with_vllm(
                generator, client, model_name, tokenizer, return_training_input=False
            )

            logprobs_t_valid = logprobs_t[response_mask.bool()]
            logprobs_t_2_valid = logprobs_t_2[response_mask_2.bool()]

            # Pre- and post-sync are two independent sampled generations
            # so truncate to the shorter sequence for the magnitude check.
            if logprobs_t_valid.shape[0] != logprobs_t_2_valid.shape[0]:
                min_len = min(logprobs_t_valid.shape[0], logprobs_t_2_valid.shape[0])
                print(
                    f"NOTE: pre/post-sync generation lengths differ "
                    f"({logprobs_t_valid.shape[0]} vs {logprobs_t_2_valid.shape[0]}); "
                    f"truncating to {min_len} for the magnitude check."
                )
                logprobs_t_valid = logprobs_t_valid[:min_len]
                logprobs_t_2_valid = logprobs_t_2_valid[:min_len]

            logprobs_diff = (logprobs_t_valid - logprobs_t_2_valid).abs()
            print(
                f"vLLM logprobs    - mean: {logprobs_t_valid.mean().item():.6f}, std: {logprobs_t_valid.std().item():.6f}"
            )
            print(
                f"vLLM logprobs after sync - mean: {logprobs_t_2_valid.mean().item():.6f}, std: {logprobs_t_2_valid.std().item():.6f}"
            )
            print(f"vLLM logprob diff mean: {logprobs_diff.mean().item():.6f}, std: {logprobs_diff.std().item():.6f}")
            assert (
                logprobs_diff.mean().item() < vllm_threshold
            ), f"Logprob diff should be less than {vllm_threshold}, but is {logprobs_diff.mean().item():.6f}"
