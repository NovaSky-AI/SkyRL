"""
Run with:
uv run --isolated --extra dev --extra megatron -- pytest -s tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_megatron_models.py

The *_full_fp8 / *_fp8_param rows are Hopper-only (pytest.mark.h100): they run
blockwise FP8 on both Megatron (fp8=e4m3 + fp8_recipe=blockwise, plus
fp8_param=true persistent params for the fp8_param row) and vLLM
(quantization=fp8 fed by fp8_weight_sync_mode=serialized_blockwise), with FP32
block scales (NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1, set by
_extra_env_vars_for_model). Select them with: -k "full_fp8 or fp8_param".
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
    is_large_moe = ("qwen3.5-35b" in model_name.lower() and "tiny" not in model_name.lower()) or (
        "nemotron-3-nano" in model_name.lower()
    )
    if is_large_moe:
        cfg.trainer.policy.inference_only_init = True
    validate_cfg(cfg)
    return cfg


def _extra_env_vars_for_model(model_name: str, fp8_mode: str | None = None) -> dict[str, str] | None:
    env: dict[str, str] = {}
    # MLA models need cuDNN fused attention (the conftest globally sets
    # NVTE_FUSED_ATTN=0; re-enable it here so the fused backend is available).
    if "moonlight" in model_name.lower() or "glm-4" in model_name.lower():
        env["NVTE_FUSED_ATTN"] = "1"
    if fp8_mode:
        # Hopper serialized-FP8 contract: FP32 block scales end-to-end, and
        # vLLM must not requantize wire scales to E8M0 (train/utils/utils.py
        # pins both in production; the test sets them explicitly because the
        # fp8 fields are applied after get_test_actor_config's validate_cfg).
        env["NVTE_FP8_BLOCK_SCALING_FP32_SCALES"] = "1"
        env["VLLM_USE_DEEP_GEMM_E8M0"] = "0"
    return env or None


def _engine_overrides_for_model(model_name: str, fp8_mode: str | None = None) -> dict:
    """Per-model overrides for vLLM engine init."""
    overrides = {"engine_init_kwargs": {}, "gpu_memory_utilization": 0.9}
    if "Nemotron-3-Nano" in model_name:
        overrides["engine_init_kwargs"]["max_model_len"] = 4096
        # Megatron policy init also needs room alongside vLLM on the same
        # GPU, so lower vLLM's pool footprint.
        overrides["gpu_memory_utilization"] = 0.5
    # Large MoE: Megatron policy init also needs room alongside vLLM on the
    # same GPU, so lower vLLM's pool footprint.
    if "qwen3.5-35b" in model_name.lower() and "tiny" not in model_name.lower():
        overrides["gpu_memory_utilization"] = 0.5
        if fp8_mode:
            # FP8 runs vLLM TP=1, so each rank holds the full ~35 GiB of FP8
            # weights; at gmu 0.5 on H100-80G the KV pool cannot cover the
            # checkpoint's 262144 max_model_len. The test generates ~640
            # tokens per sequence.
            overrides["engine_init_kwargs"]["max_model_len"] = 4096
            # GDN hybrid: one Mamba cache block per decode seq; the slim KV
            # pool fits ~163 blocks, and the vLLM default max_num_seqs=1024
            # fails CUDA-graph capture. The test runs <= 80 concurrent seqs.
            overrides["max_num_seqs"] = 128
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
            pad_token_id=tokenizer.pad_token_id,
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
            }
        )
        training_input.metadata = {"response_length": num_actions}
        return (response_mask, logprobs_t, generator_output), training_input
    else:
        return (response_mask, logprobs_t, generator_output)


async def construct_training_input_from_generator_output(generator_output, tokenizer):
    return convert_prompts_responses_to_batch_tensors(
        pad_token_id=tokenizer.pad_token_id,
        prompts=generator_output["prompt_token_ids"],
        responses=generator_output["response_ids"],
        rewards=generator_output["rewards"],
        loss_masks=generator_output["loss_masks"],
    )


@pytest.mark.asyncio
@pytest.mark.megatron_models
@pytest.mark.parametrize(
    "tp,pp,cp,ep,etp,inference_tp,num_gpus,model_name,vllm_threshold,megatron_threshold,fp8_mode",
    [
        pytest.param(2, 1, 1, 2, 1, 2, 4, "eatang/qwen3-moe-tiny-random", 1e-1, 2e-1, None, id="qwen3-moe_tp2_ep2"),
        pytest.param(1, 2, 2, 1, None, 2, 4, "eatang/qwen3-moe-tiny-random", 1e-1, 2e-1, None, id="qwen3-moe_pp2_cp2"),
        pytest.param(
            2,
            1,
            1,
            2,
            1,
            2,
            4,
            "eatang/glm-4.7-flash-tiny-random",
            1e-1,
            2e-2,
            None,
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
            4,
            "eatang/qwen3.5-moe-tiny-random",
            1e-1,
            2e-1,
            None,
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
            2,
            "Qwen/Qwen3.5-0.8B",
            1e-1,
            5e-2,
            None,
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
            4,
            "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            5e-1,
            5e-2,
            None,
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
            4,
            "Qwen/Qwen3.5-35B-A3B",
            3e-1,
            5e-2,
            None,
            id="qwen3.5-35b-a3b_h100_tp4_ep4",
            marks=pytest.mark.h100,
        ),
        # Full-FP8 rows: blockwise FP8 Megatron compute + FP8 vLLM rollout fed
        # by serialized blockwise weight sync; the fp8_param row additionally
        # keeps persistent FP8 Megatron params with exact optimizer-master
        # init from unquantized checkpoint shards. Hopper-only: the wire
        # contract and fp8_param require FP32 block scales
        # (NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1); Blackwell runs power-of-2
        # scales with fp8_param=false. Thresholds mirror the matching bf16
        # rows; tune as we accumulate measured diffs.
        pytest.param(
            2,
            1,
            1,
            1,
            None,
            2,
            2,
            "Qwen/Qwen3.5-0.8B",
            1e-1,
            5e-2,
            "full_fp8",
            id="qwen3.5-0.8b-dense_tp2_full_fp8",
            marks=pytest.mark.h100,
        ),
        pytest.param(
            2,
            1,
            1,
            1,
            None,
            2,
            2,
            "Qwen/Qwen3.5-0.8B",
            1e-1,
            5e-2,
            "fp8_param",
            id="qwen3.5-0.8b-dense_tp2_fp8_param",
            marks=pytest.mark.h100,
        ),
        # vLLM must run TP=1 for this model under blockwise FP8: at TP=4 a
        # 4304-wide projection shards to 1076 per rank, which vLLM rejects
        # ("output_partition_size = 1076 is not divisible by ... block_n =
        # 128"). TP=1 x 4 engines matches the validated production layout
        # (Megatron TP/EP shards -> full-width vLLM ranks).
        pytest.param(
            4,
            1,
            1,
            4,
            1,
            1,
            4,
            "Qwen/Qwen3.5-35B-A3B",
            3e-1,
            5e-2,
            "full_fp8",
            id="qwen3.5-35b-a3b_h100_tp4_ep4_full_fp8",
            marks=pytest.mark.h100,
        ),
    ],
)
async def test_logprobs_matching_roundtrip(
    tp, pp, cp, ep, etp, inference_tp, num_gpus, model_name, vllm_threshold, megatron_threshold, fp8_mode
):
    """
    Check that logprob diff matches acrosss vllm and megatron.
    """
    with ray_init(extra_env_vars=_extra_env_vars_for_model(model_name, fp8_mode)):
        cfg = get_test_actor_config(model_name=model_name)
        cfg.trainer.strategy = "megatron"
        cfg.generator.inference_engine.tensor_parallel_size = inference_tp
        cfg.generator.inference_engine.num_engines = num_gpus // inference_tp
        cfg.generator.sampling_params = SamplingParams(
            max_generate_length=MAX_GENERATE_LENGTH,
            logprobs=1,
            temperature=0.0,
        )
        cfg.generator.batched = False
        cfg.generator.max_turns = 1

        if fp8_mode:
            # Megatron: blockwise FP8 compute; the fp8_param variant keeps
            # persistent FP8 params (requires fp8_param_gather so updated FP32
            # masters requantize into the FP8 compute weights).
            mcfg = cfg.trainer.policy.megatron_config
            transformer_config_kwargs = dict(mcfg.transformer_config_kwargs or {})
            transformer_config_kwargs.update(
                {
                    "fp8": "e4m3",
                    "fp8_recipe": "blockwise",
                    "fp8_amax_compute_algo": "most_recent",
                    "fp8_param": fp8_mode == "fp8_param",
                }
            )
            mcfg.transformer_config_kwargs = transformer_config_kwargs
            if fp8_mode == "fp8_param":
                mcfg.ddp_config.fp8_param_gather = True
            # vLLM: FP8 rollout fed by serialized blockwise weight sync
            # (_apply_serialized_fp8_weight_sync_defaults injects
            # quantization=fp8, load_format=dummy and the blockwise
            # quantization_config into the engine kwargs).
            cfg.generator.inference_engine.fp8_weight_sync_mode = "serialized_blockwise"
            # The validated FP8 production runs use the mp executor; with the
            # ray executor, vLLM 0.23's ray_executor_v2 ignores
            # VLLM_RAY_BUNDLE_INDICES, so multi-engine colocate (e.g. the 35B
            # row's 4 x TP=1) stacks every engine's worker on GPU 0 and OOMs.
            cfg.generator.inference_engine.distributed_executor_backend = "mp"

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        engine_overrides = _engine_overrides_for_model(model_name, fp8_mode)
        async with InferenceEngineState.create(
            cfg=cfg,
            model=model_name,
            use_local=True,
            colocate_all=True,
            backend="vllm",
            sleep_level=2,  # full sleep — this test explicitly syncs weights
            gpu_memory_utilization=engine_overrides["gpu_memory_utilization"],
            engine_init_kwargs=engine_overrides["engine_init_kwargs"],
            max_num_seqs=engine_overrides.get("max_num_seqs"),
        ) as engines:
            client, pg = engines.client, engines.pg

            generator = SkyRLGymGenerator(
                generator_cfg=cfg.generator,
                skyrl_gym_cfg=cfg.environment.skyrl_gym,
                inference_engine_client=client,
                tokenizer=tokenizer,
            )

            cfg.trainer.placement.policy_num_gpus_per_node = num_gpus
            cfg.trainer.policy.megatron_config.tensor_model_parallel_size = tp
            cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = pp
            cfg.trainer.policy.megatron_config.context_parallel_size = cp
            cfg.trainer.policy.megatron_config.expert_model_parallel_size = ep
            cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = etp
            cfg.trainer.micro_forward_batch_size_per_gpu = 2
            cfg.trainer.micro_train_batch_size_per_gpu = 2

            policy = None
            if fp8_mode:
                # Serialized FP8 boots vLLM with load_format="dummy", so real
                # weights must be synced from Megatron before generating
                # (mirrors the trainer, which always syncs before the first
                # rollout). Build the policy with the engines asleep, then
                # run the same offload/wake/broadcast dance as the sync below.
                await client.sleep()
                policy = init_worker_with_type(
                    "policy",
                    shared_pg=pg,
                    colocate_all=True,
                    num_gpus_per_node=num_gpus,
                    cfg=cfg,
                )
                ray.get(
                    policy.async_run_ray_method(
                        "pass_through", "init_weight_sync_state", client, cfg.generator.inference_engine
                    )
                )
                policy.offload_to_cpu(offload_optimizer=True, offload_model=False)
                await client.wake_up(tags=["weights"])
                with Timer("initial_sync_weights"):
                    ray.get(
                        policy.async_run_ray_method(
                            "pass_through", "broadcast_to_inference_engines", client, cfg.generator.inference_engine
                        )
                    )
                policy.offload_to_cpu(offload_optimizer=False, offload_model=True)
                await client.wake_up(tags=["kv_cache"])
            else:
                await client.wake_up()

            (response_mask, logprobs_t, gen_out_1), training_input = await generate_with_vllm(
                generator, client, model_name, tokenizer, return_training_input=True
            )
            await client.sleep()

            if policy is None:
                policy = init_worker_with_type(
                    "policy",
                    shared_pg=pg,
                    colocate_all=True,
                    num_gpus_per_node=num_gpus,
                    cfg=cfg,
                )
                ray.get(
                    policy.async_run_ray_method(
                        "pass_through", "init_weight_sync_state", client, cfg.generator.inference_engine
                    )
                )
            else:
                policy.backload_to_gpu(backload_optimizer=False, backload_model=True)

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

            response_mask_2, logprobs_t_2, gen_out_2 = await generate_with_vllm(
                generator, client, model_name, tokenizer, return_training_input=False
            )

            if fp8_mode:
                # In the FP8 flow both generations ran on identical synced
                # weights, so compare logprobs only on each sequence's common
                # prefix: once greedy decoding diverges at a near-tie token,
                # later positions score different tokens and their diff is
                # pure noise (measured up to ~0.14 mean on identical weights,
                # vs ~1e-3 on common prefixes).
                ids_1, lp_1 = gen_out_1["response_ids"], gen_out_1["rollout_logprobs"]
                ids_2, lp_2 = gen_out_2["response_ids"], gen_out_2["rollout_logprobs"]
                assert lp_1 is not None and lp_2 is not None, "resync check needs rollout logprobs"
                diffs = []
                divergent = 0
                for s1, s2, l1, l2 in zip(ids_1, ids_2, lp_1, lp_2):
                    n = 0
                    for a, b in zip(s1, s2):
                        if a != b:
                            break
                        n += 1
                    if n < min(len(s1), len(s2)):
                        divergent += 1
                    diffs.extend(abs(x - y) for x, y in zip(l1[:n], l2[:n]))
                assert diffs, "no common-prefix tokens between pre/post-sync generations"
                logprobs_diff = torch.tensor(diffs)
                print(
                    f"vLLM resync common-prefix logprob diff mean: {logprobs_diff.mean().item():.6f}, "
                    f"std: {logprobs_diff.std().item():.6f} over {len(diffs)} tokens "
                    f"({divergent}/{len(ids_1)} sequences diverged at a near-tie token)"
                )
            else:
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
                print(
                    f"vLLM logprob diff mean: {logprobs_diff.mean().item():.6f}, std: {logprobs_diff.std().item():.6f}"
                )
            assert (
                logprobs_diff.mean().item() < vllm_threshold
            ), f"Logprob diff should be less than {vllm_threshold}, but is {logprobs_diff.mean().item():.6f}"
