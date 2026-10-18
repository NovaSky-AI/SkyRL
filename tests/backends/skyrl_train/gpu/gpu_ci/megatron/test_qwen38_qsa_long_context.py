"""
Run with:
SKYRL_QWEN38_4LAYER_PATH=<slice> uv run --isolated --extra dev --extra megatron pytest -m h100 -s \
    tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_qwen38_qsa_long_context.py

Qwen3.8-Flash-Next QSA sparse regime: vLLM vs Megatron logprobs on ~8K-token sequences.

test_megatron_models.py keeps every sequence under indexer_budget + compress_ratio - 1 = 2051
tokens, where the QSA indexer selects every key and attention is exactly dense. Here each prompt
is ~7.7K tokens (concatenated GSM8K questions), so every response token attends through the
indexer's top-512-of-~1.9K block selection on both sides: vLLM's QSA kernels and Megatron's
Qwen4ExpQSAIndexer + qsa_sparse_attention. Forward only, no weight sync (the roundtrip row
covers that).
"""

import pytest
import ray
from transformers import AutoTokenizer

from skyrl.backends.skyrl_train.distributed.dispatch import (
    WorkerOutput,
    loss_fn_outputs_to_tensor,
)
from skyrl.train.config import SamplingParams
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator
from tests.backends.skyrl_train.gpu.gpu_ci.conftest import ray_init
from tests.backends.skyrl_train.gpu.gpu_ci.megatron.test_megatron_models import (
    QWEN38_4LAYER,
    _engine_overrides_for_model,
    _extra_env_vars_for_model,
    generate_with_vllm,
    get_test_actor_config,
)
from tests.backends.skyrl_train.gpu.utils import (
    InferenceEngineState,
    Timer,
    get_test_generator_input,
    init_worker_with_type,
)

PROMPT_TOKENS = 7680
GENERATE_LENGTH = 256
MAX_MODEL_LEN = 8448 + 512  # prompt + response + chat-template headroom
NUM_PROMPTS = 8  # two per DP rank at EP=4 (DP=4)


def _long_context_input(model_name, tokenizer):
    """GSM8K prompts whose user turn is replaced by ~PROMPT_TOKENS tokens of concatenated questions."""
    batch = get_test_generator_input(model=model_name, num_prompts=400, n_samples_per_prompt=1)
    questions = [p[-1]["content"] for p in batch["prompts"]]
    prompts = []
    for i in range(NUM_PROMPTS):
        parts, n, j = [], 0, i * 37
        while n < PROMPT_TOKENS:
            q = questions[j % len(questions)]
            parts.append(q)
            n += len(tokenizer.encode(q, add_special_tokens=False)) + 2
            j += 1
        prompts.append([{"role": "user", "content": "\n\n".join(parts)}])
    return {
        "prompts": prompts,
        "env_classes": batch["env_classes"][:NUM_PROMPTS],
        "env_extras": batch["env_extras"][:NUM_PROMPTS],
        "trajectory_ids": batch["trajectory_ids"][:NUM_PROMPTS],
    }


@pytest.mark.asyncio
@pytest.mark.megatron_models
@pytest.mark.h100
@pytest.mark.parametrize("threshold", [pytest.param(1e-1, id="qwen3.8-flash-next-4layer_h100_ep4_qsa8k")])
async def test_qwen38_qsa_long_context_logprobs(threshold, monkeypatch):
    model_name = QWEN38_4LAYER
    tp, ep, inference_tp, num_gpus = 1, 4, 4, 4
    with ray_init(extra_env_vars=_extra_env_vars_for_model(model_name)):
        cfg = get_test_actor_config(model_name=model_name)
        cfg.trainer.strategy = "megatron"
        cfg.trainer.max_prompt_length = MAX_MODEL_LEN
        cfg.generator.max_input_length = MAX_MODEL_LEN
        cfg.generator.inference_engine.tensor_parallel_size = inference_tp
        cfg.generator.inference_engine.num_engines = num_gpus // inference_tp
        cfg.generator.sampling_params = SamplingParams(max_generate_length=GENERATE_LENGTH, logprobs=1, temperature=0.0)
        cfg.generator.batched = False
        cfg.generator.max_turns = 1
        cfg.trainer.placement.policy_num_gpus_per_node = num_gpus
        mcfg = cfg.trainer.policy.megatron_config
        mcfg.tensor_model_parallel_size = tp
        mcfg.pipeline_model_parallel_size = 1
        mcfg.context_parallel_size = 1
        mcfg.expert_model_parallel_size = ep
        mcfg.expert_tensor_parallel_size = 1
        cfg.trainer.micro_forward_batch_size_per_gpu = 1
        cfg.trainer.max_tokens_per_microbatch = MAX_MODEL_LEN

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        long_input = _long_context_input(model_name, tokenizer)

        engine_overrides = _engine_overrides_for_model(model_name)
        engine_overrides["engine_init_kwargs"]["max_model_len"] = MAX_MODEL_LEN
        async with InferenceEngineState.create(
            cfg=cfg,
            model=model_name,
            use_local=True,
            colocate_all=True,
            backend="vllm",
            sleep_level=1,
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
                policy_model_name=None,
            )
            await client.wake_up()
            # generate_with_vllm builds its batch with get_test_generator_input; hand it ours.
            monkeypatch.setattr(
                "tests.backends.skyrl_train.gpu.gpu_ci.megatron.test_megatron_models.get_test_generator_input",
                lambda **_: long_input,
            )
            (response_mask, logprobs_t, gen_out), training_input = await generate_with_vllm(
                generator,
                client,
                model_name,
                tokenizer,
                return_training_input=True,
                max_generate_length=GENERATE_LENGTH,
            )
            await client.sleep()
            prompt_lens = [len(p) for p in gen_out["prompt_token_ids"]]
            response_lens = [len(r) for r in gen_out["response_ids"]]
            print(f"prompt tokens {prompt_lens}, response tokens {response_lens}")
            assert min(prompt_lens) > 2051, "prompts must exceed the QSA dense regime"

            policy = init_worker_with_type(
                "policy", shared_pg=pg, colocate_all=True, num_gpus_per_node=num_gpus, cfg=cfg
            )
            with Timer("megatron_forward_8k"):
                results = ray.get(policy.async_run_ray_method("mesh", "forward", data=training_input))
            output = WorkerOutput.cat(policy.actor_infos, results)
            logprobs_megatron = loss_fn_outputs_to_tensor(output.loss_fn_outputs, key="logprobs")

            mask = response_mask.bool()
            diff = (logprobs_t[mask] - logprobs_megatron[mask]).abs()
            print(
                f"QSA sparse (~{PROMPT_TOKENS} prompt tokens): vLLM vs Megatron logprob diff mean "
                f"{diff.mean().item():.6f}, std {diff.std().item():.6f}, max {diff.max().item():.6f} "
                f"over {diff.numel()} tokens"
            )
            assert diff.mean().item() < threshold, f"Logprob diff should be < {threshold}, is {diff.mean().item():.6f}"
