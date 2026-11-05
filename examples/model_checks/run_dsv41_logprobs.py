"""Score fixed probe tokens with the Megatron trainer and the vLLM sampler for DeepSeek-V4.1-Flash; report the gap.

Uses the async LoRA recipe's layout and settings (Megatron TP1 / EP8 on one node, one vLLM TP8 engine on another,
LoRA r32 on the same modules, fused LM-head logprob, no packed sequences), with no routing replay: the probe tokens
are not sampled, so the gap includes any router disagreement. Switches isolate one factor at a time:

    uv run --isolated --extra megatron -m examples.model_checks.run_dsv41_logprobs --model <dir> [--no-lora]
        [--enforce-eager] [--no-fused-lm-head]

Prints mean / max abs logprob error and the mean signed error (trainer - inference) over all probe tokens.
"""

import argparse
import asyncio
import os

from examples.model_checks.logprob_checks import (
    build_probe_sequences,
    compare_logprobs,
    replicate_to_multiple,
)
from examples.model_checks.run_nemotron_logprobs import (
    build_batch,
    open_runtime,
    publish,
    score_inference,
    score_trainer,
)
from skyrl.backends.skyrl_train.inference_servers.utils import resolve_policy_model_name
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import validate_cfg
from skyrl.utils.tok import get_tokenizer

LORA_TARGET_MODULES = [
    "linear_q_down_proj",
    "linear_q_up_proj",
    "linear_kv_proj",
    "linear_proj",
    "linear_fc1",
    "linear_fc2",
]
VLLM_LORA_TARGET_MODULES = ["fused_wqa_wkv", "wq_b", "wo_b", "gate_up_proj", "down_proj", "experts"]


def build_config(args: argparse.Namespace) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.policy.model.path = args.model
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = 8
    cfg.trainer.placement.ref_num_nodes = 1
    cfg.trainer.placement.ref_num_gpus_per_node = 8
    cfg.trainer.policy.inference_only_init = True
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.remove_microbatch_padding = False
    cfg.trainer.fused_lm_head_logprob = not args.no_fused_lm_head
    cfg.trainer.fused_lm_head_logprob_backend = "triton"
    cfg.trainer.logger = "console"
    parallel = cfg.trainer.policy.megatron_config
    parallel.tensor_model_parallel_size = 1
    parallel.pipeline_model_parallel_size = 1
    parallel.context_parallel_size = 1
    parallel.expert_model_parallel_size = 8
    parallel.expert_tensor_parallel_size = 1
    parallel.mtp_num_layers = 0
    parallel.moe_grouped_gemm = True
    parallel.moe_token_dispatcher_type = "alltoall"
    parallel.lora_config.merge_lora = False
    lora = cfg.trainer.policy.model.lora
    lora.rank = 0 if args.no_lora else 32
    lora.alpha = 32
    lora.target_modules = LORA_TARGET_MODULES
    if not args.no_lora:
        lora.sync_mode = "memory"
    engine = cfg.generator.inference_engine
    engine.run_engines_locally = True
    engine.tensor_parallel_size = 8
    engine.num_engines = 1
    engine.enforce_eager = args.enforce_eager
    engine.weight_sync_backend = "nccl"
    engine.gpu_memory_utilization = 0.85
    engine.engine_init_kwargs = {
        "max_model_len": args.max_model_len,
        "swa_bounded_replay": False,
        **({} if args.no_lora else {"lora_target_modules": VLLM_LORA_TARGET_MODULES}),
    }
    validate_cfg(cfg)
    return cfg


async def run(args: argparse.Namespace) -> None:
    cfg = build_config(args)
    tokenizer = get_tokenizer(cfg.trainer.policy.model.path, trust_remote_code=True)
    async with open_runtime(cfg, tokenizer) as (policy, client):
        sequences = replicate_to_multiple(build_probe_sequences(tokenizer), policy.get_dp_size())
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        batch = build_batch(sequences, pad_token_id)
        if not args.no_lora:
            await publish(policy, client, cfg)
        model = resolve_policy_model_name(cfg)
        trainer = score_trainer(policy, batch)
        inference = await score_inference(client, sequences, model)
        stats = compare_logprobs(trainer, inference)
        signed = sum(t - i for t, i in zip(trainer, inference)) / len(trainer)
        tag = os.environ.get("RUN_TAG", "")
        print(f"[dsv41-logprobs] {tag} n={len(trainer)} {stats} mean_signed(trainer-inference)={signed:.4f}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--no-fused-lm-head", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=8192)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
