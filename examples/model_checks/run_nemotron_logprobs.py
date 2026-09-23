"""Compare trainer and inference logprobs for a large model through one weight sync.

Defaults target Nemotron-3-Super-120B on 8 trainer GPUs (Megatron TP8 EP8) plus 8
inference GPUs (vLLM TP8), non-colocated across two nodes:

    uv run --isolated --extra megatron -m examples.model_checks.run_nemotron_logprobs
    uv run --isolated --extra megatron -m examples.model_checks.run_nemotron_logprobs --full-ft

Flow (the same for LoRA and full fine-tuning; only the perturbed tensors differ):

    base     score trainer and inference on fixed probe tokens; must agree
    stale    perturb the trainer (LoRA B tensors, or every weight); it must now differ from inference
    publish  sync the trainer to the inference engines
    updated  score inference twice; must agree with the trainer, and with itself (repeat)

No optimizer is created. With LoRA the adapter is exported to ``--lora-sync-path``,
which every node's inference engines must be able to read (a shared mount).
A JSON report with every phase's statistics is written to ``--output``.
"""

import argparse
import asyncio
import json
import math
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import ray
import torch

from examples.model_checks.logprob_checks import (
    build_probe_sequences,
    check_agreement,
    compare_logprobs,
    perturb_full_weights,
    perturb_lora_b,
)
from skyrl.backends.skyrl_train.distributed.dispatch import (
    WorkerOutput,
    loss_fn_outputs_to_tensor,
)
from skyrl.backends.skyrl_train.inference_servers.setup import (
    build_new_inference_client,
)
from skyrl.backends.skyrl_train.inference_servers.utils import resolve_policy_model_name
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
    MegatronPolicyWorkerBase,
)
from skyrl.backends.skyrl_train.workers.worker import PPORayActorGroup
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import initialize_ray, validate_cfg
from skyrl.utils.tok import get_tokenizer

DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"


class LogprobCheckWorkerBase(MegatronPolicyWorkerBase):
    def perturb(self, mode: str, multiplier: float) -> Dict[str, float]:
        # One model chunk per virtual pipeline stage.
        named_parameters = (
            (f"chunk{index}.{name}", parameter)
            for index, chunk in enumerate(self.actor_module)
            for name, parameter in chunk.named_parameters()
        )
        if mode == "lora":
            return perturb_lora_b(named_parameters, multiplier=multiplier)
        return perturb_full_weights(named_parameters, multiplier=multiplier)


LogprobCheckWorker = ray.remote(num_gpus=1)(LogprobCheckWorkerBase)


def build_config(args: argparse.Namespace) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.policy.model.path = args.model
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = args.trainer_gpus
    # No reference model is built; its placement only has to agree with the policy's.
    cfg.trainer.placement.ref_num_nodes = 1
    cfg.trainer.placement.ref_num_gpus_per_node = args.trainer_gpus
    # Forward and weight sync only: no fp32 masters or optimizer state.
    cfg.trainer.policy.inference_only_init = True
    cfg.trainer.micro_forward_batch_size_per_gpu = 2
    cfg.trainer.remove_microbatch_padding = True
    cfg.trainer.logger = "console"
    cfg.trainer.log_path = str(args.output.parent / "runtime-logs")
    parallel = cfg.trainer.policy.megatron_config
    parallel.tensor_model_parallel_size = args.tp
    parallel.expert_model_parallel_size = args.ep
    parallel.expert_tensor_parallel_size = args.etp
    parallel.lora_config.merge_lora = False
    lora = cfg.trainer.policy.model.lora
    lora.rank = 0 if args.full_ft else args.lora_rank
    lora.alpha = 2 * args.lora_rank
    # MLA and NemotronH attention have no ``linear_qkv``; adapt the output projection and the MLP.
    lora.target_modules = ["linear_proj", "linear_fc1", "linear_fc2"]
    lora.lora_sync_path = args.lora_sync_path
    engine = cfg.generator.inference_engine
    engine.run_engines_locally = True
    engine.tensor_parallel_size = args.inference_tp
    engine.num_engines = 1
    engine.engine_init_kwargs = {"max_model_len": args.max_model_len}
    validate_cfg(cfg)
    return cfg


@asynccontextmanager
async def open_runtime(cfg: SkyRLTrainConfig, tokenizer):
    """Start the inference engines and the policy workers; tear both down on exit."""
    if ray.is_initialized():
        raise RuntimeError("Run this script in a fresh driver process")
    initialize_ray(cfg)
    try:
        client, setup = build_new_inference_client(cfg, tokenizer)
        try:
            policy = PPORayActorGroup(
                cfg.trainer,
                num_nodes=cfg.trainer.placement.policy_num_nodes,
                num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
                ray_actor_type=LogprobCheckWorker,
                sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
                record_memory=cfg.trainer.policy.record_memory,
            )
            ray.get(policy.async_init_model(cfg.trainer.policy.model.path))
            ray.get(
                policy.async_run_ray_method(
                    "pass_through", "init_weight_sync_state", client, cfg.generator.inference_engine
                )
            )
            yield policy, client
        finally:
            try:
                await client.aclose()
            finally:
                if setup.router is not None:
                    setup.router.shutdown()
                for group in setup.server_groups:
                    group.shutdown()
    finally:
        ray.shutdown()


def build_batch(sequences: List[List[int]], pad_token_id: int) -> TrainingInputBatch:
    """Right-padded batch whose response is every token after the first."""
    batch_size = len(sequences)
    seq_len = max(len(tokens) for tokens in sequences)
    num_actions = seq_len - 1
    input_ids = torch.full((batch_size, seq_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)
    response_mask = torch.zeros((batch_size, num_actions), dtype=torch.long)
    for row, tokens in enumerate(sequences):
        input_ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        attention_mask[row, : len(tokens)] = 1
        response_mask[row, : len(tokens) - 1] = 1
    zeros = torch.zeros((batch_size, num_actions), dtype=torch.float32)
    batch = TrainingInputBatch(
        {
            "sequences": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "loss_mask": response_mask.clone(),
            "rewards": zeros.clone(),
            "rollout_logprobs": zeros.clone(),
            "rollout_expert_indices": None,
            "action_log_probs": zeros.clone(),
            "base_action_log_probs": zeros.clone(),
            "advantages": zeros.clone(),
        }
    )
    batch.metadata = {"response_length": num_actions}
    return batch


def score_trainer(policy: PPORayActorGroup, batch: TrainingInputBatch) -> List[float]:
    """Trainer logprob of every response token, flattened in batch order."""
    results = ray.get(policy.async_run_ray_method("mesh", "forward", data=batch))
    output = WorkerOutput.cat(policy.actor_infos, results)
    logprobs = loss_fn_outputs_to_tensor(output.loss_fn_outputs, key="logprobs")
    scores = logprobs[batch["response_mask"].bool()].tolist()
    if not all(map(math.isfinite, scores)):
        raise ValueError("nonfinite trainer logprobs")
    return scores


async def score_inference(client, sequences: List[List[int]], model: str) -> List[float]:
    """Inference prompt logprob of every token after the first, flattened in batch order."""
    await client.reset_prefix_cache()
    scores: List[float] = []
    for tokens in sequences:
        result = await client.sample(
            {
                "json": {
                    "model": model,
                    "prompt": {"chunks": [{"type": "encoded_text", "tokens": tokens}]},
                    "sampling_params": {"max_tokens": 1, "temperature": 1.0},
                    "num_samples": 1,
                    "prompt_logprobs": True,
                }
            }
        )
        values = result["prompt_logprobs"]
        if values is None or len(values) != len(tokens) or values[0] is not None or any(v is None for v in values[1:]):
            raise ValueError(f"unexpected prompt_logprobs for {len(tokens)} tokens: {values}")
        if not all(map(math.isfinite, values[1:])):
            raise ValueError("nonfinite inference logprobs")
        scores.extend(values[1:])
    return scores


async def publish(policy: PPORayActorGroup, client, cfg: SkyRLTrainConfig) -> None:
    """Sync the trainer weights (or adapter) to the inference engines."""
    await client.pause_generation()
    try:
        ray.get(
            policy.async_run_ray_method(
                "pass_through", "broadcast_to_inference_engines", client, cfg.generator.inference_engine
            )
        )
    finally:
        await client.resume_generation()


async def run_check(policy, client, cfg: SkyRLTrainConfig, tokenizer, args: argparse.Namespace, report: Dict[str, Any]):
    sequences = build_probe_sequences(tokenizer)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    batch = build_batch(sequences, pad_token_id)
    lora = cfg.trainer.policy.model.lora.rank > 0
    model = resolve_policy_model_name(cfg)
    report["tokens"] = sequences
    report["model"] = model

    if lora:
        # The adapter name only exists on the engines after a sync.
        await publish(policy, client, cfg)
    trainer_base = score_trainer(policy, batch)
    inference_base = await score_inference(client, sequences, model)
    report["base"] = compare_logprobs(trainer_base, inference_base)
    print(f"base: {report['base']}", flush=True)
    check_agreement(report["base"], args.mean_atol, args.max_atol, "base")

    report["perturbation"] = ray.get(
        policy.async_run_ray_method("pass_through", "perturb", "lora" if lora else "full", args.perturb_multiplier)
    )[0]
    trainer_updated = score_trainer(policy, batch)
    report["stale"] = compare_logprobs(trainer_updated, inference_base)
    print(f"stale: {report['stale']}", flush=True)
    if report["stale"]["mean_abs"] <= args.mean_atol:
        raise AssertionError(
            f"perturbed trainer differs from the stale inference scores by only {report['stale']['mean_abs']:.6f}; "
            f"raise --perturb-multiplier so a missed sync fails the {args.mean_atol} agreement check"
        )

    await publish(policy, client, cfg)
    inference_updated = await score_inference(client, sequences, model)
    inference_repeat = await score_inference(client, sequences, model)
    report["updated"] = compare_logprobs(trainer_updated, inference_updated)
    report["repeat"] = compare_logprobs(inference_updated, inference_repeat)
    print(f"updated: {report['updated']}", flush=True)
    print(f"repeat: {report['repeat']}", flush=True)
    check_agreement(report["updated"], args.mean_atol, args.max_atol, "updated")
    if report["repeat"]["max_abs"] > args.repeat_atol:
        raise AssertionError(f"repeated inference scoring differs by {report['repeat']['max_abs']:.2e}")
    report["scores"] = {
        "trainer_base": trainer_base,
        "inference_base": inference_base,
        "trainer_updated": trainer_updated,
        "inference_updated": inference_updated,
    }


def write_report(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2))
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--full-ft", action="store_true", help="perturb and sync every weight instead of a LoRA adapter"
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-sync-path", default="/shared/skyrl-logprob-adapter")
    parser.add_argument("--trainer-gpus", type=int, default=8)
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--ep", type=int, default=8)
    parser.add_argument("--etp", type=int, default=1)
    parser.add_argument("--inference-tp", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--mean-atol", type=float, default=0.05, help="mean abs logprob error allowed")
    parser.add_argument("--max-atol", type=float, default=0.5, help="max abs logprob error allowed")
    parser.add_argument(
        "--repeat-atol", type=float, default=1e-6, help="max abs error allowed between two inference scorings"
    )
    parser.add_argument(
        "--perturb-multiplier",
        type=float,
        default=10.0,
        help="scales the perturbation (1e-3 noise std on LoRA B; 1e-3 relative noise on full weights)",
    )
    parser.add_argument("--output", type=Path, default=Path("logprob_check/logprobs.json"))
    args = parser.parse_args()
    for name in ("mean_atol", "max_atol", "repeat_atol", "perturb_multiplier"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    args.output = args.output.resolve()
    return args


async def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    tokenizer = get_tokenizer(cfg.trainer.policy.model.path)
    report: Dict[str, Any] = {"passed": False, "args": {k: str(v) for k, v in vars(args).items()}}
    started = time.perf_counter()
    try:
        async with open_runtime(cfg, tokenizer) as (policy, client):
            await run_check(policy, client, cfg, tokenizer, args, report)
        report["passed"] = True
    finally:
        report["seconds"] = time.perf_counter() - started
        write_report(args.output, report)
        print(f"report written to {args.output} (passed={report['passed']})", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
