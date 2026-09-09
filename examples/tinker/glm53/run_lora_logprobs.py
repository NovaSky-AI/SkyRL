"""Check native LoRA publication on an owned Ray cluster, without TCLI or an API server."""

import argparse
import asyncio
import json
import math
from pathlib import Path
from time import perf_counter

from examples.model_checks.megatron_lora import (
    build_batch,
    open_runtime,
    perturb_trainer,
    publish,
    score_sampler,
    score_trainer,
)
from skyrl.backends.skyrl_train.inference_servers.utils import resolve_policy_model_name
from skyrl.tinker.logprob_checks import build_probe_sequences as build_sequences
from skyrl.tinker.logprob_checks import (
    check_initial_adapter,
    check_update_stimulus,
    check_updated_adapter,
    check_withheld_publication,
)
from skyrl.train.config import SkyRLTrainConfig
from skyrl.utils.tok import get_tokenizer


async def run(args, report):
    cfg = load_config(args.backend_config, args.output_dir)
    tokenizer = get_tokenizer(cfg.trainer.policy.model.path)
    sequences = build_sequences(tokenizer)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    batch = build_batch(sequences, pad_token_id)
    report.update(
        tokens=sequences,
        scored_positions=[len(tokens) - 1 for tokens in sequences],
        mean_atol=args.mean_atol,
        lora_b_multiplier=args.lora_b_multiplier,
        model=cfg.trainer.policy.model.path,
    )
    adapter = resolve_policy_model_name(cfg)

    async with open_runtime(cfg, tokenizer) as (policy, client):
        try:
            await check_zero_initialized_policy(policy, client, cfg, batch, sequences, report, args.mean_atol)
            apply_trainer_update(policy, batch, report, args.lora_b_multiplier)
            await check_unpublished_sampler(client, sequences, adapter, report)
            check_update_stimulus(report, args.mean_atol)
            await publish(policy, client, cfg)
            await check_published_update(client, sequences, adapter, report, args.mean_atol)
        finally:
            # Preserve failed assertions even if subsequent runtime cleanup hangs.
            write_report(args.output_dir, report)


def write_report(output_dir, report):
    temporary = output_dir / "logprobs.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False))
    temporary.replace(output_dir / "logprobs.json")


async def check_zero_initialized_policy(policy, client, cfg, batch, sequences, report, atol):
    report["base"] = await score_sampler(client, sequences, client.model_name)
    await publish(policy, client, cfg)
    adapter = resolve_policy_model_name(cfg)
    report["zero"] = await score_sampler(client, sequences, adapter)
    report["trainer_zero"] = score_trainer(policy, batch)
    report["repeat"] = await score_sampler(client, sequences, adapter)
    report["trainer_repeat"] = score_trainer(policy, batch)
    check_initial_adapter(report, atol)


def apply_trainer_update(policy, batch, report, multiplier=10):
    report["perturbation"] = perturb_trainer(policy, multiplier)
    report["trainer_updated"] = score_trainer(policy, batch)


async def check_unpublished_sampler(client, sequences, adapter, report):
    report["stale"] = await score_sampler(client, sequences, adapter)
    check_withheld_publication(report)


async def check_published_update(client, sequences, adapter, report, atol):
    report["updated"] = await score_sampler(client, sequences, adapter)
    check_updated_adapter(report, atol)


def validate_config(overrides):
    if overrides["strategy"] != "megatron":
        raise ValueError("This diagnostic requires Megatron")
    if overrides["trainer.placement.colocate_all"]:
        raise ValueError("This diagnostic requires disaggregated trainer/inference GPUs")
    if overrides["trainer.policy.model.lora.rank"] <= 0:
        raise ValueError("A positive LoRA rank is required")
    if overrides["trainer.policy.megatron_config.lora_config.merge_lora"]:
        raise ValueError("Separate adapter publication is required")
    if not overrides["generator.inference_engine.run_engines_locally"]:
        raise ValueError("This command starts its own inference engines")
    for key in (
        "generator.inference_engine.external_proxy_url",
        "generator.inference_engine.external_server_urls",
        "generator.inference_engine.enable_pd",
    ):
        if overrides.get(key):
            raise ValueError(f"This owned-runtime diagnostic does not support {key}")


def validate_scoring_config(trainer):
    if trainer.algorithm.temperature != 1.0:
        raise ValueError("Prompt-logprob comparison requires trainer temperature=1")
    placement = trainer.placement
    world_size = placement.policy_num_nodes * placement.policy_num_gpus_per_node
    parallel = trainer.policy.megatron_config
    model_size = (
        parallel.tensor_model_parallel_size * parallel.pipeline_model_parallel_size * parallel.context_parallel_size
    )
    if world_size <= 0 or model_size <= 0 or world_size % model_size:
        raise ValueError("Trainer GPU count must be divisible by TP * PP * CP")
    if world_size // model_size not in (1, 2):
        raise ValueError("The two probe sequences require trainer DP=1 or DP=2")


def load_config(path, output_dir):
    overrides = json.loads(path.read_text())
    validate_config(overrides)
    overrides["trainer.strategy"] = overrides.pop("strategy")
    # Disable measurement overhead; preserve model, topology and kernel settings.
    overrides["trainer.policy.torch_profiler_config"] = {"enable": False}
    overrides["trainer.log_path"] = str(output_dir / "runtime-logs")
    cfg = SkyRLTrainConfig.from_cli_overrides(overrides)
    validate_scoring_config(cfg.trainer)
    with (output_dir / "backend-config.json").open("x") as receipt:
        json.dump(overrides, receipt, indent=2)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-config", type=Path, required=True, help="Rendered run_server.py backend config")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mean-atol", type=float, required=True, help="Reviewed mean logprob error budget")
    parser.add_argument(
        "--lora-b-multiplier", type=float, default=10, help="Predeclared test stimulus, not an adaptive acceptance knob"
    )
    args = parser.parse_args()
    if not math.isfinite(args.mean_atol) or args.mean_atol <= 0:
        parser.error("logprob budget must be positive and finite")
    if not math.isfinite(args.lora_b_multiplier) or args.lora_b_multiplier <= 0:
        parser.error("LoRA B multiplier must be positive and finite")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"passed": False}
    started = perf_counter()
    try:
        asyncio.run(run(args, report))
        report["passed"] = True
    finally:
        report["seconds"] = perf_counter() - started
        write_report(args.output_dir, report)


if __name__ == "__main__":
    main()
