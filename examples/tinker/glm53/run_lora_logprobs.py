"""Check native LoRA publication on an owned Ray cluster, without TCLI or an API server."""

import argparse
import asyncio
import json
import math
from itertools import cycle, islice
from pathlib import Path
from time import perf_counter

from examples.model_checks.lora_logprobs import (
    check_initial_adapter,
    check_updated_adapter,
    check_withheld_publication,
)


async def run(args, report):
    cfg = load_config(args.backend_config, args.output_dir)
    from examples.model_checks.megatron_lora import (  # noqa: PLC0415
        build_batch,
        open_runtime,
        perturb_trainer,
        publish,
        score_sampler,
        score_trainer,
    )
    from skyrl.backends.skyrl_train.inference_servers.utils import resolve_policy_model_name  # noqa: PLC0415
    from skyrl.utils.tok import get_tokenizer  # noqa: PLC0415

    tokenizer = get_tokenizer(cfg.trainer.policy.model.path)
    sequences = build_sequences(tokenizer)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    batch = build_batch(sequences, pad_token_id)
    report.update(
        tokens=sequences,
        scored_positions=[len(tokens) - 1 for tokens in sequences],
        mean_atol=args.mean_atol,
        model=cfg.trainer.policy.model.path,
    )
    adapter = resolve_policy_model_name(cfg)

    async with open_runtime(cfg, tokenizer) as (policy, client):
        # Zero-init LoRA must preserve base scores and agree across both engines.
        report["base"] = await score_sampler(client, sequences, client.model_name)
        await publish(policy, client, cfg)
        report["zero"] = await score_sampler(client, sequences, adapter)
        report["trainer_zero"] = score_trainer(policy, batch)
        report["repeat"] = await score_sampler(client, sequences, adapter)
        check_initial_adapter(report, args.mean_atol)

        # The trainer changes; inference must not change until publication.
        report["perturbation"] = perturb_trainer(policy)
        report["trainer_updated"] = score_trainer(policy, batch)
        report["stale"] = await score_sampler(client, sequences, adapter)
        check_withheld_publication(report)

        await publish(policy, client, cfg)
        report["updated"] = await score_sampler(client, sequences, adapter)
        check_updated_adapter(report, args.mean_atol)


def build_sequences(tokenizer):
    return [
        list(islice(cycle(tokenizer.encode(text, add_special_tokens=False)), length))
        for text, length in [("A river flows beneath a bridge. ", 65), ("Calculate seven times eight. ", 129)]
    ]


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


def load_config(path, output_dir):
    from skyrl.train.config import SkyRLTrainConfig  # noqa: PLC0415

    overrides = json.loads(path.read_text())
    validate_config(overrides)
    overrides["trainer.strategy"] = overrides.pop("strategy")
    # Disable measurement overhead; preserve model, topology and kernel settings.
    overrides["trainer.policy.torch_profiler_config"] = {"enable": False}
    overrides["trainer.log_path"] = str(output_dir / "runtime-logs")
    cfg = SkyRLTrainConfig.from_cli_overrides(overrides)
    with (output_dir / "backend-config.json").open("x") as receipt:
        json.dump(overrides, receipt, indent=2)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-config", type=Path, required=True, help="Rendered run_server.py backend config")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mean-atol", type=float, required=True, help="Reviewed mean logprob error budget")
    args = parser.parse_args()
    if not math.isfinite(args.mean_atol) or args.mean_atol <= 0:
        parser.error("mean-atol must be positive and finite")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"passed": False}
    started = perf_counter()
    try:
        asyncio.run(run(args, report))
        report["passed"] = True
    finally:
        report["seconds"] = perf_counter() - started
        with (args.output_dir / "logprobs.json").open("x") as receipt:
            json.dump(report, receipt, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
