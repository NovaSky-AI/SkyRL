"""Profile full-context GSPO through SkyRL’s Tinker API."""

import argparse
import math
from pathlib import Path

import tinker
from tinker import types

from skyrl.tinker import sdk_checks as checks


def run(args) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=False)
    # This example targets the local, unauthenticated SkyRL API only.
    service = tinker.ServiceClient(base_url=args.base_url, api_key="tml-dummy")
    with (args.output_dir / "phases.jsonl").open("w") as report:
        trainer = checks.create_training_client(service, args.model_path, report)
        try:
            datums = checks.prepare_full_context_inputs(trainer, args, report)
            prompt = types.ModelInput.from_ints(datums[0].model_input.to_ints()[:128])
            checks.publish_and_sample(trainer, prompt, report, "initial")
            inference_profile_url = args.inference_profile_url
            if args.inference_profile_url_file is not None:
                inference_profile_url = args.inference_profile_url_file.read_text().strip()
                if not inference_profile_url:
                    raise ValueError("inference profiling endpoint file is empty")
            for step in range(args.steps):
                step_phase = "warmup" if step == 0 else f"step_{step}"
                with checks.measure_phase(report, step_phase):
                    batch = checks.score_reference_batch(trainer, datums, args, report, step)
                    checks.train_batch(trainer, batch, args, report, step_phase)
                    checks.update_optimizer(trainer, args.learning_rate, report, step_phase)
                    checks.publish_and_sample(trainer, prompt, report, step_phase, inference_profile_url)
            with checks.measure_phase(report, "checkpoint") as record:
                record["path"] = trainer.save_state("full-context-final").result().path
        finally:
            with checks.measure_phase(report, "unload"):
                checks.unload_model(args.base_url, trainer.model_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model-path", required=True)
    profiling = parser.add_mutually_exclusive_group()
    profiling.add_argument("--inference-profile-url", help="Owned vLLM engine URL with profiling enabled")
    profiling.add_argument(
        "--inference-profile-url-file",
        type=Path,
        help="Owned engine URL, written by the launcher before initial publication completes",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New directory for inputs, phase timings and metrics"
    )
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=2, help="Total full-context sequences per update")
    parser.add_argument("--steps", type=int, default=3, help="Total updates: one warmup, then measured updates")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    args = parser.parse_args()
    if args.context < 2 or args.batch_size < 2 or args.steps < 2:
        parser.error("context >= 2, batch-size >= 2 and steps >= 2 are required")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be positive and finite")
    run(args)


if __name__ == "__main__":
    main()
