"""Profile full-context GSPO through SkyRL’s Tinker API."""

import argparse
import json
import math
import os
import time
from pathlib import Path

import tinker
from tinker import types

from skyrl.tinker import sdk_checks as checks
from skyrl.tinker.profiling_receipts import (
    PhaseReceipt,
    ReceiptRecorder,
    derive_receipt_table,
    load_receipt_metadata,
    verify_checksum_manifest,
    write_checksum_manifest,
)


def run(args) -> None:
    config = checks.TrainingCheckConfig(
        args.output_dir,
        args.context,
        args.batch_size,
        args.steps,
        args.learning_rate,
        args.profile_mode,
    )
    config.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = load_receipt_metadata(args.receipt_metadata)
    if metadata.profiler_mode != args.profile_mode:
        raise ValueError("receipt metadata profiler_mode does not match --profile-mode")
    failure = None
    try:
        run_workload(args, config, metadata)
    except Exception as error:
        failure = error
        raise
    finally:
        receipt_path = args.output_dir / "receipts.jsonl"
        if receipt_path.exists():
            try:
                finalize_run_artifacts(args.output_dir)
            except Exception as finalization_error:
                if failure is None:
                    raise
                failure.add_note(f"Artifact finalization failed: {finalization_error}")


def run_workload(args, config, metadata) -> None:
    with (
        ReceiptRecorder(args.output_dir / "receipts.jsonl", metadata) as receipts,
        (args.output_dir / "phases.jsonl").open("x") as report,
    ):
        # This example targets the local, unauthenticated SkyRL API only.
        with receipts.record("service_connection", 0, None):
            service = tinker.ServiceClient(base_url=args.base_url, api_key="tml-dummy")
        with receipts.record("model_creation", 0, None):
            trainer = checks.create_training_client(service, args.model_path, report)
        failure = None
        generation = 0
        try:
            datums = checks.prepare_full_context_inputs(trainer, config, report)
            prompt = types.ModelInput.from_ints(datums[0].model_input.to_ints()[:128])
            checks.publish_and_sample(trainer, prompt, report, "initial", receipts=receipts, generation=0)
            profile_url = resolve_inference_profile_url(args, report) if args.profile_mode == "receiver" else None
            for step in range(config.steps):
                step_phase = "warmup" if step == 0 else f"step_{step}"
                generation = step + 1
                with (
                    checks.profile_training(
                        args.base_url if args.profile_mode == "trainer" else None,
                        trainer.model_id,
                        report,
                        step_phase,
                        step,
                        f"{args.output_dir.name}-{step_phase}",
                        receipts,
                        generation,
                        step,
                    ),
                    checks.measure_phase(report, step_phase),
                ):
                    with receipts.record("reference_forward", generation, step):
                        batch = checks.score_reference_batch(trainer, datums, config, report, step)
                    with receipts.record("training_forward_backward", generation, step):
                        checks.train_batch(trainer, batch, config, report, step_phase)
                    with receipts.record("optimizer", generation, step):
                        checks.update_optimizer(trainer, config.learning_rate, report, step_phase)
                    checks.publish_and_sample(
                        trainer,
                        prompt,
                        report,
                        step_phase,
                        profile_url,
                        receipts,
                        generation,
                        step,
                    )
            with receipts.record("checkpoint", generation, config.steps - 1):
                with checks.measure_phase(report, "checkpoint") as record:
                    record["path"] = trainer.save_state("full-context-final").result().path
        except Exception as error:
            failure = error
            raise
        finally:
            try:
                with receipts.record("unload", generation, config.steps - 1):
                    with checks.measure_phase(report, "unload"):
                        checks.unload_model(args.base_url, trainer.model_id)
            except Exception as cleanup_error:
                if failure is None:
                    raise
                failure.add_note(f"Model unload failed: {cleanup_error}")


def finalize_run_artifacts(output_dir: Path) -> None:
    receipt_path = output_dir / "receipts.jsonl"
    receipts = [PhaseReceipt.model_validate_json(line) for line in receipt_path.read_text().splitlines()]
    derived_path = output_dir / "derived.json"
    with derived_path.open("x") as output:
        json.dump(
            derive_receipt_table(receipts, ["receipts.jsonl"]),
            output,
            indent=2,
            sort_keys=True,
        )
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    manifest_path = output_dir / "SHA256SUMS"
    files = [path for path in output_dir.iterdir() if path.is_file() and path != manifest_path]
    write_checksum_manifest(manifest_path, files, output_dir)
    verify_checksum_manifest(manifest_path, output_dir)


def resolve_inference_profile_url(args, report):
    if args.inference_profile_url_file is None:
        return args.inference_profile_url
    with checks.measure_phase(report, "inference_profiler_endpoint"):
        return wait_for_inference_profile_url(args.inference_profile_url_file)


def wait_for_inference_profile_url(path: Path) -> str:
    """Wait at most 30 seconds for the launcher's atomic endpoint-file handoff."""
    deadline = time.monotonic() + 30
    while True:
        try:
            url = path.read_text().strip()
        except FileNotFoundError as error:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("inference profiling endpoint file was not ready within 30 seconds") from error
            time.sleep(min(0.25, remaining))
            continue
        if not url:
            raise ValueError("inference profiling endpoint file is empty")
        return url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--profile-mode",
        choices=("none", "trainer", "receiver"),
        default="none",
        help="Same benchmark operations with no capture, trainer capture, or receiver capture",
    )
    profiling = parser.add_mutually_exclusive_group()
    profiling.add_argument("--inference-profile-url", help="Owned vLLM engine URL with profiling enabled")
    profiling.add_argument(
        "--inference-profile-url-file",
        type=Path,
        help="Owned engine URL, written by the launcher before initial publication completes",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory for inputs, phase timings and metrics",
    )
    parser.add_argument("--receipt-metadata", type=Path, required=True)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Total full-context sequences per update",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=3,
        help="Total updates: one warmup, then measured updates",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    args = parser.parse_args()
    if args.profile_mode == "receiver" and not (args.inference_profile_url or args.inference_profile_url_file):
        parser.error("receiver mode requires an inference profiler URL or URL file")
    if args.profile_mode != "receiver" and (args.inference_profile_url or args.inference_profile_url_file):
        parser.error("inference profiler URLs require --profile-mode receiver")
    if args.context < 2 or args.batch_size < 2 or args.steps < 2:
        parser.error("context >= 2, batch-size >= 2 and steps >= 2 are required")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be positive and finite")
    run(args)


if __name__ == "__main__":
    main()
