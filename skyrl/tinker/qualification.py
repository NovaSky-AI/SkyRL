"""Public transport-neutral LoRA qualification for caller-owned SkyRL clients."""

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

from tinker import types

from skyrl.tinker import sdk_checks, sdk_logprobs
from skyrl.tinker.logprob_checks import check_initial_adapter, compare_logprobs
from skyrl.tinker.profiling_receipts import (
    PhaseReceipt,
    ProfilerMode,
    ReceiptMetadata,
    ReceiptRecorder,
    derive_receipt_table,
    sha256_file,
    verify_checksum_manifest,
    write_checksum_manifest,
)


@dataclass(frozen=True)
class LoraQualificationConfig:
    """Inputs for one qualification of already-claimed service and trainer clients."""

    output_dir: Path
    context: int
    batch_size: int
    steps: int
    lr: float
    mean_atol: float
    max_atol: float
    receipt_metadata: ReceiptMetadata
    profiler_mode: ProfilerMode = "none"
    trainer_profiler_url: str | None = None
    receiver_profiler_url: str | None = None

    def __post_init__(self) -> None:
        if self.context < 2 or self.batch_size < 2 or self.steps < 2:
            raise ValueError("context, batch_size, and steps must each be at least two")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be positive and finite")
        if not math.isfinite(self.mean_atol) or self.mean_atol <= 0:
            raise ValueError("mean_atol must be positive and finite")
        if not math.isfinite(self.max_atol) or self.max_atol <= 0:
            raise ValueError("max_atol must be positive and finite")
        if self.receipt_metadata.profiler_mode != self.profiler_mode:
            raise ValueError("receipt metadata profiler mode does not match the qualification mode")
        urls = {
            "trainer": self.trainer_profiler_url,
            "receiver": self.receiver_profiler_url,
        }
        required = urls.get(self.profiler_mode)
        forbidden = [url for mode, url in urls.items() if mode != self.profiler_mode and url]
        if self.profiler_mode != "none" and not required:
            raise ValueError(f"{self.profiler_mode} mode requires its profiler URL")
        if self.profiler_mode == "none" and any(urls.values()):
            raise ValueError("profiler URLs require a profiling mode")
        if forbidden:
            raise ValueError("only the selected profiler owner may receive a URL")


@dataclass(frozen=True)
class LoraQualificationSummary:
    """Small successful-run result; detailed evidence stays in the output manifest."""

    output_dir: Path
    steps: int
    discriminating_steps: tuple[int, ...]
    checkpoint_path: str
    manifest_path: Path


def run_lora_qualification(
    service: object,
    trainer: object,
    tokenizer: object,
    config: LoraQualificationConfig,
) -> LoraQualificationSummary:
    """Run #2182 checks and atomically publish checksummed evidence.

    The caller owns service allocation and cleanup. This function owns all model work from
    fixed inputs through the final checkpoint and never imports a CLI or Trajectory layer.
    """
    config.output_dir.mkdir(parents=True, exist_ok=False)
    state = {
        "passed": False,
        "completed_steps": [],
        "step_reports": [],
        "checkpoint_path": None,
    }
    failure = None
    try:
        _run_qualification(service, trainer, tokenizer, config, state)
        state["passed"] = True
    except Exception as error:
        failure = error
        state.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        try:
            _finalize_artifacts(config, state)
        except Exception as finalization_error:
            if failure is None:
                raise
            failure.add_note(f"Qualification artifact finalization failed: {finalization_error}")

    discriminating_steps = tuple(
        step for step, scores in enumerate(state["step_reports"]) if _is_discriminating(scores, config)
    )
    return LoraQualificationSummary(
        output_dir=config.output_dir,
        steps=config.steps,
        discriminating_steps=discriminating_steps,
        checkpoint_path=state["checkpoint_path"],
        manifest_path=config.output_dir / "manifest.json",
    )


def _run_qualification(service, trainer, tokenizer, config, state) -> None:
    training = sdk_checks.TrainingCheckConfig(
        output_dir=config.output_dir,
        context=config.context,
        batch_size=config.batch_size,
        steps=config.steps,
        learning_rate=config.lr,
        profile_mode=config.profiler_mode,
    )
    with (
        ReceiptRecorder(config.output_dir / "receipts.jsonl", config.receipt_metadata) as receipts,
        (config.output_dir / "phases.jsonl").open("x") as phases,
    ):
        datums = sdk_checks.prepare_full_context_inputs(trainer, training, phases, tokenizer)
        probes = sdk_logprobs.prepare_probes(trainer, tokenizer)
        _write_token_evidence(config.output_dir / "token-evidence.json", datums, probes)
        with sdk_checks.measure_phase(phases, "qualification/base_logprobs"):
            info = trainer.get_info()
            base_sampler = service.create_sampling_client(base_model=info.model_data.model_name)
            base_scores = sdk_logprobs.score_sampler(base_sampler, probes)
        prompt = types.ModelInput.from_ints(datums[0].model_input.to_ints()[:128])
        sampler = sdk_checks.publish_and_sample(
            trainer,
            prompt,
            phases,
            "initial",
            receipts=receipts,
            generation=0,
        )
        for step in range(config.steps):
            phase = "warmup" if step == 0 else f"step_{step}"
            generation = step + 1
            scores = {"base": base_scores} if step == 0 else {}
            step_failure = None
            try:
                with (
                    sdk_checks.profile_training(
                        config.trainer_profiler_url,
                        trainer.model_id,
                        phases,
                        phase,
                        step,
                        f"{config.output_dir.name}-{phase}",
                        receipts,
                        generation,
                        step,
                    ),
                    sdk_checks.measure_phase(phases, phase),
                ):
                    with sdk_checks.measure_phase(phases, f"{phase}/logprobs_before"):
                        sdk_logprobs.score_before_update(
                            trainer,
                            sampler,
                            probes,
                            scores,
                            config.mean_atol,
                            config.max_atol,
                        )
                        if step == 0:
                            check_initial_adapter(scores, config.mean_atol, config.max_atol)
                    with receipts.record("reference_forward", generation, step):
                        batch = sdk_checks.score_reference_batch(trainer, datums, training, phases, step)
                    with receipts.record("training_forward_backward", generation, step):
                        sdk_checks.train_batch(trainer, batch, training, phases, phase)
                    with receipts.record("optimizer", generation, step):
                        sdk_checks.update_optimizer(trainer, config.lr, phases, phase)
                    with sdk_checks.measure_phase(phases, f"{phase}/logprobs_stale"):
                        sdk_logprobs.check_withheld_publication(trainer, sampler, probes, scores)
                    sampler = sdk_checks.publish_and_sample(
                        trainer,
                        prompt,
                        phases,
                        phase,
                        config.receiver_profiler_url,
                        receipts,
                        generation,
                        step,
                    )
                    with sdk_checks.measure_phase(phases, f"{phase}/logprobs_updated"):
                        sdk_logprobs.check_published_update(
                            sampler,
                            probes,
                            scores,
                            config.mean_atol,
                            config.max_atol,
                        )
                state["step_reports"].append(scores)
                state["completed_steps"].append(step)
            except Exception as error:
                step_failure = error
                raise
            finally:
                try:
                    _write_json_atomic(config.output_dir / f"step_{step}_logprobs.json", scores)
                except Exception as receipt_error:
                    if step_failure is None:
                        raise
                    step_failure.add_note(f"Per-token receipt failed: {receipt_error}")
        with receipts.record("checkpoint", config.steps, config.steps - 1):
            with sdk_checks.measure_phase(phases, "checkpoint"):
                checkpoint_path = trainer.save_state("full-context-final").result().path
        state["checkpoint_path"] = checkpoint_path
        _write_json_atomic(config.output_dir / "checkpoint.json", {"path": checkpoint_path})


def _finalize_artifacts(config: LoraQualificationConfig, state: dict) -> None:
    step_reports = state["step_reports"]
    discriminating_steps = [step for step, scores in enumerate(step_reports) if _is_discriminating(scores, config)]
    qualification = {
        "passed": state["passed"],
        "tolerances": {
            "mean_atol": config.mean_atol,
            "max_atol": config.max_atol,
        },
        "training_mechanics": "passed" if state["passed"] else "failed",
        "strong_publication_discrimination": ("passed" if discriminating_steps else "inconclusive"),
        "discriminating_steps": discriminating_steps,
        "completed_steps": state["completed_steps"],
        "checkpoint_path": state["checkpoint_path"],
    }
    if step_reports:
        qualification["initial_sampler_vs_final_trainer_diagnostic"] = compare_logprobs(
            step_reports[-1]["trainer_updated"], step_reports[0]["zero"]
        )
    if not state["passed"]:
        qualification.update(
            error_type=state.get("error_type", "UnknownError"),
            error=state.get("error", "qualification did not complete"),
        )
    _write_json_atomic(config.output_dir / "qualification.json", qualification)

    receipt_path = config.output_dir / "receipts.jsonl"
    receipts = [PhaseReceipt.model_validate_json(line) for line in receipt_path.read_text().splitlines()]
    _write_json_atomic(
        config.output_dir / "derived.json",
        derive_receipt_table(receipts, ["receipts.jsonl"]),
    )
    manifest_path = config.output_dir / "SHA256SUMS"
    files = [
        path
        for path in config.output_dir.iterdir()
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.tmp", "manifest.json", "manifest.json.tmp"}
    ]
    temporary_manifest = manifest_path.with_name("SHA256SUMS.tmp")
    write_checksum_manifest(temporary_manifest, files, config.output_dir)
    temporary_manifest.replace(manifest_path)
    _fsync_directory(config.output_dir)
    verify_checksum_manifest(manifest_path, config.output_dir)
    artifacts = [
        {"path": relative, "sha256": digest}
        for digest, relative in (line.split("  ", 1) for line in manifest_path.read_text().splitlines())
    ]
    _write_json_atomic(
        config.output_dir / "manifest.json",
        {
            "schema_version": 1,
            "kind": "skyrl.lora_qualification",
            "status": "passed" if state["passed"] else "failed",
            "receipt_schema_version": "1.0.0",
            "tolerances": {
                "mean_atol": config.mean_atol,
                "max_atol": config.max_atol,
            },
            "sha256sums_sha256": sha256_file(manifest_path),
            "token_evidence": {
                "path": "token-evidence.json",
                "sha256": sha256_file(config.output_dir / "token-evidence.json"),
            },
            "artifacts": artifacts,
        },
    )


def _is_discriminating(scores: dict, config: LoraQualificationConfig) -> bool:
    comparison = scores["stale_parity"]
    return comparison["mean_abs"] >= config.mean_atol or comparison["max_abs"] >= config.max_atol


def _write_token_evidence(path: Path, datums: list[types.Datum], probes: list[types.Datum]) -> None:
    def serialize(datum: types.Datum) -> dict:
        weights = datum.loss_fn_inputs["weights"].data
        return {
            "input_token_ids": datum.model_input.to_ints(),
            "target_token_ids": datum.loss_fn_inputs["target_tokens"].data,
            "scoring_weights": weights,
            "scoring_mask": [weight != 0 for weight in weights],
        }

    _write_json_atomic(
        path,
        {
            "schema_version": 1,
            "training_datums": [serialize(datum) for datum in datums],
            "logprob_probes": [serialize(datum) for datum in probes],
        },
    )


def _write_json_atomic(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as output:
        json.dump(value, output, indent=2, allow_nan=False, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
