"""Focused checks for the public transport-neutral LoRA qualification boundary."""

import json
from types import SimpleNamespace

import pytest
from tinker import types

from skyrl.tinker.profiling_receipts import (
    PhaseReceipt,
    ReceiptMetadata,
    verify_checksum_manifest,
)
from skyrl.tinker.qualification import (
    LoraQualificationConfig,
    LoraQualificationSummary,
    run_lora_qualification,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [11, 12, 13] if "river" in text else [21, 22, 23]


class FakeSampler:
    def __init__(self, version, events):
        self.version = version
        self.events = events

    def compute_logprobs(self, model_input):
        tokens = model_input.to_ints()

        def result():
            self.events.append(f"sampler_logprobs:{self.version}")
            return [None] + [-2.0 + 0.2 * self.version] * (len(tokens) - 1)

        return SimpleNamespace(result=result)

    def sample(self, *args, **kwargs):
        def result():
            self.events.append(f"sample:{self.version}")
            return SimpleNamespace(sequences=[SimpleNamespace(tokens=[self.version, 7])])

        return SimpleNamespace(result=result)


class FakeTrainer:
    model_id = "model-test"

    def __init__(self, events, fail_optimizer=None, fail_publish=None):
        self.events = events
        self.fail_optimizer = fail_optimizer
        self.fail_publish = fail_publish
        self.optimizer_calls = 0
        self.version = 0
        self.info = types.GetInfoResponse.model_validate(
            {
                "model_id": self.model_id,
                "model_data": {"model_name": "test-model"},
                "is_lora": True,
                "lora_rank": 32,
            }
        )

    def get_info(self):
        return self.info

    def forward(self, data, loss):
        assert loss == "cross_entropy"
        probe = len(data[0].model_input.to_ints()) != 7
        label = "trainer_logprobs" if probe else "reference"

        def result():
            self.events.append(f"{label}:{self.version}")
            return SimpleNamespace(
                loss_fn_outputs=[
                    {
                        "logprobs": types.TensorData(
                            data=[-2.0 + 0.2 * self.version] * len(datum.model_input.to_ints()),
                            dtype="float32",
                            shape=[len(datum.model_input.to_ints())],
                        )
                    }
                    for datum in data
                ],
                metrics={"loss": 0.0},
            )

        return SimpleNamespace(result=result)

    def forward_backward(self, data, loss):
        assert loss == "gspo"

        def result():
            self.events.append(f"forward_backward:{self.version}")
            return SimpleNamespace(
                loss_fn_outputs=[
                    {
                        "logprobs": types.TensorData(
                            data=[-2.0] * len(datum.model_input.to_ints()),
                            dtype="float32",
                            shape=[len(datum.model_input.to_ints())],
                        )
                    }
                    for datum in data
                ],
                metrics={"loss": 0.0},
            )

        return SimpleNamespace(result=result)

    def optim_step(self, params):
        assert params.learning_rate == 1e-5
        call = self.optimizer_calls
        self.optimizer_calls += 1

        def result():
            self.events.append(f"optimizer:{call}")
            if self.fail_optimizer == call:
                raise RuntimeError("optimizer failed")
            self.version += 1
            return SimpleNamespace(metrics={"skyrl.ai/grad_norm": 1.0 + call})

        return SimpleNamespace(result=result)

    def save_weights_and_get_sampling_client(self):
        self.events.append(f"publish:{self.version}")
        if self.fail_publish == self.version:
            raise RuntimeError("publication failed")
        return FakeSampler(self.version, self.events)

    def save_state(self, name):
        assert name == "full-context-final"

        def result():
            self.events.append("checkpoint")
            return SimpleNamespace(path="tinker://test/full-context-final")

        return SimpleNamespace(result=result)


class FakeService:
    def __init__(self, events):
        self.events = events

    def create_sampling_client(self, base_model):
        assert base_model == "test-model"
        self.events.append("base_sampler")
        return FakeSampler(0, self.events)


def make_config(tmp_path, metadata, **changes):
    values = {
        "output_dir": tmp_path / "qualification",
        "context": 7,
        "batch_size": 2,
        "steps": 2,
        "lr": 1e-5,
        "mean_atol": 0.05,
        "max_atol": 0.05,
        "receipt_metadata": metadata,
    }
    values.update(changes)
    return LoraQualificationConfig(**values)


@pytest.fixture
def metadata():
    return ReceiptMetadata(
        run_id="qualification-test",
        model={
            "name": "test-model",
            "revision": "model-revision",
            "config_sha256": "a" * 64,
        },
        provenance={
            "repository_url": "https://github.com/NovaSky-AI/SkyRL",
            "commit": "b" * 40,
            "dirty": False,
            "image_uri": "registry.example.com/skyrl:test",
            "image_digest": f"sha256:{'c' * 64}",
        },
        arm="baseline_a",
        transport={"implementation": "safetensors", "revision": "baseline-v1"},
        profiler_mode="none",
    )


def test_public_qualification_publishes_before_base_sampling_and_owns_exact_artifacts(tmp_path, metadata):
    events = []
    config = make_config(tmp_path, metadata)
    summary = run_lora_qualification(FakeService(events), FakeTrainer(events), FakeTokenizer(), config)
    assert isinstance(summary, LoraQualificationSummary)
    assert summary.discriminating_steps == (0, 1)
    assert summary.checkpoint_path == "tinker://test/full-context-final"
    assert events == [
        "publish:0",
        "sample:0",
        "base_sampler",
        "sampler_logprobs:0",
        "sampler_logprobs:0",
        "trainer_logprobs:0",
        "sampler_logprobs:0",
        "sampler_logprobs:0",
        "trainer_logprobs:0",
        "sampler_logprobs:0",
        "sampler_logprobs:0",
        "reference:0",
        "forward_backward:0",
        "optimizer:0",
        "trainer_logprobs:1",
        "sampler_logprobs:0",
        "sampler_logprobs:0",
        "publish:1",
        "sample:1",
        "sampler_logprobs:1",
        "sampler_logprobs:1",
        "trainer_logprobs:1",
        "sampler_logprobs:1",
        "sampler_logprobs:1",
        "trainer_logprobs:1",
        "sampler_logprobs:1",
        "sampler_logprobs:1",
        "reference:1",
        "forward_backward:1",
        "optimizer:1",
        "trainer_logprobs:2",
        "sampler_logprobs:1",
        "sampler_logprobs:1",
        "publish:2",
        "sample:2",
        "sampler_logprobs:2",
        "sampler_logprobs:2",
        "checkpoint",
    ]
    assert {path.name for path in config.output_dir.iterdir()} == {
        "SHA256SUMS",
        "checkpoint.json",
        "datums.json",
        "derived.json",
        "manifest.json",
        "phases.jsonl",
        "qualification.json",
        "receipts.jsonl",
        "run.json",
        "step_0_batch.json",
        "step_0_logprobs.json",
        "step_1_batch.json",
        "step_1_logprobs.json",
        "token-evidence.json",
    }
    verify_checksum_manifest(config.output_dir / "SHA256SUMS", config.output_dir)
    manifest = json.loads((config.output_dir / "manifest.json").read_text())
    assert manifest["status"] == "passed"
    assert manifest["tolerances"] == {"mean_atol": 0.05, "max_atol": 0.05}
    token_evidence = json.loads((config.output_dir / "token-evidence.json").read_text())
    assert token_evidence["training_datums"][0] == {
        "input_token_ids": [11, 12, 13, 11, 12, 13, 11],
        "target_token_ids": [12, 13, 11, 12, 13, 11, 12],
        "scoring_weights": [1.0] * 7,
        "scoring_mask": [True] * 7,
    }
    assert manifest["token_evidence"] == next(
        artifact for artifact in manifest["artifacts"] if artifact["path"] == "token-evidence.json"
    )
    assert not list(config.output_dir.glob("*.tmp"))
    receipts = [
        PhaseReceipt.model_validate_json(line)
        for line in (config.output_dir / "receipts.jsonl").read_text().splitlines()
    ]
    assert [receipt.operation for receipt in receipts] == [
        "adapter_publication",
        "sample",
        "reference_forward",
        "training_forward_backward",
        "optimizer",
        "adapter_publication",
        "sample",
        "reference_forward",
        "training_forward_backward",
        "optimizer",
        "adapter_publication",
        "sample",
        "checkpoint",
    ]


def test_failed_publication_preserves_previous_generation_without_activation_row(tmp_path, metadata):
    events = []
    config = make_config(tmp_path, metadata)
    with pytest.raises(RuntimeError, match="publication failed"):
        run_lora_qualification(FakeService(events), FakeTrainer(events, fail_publish=1), FakeTokenizer(), config)
    rows = [
        PhaseReceipt.model_validate_json(line)
        for line in (config.output_dir / "receipts.jsonl").read_text().splitlines()
    ]
    publication = rows[-1]
    assert publication.operation == "adapter_publication"
    assert publication.outcome.success is False
    assert publication.outcome.active_generation_after == 0
    assert not any(row.operation == "adapter_activation" for row in rows)


def test_failed_qualification_preserves_primary_error_and_receipts(tmp_path, metadata):
    events = []
    config = make_config(tmp_path, metadata)
    with pytest.raises(RuntimeError, match="optimizer failed"):
        run_lora_qualification(
            FakeService(events),
            FakeTrainer(events, fail_optimizer=1),
            FakeTokenizer(),
            config,
        )
    assert (config.output_dir / "step_1_logprobs.json").is_file()
    assert json.loads((config.output_dir / "qualification.json").read_text()) == {
        "checkpoint_path": None,
        "completed_steps": [0],
        "discriminating_steps": [0],
        "error": "optimizer failed",
        "error_type": "RuntimeError",
        "initial_sampler_vs_final_trainer_diagnostic": {
            "max_abs": pytest.approx(0.2),
            "mean_abs": pytest.approx(0.2),
            "p99_abs": pytest.approx(0.2),
            "tokens": 192,
        },
        "passed": False,
        "strong_publication_discrimination": "passed",
        "tolerances": {"max_atol": 0.05, "mean_atol": 0.05},
        "training_mechanics": "failed",
    }
    receipts = [
        PhaseReceipt.model_validate_json(line)
        for line in (config.output_dir / "receipts.jsonl").read_text().splitlines()
    ]
    assert receipts[-1].operation == "optimizer"
    assert receipts[-1].outcome.success is False
    assert receipts[-1].outcome.error == "optimizer failed"
    assert json.loads((config.output_dir / "manifest.json").read_text())["status"] == "failed"
    verify_checksum_manifest(config.output_dir / "SHA256SUMS", config.output_dir)
    assert not list(config.output_dir.glob("*.tmp"))


def test_per_token_receipt_failure_does_not_mask_operation_failure(tmp_path, metadata, monkeypatch):
    from skyrl.tinker import qualification

    config = make_config(tmp_path, metadata)
    write_json = qualification._write_json_atomic

    def fail_one_receipt(path, value):
        if path.name == "step_1_logprobs.json":
            raise OSError("receipt disk failed")
        write_json(path, value)

    monkeypatch.setattr(qualification, "_write_json_atomic", fail_one_receipt)
    with pytest.raises(RuntimeError, match="optimizer failed") as caught:
        run_lora_qualification(
            FakeService([]),
            FakeTrainer([], fail_optimizer=1),
            FakeTokenizer(),
            config,
        )
    assert caught.value.__notes__ == ["Per-token receipt failed: receipt disk failed"]
    assert json.loads((config.output_dir / "manifest.json").read_text())["status"] == "failed"


def test_localized_logprob_corruption_fails_initial_max_gate_below_mean_gate():
    from skyrl.tinker.logprob_checks import check_initial_adapter

    reference = [0.0] * 200
    corrupted = reference.copy()
    corrupted[37] = 0.1
    report = {
        "base": reference,
        "trainer_zero": corrupted,
        "zero": reference,
        "trainer_repeat": reference,
        "repeat": reference,
    }
    with pytest.raises(AssertionError, match="parity tolerance"):
        check_initial_adapter(report, mean_atol=0.01, max_atol=0.05)
    assert report["zero_parity"]["mean_abs"] < 0.01
    assert report["zero_parity"]["max_abs"] >= 0.05


def test_localized_logprob_corruption_fails_withheld_max_gate_below_mean_gate():
    from skyrl.tinker.logprob_checks import check_withheld_publication

    reference = [0.0] * 200
    corrupted = reference.copy()
    corrupted[37] = 0.1
    report = {
        "repeat": reference,
        "stale": corrupted,
        "repeat_noise": {"mean_abs": 0.0, "max_abs": 0.0},
    }
    with pytest.raises(AssertionError, match="unpublished trainer update"):
        check_withheld_publication(report)
    assert report["withheld_publication"]["mean_abs"] < 0.01
    assert report["withheld_publication"]["max_abs"] >= 0.05


def test_localized_logprob_corruption_fails_updated_max_gate_below_mean_gate():
    from skyrl.tinker.logprob_checks import check_updated_adapter

    zero = [0.0] * 200
    updated = [0.2] * 200
    corrupted = updated.copy()
    corrupted[37] += 0.1
    report = {
        "trainer_zero": zero,
        "zero": zero,
        "trainer_updated": updated,
        "updated": corrupted,
        "stale": zero,
        "repeat_noise": {"mean_abs": 0.0},
        "trainer_repeat_noise": {"mean_abs": 0.0},
    }
    with pytest.raises(AssertionError, match="published adapter"):
        check_updated_adapter(report, mean_atol=0.01, max_atol=0.05)
    assert report["updated_parity"]["mean_abs"] < 0.01
    assert report["updated_parity"]["max_abs"] >= 0.05


@pytest.mark.parametrize("max_atol", [0.0, float("nan"), float("inf")])
def test_qualification_rejects_invalid_max_tolerance(tmp_path, metadata, max_atol):
    with pytest.raises(ValueError, match="max_atol"):
        make_config(tmp_path, metadata, max_atol=max_atol)
