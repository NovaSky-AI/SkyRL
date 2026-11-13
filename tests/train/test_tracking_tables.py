"""Exercise sample tables through W&B's real offline serializer."""

import json
from pathlib import Path

import pytest
import wandb

from skyrl.train.utils.tracking import Tracking


@pytest.fixture
def offline_tracker(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    for name in ("WANDB_RUN_ID", "WANDB_RESUME", "WANDB_SWEEP_ID"):
        monkeypatch.delenv(name, raising=False)
    tracker = Tracking("skyrl-table-tests", "samples", backend="wandb", config={})
    try:
        yield tracker
    finally:
        tracker.finish()


@pytest.fixture
def table_artifacts(monkeypatch):
    """Capture the table JSON handed to the SDK's real artifact logger."""
    artifacts = []
    log_artifact = wandb.Run.log_artifact

    def capture(run, artifact, *args, **kwargs):
        for name, entry in artifact.manifest.entries.items():
            if name.endswith(".table.json"):
                artifacts.append((artifact.name, json.loads(Path(entry.local_path).read_text())))
        return log_artifact(run, artifact, *args, **kwargs)

    monkeypatch.setattr(wandb.Run, "log_artifact", capture)
    return artifacts


def test_tables_serialize_new_rows_and_share_the_scalar_step(offline_tracker, table_artifacts):
    tracker = offline_tracker
    run = wandb.run
    for step in range(3):
        tracker.log_samples_to_table("train/samples", ["step", "text"], [(step, "train")], step)
        assert run.step == step
        tracker.log_samples_to_table("eval/samples", ["step", "reward"], [(step, 0.5)], step)
        assert run.step == step
        tracker.log({"loss": float(step)}, step=step, commit=True)
        assert run.step == step + 1

    assert len(table_artifacts) == 6
    assert all(payload["log_mode"] == "INCREMENTAL" for _, payload in table_artifacts)
    assert [payload["data"] for _, payload in table_artifacts] == [
        [[0, "train"]],
        [[0, 0.5]],
        [[1, "train"]],
        [[1, 0.5]],
        [[2, "train"]],
        [[2, 0.5]],
    ]
    assert len({name for name, _ in table_artifacts}) == 2


def test_table_columns_cannot_change_between_calls(offline_tracker):
    tracker = offline_tracker
    columns = ["step", "text"]
    tracker.log_samples_to_table("samples", columns, [(0, "first")], step=0)
    columns[1] = "renamed"
    with pytest.raises(ValueError, match="columns"):
        tracker.log_samples_to_table("samples", columns, [(1, "second")], step=1)
    tracker.log_samples_to_table("samples", ["step", "text"], [(1, "second")], step=1)
    assert tracker._sample_tables["samples"].data == [[0, "first"], [1, "second"]]


def test_table_keeps_native_row_and_type_validation(offline_tracker):
    tracker = offline_tracker
    tracker.log_samples_to_table("samples", ["step", "text"], [(0, "first")], step=0)
    with pytest.raises(ValueError, match="expects 2 columns"):
        tracker.log_samples_to_table("samples", ["step", "text"], [(1,)], step=1)
    with pytest.raises(TypeError, match="incompatible types"):
        tracker.log_samples_to_table("samples", ["step", "text"], [("wrong", "type")], step=1)
    assert tracker._sample_tables["samples"].data == [[0, "first"]]


def test_new_table_uses_native_resumed_increment_state(offline_tracker, table_artifacts, monkeypatch):
    tracker = offline_tracker
    run = wandb.run
    # Offline init does not resume. Supply the metadata an online resume returns,
    # then exercise the SDK's normal logging and restoration implementation.
    monkeypatch.setattr(run._settings, "resumed", True)
    run.summary.update(
        {
            "samples": {
                "_type": "incremental-table-file",
                "increment_num": 7,
                "artifact_path": "wandb-client-artifact://previous/7.samples.table.json",
                "previous_increments_paths": ["wandb-client-artifact://older/6.samples.table.json"],
            }
        }
    )
    tracker.log_samples_to_table("samples", ["step"], [(8,)], step=8)
    payload = table_artifacts[-1][1]
    assert payload["data"] == [[8]]
    assert payload["increment_num"] == 8
    assert payload["previous_increments_paths"] == [
        "wandb-client-artifact://older/6.samples.table.json",
        "wandb-client-artifact://previous/7.samples.table.json",
    ]


def test_non_wandb_backends_do_not_create_tables():
    tracker = Tracking("project", "experiment", backend="console")
    tracker.log_samples_to_table("samples", ["step"], [(0,)], step=0)
    assert not hasattr(tracker, "_sample_tables")
