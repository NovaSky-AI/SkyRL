"""Each step's skycap documents, uploaded to W&B as one version of a ``skycap-records-<phase>-<run>`` artifact."""

import json
import re
import tempfile
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import orjson
import zstandard
from loguru import logger

from skyrl.train.utils.callbacks import CallbackInput, TrainingCallback, TrainingControl

ARTIFACT_TYPE = "skycap-records"
FETCH_TIMEOUT = 60.0


@dataclass
class Created:
    """One skycap trajectory the generator opened: a trial attempt."""

    server: str
    id: str
    instance_id: str
    repetition_id: int
    attempt: int


class RecordLog:
    """The trajectories created since the last upload, per training phase."""

    def __init__(self) -> None:
        self._created: Dict[str, List[Created]] = {}

    def add(self, phase: str, created: Created) -> None:
        self._created.setdefault(phase, []).append(created)

    def take(self, phase: str) -> List[Created]:
        return self._created.pop(phase, [])


def artifact_name(phase: str, run_id: str) -> str:
    return f"{ARTIFACT_TYPE}-{phase}-{re.sub(r'[^a-zA-Z0-9_.-]', '-', run_id)}"


def fetch_document(created: Created) -> Optional[dict]:
    """The stored document, from the server that wrote it; None unless it is on that server's disk."""
    url = f"{created.server}/trajectories/{created.id}"
    try:
        with urllib.request.urlopen(url, timeout=FETCH_TIMEOUT) as response:
            document = json.load(response)
    except Exception as error:  # noqa: BLE001 - counted as missing
        logger.warning(f"skycap artifact: fetching {url} failed: {type(error).__name__}: {error}")
        return None
    # A document still in memory (its write failed) has no format_version and no sidecars manifest.
    return document if "format_version" in document else None


def upload(
    wandb: Any, run_id: str, phase: str, step: int, created: List[Created], trained: Optional[set]
) -> Optional[str]:
    """Upload the documents and a ``step.json`` index, aliased ``step-N`` and ``latest``. Returns the artifact name."""
    last_attempt: Dict[tuple, int] = {}
    for entry in created:
        key = (entry.instance_id, entry.repetition_id)
        last_attempt[key] = max(last_attempt.get(key, -1), entry.attempt)
    index = []
    with tempfile.TemporaryDirectory() as tmp:
        artifact = wandb.Artifact(name=artifact_name(phase, run_id), type=ARTIFACT_TYPE)
        for entry in created:
            document = fetch_document(entry)
            key = (entry.instance_id, entry.repetition_id)
            row = asdict(entry)
            del row["server"]
            row["uploaded"] = document is not None
            # A superseded attempt never trains; without dynamic sampling every final attempt does.
            row["trained"] = entry.attempt == last_attempt[key] and (trained is None or key in trained)
            index.append(row)
            if document is None:
                continue
            path = Path(tmp) / f"{entry.id}.json.zst"
            path.write_bytes(zstandard.ZstdCompressor().compress(orjson.dumps(document)))
            artifact.add_file(str(path), name=path.name)
        uploaded = sum(row["uploaded"] for row in index)
        missing = len(index) - uploaded
        if missing:
            logger.warning(f"skycap artifact: {missing} of {len(index)} {phase} documents of step {step} are missing")
        if not uploaded:
            return None
        (Path(tmp) / "step.json").write_bytes(orjson.dumps(index))
        artifact.add_file(str(Path(tmp) / "step.json"), name="step.json")
        artifact.metadata.update(
            {
                "global_step": step,
                "training_phase": phase,
                "run_id": run_id,
                "contents": "documents",
                "num_uploaded": uploaded,
                "num_missing": missing,
                "num_trained": sum(row["trained"] for row in index),
            }
        )
        wandb.log_artifact(artifact, aliases=[f"step-{step}", "latest"])
    logger.info(f"skycap artifact {artifact.name}:step-{step}: {uploaded} documents")
    return artifact.name


class SkycapUploads(TrainingCallback):
    """Uploads the train records after each step and the eval records after each eval pass, off the step."""

    def __init__(self, records: RecordLog, phases: List[str]) -> None:
        self.records = records
        self.phases = set(phases)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="skycap-artifacts")

    def _submit(self, trainer: Any, phase: str, step: int, trained: Optional[set]) -> None:
        created = self.records.take(phase)
        tracker = trainer.tracker
        if phase not in self.phases or not created or tracker is None or tracker.backend != "wandb":
            return
        wandb = tracker.logger
        future = self._executor.submit(upload, wandb, wandb.run.id, phase, step, created, trained)
        future.add_done_callback(_log_failure)

    def on_step_end(self, trainer: Any, callback_input: CallbackInput, control: TrainingControl) -> None:
        ids = callback_input.trajectory_ids
        trained = None if ids is None else {(str(t.instance_id), t.repetition_id) for t in ids}
        self._submit(trainer, "train", callback_input.global_step, trained)

    def on_eval_end(self, trainer: Any, callback_input: CallbackInput, control: TrainingControl) -> None:
        self._submit(trainer, "eval", callback_input.global_step, None)

    def on_train_end(self, trainer: Any, callback_input: CallbackInput, control: TrainingControl) -> None:
        # The tracker finishes the W&B run right after this event.
        self._executor.shutdown(wait=True)


def _log_failure(future: Future) -> None:
    if future.exception() is not None:
        logger.opt(exception=future.exception()).error("uploading skycap records failed")
