"""Each training step's skycap records, indexed in W&B.

skycap stores the records: each server writes a trajectory to its
``record_dir`` when it ends and, with ``skycap.record_mirror``, copies it to
remote storage. What only the trainer knows is which trajectories made up a
step, in which phase, and which of them trained. ``SkycapRecordIndex`` logs
that once per step, as one version of the W&B artifact
``skycap-records-<phase>-<run id>``, aliased ``<phase>-step-N`` and ``latest``:

- ``step.json``: the step's run index, as ``run_index.md`` (next to this file)
  specifies it: ``format_version``, ``run``, ``phase`` and ``step``,
  and a row per trajectory the generator opened during the step (every
  attempt, retries included) with its ``instance_id``, ``repetition_id``,
  ``attempt``, ``status``, the annotations it was finished with, whether a
  later attempt ``superseded`` it, whether it ``trained``, and its ``record``
  location from ``FinishResult.record`` (``path``, ``mirror``, ``files``).
- For a mirrored record, a reference entry ``records/<name>`` per file in its
  ``record.files``, to that file beside the mirrored document, added with
  ``checksum=False``: W&B stores the URI and neither reads the store nor
  copies bytes. A local-only record is in the index only.

Nothing here is Harbor's: any skycap generator that adds its trajectories to a
``RecordLog`` gets the index.

Logging fails open. The index is built on the trainer's thread, so a bug in it
raises into the step; the W&B calls run on a background thread, and W&B being
slow, down or erroring never fails a step. A call that overruns ``timeout`` is
abandoned with a warning and not retried (W&B may still create the version,
and a retry would log it twice). Errors that may pass are retried up to
``attempts`` times, but only before W&B accepted the version. The queue holds
``queue_size`` steps and drops new ones when full, and ``on_train_end`` waits up
to ``shutdown_timeout`` before the tracker finishes the run, then drops what is
left. Every loss is logged and counted in ``stats()``.
"""

import json
import queue
import re
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from loguru import logger

from skyrl.train.utils.callbacks import CallbackInput, TrainingCallback, TrainingControl

ARTIFACT_TYPE = "skycap-records"
PHASES = ("train", "eval")

#: The run index's ``format_version`` (``run_index.md``).
INDEX_FORMAT_VERSION = 1

#: ``OSError`` subclasses no retry can fix.
_PERMANENT = (PermissionError, FileNotFoundError, IsADirectoryError, NotADirectoryError, FileExistsError)
_STOP = object()


@dataclass
class RecordEntry:
    """One skycap trajectory a generator opened: a trial attempt."""

    id: str
    instance_id: str
    repetition_id: int
    attempt: int
    #: skycap's status at finish; None when no finish was answered.
    status: Optional[str] = None
    #: What the trajectory was finished with (e.g. the reward).
    annotations: Optional[Dict[str, Any]] = None
    #: ``FinishResult.record``: ``{"host", "path", "mirror", "files"}``, or None when skycap wrote no record.
    record: Optional[Dict[str, Any]] = None


class RecordLog:
    """The trajectories opened since the last index, per training phase.

    The generator adds to it, and the callback takes a phase's entries at the end of the step.
    """

    def __init__(self) -> None:
        self._entries: Dict[str, List[RecordEntry]] = {}
        self._lock = threading.Lock()

    def add(self, phase: str, trajectory_id: Any, attempt: int, trajectory: Any) -> None:
        """Log a ``skycap.Trajectory`` opened for ``trajectory_id``, finished or not."""
        result = trajectory.result
        location = None
        if result is not None and result.record is not None:
            location = {
                "host": result.record.host,
                "path": result.record.path,
                "mirror": result.record.mirror,
                "files": list(result.record.files),
            }
        entry = RecordEntry(
            id=trajectory.id,
            instance_id=str(trajectory_id.instance_id),
            repetition_id=trajectory_id.repetition_id,
            attempt=attempt,
            status=None if result is None else result.status,
            annotations=trajectory.finishing,
            record=location,
        )
        with self._lock:
            self._entries.setdefault(phase, []).append(entry)

    def take(self, phase: str) -> List[RecordEntry]:
        with self._lock:
            return self._entries.pop(phase, [])


def artifact_name(phase: str, run_id: str) -> str:
    return f"{ARTIFACT_TYPE}-{phase}-{re.sub(r'[^a-zA-Z0-9_.-]', '-', run_id)}"


def trained_keys(callback_input: CallbackInput) -> Optional[Set[Tuple[str, int]]]:
    """The ``(instance_id, repetition_id)`` of every trajectory with a trained token in the step's batch.

    None when the trainer didn't pass the batch's ``trajectory_ids``. Rows past them are the trainer's padding.
    """
    ids = callback_input.trajectory_ids
    if ids is None:
        return None
    loss_mask = None if callback_input.batch is None else callback_input.batch.get("loss_mask")
    if loss_mask is None:
        return {(str(t.instance_id), t.repetition_id) for t in ids}
    tokens = loss_mask[: len(ids)].sum(dim=-1).tolist()
    return {(str(t.instance_id), t.repetition_id) for t, count in zip(ids, tokens) if count > 0}


def index_rows(entries: List[RecordEntry], phase: str, trained: Optional[Set[Tuple[str, int]]]) -> List[dict]:
    """The run index's rows. ``trained`` is None when unknown, and then so is each final attempt's."""
    last: Dict[Tuple[str, int], int] = {}
    for entry in entries:
        key = (entry.instance_id, entry.repetition_id)
        last[key] = max(last.get(key, -1), entry.attempt)
    rows = []
    for entry in entries:
        key = (entry.instance_id, entry.repetition_id)
        superseded = entry.attempt < last[key]
        if phase != "train" or superseded:
            was_trained: Optional[bool] = False
        else:
            was_trained = None if trained is None else key in trained
        row = asdict(entry)
        record = row.pop("record")
        rows.append({**row, "superseded": superseded, "trained": was_trained, "record": record})
    return rows


def step_index(run_id: str, phase: str, step: int, rows: List[dict]) -> Dict[str, Any]:
    """The run index of one step and phase: the object ``index/<phase>/step-<N>.json`` holds."""
    return {"format_version": INDEX_FORMAT_VERSION, "run": run_id, "phase": phase, "step": step, "rows": rows}


def record_references(rows: List[dict]) -> List[Tuple[str, str]]:
    """``(uri, name in the artifact)`` per file of every mirrored record: ``records/<file name>``.

    The files are beside the mirrored document. A record from a server that doesn't list ``files`` is
    referenced by its document alone.
    """
    references = []
    for row in rows:
        record = row["record"]
        if record is None or record.get("mirror") is None:
            continue
        mirror_dir, document = record["mirror"].rsplit("/", 1)
        for name in record.get("files") or [document]:
            references.append((f"{mirror_dir}/{name}", f"records/{name}"))
    return references


@dataclass
class IndexVersion:
    """One artifact version to log: everything W&B is handed, built before any W&B call."""

    name: str
    aliases: List[str]
    metadata: Dict[str, Any]
    index: bytes
    #: ``(uri, name in the artifact)`` per file of each mirrored record.
    references: List[Tuple[str, str]]


def build_version(run_id: str, phase: str, step: int, rows: List[dict]) -> IndexVersion:
    references = record_references(rows)
    return IndexVersion(
        name=artifact_name(phase, run_id),
        aliases=[f"{phase}-step-{step}", "latest"],
        metadata={
            "global_step": step,
            "training_phase": phase,
            "run_id": run_id,
            "num_trajectories": len(rows),
            "num_trained": sum(row["trained"] is True for row in rows),
            "num_superseded": sum(row["superseded"] for row in rows),
            "num_recorded": sum(row["record"] is not None for row in rows),
            "num_mirrored": sum(row["record"] is not None and row["record"]["mirror"] is not None for row in rows),
            "num_referenced": len(references),
        },
        index=json.dumps(step_index(run_id, phase, step, rows)).encode(),
        references=references,
    )


class _Abandoned(Exception):
    """A W&B call overran its timeout."""


class _AfterSend(Exception):
    """W&B failed after it accepted the version, so a retry could log it twice."""


class SkycapRecordIndex(TrainingCallback):
    """Logs the train records after each step, and the eval records after each eval pass, off the step.

    Args:
        records: the log the generator adds to.
        phases: the training phases to index: ``train``, ``eval``.
        queue_size: versions waiting to be logged, beyond which new ones are dropped.
        timeout: seconds one version's W&B calls may take before they are abandoned.
        attempts: tries per version for an error that may pass.
        backoff: seconds before the first retry, doubled for each one after.
        shutdown_timeout: seconds ``on_train_end`` waits for the queue.
    """

    def __init__(
        self,
        records: RecordLog,
        phases: Sequence[str],
        *,
        queue_size: int = 16,
        timeout: float = 300.0,
        attempts: int = 3,
        backoff: float = 5.0,
        shutdown_timeout: float = 120.0,
    ) -> None:
        unknown = set(phases) - set(PHASES)
        if unknown:
            raise ValueError(f"skycap.wandb.phases: unknown phases {sorted(unknown)}; choose from {list(PHASES)}")
        self.records = records
        self.phases = set(phases)
        self.timeout = timeout
        self.attempts = attempts
        self.backoff = backoff
        self.shutdown_timeout = shutdown_timeout
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=queue_size)
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._closed = False
        self._in_flight = 0
        self._counts = {"logged": 0, "failed": 0, "timed_out": 0, "dropped": 0, "retried": 0}

    # -- events -----------------------------------------------------------------
    def on_step_end(self, trainer: Any, callback_input: CallbackInput, control: TrainingControl) -> None:
        self._index(trainer, "train", callback_input.global_step, trained_keys(callback_input))

    def on_eval_end(self, trainer: Any, callback_input: CallbackInput, control: TrainingControl) -> None:
        self._index(trainer, "eval", callback_input.global_step, set())

    def on_train_end(self, trainer: Any, callback_input: CallbackInput, control: TrainingControl) -> None:
        # The tracker finishes the W&B run right after this event.
        self.close()

    def stats(self) -> Dict[str, int]:
        """Versions ``logged``, ``failed`` (``timed_out`` included), ``dropped`` and ``pending``; ``retried`` calls."""
        with self._lock:
            return {**self._counts, "pending": self._queue.qsize() + self._in_flight}

    def close(self, timeout: Optional[float] = None) -> bool:
        """Wait up to ``timeout`` (default ``shutdown_timeout``) for queued versions, then drop the rest."""
        deadline = time.monotonic() + (self.shutdown_timeout if timeout is None else timeout)
        with self._lock:
            self._closed = True
            worker = self._worker
        if worker is not None:
            try:
                self._queue.put(_STOP, timeout=max(0.0, deadline - time.monotonic()))
            except queue.Full:
                pass
            worker.join(max(0.0, deadline - time.monotonic()))
        self._stopping.set()
        left = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            left += item is not _STOP
        with self._lock:
            self._counts["dropped"] += left
            in_flight = self._in_flight
        if left or in_flight:
            logger.warning(
                f"skycap record index: shutdown deadline reached; dropped {left} queued versions, "
                f"{in_flight} still logging"
            )
        return not left and not in_flight

    # -- building, on the trainer's thread ----------------------------------------
    def _index(self, trainer: Any, phase: str, step: int, trained: Optional[Set[Tuple[str, int]]]) -> None:
        entries = self.records.take(phase)
        if phase not in self.phases or not entries:
            return
        tracker = getattr(trainer, "tracker", None)
        if tracker is None or tracker.backend != "wandb":
            return
        wandb = tracker.logger
        run = getattr(wandb, "run", None)
        if run is None:
            logger.warning(f"skycap record index: the tracker has no W&B run; not indexing {phase} step {step}")
            self._count("failed")
            return
        version = build_version(run.id, phase, step, index_rows(entries, phase, trained))
        self._submit((wandb, run, version))

    def _submit(self, item: Tuple[Any, Any, IndexVersion]) -> None:
        with self._lock:
            closed = self._closed
            if not closed and self._worker is None:
                self._worker = threading.Thread(target=self._work, name="skycap-record-index", daemon=True)
                self._worker.start()
        name = f"{item[2].name}:{item[2].aliases[0]}"
        if closed:
            logger.warning(f"skycap record index: closed; dropping {name}")
            self._count("dropped")
            return
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            logger.warning(f"skycap record index: queue full ({self._queue.maxsize}); dropping {name}")
            self._count("dropped")

    # -- logging, on the worker ---------------------------------------------------
    def _work(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            if self._stopping.is_set():
                self._count("dropped")
                continue
            with self._lock:
                self._in_flight += 1
            try:
                self._count(self._log_with_retries(*item))
            finally:
                with self._lock:
                    self._in_flight -= 1

    def _log_with_retries(self, wandb: Any, run: Any, version: IndexVersion) -> str:
        """Log one version. Returns the counter it lands in; never raises."""
        name = f"{version.name}:{version.aliases[0]}"
        for attempt in range(1, self.attempts + 1):
            try:
                self._bounded(_log_version, wandb, run, version, self.timeout)
                logger.info(f"skycap record index: logged {name} ({len(version.references)} references)")
                return "logged"
            except _Abandoned as error:
                logger.warning(f"skycap record index: abandoned {name}: {error}")
                return "timed_out"
            except Exception as error:  # noqa: BLE001 - W&B's error, logged and counted
                retry = attempt < self.attempts and _transient(error, wandb) and not self._stopping.is_set()
                logger.warning(
                    f"skycap record index: logging {name} failed ({type(error).__name__}: {error})"
                    + (", retrying" if retry else "")
                )
                if not retry:
                    return "failed"
                self._count("retried")
                self._stopping.wait(self.backoff * 2 ** (attempt - 1))
        return "failed"

    def _bounded(self, call: Callable[..., Any], *args: Any) -> None:
        """Run ``call`` on a thread of its own and give up waiting after ``timeout``."""
        outcome: Dict[str, BaseException] = {}

        def run() -> None:
            try:
                call(*args)
            except BaseException as error:  # noqa: BLE001 - handed to the waiting worker
                outcome["error"] = error

        thread = threading.Thread(target=run, name="skycap-record-index-call", daemon=True)
        thread.start()
        thread.join(self.timeout)
        if thread.is_alive():
            raise _Abandoned(f"no answer from W&B within {self.timeout}s")
        if "error" in outcome:
            raise outcome["error"]

    def _count(self, name: str) -> None:
        with self._lock:
            self._counts[name] += 1
            if name == "timed_out":
                self._counts["failed"] += 1


def _log_version(wandb: Any, run: Any, version: IndexVersion, timeout: float) -> None:
    artifact = wandb.Artifact(name=version.name, type=ARTIFACT_TYPE, metadata=version.metadata)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "step.json"
        path.write_bytes(version.index)
        artifact.add_file(str(path), name="step.json")
        for uri, name in version.references:
            artifact.add_reference(uri, name=name, checksum=False)
        logged = run.log_artifact(artifact, aliases=version.aliases)
        try:
            logged.wait(timeout=timeout)
        except Exception as error:
            raise _AfterSend(f"{type(error).__name__}: {error}") from error


def _transient(error: BaseException, wandb: Any) -> bool:
    """Whether a retry may succeed: never once W&B accepted the version."""
    if isinstance(error, _AfterSend):
        return False
    comm_error = getattr(getattr(wandb, "errors", None), "CommError", None)
    if isinstance(comm_error, type) and isinstance(error, comm_error):
        return True
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    return isinstance(error, OSError) and not isinstance(error, _PERMANENT)
