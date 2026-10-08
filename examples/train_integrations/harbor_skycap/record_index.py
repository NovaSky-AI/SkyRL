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

import argparse
import hashlib
import json
import os
import queue
import sys
import re
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
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
#: fsspec protocols that read this machine's own files.
LOCAL_PROTOCOLS = frozenset({"file", "local"})
#: Abandoned W&B calls still running, beyond which new calls are refused (``SkycapRecordIndex._bounded``).
MAX_ABANDONED = 2


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
    """The run index of one step and phase: the object ``<phase>/index/step-<N>.json`` holds once pulled."""
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
        #: W&B calls abandoned at their timeout and still running: each holds a thread and a temp directory.
        self._abandoned: List[threading.Thread] = []
        self._closed = False
        self._in_flight = 0
        #: Versions queued and not yet settled (logged, failed or dropped), whether in the queue or logging.
        self._pending = 0
        self._counts = {"logged": 0, "failed": 0, "timed_out": 0, "dropped": 0, "retried": 0, "discarded": 0}

    # -- events -----------------------------------------------------------------
    def on_step_start(self, trainer: Any, callback_input: CallbackInput, control: TrainingControl) -> None:
        # Each step's train trajectories are taken at its end, so any still here are from a batch the trainer
        # dropped unfinished (dynamic sampling at an epoch's end), under the same global step as this one.
        stale = self.records.take("train")
        if stale:
            logger.info(f"skycap record index: {len(stale)} trajectories of an unfinished step aren't indexed")
            with self._lock:
                self._counts["discarded"] += len(stale)

    def on_step_end(self, trainer: Any, callback_input: CallbackInput, control: TrainingControl) -> None:
        self._index(trainer, "train", callback_input.global_step, trained_keys(callback_input))

    def on_eval_end(self, trainer: Any, callback_input: CallbackInput, control: TrainingControl) -> None:
        self._index(trainer, "eval", callback_input.global_step, set())

    def on_train_end(self, trainer: Any, callback_input: CallbackInput, control: TrainingControl) -> None:
        # The tracker finishes the W&B run right after this event.
        self.close()

    def stats(self) -> Dict[str, int]:
        """Versions ``logged``, ``failed`` (``timed_out`` included), ``dropped`` and ``pending``; ``retried`` calls;
        ``discarded`` trajectories, of a step the trainer dropped unfinished."""
        with self._lock:
            return {**self._counts, "pending": self._pending}

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
            self._pending -= left
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
        name = f"{item[2].name}:{item[2].aliases[0]}"
        # Checking `_closed` and queueing under the lock `close` sets it with: a version is either queued
        # before close, and so logged or counted by it, or refused here.
        with self._lock:
            if self._closed:
                reason = "closed"
            else:
                if self._worker is None:
                    self._worker = threading.Thread(target=self._work, name="skycap-record-index", daemon=True)
                    self._worker.start()
                try:
                    self._queue.put_nowait(item)
                except queue.Full:
                    reason = f"queue full ({self._queue.maxsize})"
                else:
                    self._pending += 1
                    return
        logger.warning(f"skycap record index: {reason}; dropping {name}")
        self._count("dropped")

    # -- logging, on the worker ---------------------------------------------------
    def _work(self) -> None:
        while True:
            try:
                # Waking up now and then: if close() couldn't queue _STOP (a full queue) or drained it past
                # its deadline, the worker exits on `_stopping` instead of waiting on an empty queue forever.
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._stopping.is_set():
                    return
                continue
            if item is _STOP:
                return
            if self._stopping.is_set():
                self._settle("dropped")
                continue
            with self._lock:
                self._in_flight += 1
            outcome = "failed"
            try:
                outcome = self._log_with_retries(*item)
            except Exception:  # noqa: BLE001 - a bug here must not stop the worker, or the run
                logger.exception(f"skycap record index: logging {item[2].name} failed")
            finally:
                with self._lock:
                    self._in_flight -= 1
                self._settle(outcome)

    def _settle(self, outcome: str) -> None:
        """Count a queued version's outcome, in the same step that stops counting it as pending."""
        with self._lock:
            self._counts[outcome] += 1
            if outcome == "timed_out":
                self._counts["failed"] += 1
            self._pending -= 1

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
        """Run ``call`` on a thread of its own and give up waiting after ``timeout``.

        An abandoned call can't be cancelled, so while ``MAX_ABANDONED`` of them are still running, new calls
        are refused outright: a W&B that keeps hanging can't pile up threads and temp directories.
        """
        self._abandoned = [thread for thread in self._abandoned if thread.is_alive()]
        if len(self._abandoned) >= MAX_ABANDONED:
            raise _Abandoned(f"{len(self._abandoned)} earlier W&B calls are still hanging")
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
            self._abandoned.append(thread)
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


# -- pulling a run back -----------------------------------------------------------
@dataclass
class PullSummary:
    """What ``pull`` brought back, and what it couldn't."""

    #: ``<artifact>:<version>`` per version whose ``step.json`` was written.
    versions: List[str] = field(default_factory=list)
    #: Versions that couldn't be read at all, with why.
    failed_versions: List[str] = field(default_factory=list)
    #: Records whose files are all in ``out_dir`` now (already there and identical included).
    records: int = 0
    #: Files moved in; files left alone because an identical one was already there.
    files: int = 0
    unchanged: int = 0
    #: ``<version>/<id>: why`` per mirrored record whose files couldn't all be fetched. Skipped, not partial.
    missing: List[str] = field(default_factory=list)
    #: ``<version>/<id>`` per row whose record has no mirror: indexed, with nothing in W&B to pull.
    local_only: List[str] = field(default_factory=list)

    @property
    def pulled_anything(self) -> bool:
        return bool(self.versions or self.records)

    def report(self) -> str:
        lines = [
            f"versions: {len(self.versions)} pulled, {len(self.failed_versions)} failed",
            f"records: {self.records} pulled ({self.files} files moved in, {self.unchanged} already there), "
            f"{len(self.missing)} missing, {len(self.local_only)} local-only (indexed, nothing to pull)",
        ]
        lines += [f"  failed version {item}" for item in self.failed_versions]
        lines += [f"  missing {item}" for item in self.missing]
        lines += [f"  local-only {item}" for item in self.local_only]
        return "\n".join(lines)


def parse_ref(ref: str) -> Tuple[str, Optional[str]]:
    """``entity/project/artifact[:alias]`` to ``(entity/project/artifact, alias or None)``."""
    path, _, alias = ref.rpartition(":") if ":" in ref.rsplit("/", 1)[-1] else (ref, "", "")
    if path.count("/") != 2 or not all(path.split("/")):
        raise ValueError(f"expected <entity>/<project>/<artifact>[:<alias>], got {ref!r}")
    return path, alias or None


def pull(ref: str, out_dir: Any, *, api: Any = None, allow_local: bool = False) -> PullSummary:
    """Pull a run's records and step index from W&B into ``out_dir``, a record directory.

    With an alias (``...:train-step-3``, ``...:v7``), that version; without one, every version. Each phase
    is a record directory of its own: a version's records go to ``out_dir/<phase>/`` and its ``step.json``
    to ``out_dir/<phase>/index/step-<N>.json``.
    A record whose files can't all be fetched is reported and skipped; the rest go on.

    Args:
        ref: ``<entity>/<project>/<artifact>[:<alias>]``.
        out_dir: the record directory to fill; created if missing.
        api: a ``wandb.Api``; one is made when None.
        allow_local: read records from a mirror on this machine's own filesystem; refused otherwise, since
            the mirror's location comes from the artifact.
    """
    path, alias = parse_ref(ref)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if api is None:
        import wandb

        api = wandb.Api()
    summary = PullSummary()
    try:
        versions = [api.artifact(f"{path}:{alias}")] if alias else list(api.artifacts(ARTIFACT_TYPE, path))
    except Exception as error:  # noqa: BLE001 - reported; nothing else to pull
        summary.failed_versions.append(f"{ref}: {type(error).__name__}: {error}")
        return summary
    for artifact in versions:
        label = f"{path.rsplit('/', 1)[-1]}:{getattr(artifact, 'version', '?')}"
        try:
            _pull_version(artifact, label, out, summary, allow_local)
        except Exception as error:  # noqa: BLE001 - one version must not stop the others
            summary.failed_versions.append(f"{label}: {type(error).__name__}: {error}")
    return summary


def _pull_version(artifact: Any, label: str, out: Path, summary: PullSummary, allow_local: bool = False) -> None:
    # A scratch directory inside out_dir, so each file moves in with an atomic rename.
    with tempfile.TemporaryDirectory(prefix=".pull-", dir=out) as tmp:
        root = Path(tmp)
        # Only the index comes from W&B. The records come from the mirror, read directly (below): W&B's own
        # download of an s3:// reference needs s3:ListBucketVersions, which a reader of the bucket may lack.
        artifact.get_entry("step.json").download(root=str(root))
        step = json.loads((root / "step.json").read_bytes())
        if not isinstance(step, dict) or not isinstance(step.get("rows"), list):
            raise ValueError("step.json is not a run index (no rows)")
        phase, number = step["phase"], int(step["step"])
        if phase not in PHASES:
            raise ValueError(f"step.json has an unknown phase {phase!r}")
        # Each phase is a record directory of its own, its step indexes beside the records.
        phase_dir = out / phase
        phase_dir.mkdir(exist_ok=True)
        for row in step["rows"]:
            _fetch_record(row, root / "records", allow_local)
            _move_record(row, root / "records", phase_dir, f"{label}/{row.get('id')}", summary)
        target = phase_dir / "index" / f"step-{number}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        _move_in(root / "step.json", target)
    summary.versions.append(label)


def _fetch_record(row: Dict[str, Any], dest: Path, allow_local: bool = False) -> None:
    """A mirrored record's files, read from the mirror into ``dest`` with the caller's own credentials.

    A file that can't be read is left out (a partial download is removed), and ``_move_record`` reports the
    record missing, naming it. The mirror's location comes from the downloaded ``step.json``, so one on this
    machine's own filesystem (``file://``, a bare path) is refused unless ``allow_local``: otherwise an
    artifact could copy a private local file into the output directory.
    """
    try:
        names = _record_files(row)
    except ValueError:
        return  # reported by _move_record
    if not names:
        return
    import fsspec

    mirror_dir = row["record"]["mirror"].rsplit("/", 1)[0]
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        uri = f"{mirror_dir}/{name}"
        try:
            fs, path = fsspec.core.url_to_fs(uri)
            protocols = {fs.protocol} if isinstance(fs.protocol, str) else set(fs.protocol)
            if protocols & LOCAL_PROTOCOLS and not allow_local:
                raise PermissionError("a mirror on this machine's filesystem needs --allow-local-mirror")
            fs.get_file(path, str(dest / name))
        except Exception as error:  # noqa: BLE001 - this record is reported missing
            (dest / name).unlink(missing_ok=True)
            logger.warning(f"skycap record pull: fetching {uri} failed: {type(error).__name__}: {error}")


def _record_files(row: Dict[str, Any]) -> List[str]:
    """A mirrored row's file names, as ``record_references`` referenced them; [] for a local-only row.

    Raises ValueError when a name isn't a plain file name: the names come from the downloaded ``step.json``,
    and one like ``../x`` would move a file out of the directory.
    """
    record = row.get("record")
    if not record or not record.get("mirror"):
        return []
    names = list(record.get("files") or [record["mirror"].rsplit("/", 1)[-1]])
    unsafe = [name for name in names if not _plain_name(name)]
    if unsafe:
        raise ValueError(f"file names that aren't plain file names: {unsafe!r}")
    return names


def _plain_name(name: Any) -> bool:
    return (
        isinstance(name, str)
        and name not in ("", ".", "..")
        and "/" not in name
        and "\\" not in name
        and "\0" not in name
    )


def _move_record(row: Dict[str, Any], downloaded: Path, out: Path, label: str, summary: PullSummary) -> None:
    record = row.get("record")
    if not record:
        return
    try:
        names = _record_files(row)
    except ValueError as error:
        summary.missing.append(f"{label}: not pulled: {error}")
        return
    if not names:
        from skycap import RecordLocation

        # Not in the mirror: say where it is, `host:path`, for whoever wants to fetch it from that node.
        summary.local_only.append(f"{label} at {RecordLocation.from_json(record).local}")
        return
    absent = [name for name in names if not (downloaded / name).is_file()]
    if absent:
        summary.missing.append(f"{label}: could not fetch {', '.join(absent)}")
        return
    # Sidecars before the document, as skycap writes them, so a document in out_dir has its sidecars.
    # A record goes in whole or not at all: a file it replaces is kept aside until the whole record is in.
    backups = downloaded.parent / "replaced"
    placed: List[str] = []
    replaced: List[str] = []
    moved = unchanged = 0
    try:
        for name in names:
            source, target = downloaded / name, out / name
            if target.is_file() and _same(target, source):
                unchanged += 1
                continue
            if target.exists():
                if not target.is_file():
                    raise IsADirectoryError(f"{target} is not a file")
                backups.mkdir(exist_ok=True)
                os.replace(target, backups / name)
                replaced.append(name)
            # The scratch directory is inside out_dir, so this is an atomic rename on one filesystem.
            os.replace(source, target)
            placed.append(name)
            moved += 1
    except OSError as error:
        # One record that can't move in (say, a directory where a file goes) mustn't stop the others, and
        # mustn't change out_dir: take back what it placed, put back what it replaced, report it, go on.
        for name in placed:
            (out / name).unlink(missing_ok=True)
        for name in replaced:
            os.replace(backups / name, out / name)
        summary.missing.append(f"{label}: could not move into {out}: {type(error).__name__}: {error}")
        return
    for name in replaced:
        (backups / name).unlink(missing_ok=True)
    summary.files += moved
    summary.unchanged += unchanged
    summary.records += 1


def _move_in(source: Path, target: Path) -> bool:
    """Rename ``source`` onto ``target`` unless an identical file is there. Returns whether it moved."""
    if target.is_file() and _same(target, source):
        return False
    # The scratch directory is inside out_dir, so this is an atomic rename on one filesystem.
    os.replace(source, target)
    return True


def _same(a: Path, b: Path) -> bool:
    """Whether two files hold the same bytes; sizes first, so most differences need no hashing."""
    return a.stat().st_size == b.stat().st_size and _digest(a) == _digest(b)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Optional[List[str]] = None, *, api: Any = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m examples.train_integrations.harbor_skycap.record_index")
    commands = parser.add_subparsers(dest="command", required=True)
    pull_parser = commands.add_parser(
        "pull", help="pull a run's records and step index from W&B into a record directory"
    )
    pull_parser.add_argument("ref", help="<entity>/<project>/<artifact>[:<alias>]; every version without an alias")
    pull_parser.add_argument("out_dir", help="the record directory to fill")
    pull_parser.add_argument(
        "--allow-local-mirror",
        action="store_true",
        help="read records from a mirror on this machine's own filesystem (file://, a path); refused by default",
    )
    args = parser.parse_args(argv)
    try:
        summary = pull(args.ref, args.out_dir, api=api, allow_local=args.allow_local_mirror)
    except ValueError as error:
        parser.error(str(error))
    print(summary.report())
    return 0 if summary.pulled_anything else 1


if __name__ == "__main__":
    sys.exit(main())
