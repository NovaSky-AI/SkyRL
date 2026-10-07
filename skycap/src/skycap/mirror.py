"""A copy of every written record in a remote store, made in the background.

    mirror = RecordMirror("s3://bucket/run-7")
    mirror.submit(record_dir, "tr_ab12")    # after record.write; never blocks
    mirror.close()                          # at shutdown, bounded by a deadline

The local ``record_dir`` stays the source of truth: a record is written there
first, as always, and then queued for the mirror, which copies its files to
``{url}/{name}`` through fsspec, sidecars before the document. So a document
in the mirror has its sidecars beside it, as on disk.

``exclude`` leaves sidecar kinds out of the mirror, e.g. ``("experts",)`` when
the remote copy is for reading rather than retraining. The mirrored document
then lists only the sidecars it has and names the others in
``omitted_sidecars``; the local record is untouched. Excluding ``tokens`` is
allowed, but a viewer reading the mirror then has message text only.

The mirror fails open. A store that is slow, down or refusing never fails a
trajectory, it only loses the copy, and every loss is logged and counted in
``stats`` (served on ``/healthz``):

- The queue is bounded. A record that finds it full is dropped.
- Each file's copy has a timeout. A copy that overruns it is abandoned, not
  retried, since it may still land and a retry would write it twice.
- Errors that may pass (connection resets, timeouts, most ``OSError``) are
  retried, a bounded number of times with exponential backoff. Anything else
  fails the record at once.
- ``close`` waits for the queue until a deadline and drops what is left.

fsspec is imported only when a mirror is made (the ``remote`` extra). A store
other than the ones fsspec ships needs its implementation installed, e.g.
``s3fs`` for ``s3://`` or ``gcsfs`` for ``gs://``.
"""

from __future__ import annotations

import logging
import queue
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from skycap import record

logger = logging.getLogger(__name__)

#: ``OSError`` subclasses that say the copy can't succeed however often it is tried.
_PERMANENT = (PermissionError, FileNotFoundError, IsADirectoryError, NotADirectoryError, FileExistsError)
_STOP = object()


class _Abandoned(Exception):
    """A copy overran its timeout, or the store already has too many that did."""


class RecordMirror:
    """Copies written records to ``url`` on background threads.

    Args:
        url: an fsspec URL, e.g. ``s3://bucket/prefix``, ``gs://bucket/prefix`` or ``memory://prefix``.
        exclude: sidecar kinds (``tokens``, ``experts``, ``sampling_mask``) not to copy.
        workers: threads copying records concurrently.
        queue_size: records waiting to be copied, beyond which new ones are dropped.
        timeout: seconds one file's copy may take before it is abandoned.
        attempts: tries per file for an error that may pass.
        backoff: seconds before the first retry, doubled for each one after.
        shutdown_timeout: seconds ``close`` waits for the queue to drain.
        storage_options: passed to the fsspec filesystem (credentials, endpoint).
    """

    def __init__(
        self,
        url: str,
        *,
        exclude: Iterable[str] = (),
        workers: int = 4,
        queue_size: int = 1024,
        timeout: float = 300.0,
        attempts: int = 3,
        backoff: float = 1.0,
        shutdown_timeout: float = 60.0,
        storage_options: dict[str, Any] | None = None,
    ) -> None:
        try:
            import fsspec
        except ImportError as error:
            raise ImportError("a record mirror needs fsspec: install skycap[remote]") from error
        if workers < 1 or queue_size < 1 or attempts < 1:
            raise ValueError("workers, queue_size and attempts must be at least 1")
        if isinstance(exclude, str):
            raise TypeError("exclude is a list of sidecar kinds, not one string")
        unknown = set(exclude) - set(record.SIDECAR_KINDS)
        if unknown:
            raise ValueError(f"unknown sidecar kinds {sorted(unknown)} in exclude; known: {list(record.SIDECAR_KINDS)}")
        self.exclude = frozenset(exclude)
        # Resolving the URL here makes a bad one, or a store whose implementation is missing, fail at startup.
        self.fs, root = fsspec.core.url_to_fs(url, **(storage_options or {}))
        self.url = url.rstrip("/")
        self._root = root.rstrip("/")
        self.workers = workers
        self.timeout = timeout
        self.attempts = attempts
        self.backoff = backoff
        self.shutdown_timeout = shutdown_timeout
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=queue_size)
        self._threads: list[threading.Thread] = []
        self._abandoned: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._closed = False
        self._counts = {"mirrored": 0, "failed": 0, "timed_out": 0, "dropped": 0, "retried": 0}
        self._in_flight = 0

    def uri(self, name: str) -> str:
        """Where the mirror puts the record file ``name``, e.g. ``tr_ab12.json.zst``."""
        return f"{self.url}/{name}"

    def names(self, record_dir: Path, trajectory_id: str) -> list[str]:
        """The file names a written record has in the mirror: its kept sidecars, then its document.

        These are the names the mirror copies the record to, whether or not the copy has landed yet.
        """
        return [path.name for path in self._kept(record.record_files(record_dir, trajectory_id))]

    def stats(self) -> dict[str, int]:
        """Records ``mirrored``, ``failed`` (``timed_out`` included), ``dropped`` and ``pending``, and ``retried`` copies."""
        with self._lock:
            return {**self._counts, "pending": self._queue.qsize() + self._in_flight}

    def submit(self, record_dir: Path, trajectory_id: str) -> bool:
        """Queue a written record. Returns False, and drops it, when the queue is full or the mirror closed."""
        with self._lock:
            closed = self._closed
            if not closed and not self._threads:
                self._start()
        if closed:
            self._drop(f"the mirror is closed; not mirroring {trajectory_id}")
            return False
        try:
            self._queue.put_nowait((record_dir, trajectory_id))
        except queue.Full:
            self._drop(f"the mirror queue is full ({self._queue.maxsize}); dropping {trajectory_id}")
            return False
        return True

    def close(self, timeout: float | None = None) -> bool:
        """Wait up to ``timeout`` (default ``shutdown_timeout``) for queued records, then drop the rest.

        Returns whether everything queued was handled in time. Copies still running at the deadline are
        left to finish on their own threads, which don't keep the process alive.
        """
        deadline = time.monotonic() + (self.shutdown_timeout if timeout is None else timeout)
        with self._lock:
            self._closed = True
            threads = list(self._threads)
        for _ in threads:
            # Behind every queued record, so each worker exits once the queue is drained.
            self._put_before(_STOP, deadline)
        for thread in threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        self._stopping.set()
        left = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is not _STOP:
                left += 1
        with self._lock:
            self._counts["dropped"] += left
            in_flight = self._in_flight
        if left or in_flight:
            logger.warning(
                "record mirror: shutdown deadline reached; dropped %d queued records, %d still copying", left, in_flight
            )
        return not left and not in_flight

    # -- the workers ------------------------------------------------------------
    def _start(self) -> None:
        for index in range(self.workers):
            thread = threading.Thread(target=self._work, name=f"skycap-mirror-{index}", daemon=True)
            thread.start()
            self._threads.append(thread)

    def _put_before(self, item: Any, deadline: float) -> None:
        try:
            self._queue.put(item, timeout=max(0.0, deadline - time.monotonic()))
        except queue.Full:
            pass

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
                self._count(self._mirror(*item))
            except Exception:  # noqa: BLE001 - a bug here must not stop the worker
                logger.exception("record mirror: mirroring %s failed", item[1])
                self._count("failed")
            finally:
                with self._lock:
                    self._in_flight -= 1

    def _mirror(self, record_dir: Path, trajectory_id: str) -> str:
        """Copy one record, sidecars first. Returns the counter it lands in."""
        files = record.record_files(record_dir, trajectory_id)
        sidecars, document = files[:-1], files[-1]
        kept = self._kept(sidecars)
        with tempfile.TemporaryDirectory(prefix="skycap-mirror-") as scratch:
            source = document
            if len(kept) < len(sidecars):
                # The mirrored document lists only the sidecars the mirror has.
                source = Path(scratch) / document.name
                source.write_bytes(record.omit_sidecars(document, self.exclude))
            return self._copy_files([*[(path, path.name) for path in kept], (source, document.name)])

    def _kept(self, files: list[Path]) -> list[Path]:
        """``files`` without the sidecars of excluded kinds."""
        return [path for path in files if record.sidecar_kind(path) not in self.exclude]

    def _copy_files(self, files: list[tuple[Path, str]]) -> str:
        """Copy ``(local file, remote name)`` pairs in order, stopping at the first failure."""
        for path, name in files:
            remote = f"{self._root}/{name}"
            try:
                self._copy(path, remote)
            except _Abandoned as error:
                # An abandoned copy may still be reading a rewritten document's scratch file when it is
                # removed; that copy then fails, which is what abandoning it already counted.
                logger.warning("record mirror: abandoned %s: %s", self.uri(name), error)
                return "timed_out"
            except Exception as error:  # noqa: BLE001 - the store's error, logged and counted
                logger.warning("record mirror: copying %s failed: %s: %s", self.uri(name), type(error).__name__, error)
                return "failed"
        return "mirrored"

    @classmethod
    def from_config(cls, url: str, config: Mapping[str, Any] | None = None) -> RecordMirror:
        """A mirror of ``url`` with ``config`` as its keyword options, e.g. ``{"exclude": ["experts"]}``."""
        return cls(url, **dict(config or {}))

    def _copy(self, path: Path, remote: str) -> None:
        for attempt in range(1, self.attempts + 1):
            try:
                self._bounded(self.fs.put_file, str(path), remote)
                return
            except _Abandoned:
                raise
            except Exception as error:
                if attempt == self.attempts or not _transient(error) or self._stopping.is_set():
                    raise
                logger.info(
                    "record mirror: copying %s failed (%s: %s), retrying",
                    self.uri(path.name),
                    type(error).__name__,
                    error,
                )
                self._count("retried")
                self._stopping.wait(self.backoff * 2 ** (attempt - 1))

    def _bounded(self, call: Callable[..., Any], *args: Any) -> None:
        """Run ``call`` on a thread of its own and give up waiting after ``timeout``."""
        # An abandoned copy can't be cancelled; a store that keeps hanging gets no new threads.
        with self._lock:
            self._abandoned = [thread for thread in self._abandoned if thread.is_alive()]
            hanging = len(self._abandoned)
        if hanging >= 2 * self.workers:
            raise _Abandoned(f"{hanging} earlier copies are still hanging")
        outcome: dict[str, BaseException] = {}

        def run() -> None:
            try:
                call(*args)
            except BaseException as error:  # noqa: BLE001 - handed to the waiting worker
                outcome["error"] = error

        thread = threading.Thread(target=run, name="skycap-mirror-copy", daemon=True)
        thread.start()
        thread.join(self.timeout)
        if thread.is_alive():
            with self._lock:
                self._abandoned.append(thread)
            raise _Abandoned(f"no answer within {self.timeout}s")
        if "error" in outcome:
            raise outcome["error"]

    def _count(self, name: str) -> None:
        with self._lock:
            self._counts[name] += 1
            if name == "timed_out":
                self._counts["failed"] += 1

    def _drop(self, message: str) -> None:
        logger.warning("record mirror: %s", message)
        self._count("dropped")


def _transient(error: BaseException) -> bool:
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    return isinstance(error, OSError) and not isinstance(error, _PERMANENT)
