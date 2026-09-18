"""An exclusive advisory file lock. A leaf module: nothing here imports SkyRL, so it can be imported
while the ``skyrl.train.utils`` package itself is still initializing (``delta_checkpoint`` needs it)."""

import fcntl
import os
from pathlib import Path


class FileLock:
    """An exclusive advisory lock on a file (``fcntl.flock``), usable as a context manager.

    Serializes the processes of one node -- TP ranks, co-located engines -- around a shared
    directory: the delta weight-sync cache and the reserved eval group's export cache.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fd: int | None = None

    def acquire(self, blocking: bool = True) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
        flags = fcntl.LOCK_EX
        if not blocking:
            flags |= fcntl.LOCK_NB
        try:
            fcntl.flock(fd, flags)
        except BlockingIOError:
            os.close(fd)
            return False
        except Exception:
            os.close(fd)
            raise
        self._fd = fd
        return True

    def release(self) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "FileLock":
        self.acquire(blocking=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
