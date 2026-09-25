"""zstd, in one place.

A trajectory's journal records and a committed record are both compressed, and
both have to be read back by anything that opens them, so the level and the
reader live here rather than being chosen twice.

**One compressor per thread.** A `ZstdCompressor` carries internal state and
cannot be used by two threads at once; sharing one module-level instance is
not a slow path, it is a segfault waiting for enough concurrency to find it.
Every write here happens in a worker thread -- journal appends and committed
records both go through `asyncio.to_thread`, and per-trajectory locking means
two trajectories compress at the same moment -- so this is reached routinely
rather than exotically.

Thread-local rather than per-call: a compressor allocates a context, and one
per record on the write path would be a cost paid for nothing. Rather than a
lock, because serializing every compression across the process would put
trajectories back in each other's way, which per-trajectory locking exists to
avoid.
"""

from __future__ import annotations

import threading

import zstandard

LEVEL = 6

_local = threading.local()


def _compressor() -> zstandard.ZstdCompressor:
    found = getattr(_local, "compressor", None)
    if found is None:
        found = _local.compressor = zstandard.ZstdCompressor(level=LEVEL)
    return found


def _decompressor() -> zstandard.ZstdDecompressor:
    found = getattr(_local, "decompressor", None)
    if found is None:
        found = _local.decompressor = zstandard.ZstdDecompressor()
    return found


def compress(data: bytes) -> bytes:
    return _compressor().compress(data)


def decompress(data: bytes) -> bytes:
    # ``max_output_size`` is unbounded here; what is compressed is already
    # bounded by the proxy's max_request_bytes.
    return _decompressor().decompress(data)
