"""Where an export's artifact goes: a directory beside the record.

An export produces one compressed JSONL file. It is written next to the run it
came from -- `<record>/exports/<export id>/<name>.jsonl.zst` -- so a record
directory holds a run and everything derived from it, and copying the
directory copies both.

There is no object-store abstraction here, and no delivery to a second
destination. Capture writes one kind of file to one place it was told about;
getting an artifact somewhere else is `cp`, or whatever a deployment already
uses to move files, and neither is something this process should be in the
middle of.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path


def safe_segment(value: str) -> str:
    """Make ``value`` safe as one path segment.

    Names that a caller chose are sanitized before they reach a path. Path
    separators and control characters become underscores, so a name can never
    introduce a new segment; runs of two or more dots are collapsed, so no
    ``..`` survives; and the result is length-bounded, so a long name cannot
    push a path past a filesystem's limit.
    """
    cleaned = [character if character.isalnum() or character in "-_." else "_" for character in value]
    text = re.sub(r"\.{2,}", "_", "".join(cleaned))
    text = re.sub(r"_{2,}", "_", text).strip("._")
    return (text or "_")[:96]


def artifact_key(export_id: str, filename: str) -> str:
    """One export's artifact, relative to the exports directory."""
    return f"{safe_segment(export_id)}/{safe_segment(filename)}"


class ArtifactStore:
    """The exports directory. Write one, read one back."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()

    def path(self, key: str) -> Path:
        """The file a key names, refusing one that would escape the root."""
        candidate = (self.root / key).resolve()
        root = self.root.resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError(f"artifact key escapes the exports directory: {key!r}")
        return candidate

    async def write(self, key: str, data: bytes) -> str:
        """Write an artifact and return its `file://` URI.

        Written to a temporary name and renamed, so a reader never sees a
        half-written artifact and a process killed mid-write leaves nothing
        that looks finished.
        """
        path = self.path(key)

        def store() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_bytes(data)
            temporary.replace(path)

        await asyncio.to_thread(store)
        return f"file://{path}"

    async def read(self, key: str) -> bytes:
        return await asyncio.to_thread(self.path(key).read_bytes)
