"""The on-disk record: written once, when a trajectory ends.

The format is specified in ``docs/format.md``, which is the contract for any
reader (the viewer reads it from Node). In short, each trajectory is up to four
zstd files in ``record_dir``, split by who reads them:

    {id}.json.zst           the document: meta, annotations, tools, failures, nodes,
                            and a manifest of the sidecars below
    {id}.tokens.zst         token ids, logprobs, and the text the tokens decode to
                            with each token's byte offset into it
    {id}.experts.zst        routed experts (R3)
    {id}.sampling_mask.zst  per sampled token, the ids it could have been drawn from

A sidecar is raw little-endian arrays, concatenated at 8-byte-aligned offsets
and compressed as one zstd frame. The document's ``sidecars`` manifest gives
each array's offset, dtype and shape, so a reader needs only a typed-array view.
Every file is written to a temporary name and renamed, sidecars first, so a
document on disk always has its sidecars beside it.

This is durability for viewing, not fault tolerance: nothing is written while a
trajectory runs, and a crash loses the trajectories that were open.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import zstandard

from skycap.graph import CallInfo, NodeTokens
from skycap.trajectory import Failure, Trajectory

FORMAT_VERSION = 1
_LEVEL = 3
_ALIGN = 8
#: The dtypes a sidecar array may have, by the name the manifest uses.
DTYPES = {name: np.dtype(name).newbyteorder("<") for name in ("uint8", "uint16", "int16", "int32", "int64", "float64")}


def document_path(record_dir: Path, trajectory_id: str) -> Path:
    return record_dir / f"{trajectory_id}.json.zst"


def sidecar_path(record_dir: Path, trajectory_id: str, kind: str) -> Path:
    return record_dir / f"{trajectory_id}.{kind}.zst"


# -- writing ------------------------------------------------------------------
def write(record_dir: Path, trajectory: Trajectory) -> Path:
    """Write the trajectory's files. Returns the document path."""
    record_dir.mkdir(parents=True, exist_ok=True)
    document = trajectory.document()
    columns = _Columns()
    for entry, node in zip(document["nodes"], trajectory.graph, strict=True):
        entry["tokens"] = None if node.tokens is None else columns.add(node.tokens)
    sidecars = {}
    for kind, arrays in columns.sidecars().items():
        name = sidecar_path(record_dir, trajectory.id, kind).name
        sidecars[kind] = {"file": name, "arrays": _write_sidecar(record_dir / name, arrays)}
    document["format_version"] = FORMAT_VERSION
    document["sidecars"] = sidecars
    path = document_path(record_dir, trajectory.id)
    _write_bytes(path, zstandard.ZstdCompressor(level=_LEVEL).compress(orjson.dumps(document)))
    return path


class _Columns:
    """Every token node's arrays, flattened in node order, and each node's slice of them."""

    def __init__(self) -> None:
        self.token_ids: list[np.ndarray] = []
        self.logprobs: list[np.ndarray] = []
        self.text_offsets: list[np.ndarray] = []
        self.text: list[bytes] = []
        self.experts: list[np.ndarray] = []
        self.mask_rows: list[list[int]] = []
        self.tokens = self.text_bytes = self.expert_rows = 0

    def add(self, tokens: NodeTokens) -> dict[str, Any]:
        length = len(tokens.token_ids)
        meta: dict[str, Any] = {
            "offset": self.tokens,
            "length": length,
            "sampled_start": tokens.sampled_start,
            "has_logprobs": tokens.logprobs is not None,
            "text_offset": None,
            "text_bytes": 0,
            "experts_offset": None,
            "experts_rows": 0,
            "mask_offset": None,
            "mask_rows": 0,
        }
        self.token_ids.append(np.asarray(tokens.token_ids, dtype=np.int32))
        self.logprobs.append(
            np.asarray(tokens.logprobs, dtype=np.float64)
            if tokens.logprobs is not None
            else np.full(length, np.nan, dtype=np.float64)
        )
        self.tokens += length
        if tokens.text is not None:
            data = tokens.text.encode("utf-8")
            offsets = _checked_offsets(tokens.text_offsets, length, len(data))
            meta["text_offset"], meta["text_bytes"] = self.text_bytes, len(data)
            self.text.append(data)
            self.text_offsets.append(offsets)
            self.text_bytes += len(data)
        else:
            self.text_offsets.append(np.zeros(length, dtype=np.int32))
        if tokens.routed_experts is not None:
            routed = np.asarray(tokens.routed_experts)
            meta["experts_offset"], meta["experts_rows"] = self.expert_rows, routed.shape[0]
            self.experts.append(routed)
            self.expert_rows += routed.shape[0]
        if tokens.sampling_mask is not None:
            meta["mask_offset"], meta["mask_rows"] = len(self.mask_rows), len(tokens.sampling_mask)
            self.mask_rows.extend(tokens.sampling_mask)
        return meta

    def sidecars(self) -> dict[str, dict[str, np.ndarray]]:
        out: dict[str, dict[str, np.ndarray]] = {}
        if self.token_ids:
            out["tokens"] = {
                "token_ids": np.concatenate(self.token_ids),
                "logprobs": np.concatenate(self.logprobs),
                "text_offsets": np.concatenate(self.text_offsets),
                "text": np.frombuffer(b"".join(self.text), dtype=np.uint8),
            }
        if self.experts:
            out["experts"] = {"routed_experts": np.concatenate(self.experts)}
        if self.mask_rows:
            offsets = np.zeros(len(self.mask_rows) + 1, dtype=np.int64)
            offsets[1:] = np.cumsum([len(row) for row in self.mask_rows])
            ids = np.fromiter((i for row in self.mask_rows for i in row), dtype=np.int32, count=int(offsets[-1]))
            out["sampling_mask"] = {"ids": ids, "offsets": offsets}
        return out


def _checked_offsets(offsets: list[int] | None, length: int, text_bytes: int) -> np.ndarray:
    """A node's per-token byte offsets into its text, or a ValueError naming what's wrong."""
    if offsets is None or len(offsets) != length:
        raise ValueError(f"text_offsets must have one entry per token ({length})")
    array = np.asarray(offsets, dtype=np.int32)
    if length and (array[0] < 0 or array[-1] > text_bytes or np.any(np.diff(array) < 0)):
        raise ValueError("text_offsets must be non-decreasing byte offsets within the text")
    return array


def _write_sidecar(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    """Write ``arrays`` as one aligned, compressed buffer. Returns their manifest."""
    manifest: dict[str, dict[str, Any]] = {}
    chunks: list[bytes] = []
    cursor = 0
    for name, array in arrays.items():
        if array.dtype.name not in DTYPES:
            raise ValueError(f"sidecar array {name!r} has unsupported dtype {array.dtype}")
        data = np.ascontiguousarray(array, dtype=DTYPES[array.dtype.name]).tobytes()
        padding = -cursor % _ALIGN
        chunks.append(b"\0" * padding)
        cursor += padding
        manifest[name] = {"dtype": array.dtype.name, "shape": list(array.shape), "offset": cursor}
        chunks.append(data)
        cursor += len(data)
    _write_bytes(path, zstandard.ZstdCompressor(level=_LEVEL).compress(b"".join(chunks)))
    return manifest


def _write_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


# -- reading ------------------------------------------------------------------
def list_ids(record_dir: Path) -> Iterator[str]:
    for path in sorted(record_dir.glob("*.json.zst")):
        yield path.name.removesuffix(".json.zst")


def read_document(record_dir: Path, trajectory_id: str) -> dict[str, Any]:
    raw = document_path(record_dir, trajectory_id).read_bytes()
    return orjson.loads(zstandard.ZstdDecompressor().decompressobj().decompress(raw))


def read_sidecar(record_dir: Path, document: dict[str, Any], kind: str) -> dict[str, np.ndarray] | None:
    """A sidecar's arrays, by name, or None if the trajectory has none of that kind."""
    entry = document.get("sidecars", {}).get(kind)
    if entry is None:
        return None
    buffer = zstandard.ZstdDecompressor().decompressobj().decompress((record_dir / entry["file"]).read_bytes())
    out = {}
    for name, spec in entry["arrays"].items():
        dtype = DTYPES[spec["dtype"]]
        count = int(np.prod(spec["shape"], dtype=np.int64))
        out[name] = np.frombuffer(buffer, dtype=dtype, count=count, offset=spec["offset"]).reshape(spec["shape"])
    return out


def load(record_dir: Path, trajectory_id: str) -> Trajectory:
    """Rebuild a written trajectory, graph and arrays included."""
    document = read_document(record_dir, trajectory_id)
    arrays = {kind: read_sidecar(record_dir, document, kind) for kind in ("tokens", "experts", "sampling_mask")}
    trajectory = Trajectory(
        id=document["id"],
        meta=document["meta"],
        capture=document.get("capture") or {},
        status=document["status"],
        annotations=document["annotations"],
        failures=[Failure(**failure) for failure in document["failures"]],
        created_at=document["created_at"],
        finished_at=document["finished_at"],
        # A record written at shutdown is still open: its finish is still to come.
        ended=document["status"] != "open",
    )
    graph = trajectory.graph
    graph.tools.update(document["tools"])
    for entry in document["nodes"]:
        node, _ = graph.add(
            entry["parent"],
            role=entry["role"],
            author=entry["author"],
            message=entry["message"],
            match_hash=entry["match_hash"],
            delta_hash=entry["delta_hash"],
            created_at=entry["created_at"],
            tokens=_node_tokens(entry.get("tokens"), arrays),
        )
        assert node.id == entry["id"], "record nodes are stored in creation order"
        node.calls.extend(CallInfo(**call) for call in entry["calls"])
    return trajectory


def _node_tokens(meta: dict[str, Any] | None, arrays: dict[str, dict[str, np.ndarray] | None]) -> NodeTokens | None:
    if meta is None:
        return None
    tokens = arrays["tokens"]
    assert tokens is not None, "a node with tokens needs the tokens sidecar"
    span = slice(meta["offset"], meta["offset"] + meta["length"])
    text = offsets = None
    if meta["text_offset"] is not None:
        start = meta["text_offset"]
        text = tokens["text"][start : start + meta["text_bytes"]].tobytes().decode("utf-8")
        offsets = tokens["text_offsets"][span].tolist()
    routed = None
    if meta["experts_offset"] is not None:
        experts = arrays["experts"]
        assert experts is not None
        start = meta["experts_offset"]
        routed = experts["routed_experts"][start : start + meta["experts_rows"]].copy()
    rows = None
    if meta["mask_offset"] is not None and meta["mask_rows"] == 0:
        rows = []
    elif meta["mask_offset"] is not None:
        mask = arrays["sampling_mask"]
        assert mask is not None
        ids, bounds = mask["ids"], mask["offsets"]
        first = meta["mask_offset"]
        rows = [ids[bounds[r] : bounds[r + 1]].tolist() for r in range(first, first + meta["mask_rows"])]
    return NodeTokens(
        token_ids=tokens["token_ids"][span].tolist(),
        sampled_start=meta["sampled_start"],
        logprobs=tokens["logprobs"][span].tolist() if meta["has_logprobs"] else None,
        routed_experts=routed,
        sampling_mask=rows,
        text=text,
        text_offsets=offsets,
    )
