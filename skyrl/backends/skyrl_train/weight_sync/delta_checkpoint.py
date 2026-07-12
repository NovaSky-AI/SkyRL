"""Checkpoint-delta publishing and receiving for disk/cloud weight sync."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import mmap
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from skyrl.backends.skyrl_train.weight_sync.base import WeightChunk
from skyrl.backends.skyrl_train.weight_sync.delta_payload import (
    bytes_tensor_to_tensor,
    bytes_to_uint8_tensor,
    compress_bytes,
    decompress_bytes,
    tensor_to_bytes_tensor,
    uint8_tensor_to_bytes,
)
from skyrl.backends.skyrl_train.weight_sync.memory_debug import (
    log_memory,
    trim_process_memory,
)
from skyrl.backends.skyrl_train.weight_sync.weight_extractor import ExtractorShardInfo

logger = logging.getLogger(__name__)


_MANIFEST_NAME = "manifest.json"
_STATE_DIR_NAME = ".skyrl_weight_sync"
_DEFAULT_WRITER_QUEUE_SIZE = 2
_DEFAULT_CHECKSUM_ALGORITHM = "xxh3-128"
_DEFAULT_PINNED_STAGING_BYTE_CAP = 32 * 1024**3
_T = TypeVar("_T")
_U = TypeVar("_U")


@dataclass
class DeltaTensorRecord:
    name: str
    dtype: str
    shape: list[int]
    uncompressed_num_bytes: int
    compressed_num_bytes: int
    payload_file: str
    payload_key: str
    checksum: str
    checksum_algorithm: str = _DEFAULT_CHECKSUM_ALGORITHM


@dataclass
class DeltaManifest:
    version: int
    base_version: int
    tensors: list[DeltaTensorRecord] = field(default_factory=list)
    total_uncompressed_num_bytes: int = 0
    total_compressed_num_bytes: int = 0
    payload_files: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "DeltaManifest":
        records = []
        for item in data.get("tensors", []):
            normalized = dict(item)
            if "checksum" not in normalized:
                normalized["checksum"] = normalized.pop("sha256")
                normalized.setdefault("checksum_algorithm", "sha256")
            else:
                normalized.pop("sha256", None)
                normalized.setdefault("checksum_algorithm", _DEFAULT_CHECKSUM_ALGORITHM)
            records.append(DeltaTensorRecord(**normalized))
        return cls(
            version=int(data["version"]),
            base_version=int(data["base_version"]),
            tensors=records,
            total_uncompressed_num_bytes=int(data.get("total_uncompressed_num_bytes", 0)),
            total_compressed_num_bytes=int(data.get("total_compressed_num_bytes", 0)),
            payload_files=list(data.get("payload_files", [])),
        )


def _version_name(version: int) -> str:
    return f"v{version:08d}"


def _version_dir(root: Path, version: int) -> Path:
    return root / "versions" / _version_name(version)


def _weights_dir(root: Path) -> Path:
    return root / "weights"


def _deltas_dir(root: Path) -> Path:
    return root / "deltas"


def _staging_dir(root: Path, version: int) -> Path:
    return root / "versions" / f"{_version_name(version)}.staging"


def _is_gs_uri(uri: str) -> bool:
    return uri.startswith("gs://")


def _join_uri(base: str, child: str) -> str:
    return f"{base.rstrip('/')}/{child}"


def _safe_path_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in value)[:160]


def _ordered_prefetch(
    items: Iterable[_T],
    fn: Callable[[_T], _U],
    prefetch_depth: int,
) -> Iterator[_U]:
    """Apply ``fn`` in order while preparing a bounded number of items ahead."""
    if prefetch_depth <= 0:
        for item in items:
            yield fn(item)
        return

    item_iter = iter(items)
    pending: deque[Future[_U]] = deque()

    def submit_next(executor: ThreadPoolExecutor) -> bool:
        try:
            item = next(item_iter)
        except StopIteration:
            return False
        pending.append(executor.submit(fn, item))
        return True

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="skyrl-delta-prefetch") as executor:
        for _ in range(prefetch_depth + 1):
            if not submit_next(executor):
                break
        while pending:
            future = pending.popleft()
            submit_next(executor)
            yield future.result()


def _run_gcloud(args: Sequence[str]) -> None:
    cmd = ["gcloud", "storage", *args]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "gcloud storage command failed: "
            f"{' '.join(cmd)}\nstdout:\n{exc.stdout or ''}\nstderr:\n{exc.stderr or ''}"
        ) from exc


def _copy_to_uri(local_path: Path, uri: str) -> None:
    if _is_gs_uri(uri):
        _run_gcloud(["cp", str(local_path), uri])
    else:
        dst = Path(uri)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        shutil.copy2(local_path, tmp_path)
        with tmp_path.open("rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp_path, dst)
        try:
            dir_fd = os.open(dst.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass


def _copy_from_uri(uri: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = local_path.with_name(f".{local_path.name}.{os.getpid()}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    if _is_gs_uri(uri):
        _run_gcloud(["cp", uri, str(tmp_path)])
    else:
        shutil.copy2(Path(uri), tmp_path)
    os.replace(tmp_path, local_path)


def publish_delta_directory(local_dir: Path, uri: str) -> None:
    """Publish payload files first and manifest last."""
    manifest_path = local_dir / _MANIFEST_NAME
    for path in sorted(local_dir.iterdir()):
        if path.name == _MANIFEST_NAME:
            continue
        _copy_to_uri(path, _join_uri(uri, path.name))
    _copy_to_uri(manifest_path, _join_uri(uri, _MANIFEST_NAME))


def publish_delta_payload_files(local_dir: Path, uri: str) -> None:
    """Publish only payload files for a source-rank submanifest."""
    for path in sorted(local_dir.iterdir()):
        if path.name == _MANIFEST_NAME:
            continue
        _copy_to_uri(path, _join_uri(uri, path.name))


def fetch_delta_directory(uri: str, cache_dir: Path) -> Tuple[DeltaManifest, Path]:
    """Fetch a published delta directory into a local cache and return its manifest."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock = FileLock(cache_dir / ".fetch.lock")
    with lock:
        manifest_path = cache_dir / _MANIFEST_NAME
        if not manifest_path.exists():
            _copy_from_uri(_join_uri(uri, _MANIFEST_NAME), manifest_path)
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = DeltaManifest.from_json(json.load(f))

        for payload_file in manifest.payload_files:
            dst = cache_dir / payload_file
            if not dst.exists():
                _copy_from_uri(_join_uri(uri, payload_file), dst)
    return manifest, cache_dir


def resolve_checkpoint_path(model_path: str) -> Path:
    """Resolve a local checkpoint path from a local path or Hugging Face repo id."""
    path = Path(model_path)
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_path, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"]))


class FileLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._fd: Optional[int] = None

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


def _read_safetensors_header(path: Path) -> tuple[int, dict]:
    with path.open("rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len))
    return 8 + header_len, header


class SafetensorsPatcher:
    """Patch tensor bytes in safetensors files without loading whole shards."""

    def __init__(self, checkpoint_dir: Path) -> None:
        self.checkpoint_dir = checkpoint_dir
        self._headers: Dict[Path, tuple[int, dict]] = {}

    def write_tensor(self, rel_file: str, name: str, tensor: torch.Tensor) -> None:
        path = self.checkpoint_dir / rel_file
        if path not in self._headers:
            self._headers[path] = _read_safetensors_header(path)
        data_start, header = self._headers[path]
        if name not in header:
            raise KeyError(f"Tensor {name!r} not found in {path}")
        offsets = header[name]["data_offsets"]
        expected_nbytes = int(offsets[1]) - int(offsets[0])
        data = uint8_tensor_to_bytes(tensor_to_bytes_tensor(tensor))
        if len(data) != expected_nbytes:
            raise ValueError(f"Tensor {name!r} has {len(data)} bytes, but {path} expects {expected_nbytes} bytes")
        with path.open("r+b") as f:
            f.seek(data_start + int(offsets[0]))
            f.write(data)


class CheckpointIndex:
    """Name-to-safetensors-file index for a local HF checkpoint."""

    def __init__(self, checkpoint_dir: Path) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.weight_map = self._load_weight_map()

    def _load_weight_map(self) -> dict[str, str]:
        index_files = sorted(self.checkpoint_dir.glob("*.safetensors.index.json"))
        if index_files:
            with index_files[0].open("r", encoding="utf-8") as f:
                return dict(json.load(f)["weight_map"])

        weight_map: dict[str, str] = {}
        for safetensors_file in sorted(self.checkpoint_dir.glob("*.safetensors")):
            with safe_open(safetensors_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    weight_map[key] = safetensors_file.name
        if not weight_map:
            raise ValueError(f"No safetensors weights found under {self.checkpoint_dir}")
        return weight_map

    def resolve_tensor_name(self, name: str) -> str:
        if name in self.weight_map:
            return name
        raise KeyError(f"Tensor {name!r} not found in checkpoint {self.checkpoint_dir}")

    def relative_file_for(self, name: str) -> str:
        try:
            return self.weight_map[self.resolve_tensor_name(name)]
        except KeyError as e:
            raise KeyError(f"Tensor {name!r} not found in checkpoint {self.checkpoint_dir}") from e

    def load_tensor(self, name: str) -> torch.Tensor:
        resolved_name = self.resolve_tensor_name(name)
        rel_file = self.weight_map[resolved_name]
        with safe_open(self.checkpoint_dir / rel_file, framework="pt", device="cpu") as f:
            return f.get_tensor(resolved_name)

    def iter_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        by_file: dict[str, list[str]] = {}
        for name, rel_file in self.weight_map.items():
            by_file.setdefault(rel_file, []).append(name)
        for rel_file in sorted(by_file):
            with safe_open(self.checkpoint_dir / rel_file, framework="pt", device="cpu") as f:
                for name in by_file[rel_file]:
                    yield name, f.get_tensor(name)


def _copy_file_reflink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["cp", "--reflink=auto", str(src), str(dst)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        shutil.copy2(src, dst)


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = src.resolve()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def mirror_checkpoint_tree(base_dir: Path, target_dir: Path, copied_rel_files: set[str]) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=False)
    for src in base_dir.rglob("*"):
        rel = src.relative_to(base_dir)
        dst = target_dir / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        elif rel.as_posix() in copied_rel_files:
            _copy_file_reflink(src, dst)
        else:
            _link_or_copy(src, dst)


def ensure_base_version(local_checkpoint_dir: Path, base_model_path: str) -> None:
    state_dir = local_checkpoint_dir / _STATE_DIR_NAME
    base_dir = _version_dir(local_checkpoint_dir, 0)
    if base_dir.exists():
        return
    with FileLock(state_dir / "init.lock"):
        if base_dir.exists():
            return
        resolved_base = resolve_checkpoint_path(base_model_path)
        mirror_checkpoint_tree(resolved_base, base_dir, copied_rel_files=set())


class CheckpointVersionWriter:
    def __init__(
        self,
        root: Path,
        base_version: int,
        target_version: int,
        changed_names: Optional[set[str]] = None,
        copied_rel_files: Optional[set[str]] = None,
    ) -> None:
        self.root = root
        self.base_version = base_version
        self.target_version = target_version
        self.base_dir = _version_dir(root, base_version)
        self.final_dir = _version_dir(root, target_version)
        self.staging_dir = _staging_dir(root, target_version)
        base_index = CheckpointIndex(self.base_dir)
        if copied_rel_files is None:
            copied_rel_files = {base_index.relative_file_for(name) for name in changed_names or set()}
        mirror_checkpoint_tree(self.base_dir, self.staging_dir, copied_rel_files=copied_rel_files)
        self.index = CheckpointIndex(self.staging_dir)
        self.patcher = SafetensorsPatcher(self.staging_dir)

    def write_tensor(self, name: str, tensor: torch.Tensor) -> None:
        resolved_name = self.index.resolve_tensor_name(name)
        self.patcher.write_tensor(self.index.relative_file_for(resolved_name), resolved_name, tensor)

    def commit(self) -> None:
        if self.final_dir.exists():
            shutil.rmtree(self.staging_dir, ignore_errors=True)
            return
        os.rename(self.staging_dir, self.final_dir)

    def abort(self) -> None:
        shutil.rmtree(self.staging_dir, ignore_errors=True)


def stage_checkpoint_tree_deferred(base_dir: Path, target_dir: Path, deferred_rel_files: set[str]) -> None:
    """Create a staging checkpoint while leaving deferred files absent.

    Deferred files are safetensors shards that will be copied/reflinked lazily
    by the writer thread before the first tensor in that shard is patched.
    """
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=False)
    for src in base_dir.rglob("*"):
        rel = src.relative_to(base_dir)
        rel_posix = rel.as_posix()
        dst = target_dir / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        elif rel_posix in deferred_rel_files:
            continue
        else:
            _link_or_copy(src, dst)


class LazyCheckpointVersionWriter:
    """Persist changed tensors into a staged checkpoint lazily by shard."""

    def __init__(
        self,
        root: Path,
        base_version: int,
        target_version: int,
        changed_names: set[str],
    ) -> None:
        self.root = root
        self.base_version = base_version
        self.target_version = target_version
        self.base_dir = _version_dir(root, base_version)
        self.final_dir = _version_dir(root, target_version)
        self.staging_dir = _staging_dir(root, target_version)
        self.index = CheckpointIndex(self.base_dir)
        self.deferred_rel_files = {self.index.relative_file_for(name) for name in changed_names}
        self._prepared_rel_files: set[str] = set()
        stage_checkpoint_tree_deferred(self.base_dir, self.staging_dir, self.deferred_rel_files)
        self.patcher = SafetensorsPatcher(self.staging_dir)

    def _prepare_rel_file(self, rel_file: str) -> None:
        if rel_file in self._prepared_rel_files:
            return
        src = self.base_dir / rel_file
        dst = self.staging_dir / rel_file
        if not dst.exists():
            _copy_file_reflink(src, dst)
        self._prepared_rel_files.add(rel_file)

    def write_tensor(self, name: str, tensor: torch.Tensor) -> None:
        resolved_name = self.index.resolve_tensor_name(name)
        rel_file = self.index.relative_file_for(resolved_name)
        self._prepare_rel_file(rel_file)
        self.patcher.write_tensor(rel_file, resolved_name, tensor)

    def commit(self) -> None:
        if self.final_dir.exists():
            shutil.rmtree(self.staging_dir, ignore_errors=True)
            return
        missing = sorted(rel_file for rel_file in self.deferred_rel_files if not (self.staging_dir / rel_file).exists())
        if missing:
            raise RuntimeError(f"Missing staged checkpoint shard(s): {missing[:5]}")
        os.rename(self.staging_dir, self.final_dir)

    def abort(self) -> None:
        shutil.rmtree(self.staging_dir, ignore_errors=True)


class AsyncCheckpointWriter:
    """Bounded CPU writer queue for checkpoint persistence side effects."""

    _STOP = object()

    def __init__(self, writer: LazyCheckpointVersionWriter, max_queue_size: int) -> None:
        self.writer = writer
        self._queue: queue.Queue[object] = queue.Queue(maxsize=max_queue_size)
        self._failure: BaseException | None = None
        self._failure_lock = threading.Lock()
        self._abort = threading.Event()
        self.write_s = 0.0
        self._thread = threading.Thread(target=self._run, name="skyrl-delta-checkpoint-writer", daemon=True)
        self._thread.start()

    def _set_failure(self, exc: BaseException) -> None:
        with self._failure_lock:
            if self._failure is None:
                self._failure = exc

    def _raise_if_failed(self) -> None:
        with self._failure_lock:
            failure = self._failure
        if failure is not None:
            raise RuntimeError("Async checkpoint writer failed") from failure

    def enqueue(self, name: str, tensor: torch.Tensor) -> None:
        self._raise_if_failed()
        cpu_tensor = tensor.detach().to("cpu", copy=True).contiguous()
        item = (name, cpu_tensor)
        while True:
            self._raise_if_failed()
            try:
                self._queue.put(item, timeout=0.1)
                break
            except queue.Full:
                continue
        self._raise_if_failed()

    def close_and_wait(self) -> None:
        if self._thread.is_alive():
            while True:
                try:
                    self._queue.put(self._STOP, timeout=0.1)
                    break
                except queue.Full:
                    self._raise_if_failed()
                    continue
        self._thread.join()
        self._raise_if_failed()

    def commit(self) -> None:
        self.writer.commit()

    def abort(self) -> None:
        self._abort.set()
        if self._thread.is_alive():
            try:
                self._queue.put_nowait(self._STOP)
            except queue.Full:
                pass
            self._thread.join()
        self.writer.abort()

    def _run(self) -> None:
        try:
            while True:
                item = self._queue.get()
                if item is self._STOP or self._abort.is_set():
                    return
                name, tensor = item
                t0 = time.perf_counter()
                self.writer.write_tensor(name, tensor)
                self.write_s += time.perf_counter() - t0
                del tensor
        except BaseException as exc:
            self._set_failure(exc)


def apply_xor_patch(
    base_tensor: torch.Tensor,
    compressed_patch: torch.Tensor,
    expected_checksum: str,
    expected_num_bytes: Optional[int] = None,
    checksum_algorithm: str = _DEFAULT_CHECKSUM_ALGORITHM,
) -> torch.Tensor:
    compressed = uint8_tensor_to_bytes(compressed_patch)
    patch_bytes = decompress_bytes(compressed, expected_num_bytes)
    base_bytes = tensor_to_bytes_tensor(base_tensor)
    patch = bytes_to_uint8_tensor(patch_bytes)
    if patch.numel() != base_bytes.numel():
        raise ValueError(f"Patch has {patch.numel()} bytes, expected {base_bytes.numel()}")
    updated = torch.bitwise_xor(base_bytes, patch)
    if _checksum(uint8_tensor_to_bytes(updated), checksum_algorithm) != expected_checksum:
        raise ValueError("Post-apply tensor checksum mismatch")
    return bytes_tensor_to_tensor(updated, base_tensor.dtype, list(base_tensor.shape)).clone()


class PayloadReader:
    def __init__(self, payload_dir: Path) -> None:
        self.payload_dir = payload_dir
        self._files: dict[str, object] = {}

    def get(self, record: DeltaTensorRecord) -> torch.Tensor:
        handle = self._files.get(record.payload_file)
        if handle is None:
            handle = safe_open(self.payload_dir / record.payload_file, framework="pt", device="cpu")
            handle.__enter__()
            self._files[record.payload_file] = handle
        return handle.get_tensor(record.payload_key)

    def close(self) -> None:
        for handle in self._files.values():
            handle.__exit__(None, None, None)
        self._files.clear()


class PlanType(str, Enum):
    Apply = "apply"
    ApplyAndPersist = "apply_and_persist"


@dataclass
class DeltaReceivePlan:
    plan_type: PlanType
    manifest: DeltaManifest
    base_dir: Path
    payload_dir: Path
    direct_dir: Optional[Path] = None
    writer: Optional[AsyncCheckpointWriter] = None
    lock: Optional[FileLock] = None
    stats: dict[str, float] = field(default_factory=dict)

    @property
    def persist(self) -> bool:
        return self.plan_type == PlanType.ApplyAndPersist


class DeltaCheckpointIterator:
    def __init__(self, plan: DeltaReceivePlan) -> None:
        self.plan = plan
        self._reader: PayloadReader | None = None

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor]]:
        if self.plan.direct_dir is not None:
            yield from CheckpointIndex(self.plan.direct_dir).iter_tensors()
            return

        self._reader = PayloadReader(self.plan.payload_dir)
        index = CheckpointIndex(self.plan.base_dir)
        records_by_resolved_name = {}
        for record in self.plan.manifest.tensors:
            try:
                records_by_resolved_name[index.resolve_tensor_name(record.name)] = record
            except KeyError:
                logger.warning(
                    "Skipping delta tensor %s because it is not present in checkpoint %s",
                    record.name,
                    self.plan.base_dir,
                )
        try:
            for name, base_tensor in index.iter_tensors():
                record = records_by_resolved_name.get(name)
                if record is None:
                    yield name, base_tensor
                    continue

                patch = self._reader.get(record)
                updated = apply_xor_patch(
                    base_tensor,
                    patch,
                    record.checksum,
                    record.uncompressed_num_bytes,
                    record.checksum_algorithm,
                )
                if plan_writer := self.plan.writer:
                    plan_writer.enqueue(name, updated)
                yield name, updated
        finally:
            self._reader.close()


def _copy_checkpoint_tree_for_mutation(src_dir: Path, dst_dir: Path) -> None:
    """Copy a checkpoint tree into a mutable local directory.

    Reflinks are safe here because they are copy-on-write. Hardlinks are not
    used: in-place mmap patching must never mutate the original base checkpoint.
    """
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=False)
    for src in src_dir.rglob("*"):
        rel = src.relative_to(src_dir)
        dst = dst_dir / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        else:
            _copy_file_reflink(src, dst)


@dataclass
class CheckpointState:
    version: int
    write_in_progress: bool = False
    target_version: Optional[int] = None
    base_model_path: str = ""

    @classmethod
    def from_json(cls, data: dict) -> "CheckpointState":
        return cls(
            version=int(data.get("version", 0)),
            write_in_progress=bool(data.get("write_in_progress", False)),
            target_version=(int(data["target_version"]) if data.get("target_version") is not None else None),
            base_model_path=str(data.get("base_model_path", "")),
        )

    def to_json(self) -> dict:
        return {
            "version": self.version,
            "write_in_progress": self.write_in_progress,
            "target_version": self.target_version,
            "base_model_path": self.base_model_path,
        }


@dataclass(frozen=True)
class TensorLocation:
    name: str
    path: Path
    offset: int
    nbytes: int
    shape: list[int]
    dtype: str


def _optional_torch_dtype(name: str) -> Optional[torch.dtype]:
    dtype = getattr(torch, name, None)
    return dtype if isinstance(dtype, torch.dtype) else None


_SAFETENSORS_DTYPE_TO_TORCH: dict[str, torch.dtype] = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}

for _safetensors_name, _torch_name in (
    ("U16", "uint16"),
    ("U32", "uint32"),
    ("U64", "uint64"),
    ("F8_E4M3", "float8_e4m3fn"),
    ("F8_E5M2", "float8_e5m2"),
):
    if _dtype := _optional_torch_dtype(_torch_name):
        _SAFETENSORS_DTYPE_TO_TORCH[_safetensors_name] = _dtype


def _safetensors_dtype_to_torch(dtype: str) -> torch.dtype:
    try:
        return _SAFETENSORS_DTYPE_TO_TORCH[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported safetensors dtype {dtype!r}") from exc


def _torch_dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _checksum(data: object, algorithm: str = _DEFAULT_CHECKSUM_ALGORITHM) -> str:
    if algorithm == "xxh3-128":
        import xxhash

        hasher = xxhash.xxh3_128()
        hasher.update(data)
        return hasher.hexdigest()
    if algorithm == "sha256":
        return hashlib.sha256(data).hexdigest()
    raise ValueError(f"Unsupported delta checkpoint checksum algorithm {algorithm!r}")


def _tensor_locations(checkpoint_dir: Path) -> dict[str, TensorLocation]:
    locations: dict[str, TensorLocation] = {}
    for path in sorted(checkpoint_dir.glob("*.safetensors")):
        data_start, header = _read_safetensors_header(path)
        for name, info in header.items():
            if name == "__metadata__":
                continue
            begin, end = info["data_offsets"]
            locations[name] = TensorLocation(
                name=name,
                path=path,
                offset=data_start + int(begin),
                nbytes=int(end) - int(begin),
                shape=list(info.get("shape", [])),
                dtype=str(info.get("dtype", "")),
            )
    if not locations:
        raise ValueError(f"No safetensors tensors found under {checkpoint_dir}")
    return locations


class LocalCheckpointStore:
    """Mutable host-local checkpoint used by fetch-before-pause delta sync."""

    def __init__(self, base_model_path: str, local_checkpoint_dir: str) -> None:
        self.base_model_path = base_model_path
        self.root = Path(local_checkpoint_dir)
        self.weights_dir = _weights_dir(self.root)
        self.deltas_dir = _deltas_dir(self.root)
        self.state_dir = self.root / _STATE_DIR_NAME
        self.state_path = self.state_dir / "state.json"
        self.lock_path = self.state_dir / "writer.lock"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.deltas_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_initialized()

    def _read_state(self) -> Optional[CheckpointState]:
        if not self.state_path.exists():
            return None
        with self.state_path.open("r", encoding="utf-8") as f:
            return CheckpointState.from_json(json.load(f))

    def _write_state(self, state: CheckpointState) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with self.state_path.open("w", encoding="utf-8") as f:
            json.dump(state.to_json(), f)
            f.flush()

    def _reset_from_base(self) -> None:
        shutil.rmtree(self.weights_dir, ignore_errors=True)
        self._write_state(
            CheckpointState(
                version=0,
                write_in_progress=False,
                target_version=None,
                base_model_path=self.base_model_path,
            )
        )

    def _ensure_initialized(self) -> None:
        with FileLock(self.lock_path):
            state = self._read_state()
            if state is not None:
                return
            self._reset_from_base()

    def _current_checkpoint_dir(self) -> Path:
        if self.weights_dir.exists():
            return self.weights_dir
        return resolve_checkpoint_path(self.base_model_path)

    def _ensure_mutable_weights_dir(self) -> None:
        if self.weights_dir.exists():
            return
        _copy_checkpoint_tree_for_mutation(resolve_checkpoint_path(self.base_model_path), self.weights_dir)

    def _delta_uri_for_version(self, target_version: int, sync_dir: Optional[str], uri: Optional[str]) -> str:
        if uri is not None:
            return uri
        if sync_dir is None:
            raise ValueError("sync_dir is required when uri is not provided")
        return _join_uri(sync_dir, _version_name(target_version))

    def fetch(self, target_version: int, sync_dir: Optional[str] = None, uri: Optional[str] = None) -> dict[str, float]:
        stats = {"fetch_s": 0.0, "apply_s": 0.0, "reset_s": 0.0}
        with FileLock(self.lock_path):
            state = self._read_state()
            if state is None or state.write_in_progress:
                t_reset = time.perf_counter()
                self._reset_from_base()
                stats["reset_s"] += time.perf_counter() - t_reset
                state = self._read_state()
                assert state is not None

            if state.version == target_version:
                return stats
            if state.version > target_version:
                raise ValueError(
                    f"Local checkpoint version {state.version} is ahead of requested target_version {target_version}"
                )

            start_version = state.version
            self._write_state(
                CheckpointState(
                    version=start_version,
                    write_in_progress=True,
                    target_version=target_version,
                    base_model_path=self.base_model_path,
                )
            )
            try:
                current_version = start_version
                for version in range(start_version + 1, target_version + 1):
                    delta_uri = self._delta_uri_for_version(
                        version,
                        sync_dir,
                        uri if version == target_version else None,
                    )
                    cache_dir = self.deltas_dir / _safe_path_name(delta_uri)
                    t_fetch = time.perf_counter()
                    manifest, payload_dir = fetch_delta_directory(delta_uri, cache_dir)
                    stats["fetch_s"] += time.perf_counter() - t_fetch
                    if manifest.version != version:
                        raise ValueError(
                            f"Manifest version {manifest.version} does not match expected version {version}"
                        )
                    if manifest.base_version != current_version:
                        raise ValueError(
                            f"Manifest base_version {manifest.base_version} does not match local version "
                            f"{current_version}"
                        )
                    t_apply = time.perf_counter()
                    self._apply_delta_manifest(manifest, payload_dir)
                    stats["apply_s"] += time.perf_counter() - t_apply
                    current_version = version

                self._write_state(
                    CheckpointState(
                        version=target_version,
                        write_in_progress=False,
                        target_version=None,
                        base_model_path=self.base_model_path,
                    )
                )
            except Exception:
                self._write_state(
                    CheckpointState(
                        version=start_version,
                        write_in_progress=True,
                        target_version=target_version,
                        base_model_path=self.base_model_path,
                    )
                )
                raise
        return stats

    def _apply_delta_manifest(self, manifest: DeltaManifest, payload_dir: Path) -> None:
        if not manifest.tensors:
            return

        self._ensure_mutable_weights_dir()
        locations = _tensor_locations(self.weights_dir)
        index = CheckpointIndex(self.weights_dir)
        reader = PayloadReader(payload_dir)
        payloads: list[tuple[DeltaTensorRecord, str, bytes]] = []
        skipped: list[str] = []
        try:
            for record in manifest.tensors:
                try:
                    resolved_name = index.resolve_tensor_name(record.name)
                except KeyError:
                    skipped.append(record.name)
                    continue
                if resolved_name not in locations:
                    raise KeyError(f"Tensor {record.name!r} resolved to {resolved_name!r}, but no byte location exists")
                compressed_tensor = reader.get(record)
                payloads.append((record, resolved_name, uint8_tensor_to_bytes(compressed_tensor)))
        finally:
            reader.close()
        if skipped:
            logger.warning(
                "Skipped %s delta tensor(s) absent from local checkpoint %s: %s",
                len(skipped),
                self.weights_dir,
                skipped[:10],
            )

        mmaps: dict[Path, tuple[object, mmap.mmap]] = {}
        mismatches: list[str] = []
        mismatch_lock = threading.Lock()
        try:
            for _, resolved_name, _ in payloads:
                path = locations[resolved_name].path
                if path not in mmaps:
                    fh = path.open("r+b")
                    mm = mmap.mmap(fh.fileno(), 0)
                    try:
                        mm.madvise(mmap.MADV_WILLNEED)
                    except (AttributeError, OSError, ValueError):
                        pass
                    mmaps[path] = (fh, mm)

            def apply_one(item: tuple[DeltaTensorRecord, str, bytes]) -> None:
                record, resolved_name, compressed = item
                loc = locations[resolved_name]
                patch_bytes = decompress_bytes(compressed, record.uncompressed_num_bytes)
                if len(patch_bytes) != loc.nbytes:
                    raise ValueError(f"Patch for {record.name!r} has {len(patch_bytes)} bytes, expected {loc.nbytes}")
                mm = mmaps[loc.path][1]
                region = np.ndarray((loc.nbytes,), dtype=np.uint8, buffer=mm, offset=loc.offset)
                patch = np.frombuffer(patch_bytes, dtype=np.uint8)
                region ^= patch
                digest = _checksum(memoryview(region), record.checksum_algorithm)
                if digest != record.checksum:
                    with mismatch_lock:
                        mismatches.append(record.name)
                del region, patch

            workers = min(len(payloads), max(1, min(32, os.cpu_count() or 8)))
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="skyrl-delta-mmap-apply") as executor:
                list(executor.map(apply_one, payloads))
        finally:
            for fh, mm in mmaps.values():
                mm.close()
                fh.close()

        if mismatches:
            raise RuntimeError(
                f"Post-apply checksum mismatch for {len(mismatches)} tensor(s): {sorted(mismatches)[:20]}"
            )

    def validate_ready(self, target_version: int) -> None:
        state = self._read_state()
        if state is None:
            raise RuntimeError("Local checkpoint state is missing")
        if state.write_in_progress:
            raise RuntimeError(
                f"Local checkpoint has write_in_progress=True for target_version={state.target_version}; "
                "run fetch_weights before reload"
            )
        if state.version != target_version:
            raise RuntimeError(
                f"Local checkpoint version {state.version} does not match target_version {target_version}"
            )
        self._current_checkpoint_dir()

    def iter_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        yield from CheckpointIndex(self._current_checkpoint_dir()).iter_tensors()


class LocalCheckpointDeltaManager:
    def __init__(
        self,
        base_model_path: str,
        local_checkpoint_dir: str,
        max_files_to_keep: int = 5,
        prefetch_depth: int = 0,
        version_wait_timeout_s: float = 7200.0,
    ) -> None:
        self.base_model_path = base_model_path
        self.root = Path(local_checkpoint_dir)
        self.max_files_to_keep = max_files_to_keep
        self.prefetch_depth = prefetch_depth
        self.version_wait_timeout_s = version_wait_timeout_s
        ensure_base_version(self.root, base_model_path)
        (self.root / _STATE_DIR_NAME / "artifacts").mkdir(parents=True, exist_ok=True)

    def _completed_versions(self) -> list[int]:
        versions_dir = self.root / "versions"
        if not versions_dir.exists():
            return []
        versions = []
        for path in versions_dir.iterdir():
            if path.is_dir() and path.name.startswith("v") and not path.name.endswith(".staging"):
                try:
                    versions.append(int(path.name[1:]))
                except ValueError:
                    pass
        return sorted(versions)

    def _cleanup_old_versions(self, base_version: int, target_version: int) -> None:
        versions = self._completed_versions()
        protected = {0, base_version, target_version}
        excess = max(0, len(versions) - self.max_files_to_keep)
        for candidate in [v for v in versions if v not in protected][:excess]:
            shutil.rmtree(_version_dir(self.root, candidate), ignore_errors=True)

    def _wait_for_version(self, version: int, timeout_s: Optional[float] = None) -> Path:
        target_dir = _version_dir(self.root, version)
        wait_timeout_s = self.version_wait_timeout_s if timeout_s is None else timeout_s
        deadline = time.perf_counter() + wait_timeout_s
        while time.perf_counter() < deadline:
            if target_dir.exists():
                return target_dir
            time.sleep(0.1)
        raise TimeoutError(f"Timed out waiting for checkpoint version {version} at {target_dir}")

    def _apply_delta_to_writer(
        self,
        manifest: DeltaManifest,
        base_dir: Path,
        payload_dir: Path,
        writer: CheckpointVersionWriter,
    ) -> None:
        index = CheckpointIndex(base_dir)
        reader = PayloadReader(payload_dir)

        def build_updated_tensor(record: DeltaTensorRecord) -> tuple[str, torch.Tensor]:
            base_tensor = index.load_tensor(record.name)
            patch = reader.get(record)
            updated = apply_xor_patch(
                base_tensor,
                patch,
                record.checksum,
                record.uncompressed_num_bytes,
                record.checksum_algorithm,
            )
            del base_tensor, patch
            return record.name, updated

        try:
            for name, updated in _ordered_prefetch(manifest.tensors, build_updated_tensor, self.prefetch_depth):
                writer.write_tensor(name, updated)
                del updated
        finally:
            reader.close()

    def prepare(self, uri: str, expected_version: Optional[int] = None) -> DeltaReceivePlan:
        cache_dir = self.root / _STATE_DIR_NAME / "artifacts" / _safe_path_name(uri)
        t_fetch = time.perf_counter()
        manifest, payload_dir = fetch_delta_directory(uri, cache_dir)
        fetch_s = time.perf_counter() - t_fetch
        if expected_version is not None and manifest.version != expected_version:
            raise ValueError(f"Manifest version {manifest.version} does not match update version {expected_version}")
        if manifest.base_version != manifest.version - 1:
            raise ValueError(
                f"Unsupported delta version relationship: version={manifest.version}, "
                f"base_version={manifest.base_version}"
            )
        base_dir = _version_dir(self.root, manifest.base_version)
        target_dir = _version_dir(self.root, manifest.version)
        staging_dir = _staging_dir(self.root, manifest.version)
        if not base_dir.exists():
            raise FileNotFoundError(f"Base checkpoint version {manifest.base_version} missing at {base_dir}")

        lock = FileLock(self.root / _STATE_DIR_NAME / "writer.lock")
        if staging_dir.exists():
            if lock.acquire(blocking=False):
                lock.release()
                raise RuntimeError(f"Found stale staging checkpoint from failed sync: {staging_dir}")
            return DeltaReceivePlan(
                plan_type=PlanType.Apply,
                manifest=manifest,
                base_dir=base_dir,
                payload_dir=payload_dir,
                stats={"fetch_s": fetch_s, "materialize_s": 0.0, "wait_s": 0.0, "commit_s": 0.0},
            )

        if lock.acquire(blocking=False):
            writer: AsyncCheckpointWriter | None = None
            try:
                if target_dir.exists():
                    lock.release()
                    return DeltaReceivePlan(
                        plan_type=PlanType.Apply,
                        manifest=manifest,
                        base_dir=base_dir,
                        payload_dir=payload_dir,
                        stats={"fetch_s": fetch_s, "materialize_s": 0.0, "wait_s": 0.0, "commit_s": 0.0},
                    )
                else:
                    self._cleanup_old_versions(manifest.base_version, manifest.version)
                    writer = AsyncCheckpointWriter(
                        LazyCheckpointVersionWriter(
                            self.root,
                            base_version=manifest.base_version,
                            target_version=manifest.version,
                            changed_names={record.name for record in manifest.tensors},
                        ),
                        max_queue_size=max(_DEFAULT_WRITER_QUEUE_SIZE, self.prefetch_depth),
                    )
                return DeltaReceivePlan(
                    plan_type=PlanType.ApplyAndPersist,
                    manifest=manifest,
                    base_dir=base_dir,
                    payload_dir=payload_dir,
                    writer=writer,
                    lock=lock,
                    stats={
                        "fetch_s": fetch_s,
                        "materialize_s": 0.0,
                        "wait_s": 0.0,
                        "commit_s": 0.0,
                    },
                )
            except Exception:
                if writer is not None:
                    writer.abort()
                lock.release()
                raise

        return DeltaReceivePlan(
            plan_type=PlanType.Apply,
            manifest=manifest,
            base_dir=base_dir,
            payload_dir=payload_dir,
            stats={"fetch_s": fetch_s, "materialize_s": 0.0, "wait_s": 0.0, "commit_s": 0.0},
        )

    def prepare_materialize_then_reload(
        self,
        uri: str,
        expected_version: Optional[int] = None,
    ) -> DeltaReceivePlan:
        cache_dir = self.root / _STATE_DIR_NAME / "artifacts" / _safe_path_name(uri)
        t_fetch = time.perf_counter()
        manifest, payload_dir = fetch_delta_directory(uri, cache_dir)
        fetch_s = time.perf_counter() - t_fetch
        if expected_version is not None and manifest.version != expected_version:
            raise ValueError(f"Manifest version {manifest.version} does not match update version {expected_version}")
        if manifest.base_version != manifest.version - 1:
            raise ValueError(
                f"Unsupported delta version relationship: version={manifest.version}, "
                f"base_version={manifest.base_version}"
            )

        base_dir = _version_dir(self.root, manifest.base_version)
        target_dir = _version_dir(self.root, manifest.version)
        staging_dir = _staging_dir(self.root, manifest.version)
        if not base_dir.exists():
            raise FileNotFoundError(f"Base checkpoint version {manifest.base_version} missing at {base_dir}")

        stats = {"fetch_s": fetch_s, "materialize_s": 0.0, "wait_s": 0.0, "commit_s": 0.0}
        lock = FileLock(self.root / _STATE_DIR_NAME / "writer.lock")
        if target_dir.exists():
            return DeltaReceivePlan(
                plan_type=PlanType.Apply,
                manifest=manifest,
                base_dir=base_dir,
                payload_dir=payload_dir,
                direct_dir=target_dir,
                stats=stats,
            )

        if lock.acquire(blocking=False):
            writer: CheckpointVersionWriter | None = None
            try:
                if target_dir.exists():
                    return DeltaReceivePlan(
                        plan_type=PlanType.Apply,
                        manifest=manifest,
                        base_dir=base_dir,
                        payload_dir=payload_dir,
                        direct_dir=target_dir,
                        stats=stats,
                    )
                if staging_dir.exists():
                    raise RuntimeError(f"Found stale staging checkpoint from failed sync: {staging_dir}")

                self._cleanup_old_versions(manifest.base_version, manifest.version)
                writer = CheckpointVersionWriter(
                    self.root,
                    base_version=manifest.base_version,
                    target_version=manifest.version,
                    changed_names={record.name for record in manifest.tensors},
                )
                t_materialize = time.perf_counter()
                self._apply_delta_to_writer(manifest, base_dir, payload_dir, writer)
                stats["materialize_s"] = time.perf_counter() - t_materialize
                t_commit = time.perf_counter()
                writer.commit()
                stats["commit_s"] = time.perf_counter() - t_commit
            except Exception:
                if writer is not None:
                    writer.abort()
                raise
            finally:
                lock.release()
        else:
            t_wait = time.perf_counter()
            target_dir = self._wait_for_version(manifest.version)
            stats["wait_s"] = time.perf_counter() - t_wait

        return DeltaReceivePlan(
            plan_type=PlanType.Apply,
            manifest=manifest,
            base_dir=base_dir,
            payload_dir=payload_dir,
            direct_dir=target_dir,
            stats=stats,
        )

    def iterator(self, plan: DeltaReceivePlan) -> DeltaCheckpointIterator:
        return DeltaCheckpointIterator(plan)

    def complete(self, plan: DeltaReceivePlan) -> None:
        if plan.writer is None:
            return
        try:
            t_wait = time.perf_counter()
            plan.writer.close_and_wait()
            plan.stats["wait_s"] = time.perf_counter() - t_wait
            plan.stats["materialize_s"] = plan.writer.write_s
            t_commit = time.perf_counter()
            plan.writer.commit()
            plan.stats["commit_s"] = time.perf_counter() - t_commit
        except Exception:
            plan.writer.abort()
            raise
        finally:
            if plan.lock is not None:
                plan.lock.release()
                plan.lock = None

    def abort(self, plan: DeltaReceivePlan) -> None:
        if plan.writer is not None:
            plan.writer.abort()
        if plan.lock is not None:
            plan.lock.release()
            plan.lock = None


@dataclass
class DeltaPublishResult:
    rank: int
    target_version: int
    base_version: int
    records: list[DeltaTensorRecord] = field(default_factory=list)
    payload_files: list[str] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)


@dataclass
class _TensorPublishResult:
    name: str
    dtype: str
    shape: list[int]
    uncompressed_num_bytes: int
    compressed: Optional[bytes]
    checksum: str
    checksum_algorithm: str
    changed_bytes: int
    timings: dict[str, float]


@dataclass
class _StagedTensorBytes:
    data: np.ndarray | torch.Tensor
    nbytes: int
    pinned: bool


class _PinnedStagingPool:
    def __init__(
        self, max_tensor_bytes: int, num_workers: int, byte_cap: int = _DEFAULT_PINNED_STAGING_BYTE_CAP
    ) -> None:
        self.enabled = False
        self.buffer_nbytes = int(max_tensor_bytes)
        self.num_buffers = 0
        self._free: queue.Queue[torch.Tensor] = queue.Queue()
        if self.buffer_nbytes <= 0 or not torch.cuda.is_available():
            return
        max_buffers_by_cap = byte_cap // self.buffer_nbytes
        if max_buffers_by_cap <= 0:
            logger.warning(
                "Pinned delta staging disabled because largest tensor is %.3f GiB, above byte cap %.3f GiB",
                self.buffer_nbytes / 1024**3,
                byte_cap / 1024**3,
            )
            return
        num_buffers = max(1, min(max(1, num_workers), max_buffers_by_cap))
        try:
            for _ in range(num_buffers):
                self._free.put(torch.empty(self.buffer_nbytes, dtype=torch.uint8, device="cpu", pin_memory=True))
        except RuntimeError as exc:
            logger.warning("Pinned delta staging buffers unavailable (%s); using pageable CPU staging", exc)
            while not self._free.empty():
                self._free.get_nowait()
            return
        self.enabled = True
        self.num_buffers = num_buffers
        logger.info(
            "Initialized pinned delta staging pool: buffers=%s buffer_nbytes=%s total_nbytes=%s",
            num_buffers,
            self.buffer_nbytes,
            num_buffers * self.buffer_nbytes,
        )

    @property
    def total_nbytes(self) -> int:
        return self.buffer_nbytes * self.num_buffers

    def acquire(self, nbytes: int) -> Optional[torch.Tensor]:
        if not self.enabled or nbytes > self.buffer_nbytes:
            return None
        return self._free.get()

    def release(self, buffer: torch.Tensor) -> None:
        if self.enabled:
            self._free.put(buffer)


class DeltaCheckpointPublisher:
    def __init__(
        self,
        base_model_path: str,
        sync_dir: str,
        local_checkpoint_dir: str,
        max_file_size_in_gb: float = 1.0,
        max_files_to_keep: int = 5,
    ) -> None:
        self.base_model_path = base_model_path
        self.sync_dir = sync_dir
        self.root = Path(local_checkpoint_dir)
        self.max_file_size_bytes = int(max_file_size_in_gb * 1024**3)
        self.max_files_to_keep = max_files_to_keep
        self.checksum_algorithm = _DEFAULT_CHECKSUM_ALGORITHM
        self.version = 0
        self.snapshot: dict[str, np.ndarray] = {}
        self._base_checkpoint_dir: Optional[Path] = None
        self._base_locations: Optional[dict[str, TensorLocation]] = None
        self._publish_executor: Optional[ThreadPoolExecutor] = None
        self._publish_executor_workers: Optional[int] = None
        self._staging_pool: Optional[_PinnedStagingPool] = None
        self._staging_pool_signature: Optional[tuple[int, int]] = None

    def _default_shard_info(self) -> ExtractorShardInfo:
        from skyrl.backends.skyrl_train.distributed.dispatch import MeshRank

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1
        return ExtractorShardInfo(
            is_source_rank=(rank == 0),
            replicate=[],
            split=[],
            mesh_rank=MeshRank(dp=rank, sp=0, tp=0, pp=0, world_size=world_size, dp_size=world_size, pp_size=1),
            replicate_world_size=1,
            source_index_in_replicate_world=0,
            rank=rank,
        )

    def _ensure_base_locations(self) -> dict[str, TensorLocation]:
        if self._base_locations is None:
            t0 = time.perf_counter()
            self._base_checkpoint_dir = resolve_checkpoint_path(self.base_model_path)
            self._base_locations = _tensor_locations(self._base_checkpoint_dir)
            logger.info(
                "delta checkpoint publisher initialized lazy base byte index: tensors=%s init_s=%.3f "
                "base_model_path=%s",
                len(self._base_locations),
                time.perf_counter() - t0,
                self.base_model_path,
            )
        return self._base_locations

    def _base_location(self, name: str) -> TensorLocation:
        locations = self._ensure_base_locations()
        if name not in locations:
            raise KeyError(f"Tensor {name!r} not found in base checkpoint {self._base_checkpoint_dir}")
        return locations[name]

    def _read_base_bytes(self, name: str) -> np.ndarray:
        loc = self._base_location(name)
        with loc.path.open("rb") as f:
            f.seek(loc.offset)
            data = f.read(loc.nbytes)
        if len(data) != loc.nbytes:
            raise ValueError(f"Read {len(data)} bytes for {name!r}, expected {loc.nbytes}")
        return np.frombuffer(data, dtype=np.uint8).copy()

    def _snapshot_bytes(self, name: str) -> np.ndarray:
        if name not in self.snapshot:
            self.snapshot[name] = self._read_base_bytes(name)
        return self.snapshot[name]

    @staticmethod
    def _stage_tensor_to_cpu_bytes(
        tensor: torch.Tensor,
        target_dtype: torch.dtype,
        target_shape: list[int],
    ) -> tuple[np.ndarray, str, list[int]]:
        current = tensor.detach()
        if list(current.shape) != target_shape:
            raise ValueError(f"Shape mismatch: current={list(current.shape)}, base={target_shape}")
        if current.dtype != target_dtype:
            current = current.to(dtype=target_dtype)
        current = current.cpu().contiguous()
        current_bytes = current.view(torch.uint8).reshape(-1).numpy().copy()
        dtype_name = _torch_dtype_name(current.dtype)
        shape = list(current.shape)
        del current
        return current_bytes, dtype_name, shape

    @staticmethod
    def _stage_tensor_for_publish(
        tensor: torch.Tensor,
        target_dtype: torch.dtype,
        target_shape: list[int],
        staging_pool: Optional[_PinnedStagingPool],
    ) -> tuple[_StagedTensorBytes, str, list[int]]:
        current = tensor.detach()
        if list(current.shape) != target_shape:
            raise ValueError(f"Shape mismatch: current={list(current.shape)}, base={target_shape}")
        if current.dtype != target_dtype:
            current = current.to(dtype=target_dtype)
        current = current.contiguous()
        dtype_name = _torch_dtype_name(current.dtype)
        shape = list(current.shape)
        flat = current.view(torch.uint8).reshape(-1)
        nbytes = int(flat.numel())
        if current.is_cuda and staging_pool is not None:
            buffer = staging_pool.acquire(nbytes)
            if buffer is not None:
                buffer[:nbytes].copy_(flat, non_blocking=True)
                torch.cuda.current_stream(current.device).synchronize()
                del current, flat
                return _StagedTensorBytes(buffer, nbytes, pinned=True), dtype_name, shape

        current_cpu = current.cpu().contiguous()
        current_bytes = current_cpu.view(torch.uint8).reshape(-1).numpy().copy()
        del current, flat, current_cpu
        return _StagedTensorBytes(current_bytes, current_bytes.nbytes, pinned=False), dtype_name, shape

    @staticmethod
    def _process_tensor_delta(
        name: str,
        staged: _StagedTensorBytes,
        base: np.ndarray,
        dtype: str,
        shape: list[int],
        staging_pool: Optional[_PinnedStagingPool],
        checksum_algorithm: str,
    ):
        timings: dict[str, float] = {}
        if staged.pinned:
            assert isinstance(staged.data, torch.Tensor)
            t = time.perf_counter()
            try:
                current = np.empty(staged.nbytes, dtype=np.uint8)
                np.copyto(current, staged.data[: staged.nbytes].numpy())
            finally:
                if staging_pool is not None:
                    staging_pool.release(staged.data)
            timings["pinned_to_numpy_s"] = time.perf_counter() - t
        else:
            assert isinstance(staged.data, np.ndarray)
            current = staged.data
            timings["pinned_to_numpy_s"] = 0.0

        if current.nbytes != base.nbytes:
            raise ValueError(f"Byte-size mismatch for {name}: current={current.nbytes}, base={base.nbytes}")

        t = time.perf_counter()
        checksum = _checksum(memoryview(current), checksum_algorithm)
        timings["checksum_s"] = time.perf_counter() - t

        t = time.perf_counter()
        np.bitwise_xor(current, base, out=current)
        timings["xor_s"] = time.perf_counter() - t

        t = time.perf_counter()
        changed_bytes = int(np.count_nonzero(current))
        timings["changed_scan_s"] = time.perf_counter() - t

        compressed = None
        if changed_bytes:
            t = time.perf_counter()
            compressed = compress_bytes(memoryview(current), level=1)
            timings["compress_s"] = time.perf_counter() - t
        else:
            timings["compress_s"] = 0.0

        # ``current`` is the XOR patch at this point. Apply it to the existing
        # snapshot in-place so repeated publishes do not replace every
        # full-model CPU snapshot allocation.
        t = time.perf_counter()
        np.bitwise_xor(base, current, out=base)
        timings["xor_restore_s"] = time.perf_counter() - t

        return _TensorPublishResult(
            name=name,
            dtype=dtype,
            shape=shape,
            uncompressed_num_bytes=int(base.nbytes),
            compressed=compressed,
            checksum=checksum,
            checksum_algorithm=checksum_algorithm,
            changed_bytes=changed_bytes,
            timings=timings,
        )

    @staticmethod
    def _empty_stats() -> dict[str, float]:
        return {
            "extract_or_gather_s": 0.0,
            "stage_to_cpu_s": 0.0,
            "snapshot_read_s": 0.0,
            "pinned_to_numpy_s": 0.0,
            "xor_s": 0.0,
            "changed_scan_s": 0.0,
            "compress_s": 0.0,
            "checksum_s": 0.0,
            "xor_restore_s": 0.0,
            "snapshot_update_s": 0.0,
            "payload_write_s": 0.0,
            "upload_s": 0.0,
            "publish_s": 0.0,
            "processed_tensors": 0.0,
            "skipped_tensors": 0.0,
            "changed_bytes": 0.0,
            "uncompressed_bytes": 0.0,
            "compressed_bytes": 0.0,
            "records": 0.0,
            "source_rank": 0.0,
        }

    @staticmethod
    def _num_publish_workers() -> int:
        default = min(8, os.cpu_count() or 1)
        raw = os.environ.get("SKYRL_DELTA_PUBLISH_NUM_WORKERS")
        if raw is None:
            return default
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"SKYRL_DELTA_PUBLISH_NUM_WORKERS must be an integer, got {raw!r}") from exc
        if value < 1:
            raise ValueError(f"SKYRL_DELTA_PUBLISH_NUM_WORKERS must be >= 1, got {value}")
        return min(value, default)

    def _publish_executor_for(self, num_workers: int) -> ThreadPoolExecutor:
        if self._publish_executor is None:
            self._publish_executor = ThreadPoolExecutor(
                max_workers=num_workers,
                thread_name_prefix="skyrl-delta-publish",
            )
            self._publish_executor_workers = num_workers
        elif self._publish_executor_workers != num_workers:
            logger.warning(
                "delta checkpoint publisher keeping existing persistent executor with %s workers; "
                "requested %s workers",
                self._publish_executor_workers,
                num_workers,
            )
        return self._publish_executor

    @staticmethod
    def _trim_publish_workers(executor: ThreadPoolExecutor, num_workers: int) -> int:
        futures = [executor.submit(trim_process_memory) for _ in range(num_workers)]
        return sum(1 for future in futures if future.result())

    def _staging_pool_for(self, max_tensor_bytes: int, num_workers: int) -> _PinnedStagingPool:
        signature = (max_tensor_bytes, num_workers)
        if self._staging_pool is None or self._staging_pool_signature != signature:
            self._staging_pool = _PinnedStagingPool(max_tensor_bytes, num_workers)
            self._staging_pool_signature = signature
        return self._staging_pool

    def _publish_local(
        self,
        chunks: Iterable[WeightChunk],
        extractor_shard_info: ExtractorShardInfo,
    ) -> DeltaPublishResult:
        base_version = self.version
        target_version = base_version + 1
        publish_uri = _join_uri(self.sync_dir, _version_name(target_version))
        rank = extractor_shard_info.rank
        stats = self._empty_stats()
        records: list[DeltaTensorRecord] = []
        payload_files: list[str] = []
        payload_tensors: dict[str, torch.Tensor] = {}
        payload_file_idx = 0
        payload_file_bytes = 0
        pending: deque[Future[_TensorPublishResult]] = deque()
        skipped_names: set[str] = set()
        t_publish = time.perf_counter()
        stats["source_rank"] = 1.0 if extractor_shard_info.is_source_rank else 0.0
        try:
            memory_log_interval = int(os.environ.get("SKYRL_DELTA_MEMORY_LOG_INTERVAL_TENSORS", "5000"))
        except ValueError:
            memory_log_interval = 5000
        memory_log_interval = max(0, memory_log_interval)

        def snapshot_bytes_total() -> int:
            return sum(int(value.nbytes) for value in self.snapshot.values())

        def current_payload_file() -> str:
            return f"payload-rank{rank:06d}-file{payload_file_idx:06d}.safetensors"

        def flush_payload_file(local_publish_dir: Path) -> None:
            nonlocal payload_tensors, payload_file_idx, payload_file_bytes
            if not payload_tensors:
                return
            t = time.perf_counter()
            file_name = current_payload_file()
            save_file(payload_tensors, str(local_publish_dir / file_name))
            stats["payload_write_s"] += time.perf_counter() - t
            payload_files.append(file_name)
            payload_tensors = {}
            payload_file_idx += 1
            payload_file_bytes = 0
            log_memory(
                logger,
                "delta_publisher_after_payload_flush",
                rank=rank,
                target_version=target_version,
                payload_files=len(payload_files),
                snapshot_bytes=snapshot_bytes_total(),
            )

        def consume_result(result: _TensorPublishResult, local_publish_dir: Path) -> None:
            nonlocal payload_file_bytes
            for key, value in result.timings.items():
                stats[key] = stats.get(key, 0.0) + value
            stats["processed_tensors"] += 1
            stats["changed_bytes"] += result.changed_bytes
            if result.compressed is None:
                return
            if payload_tensors and payload_file_bytes + len(result.compressed) > self.max_file_size_bytes:
                flush_payload_file(local_publish_dir)
            key = result.name
            payload_tensors[key] = bytes_to_uint8_tensor(result.compressed)
            payload_file_bytes += len(result.compressed)
            record = DeltaTensorRecord(
                name=result.name,
                dtype=result.dtype,
                shape=result.shape,
                uncompressed_num_bytes=result.uncompressed_num_bytes,
                compressed_num_bytes=len(result.compressed),
                payload_file=current_payload_file(),
                payload_key=key,
                checksum=result.checksum,
                checksum_algorithm=result.checksum_algorithm,
            )
            records.append(record)
            stats["uncompressed_bytes"] += record.uncompressed_num_bytes
            stats["compressed_bytes"] += record.compressed_num_bytes
            stats["records"] += 1
            if memory_log_interval and int(stats["processed_tensors"]) % memory_log_interval == 0:
                log_memory(
                    logger,
                    "delta_publisher_progress",
                    rank=rank,
                    target_version=target_version,
                    processed_tensors=int(stats["processed_tensors"]),
                    records=len(records),
                    payload_file_bytes=payload_file_bytes,
                    pending=len(pending),
                    snapshot_tensors=len(self.snapshot),
                    snapshot_bytes=snapshot_bytes_total(),
                    uncompressed_bytes=int(stats["uncompressed_bytes"]),
                    compressed_bytes=int(stats["compressed_bytes"]),
                )

        def drain_one(local_publish_dir: Path) -> None:
            result = pending.popleft().result()
            consume_result(result, local_publish_dir)

        try:
            with tempfile.TemporaryDirectory(prefix=f"skyrl-delta-publish-rank{rank}-") as tmpdir:
                local_publish_dir = Path(tmpdir)
                num_workers = self._num_publish_workers()
                max_inflight = max(1, num_workers)
                staging_pool: Optional[_PinnedStagingPool] = None
                log_memory(
                    logger,
                    "delta_publisher_start",
                    rank=rank,
                    target_version=target_version,
                    is_source_rank=extractor_shard_info.is_source_rank,
                    existing_snapshot_tensors=len(self.snapshot),
                    existing_snapshot_bytes=snapshot_bytes_total(),
                    num_workers=num_workers,
                    max_inflight=max_inflight,
                )
                if extractor_shard_info.is_source_rank:
                    locations = self._ensure_base_locations()
                    max_tensor_bytes = max((loc.nbytes for loc in locations.values()), default=0)
                    staging_pool = self._staging_pool_for(max_tensor_bytes, num_workers)
                    log_memory(
                        logger,
                        "delta_publisher_after_staging_pool",
                        rank=rank,
                        target_version=target_version,
                        max_tensor_bytes=max_tensor_bytes,
                        staging_pool_enabled=bool(staging_pool and staging_pool.enabled),
                        staging_pool_buffer_nbytes=(staging_pool.buffer_nbytes if staging_pool else 0),
                        staging_pool_total_nbytes=(staging_pool.total_nbytes if staging_pool else 0),
                    )
                executor = self._publish_executor_for(num_workers) if extractor_shard_info.is_source_rank else None
                chunk_iter = iter(chunks)
                chunk_index = 0
                while True:
                    t = time.perf_counter()
                    try:
                        chunk = next(chunk_iter)
                    except StopIteration:
                        stats["extract_or_gather_s"] += time.perf_counter() - t
                        break
                    stats["extract_or_gather_s"] += time.perf_counter() - t
                    if not extractor_shard_info.is_source_rank:
                        chunk_index += 1
                        continue
                    for name, tensor in zip(chunk.names, chunk.tensors):
                        t = time.perf_counter()
                        try:
                            loc = self._base_location(name)
                            base = self._snapshot_bytes(name)
                        except KeyError:
                            skipped_names.add(name)
                            stats["skipped_tensors"] += 1
                            continue
                        stats["snapshot_read_s"] += time.perf_counter() - t

                        t = time.perf_counter()
                        staged, dtype_name, shape = self._stage_tensor_for_publish(
                            tensor,
                            target_dtype=_safetensors_dtype_to_torch(loc.dtype),
                            target_shape=loc.shape,
                            staging_pool=staging_pool,
                        )
                        stats["stage_to_cpu_s"] += time.perf_counter() - t
                        if executor is None:
                            raise RuntimeError("Delta checkpoint source rank is missing a publish executor")
                        pending.append(
                            executor.submit(
                                self._process_tensor_delta,
                                name,
                                staged,
                                base,
                                dtype_name,
                                shape,
                                staging_pool,
                                self.checksum_algorithm,
                            )
                        )
                        if len(pending) >= max_inflight:
                            drain_one(local_publish_dir)
                    chunk_index += 1
                while pending:
                    drain_one(local_publish_dir)
                trimmed_workers = self._trim_publish_workers(executor, num_workers) if executor is not None else 0
                trimmed = trim_process_memory()
                log_memory(
                    logger,
                    "delta_publisher_after_drain",
                    rank=rank,
                    target_version=target_version,
                    records=len(records),
                    payload_file_bytes=payload_file_bytes,
                    snapshot_tensors=len(self.snapshot),
                    snapshot_bytes=snapshot_bytes_total(),
                    memory_trimmed=trimmed,
                    publish_worker_trims=trimmed_workers,
                )
                flush_payload_file(local_publish_dir)
                trimmed = trim_process_memory()

                if skipped_names:
                    logger.warning(
                        "delta checkpoint publisher rank=%s skipped %s tensor(s) absent from base checkpoint: %s",
                        rank,
                        len(skipped_names),
                        sorted(skipped_names)[:10],
                    )

                if payload_files:
                    t = time.perf_counter()
                    log_memory(
                        logger,
                        "delta_publisher_before_payload_upload",
                        rank=rank,
                        target_version=target_version,
                        payload_files=len(payload_files),
                        snapshot_bytes=snapshot_bytes_total(),
                        memory_trimmed=trimmed,
                    )
                    publish_delta_payload_files(local_publish_dir, publish_uri)
                    stats["upload_s"] = time.perf_counter() - t
                    trimmed = trim_process_memory()
                    log_memory(
                        logger,
                        "delta_publisher_after_payload_upload",
                        rank=rank,
                        target_version=target_version,
                        payload_files=len(payload_files),
                        snapshot_bytes=snapshot_bytes_total(),
                        memory_trimmed=trimmed,
                    )
        except Exception:
            raise

        self.version = target_version
        stats["publish_s"] = time.perf_counter() - t_publish
        trimmed = trim_process_memory()
        log_memory(
            logger,
            "delta_publisher_done",
            rank=rank,
            target_version=target_version,
            records=len(records),
            payload_files=len(payload_files),
            snapshot_tensors=len(self.snapshot),
            snapshot_bytes=snapshot_bytes_total(),
            compressed_bytes=int(stats["compressed_bytes"]),
            uncompressed_bytes=int(stats["uncompressed_bytes"]),
            memory_trimmed=trimmed,
        )
        ratio = stats["compressed_bytes"] / stats["uncompressed_bytes"] if stats["uncompressed_bytes"] else 0.0
        logger.info(
            "delta checkpoint publish local: rank=%s version=%s base_version=%s tensors=%s payload_files=%s "
            "uncompressed_bytes=%s compressed_bytes=%s compression_ratio=%.6f publish_s=%.3f "
            "extract_or_gather_s=%.3f stage_to_cpu_s=%.3f snapshot_read_s=%.3f pinned_to_numpy_s=%.3f "
            "xor_s=%.3f scan_s=%.3f compress_s=%.3f checksum_s=%.3f xor_restore_s=%.3f "
            "payload_write_s=%.3f upload_s=%.3f",
            rank,
            target_version,
            base_version,
            len(records),
            len(payload_files),
            int(stats["uncompressed_bytes"]),
            int(stats["compressed_bytes"]),
            ratio,
            stats["publish_s"],
            stats["extract_or_gather_s"],
            stats["stage_to_cpu_s"],
            stats["snapshot_read_s"],
            stats["pinned_to_numpy_s"],
            stats["xor_s"],
            stats["changed_scan_s"],
            stats["compress_s"],
            stats["checksum_s"],
            stats["xor_restore_s"],
            stats["payload_write_s"],
            stats["upload_s"],
        )
        return DeltaPublishResult(
            rank=rank,
            target_version=target_version,
            base_version=base_version,
            records=records,
            payload_files=payload_files,
            stats=stats,
        )

    def finalize_publish(self, results: Sequence[DeltaPublishResult]) -> dict:
        if not results:
            raise ValueError("No delta publish results to finalize")
        target_version = results[0].target_version
        base_version = results[0].base_version
        publish_uri = _join_uri(self.sync_dir, _version_name(target_version))
        records: list[DeltaTensorRecord] = []
        payload_files: list[str] = []
        aggregate: dict[str, float] = {}
        for result in results:
            if result.target_version != target_version or result.base_version != base_version:
                raise ValueError(
                    "Mismatched delta publish versions: "
                    f"expected target={target_version}, base={base_version}; "
                    f"got target={result.target_version}, base={result.base_version} from rank={result.rank}"
                )
            records.extend(result.records)
            payload_files.extend(result.payload_files)
            for key, value in result.stats.items():
                aggregate[key] = aggregate.get(key, 0.0) + float(value)

        manifest = DeltaManifest(
            version=target_version,
            base_version=base_version,
            tensors=records,
            total_uncompressed_num_bytes=sum(record.uncompressed_num_bytes for record in records),
            total_compressed_num_bytes=sum(record.compressed_num_bytes for record in records),
            payload_files=payload_files,
        )
        with tempfile.TemporaryDirectory(prefix="skyrl-delta-manifest-") as tmpdir:
            local_publish_dir = Path(tmpdir)
            with (local_publish_dir / _MANIFEST_NAME).open("w", encoding="utf-8") as f:
                json.dump(manifest.to_json(), f)
            t = time.perf_counter()
            publish_delta_directory(local_publish_dir, publish_uri)
            manifest_upload_s = time.perf_counter() - t

        ratio = (
            manifest.total_compressed_num_bytes / manifest.total_uncompressed_num_bytes
            if manifest.total_uncompressed_num_bytes
            else 0.0
        )
        max_publish_s = max((result.stats.get("publish_s", 0.0) for result in results), default=0.0)
        sum_cpu_s = sum(
            result.stats.get("pinned_to_numpy_s", 0.0)
            + result.stats.get("xor_s", 0.0)
            + result.stats.get("changed_scan_s", 0.0)
            + result.stats.get("compress_s", 0.0)
            + result.stats.get("checksum_s", 0.0)
            + result.stats.get("xor_restore_s", 0.0)
            for result in results
        )
        logger.info(
            "delta checkpoint publish finalized: version=%s base_version=%s source_ranks=%s tensors=%s "
            "payload_files=%s uncompressed_bytes=%s compressed_bytes=%s compression_ratio=%.6f "
            "max_per_rank_publish_s=%.3f sum_per_rank_cpu_s=%.3f manifest_upload_s=%.3f "
            "extract_or_gather_s=%.3f stage_to_cpu_s=%.3f snapshot_read_s=%.3f pinned_to_numpy_s=%.3f "
            "xor_s=%.3f scan_s=%.3f compress_s=%.3f checksum_s=%.3f xor_restore_s=%.3f",
            target_version,
            base_version,
            len([result for result in results if result.stats.get("source_rank", 0.0) > 0.0]),
            len(records),
            len(payload_files),
            manifest.total_uncompressed_num_bytes,
            manifest.total_compressed_num_bytes,
            ratio,
            max_publish_s,
            sum_cpu_s,
            manifest_upload_s,
            aggregate.get("extract_or_gather_s", 0.0),
            aggregate.get("stage_to_cpu_s", 0.0),
            aggregate.get("snapshot_read_s", 0.0),
            aggregate.get("pinned_to_numpy_s", 0.0),
            aggregate.get("xor_s", 0.0),
            aggregate.get("changed_scan_s", 0.0),
            aggregate.get("compress_s", 0.0),
            aggregate.get("checksum_s", 0.0),
            aggregate.get("xor_restore_s", 0.0),
        )
        return {
            "target_version": target_version,
            "version": target_version,
            "sync_dir": self.sync_dir,
            "uri": publish_uri,
        }

    def publish(
        self,
        chunks: Iterable[WeightChunk],
        extractor_shard_info: Optional[ExtractorShardInfo] = None,
    ) -> DeltaPublishResult | dict:
        shard_info = extractor_shard_info or self._default_shard_info()
        result = self._publish_local(chunks, shard_info)
        if extractor_shard_info is None:
            return self.finalize_publish([result])
        return result
