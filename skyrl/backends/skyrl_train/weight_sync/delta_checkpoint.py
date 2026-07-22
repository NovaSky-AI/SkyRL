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
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save, save_file

from skyrl.backends.skyrl_train.weight_sync.base import WeightChunk
from skyrl.backends.skyrl_train.weight_sync.delta_payload import (
    bytes_to_uint8_tensor,
    compress_bytes,
    decompress_bytes,
    uint8_tensor_to_bytes,
)

logger = logging.getLogger(__name__)


_MANIFEST_NAME = "manifest.json"
_STATE_DIR_NAME = ".skyrl_weight_sync"
_DEFAULT_CHECKSUM_ALGORITHM = "xxh3-128"
_DEFAULT_PINNED_STAGING_BYTE_CAP = 32 * 1024**3
SUPPORTED_CHECKPOINT_LOAD_FORMATS = frozenset(
    {
        "vllm_fastsafetensors",
        "vllm_multi_thread_safetensors",
    }
)


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
    return f"delta-{version:08d}"


def _weights_dir(root: Path) -> Path:
    return root / "weights"


def _deltas_dir(root: Path) -> Path:
    return root / "deltas"


def _is_gs_uri(uri: str) -> bool:
    return uri.startswith("gs://")


def _is_s3_uri(uri: str) -> bool:
    return uri.startswith("s3://")


def _is_cloud_uri(uri: str) -> bool:
    return _is_gs_uri(uri) or _is_s3_uri(uri)


def _join_uri(base: str, child: str) -> str:
    return f"{base.rstrip('/')}/{child}"


def _safe_path_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in value)[:160]


def _uri_basename(uri: str) -> str:
    return uri.rstrip("/").rsplit("/", 1)[-1] or "object"


def _cloud_cp_command(src: str, dst: str) -> list[str]:
    if _is_gs_uri(src) or _is_gs_uri(dst):
        executable = "gcloud"
        if shutil.which(executable) is None:
            raise RuntimeError("GCS delta transfer requires the gcloud CLI to be installed on this node")
        return [executable, "storage", "cp", src, dst]
    if _is_s3_uri(src) or _is_s3_uri(dst):
        executable = "s5cmd"
        if shutil.which(executable) is None:
            raise RuntimeError("S3 delta transfer requires the s5cmd CLI to be installed on this node")
        return [executable, "cp", src, dst]
    raise ValueError(f"Unsupported cloud transfer: {src!r} -> {dst!r}")


def _run_cloud_cp(src: str, dst: str) -> None:
    cmd = _cloud_cp_command(src, dst)
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Cloud delta transfer failed: " f"{' '.join(cmd)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def _default_publish_staging_dir(sync_dir: str) -> Path:
    return Path(tempfile.gettempdir()) / "skyrl_delta_publish_staging" / _safe_path_name(sync_dir)


def _atomic_write_bytes_local(data: bytes, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with tmp_path.open("wb") as f:
        f.write(data)
        f.flush()
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


def _write_bytes_to_uri(data: bytes, uri: str, staging_dir: Optional[Path] = None) -> None:
    if _is_cloud_uri(uri):
        staging_root = staging_dir or _default_publish_staging_dir(uri)
        staging_root.mkdir(parents=True, exist_ok=True)
        name = _uri_basename(uri)
        tmp_path = staging_root / f".{name}.{os.getpid()}.tmp"
        try:
            with tmp_path.open("wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            _run_cloud_cp(str(tmp_path), uri)
        finally:
            tmp_path.unlink(missing_ok=True)
        return

    _atomic_write_bytes_local(data, Path(uri))


def _write_tensors_to_uri(tensors: dict[str, torch.Tensor], uri: str, staging_dir: Optional[Path] = None) -> None:
    if _is_cloud_uri(uri):
        staging_root = staging_dir or _default_publish_staging_dir(uri)
        staging_root.mkdir(parents=True, exist_ok=True)
        name = _uri_basename(uri)
        tmp_path = staging_root / f".{name}.{os.getpid()}.tmp"
        local_path = staging_root / name
        try:
            save_file(tensors, str(tmp_path))
            os.replace(tmp_path, local_path)
            _run_cloud_cp(str(local_path), uri)
        finally:
            tmp_path.unlink(missing_ok=True)
            local_path.unlink(missing_ok=True)
        return

    _write_bytes_to_uri(save(tensors), uri)


def _copy_from_uri(uri: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = local_path.with_name(f".{local_path.name}.{os.getpid()}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    if _is_cloud_uri(uri):
        _run_cloud_cp(uri, str(tmp_path))
    else:
        shutil.copy2(Path(uri), tmp_path)
    os.replace(tmp_path, local_path)


def fetch_delta_directory(uri: str, cache_dir: Path, gcs_download_workers: int = 4) -> Tuple[DeltaManifest, Path]:
    """Fetch a published delta directory into a local cache and return its manifest."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock = FileLock(cache_dir / ".fetch.lock")
    with lock:
        manifest_path = cache_dir / _MANIFEST_NAME
        if not manifest_path.exists():
            _copy_from_uri(_join_uri(uri, _MANIFEST_NAME), manifest_path)
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = DeltaManifest.from_json(json.load(f))

        missing_payload_files = []
        for payload_file in manifest.payload_files:
            dst = cache_dir / payload_file
            if not dst.exists():
                missing_payload_files.append(payload_file)
        if _is_gs_uri(uri) and len(missing_payload_files) > 1 and gcs_download_workers > 1:
            workers = min(len(missing_payload_files), max(1, gcs_download_workers))
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="skyrl-delta-gcs-fetch") as executor:
                futures = [
                    executor.submit(_copy_from_uri, _join_uri(uri, payload_file), cache_dir / payload_file)
                    for payload_file in missing_payload_files
                ]
                for future in futures:
                    future.result()
        else:
            for payload_file in missing_payload_files:
                _copy_from_uri(_join_uri(uri, payload_file), cache_dir / payload_file)
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


def _checkpoint_safetensors_files(checkpoint_dir: Path) -> list[str]:
    index_files = sorted(checkpoint_dir.glob("*.safetensors.index.json"))
    if index_files:
        with index_files[0].open("r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        return [str(checkpoint_dir / name) for name in sorted(set(weight_map.values()))]

    files = sorted(checkpoint_dir.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors weights found under {checkpoint_dir}")
    return [str(path) for path in files]


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

    def __init__(self, base_model_path: str, local_checkpoint_dir: str, gcs_download_workers: int = 4) -> None:
        self.base_model_path = base_model_path
        self.root = Path(local_checkpoint_dir)
        self.weights_dir = _weights_dir(self.root)
        self.deltas_dir = _deltas_dir(self.root)
        self.state_dir = self.root / _STATE_DIR_NAME
        self.state_path = self.state_dir / "state.json"
        self.lock_path = self.state_dir / "writer.lock"
        self.gcs_download_workers = max(1, int(gcs_download_workers))
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
                    manifest, payload_dir = fetch_delta_directory(
                        delta_uri,
                        cache_dir,
                        gcs_download_workers=self.gcs_download_workers,
                    )
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

    def iter_tensors(
        self,
        load_format: str = "vllm_fastsafetensors",
        multi_thread_safetensors_max_workers: int = 8,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        checkpoint_dir = self._current_checkpoint_dir()
        files = _checkpoint_safetensors_files(checkpoint_dir)
        if load_format == "vllm_multi_thread_safetensors":
            from vllm.model_executor.model_loader.weight_utils import (
                multi_thread_safetensors_weights_iterator,
            )

            yield from multi_thread_safetensors_weights_iterator(
                files,
                use_tqdm_on_load=False,
                max_workers=multi_thread_safetensors_max_workers,
            )
            return

        if load_format == "vllm_fastsafetensors":
            # NOTE (sumanthrh): The fastsafetensors iterator can lead to large temporary memory usage
            # during weight loading due to out of order loading + layerwise reloading interaction
            # For more details, see: https://github.com/vllm-project/vllm/issues/48644
            from vllm.model_executor.model_loader.weight_utils import (
                fastsafetensors_weights_iterator,
            )

            yield from fastsafetensors_weights_iterator(files, use_tqdm_on_load=False)
            return

        raise ValueError(
            "Unknown checkpoint_load_format "
            f"{load_format!r}; expected one of {sorted(SUPPORTED_CHECKPOINT_LOAD_FORMATS)}"
        )


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
        publish_staging_dir: Optional[str] = None,
        max_file_size_in_gb: float = 1.0,
        publish_num_workers: Optional[int] = None,
    ) -> None:
        self.base_model_path = base_model_path
        self.sync_dir = sync_dir
        if publish_staging_dir is not None and _is_cloud_uri(publish_staging_dir):
            raise ValueError("publish_staging_dir must be a local filesystem path")
        self.publish_staging_dir = (
            Path(publish_staging_dir) if publish_staging_dir else _default_publish_staging_dir(sync_dir)
        )
        if publish_num_workers is not None and publish_num_workers < 1:
            raise ValueError(f"publish_num_workers must be >= 1, got {publish_num_workers}")
        self.publish_num_workers = publish_num_workers
        self.max_file_size_bytes = int(max_file_size_in_gb * 1024**3)
        self.checksum_algorithm = _DEFAULT_CHECKSUM_ALGORITHM
        self.version = 0
        self.snapshot: dict[str, np.ndarray] = {}
        self._base_checkpoint_dir: Optional[Path] = None
        self._base_locations: Optional[dict[str, TensorLocation]] = None
        self._publish_executor: Optional[ThreadPoolExecutor] = None
        self._publish_executor_workers: Optional[int] = None
        self._staging_pool: Optional[_PinnedStagingPool] = None
        self._staging_pool_signature: Optional[tuple[int, int]] = None

    @staticmethod
    def _current_rank() -> int:
        """Global rank of this publisher process (0 when not running distributed).

        Only rank 0 is the delta source: it computes and uploads deltas. Other
        ranks still drain the chunk stream (to participate in the extractor's
        collectives) but publish nothing.
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        return 0

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
        t_total = time.perf_counter()
        timings: dict[str, float] = {}
        if staged.pinned:
            assert isinstance(staged.data, torch.Tensor)
            try:
                current = np.empty(staged.nbytes, dtype=np.uint8)
                np.copyto(current, staged.data[: staged.nbytes].numpy())
            finally:
                if staging_pool is not None:
                    staging_pool.release(staged.data)
        else:
            assert isinstance(staged.data, np.ndarray)
            current = staged.data

        if current.nbytes != base.nbytes:
            raise ValueError(f"Byte-size mismatch for {name}: current={current.nbytes}, base={base.nbytes}")

        checksum = _checksum(memoryview(current), checksum_algorithm)

        np.bitwise_xor(current, base, out=current)

        changed_bytes = int(np.count_nonzero(current))

        compressed = None
        if changed_bytes:
            compressed = compress_bytes(memoryview(current), level=1)

        # ``current`` is the XOR patch at this point. Apply it to the existing
        # snapshot in-place so repeated publishes do not replace every
        # full-model CPU snapshot allocation.
        np.bitwise_xor(base, current, out=base)
        timings["delta_compute_s"] = time.perf_counter() - t_total

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
            "cpu_stage_s": 0.0,
            "delta_compute_s": 0.0,
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

    def _num_publish_workers(self) -> int:
        default = min(8, os.cpu_count() or 1)
        return self.publish_num_workers or default

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

    def _staging_pool_for(self, max_tensor_bytes: int, num_workers: int) -> _PinnedStagingPool:
        signature = (max_tensor_bytes, num_workers)
        if self._staging_pool is None or self._staging_pool_signature != signature:
            self._staging_pool = _PinnedStagingPool(max_tensor_bytes, num_workers)
            self._staging_pool_signature = signature
        return self._staging_pool

    def create_delta_files(
        self,
        chunks: Iterable[WeightChunk],
    ) -> DeltaPublishResult:
        """Iterate over weight chunks and create delta files locally"""
        base_version = self.version
        target_version = base_version + 1
        publish_uri = _join_uri(self.sync_dir, _version_name(target_version))
        rank = self._current_rank()
        is_source_rank = rank == 0
        stats = self._empty_stats()
        records: list[DeltaTensorRecord] = []
        payload_files: list[str] = []
        payload_tensors: dict[str, torch.Tensor] = {}
        payload_file_idx = 0
        payload_file_bytes = 0
        pending: deque[Future[_TensorPublishResult]] = deque()
        skipped_names: set[str] = set()
        t_publish = time.perf_counter()
        stats["source_rank"] = 1.0 if is_source_rank else 0.0

        def current_payload_file() -> str:
            return f"delta-rank{rank:05d}-file{payload_file_idx:06d}.safetensors"

        def flush_payload_file() -> None:
            nonlocal payload_tensors, payload_file_idx, payload_file_bytes
            if not payload_tensors:
                return
            t = time.perf_counter()
            file_name = current_payload_file()
            staging_dir = self.publish_staging_dir / _version_name(target_version) / f"rank{rank:06d}"
            _write_tensors_to_uri(payload_tensors, _join_uri(publish_uri, file_name), staging_dir=staging_dir)
            stats["upload_s"] += time.perf_counter() - t
            payload_files.append(file_name)
            payload_tensors = {}
            payload_file_idx += 1
            payload_file_bytes = 0

        def consume_result(result: _TensorPublishResult) -> None:
            nonlocal payload_file_bytes
            for key, value in result.timings.items():
                stats[key] = stats.get(key, 0.0) + value
            stats["processed_tensors"] += 1
            stats["changed_bytes"] += result.changed_bytes
            if result.compressed is None:
                return
            if payload_tensors and payload_file_bytes + len(result.compressed) > self.max_file_size_bytes:
                flush_payload_file()
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

        def drain_one() -> None:
            result = pending.popleft().result()
            consume_result(result)

        try:
            num_workers = self._num_publish_workers()
            max_inflight = max(1, num_workers)
            staging_pool: Optional[_PinnedStagingPool] = None
            if is_source_rank:
                locations = self._ensure_base_locations()
                max_tensor_bytes = max((loc.nbytes for loc in locations.values()), default=0)
                staging_pool = self._staging_pool_for(max_tensor_bytes, num_workers)
            executor = self._publish_executor_for(num_workers) if is_source_rank else None
            chunk_iter = iter(chunks)
            while True:
                try:
                    chunk = next(chunk_iter)
                except StopIteration:
                    break
                if not is_source_rank:
                    continue
                for name, tensor in zip(chunk.names, chunk.tensors):
                    try:
                        loc = self._base_location(name)
                        base = self._snapshot_bytes(name)
                    except KeyError:
                        skipped_names.add(name)
                        stats["skipped_tensors"] += 1
                        continue

                    t = time.perf_counter()
                    staged, dtype_name, shape = self._stage_tensor_for_publish(
                        tensor,
                        target_dtype=_safetensors_dtype_to_torch(loc.dtype),
                        target_shape=loc.shape,
                        staging_pool=staging_pool,
                    )
                    stats["cpu_stage_s"] += time.perf_counter() - t
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
                        drain_one()
            while pending:
                drain_one()
            flush_payload_file()

            if skipped_names:
                logger.warning(
                    "delta checkpoint publisher rank=%s skipped %s tensor(s) absent from base checkpoint: %s",
                    rank,
                    len(skipped_names),
                    sorted(skipped_names)[:10],
                )
        except Exception:
            raise

        self.version = target_version
        stats["publish_s"] = time.perf_counter() - t_publish
        ratio = stats["compressed_bytes"] / stats["uncompressed_bytes"] if stats["uncompressed_bytes"] else 0.0
        logger.info(
            "delta checkpoint publish local: rank=%s version=%s base_version=%s tensors=%s payload_files=%s "
            "uncompressed_bytes=%s compressed_bytes=%s compression_ratio=%.6f publish_s=%.3f "
            "cpu_stage_s=%.3f delta_compute_s=%.3f upload_s=%.3f",
            rank,
            target_version,
            base_version,
            len(records),
            len(payload_files),
            int(stats["uncompressed_bytes"]),
            int(stats["compressed_bytes"]),
            ratio,
            stats["publish_s"],
            stats["cpu_stage_s"],
            stats["delta_compute_s"],
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

    def publish(self, results: Union[DeltaPublishResult, Sequence[DeltaPublishResult]]) -> dict:
        """Publish one or more `DeltaPublishResult`s to cloud storage"""
        if not results:
            raise ValueError("No delta publish results to finalize")
        if isinstance(results, DeltaPublishResult):
            results: Sequence[DeltaPublishResult] = [results]

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
        t = time.perf_counter()
        manifest_staging_dir = self.publish_staging_dir / _version_name(target_version) / "manifest"
        _write_bytes_to_uri(
            json.dumps(manifest.to_json()).encode("utf-8"),
            _join_uri(publish_uri, _MANIFEST_NAME),
            staging_dir=manifest_staging_dir,
        )
        manifest_upload_s = time.perf_counter() - t

        ratio = (
            manifest.total_compressed_num_bytes / manifest.total_uncompressed_num_bytes
            if manifest.total_uncompressed_num_bytes
            else 0.0
        )
        max_publish_s = max((result.stats.get("publish_s", 0.0) for result in results), default=0.0)
        total_upload_s = aggregate.get("upload_s", 0.0) + manifest_upload_s
        logger.info(
            "delta checkpoint publish finalized: version=%s base_version=%s source_ranks=%s tensors=%s "
            "payload_files=%s uncompressed_bytes=%s compressed_bytes=%s compression_ratio=%.6f "
            "publish_s=%.3f cpu_stage_s=%.3f delta_compute_s=%.3f upload_s=%.3f",
            target_version,
            base_version,
            len([result for result in results if result.stats.get("source_rank", 0.0) > 0.0]),
            len(records),
            len(payload_files),
            manifest.total_uncompressed_num_bytes,
            manifest.total_compressed_num_bytes,
            ratio,
            max_publish_s,
            aggregate.get("cpu_stage_s", 0.0),
            aggregate.get("delta_compute_s", 0.0),
            total_upload_s,
        )
        return {
            "target_version": target_version,
            "version": target_version,
            "sync_dir": self.sync_dir,
            "uri": publish_uri,
        }
