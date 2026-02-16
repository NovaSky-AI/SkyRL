"""
File I/O utilities for handling both local filesystem and cloud storage (S3/GCS).

This module provides a unified interface for file operations that works with:
- Local filesystem paths
- S3 paths (s3://bucket/path)
- Google Cloud Storage paths (gs://bucket/path or gcs://bucket/path)

Uses fsspec for cloud storage abstraction.
"""

import os
import tempfile
from contextlib import contextmanager
import fsspec
from loguru import logger
from .s3fs import get_s3_fs, s3_refresh_if_expiring, call_with_s3_retry, ClientError


def is_cloud_path(path: str) -> bool:
    """Check if the given path is a cloud storage path."""
    return path.startswith(("s3://", "gs://", "gcs://"))


def _get_filesystem(path: str):
    """Get the appropriate filesystem for the given path."""
    if not is_cloud_path(path):
        return fsspec.filesystem("file")

    proto = path.split("://", 1)[0]
    if proto == "s3":
        fs = get_s3_fs()
        s3_refresh_if_expiring(fs)
        return fs
    return fsspec.filesystem(proto)


def open_file(path: str, mode: str = "rb"):
    """Open a file using fsspec, works with both local and cloud paths."""
    if not is_cloud_path(path):
        return fsspec.open(path, mode)

    fs = _get_filesystem(path)
    norm = fs._strip_protocol(path)
    try:
        return fs.open(norm, mode)
    except ClientError as e:
        code = getattr(e, "response", {}).get("Error", {}).get("Code")
        if code in {"ExpiredToken", "ExpiredTokenException", "RequestExpired"} and hasattr(fs, "connect"):
            try:
                fs.connect(refresh=True)
            except Exception:
                pass
            return fs.open(norm, mode)
        raise


def makedirs(path: str, exist_ok: bool = True) -> None:
    """Create directories. Only applies to local filesystem paths."""
    if not is_cloud_path(path):
        os.makedirs(path, exist_ok=exist_ok)


def exists(path: str) -> bool:
    """Check if a file or directory exists."""
    fs = _get_filesystem(path)
    if is_cloud_path(path) and path.startswith("s3://"):
        return call_with_s3_retry(fs, fs.exists, path)
    return fs.exists(path)


def isdir(path: str) -> bool:
    """Check if path is a directory."""
    fs = _get_filesystem(path)
    if is_cloud_path(path) and path.startswith("s3://"):
        return call_with_s3_retry(fs, fs.isdir, path)
    return fs.isdir(path)


def list_dir(path: str) -> list[str]:
    """List contents of a directory."""
    fs = _get_filesystem(path)
    if is_cloud_path(path) and path.startswith("s3://"):
        return call_with_s3_retry(fs, fs.ls, path, detail=False)
    return fs.ls(path, detail=False)


def remove(path: str) -> None:
    """Remove a file or directory."""
    fs = _get_filesystem(path)
    if is_cloud_path(path) and path.startswith("s3://"):
        if call_with_s3_retry(fs, fs.isdir, path):
            call_with_s3_retry(fs, fs.rm, path, recursive=True)
        else:
            call_with_s3_retry(fs, fs.rm, path)
        return
    if fs.isdir(path):
        fs.rm(path, recursive=True)
    else:
        fs.rm(path)


def upload_directory(local_path: str, cloud_path: str) -> None:
    """Upload a local directory to cloud storage."""
    if not is_cloud_path(cloud_path):
        raise ValueError(f"Destination must be a cloud path, got: {cloud_path}")

    fs = _get_filesystem(cloud_path)
    if cloud_path.startswith("s3://"):
        call_with_s3_retry(fs, fs.put, local_path, fs._strip_protocol(cloud_path), recursive=True)
    else:
        fs.put(local_path, cloud_path, recursive=True)
    logger.info(f"Uploaded {local_path} to {cloud_path}")


def download_directory(cloud_path: str, local_path: str) -> None:
    """Download a cloud directory to local storage."""
    if not is_cloud_path(cloud_path):
        raise ValueError(f"Source must be a cloud path, got: {cloud_path}")

    fs = _get_filesystem(cloud_path)
    if cloud_path.startswith("s3://"):
        call_with_s3_retry(fs, fs.get, fs._strip_protocol(cloud_path), local_path, recursive=True)
    else:
        fs.get(cloud_path, local_path, recursive=True)
    logger.info(f"Downloaded {cloud_path} to {local_path}")


@contextmanager
def local_work_dir(output_path: str):
    """
    Context manager that provides a local working directory.

    For local paths, returns the path directly.
    For cloud paths, creates a temporary directory and uploads content at the end.

    Args:
        output_path: The final destination path (local or cloud)

    Yields:
        str: Local directory path to work with

    Example:
        with local_work_dir("s3://bucket/model") as work_dir:
            # Save files to work_dir
            model.save_pretrained(work_dir)
            # Files are automatically uploaded to s3://bucket/model at context exit
    """
    if is_cloud_path(output_path):
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                yield temp_dir
            finally:
                # Upload everything from temp_dir to cloud path
                upload_directory(temp_dir, output_path)
                logger.info(f"Uploaded directory contents to {output_path}")
    else:
        # For local paths, ensure directory exists and use it directly
        makedirs(output_path, exist_ok=True)
        yield output_path


@contextmanager
def local_read_dir(input_path: str):
    """
    Context manager that provides a local directory with content from input_path.

    For local paths, returns the path directly.
    For cloud paths, downloads content to a temporary directory.

    Args:
        input_path: The source path (local or cloud)

    Yields:
        str: Local directory path containing the content

    Example:
        with local_read_dir("s3://bucket/model") as read_dir:
            # Load files from read_dir
            model = AutoModel.from_pretrained(read_dir)
    """
    if is_cloud_path(input_path):
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download everything from cloud path to temp_dir
            download_directory(input_path, temp_dir)
            logger.info(f"Downloaded directory contents from {input_path}")
            yield temp_dir
    else:
        # For local paths, use directly (but check it exists)
        if not exists(input_path):
            raise FileNotFoundError(f"Path does not exist: {input_path}")
        yield input_path
