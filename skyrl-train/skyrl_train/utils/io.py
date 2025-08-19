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
import shutil
import json
from pathlib import Path
from typing import Any, Union, BinaryIO, TextIO, cast
from contextlib import contextmanager
import torch
import fsspec
from loguru import logger


def is_cloud_path(path: str) -> bool:
    """Check if the given path is a cloud storage path."""
    return path.startswith(("s3://", "gs://", "gcs://"))


def _get_filesystem(path: str):
    """Get the appropriate fsspec filesystem for the given path."""
    if is_cloud_path(path):
        return fsspec.filesystem(path.split("://")[0])
    else:
        return fsspec.filesystem("file")


def open_file(path: str, mode: str = "rb") -> Union[BinaryIO, TextIO]:
    """Open a file using fsspec, works with both local and cloud paths."""
    fs = _get_filesystem(path)
    return fs.open(path, mode)


def save_file(obj: Any, path: str) -> None:
    """Save a PyTorch object to local or cloud path using torch.save."""
    if is_cloud_path(path):
        with open_file(path, "wb") as f:
            torch.save(obj, cast(BinaryIO, f))
    else:
        torch.save(obj, path)


def load_file(path: str, map_location: str = "cpu", weights_only: bool = False) -> Any:
    """Load a PyTorch object from local or cloud path using torch.load."""
    if is_cloud_path(path):
        with open_file(path, "rb") as f:
            return torch.load(cast(BinaryIO, f), map_location=map_location, weights_only=weights_only)
    else:
        return torch.load(path, map_location=map_location, weights_only=weights_only)


def makedirs(path: str, exist_ok: bool = True) -> None:
    """Create directories. Only applies to local filesystem paths."""
    if not is_cloud_path(path):
        os.makedirs(path, exist_ok=exist_ok)


def exists(path: str) -> bool:
    """Check if a file or directory exists."""
    fs = _get_filesystem(path)
    return fs.exists(path)


def isdir(path: str) -> bool:
    """Check if path is a directory."""
    fs = _get_filesystem(path)
    return fs.isdir(path)


def list_dir(path: str) -> list[str]:
    """List contents of a directory."""
    fs = _get_filesystem(path)
    return fs.ls(path)


def remove(path: str) -> None:
    """Remove a file or directory."""
    fs = _get_filesystem(path)
    if fs.isdir(path):
        fs.rm(path, recursive=True)
    else:
        fs.rm(path)


def copy_tree(src: str, dst: str) -> None:
    """Copy directory tree from src to dst. Handles local-to-cloud, cloud-to-local, and cloud-to-cloud."""
    src_fs = _get_filesystem(src)
    dst_fs = _get_filesystem(dst)

    if not src_fs.isdir(src):
        raise ValueError(f"Source path is not a directory: {src}")

    # Create destination directory
    if not is_cloud_path(dst):
        makedirs(dst, exist_ok=True)

    # Copy all files recursively
    for src_file in src_fs.find(src):
        # Calculate relative path from src
        rel_path = os.path.relpath(src_file, src)
        dst_file = os.path.join(dst, rel_path).replace("\\", "/")  # Normalize path separators

        # Create parent directory for destination file
        dst_parent = os.path.dirname(dst_file)
        if not is_cloud_path(dst):
            makedirs(dst_parent, exist_ok=True)

        # Copy the file
        with src_fs.open(src_file, "rb") as src_f:
            with dst_fs.open(dst_file, "wb") as dst_f:
                shutil.copyfileobj(src_f, dst_f)


def write_text(path: str, content: str) -> None:
    """Write text content to a file."""
    if is_cloud_path(path):
        with open_file(path, "w") as f:
            cast(TextIO, f).write(content)
    else:
        with open(path, "w") as f:
            f.write(content)


def read_text(path: str) -> str:
    """Read text content from a file."""
    if is_cloud_path(path):
        with open_file(path, "r") as f:
            return cast(TextIO, f).read()
    else:
        with open(path, "r") as f:
            return f.read()




def save_json(obj: Any, path: str) -> None:
    """Save a JSON-serializable object to a file."""

    if is_cloud_path(path):
        with open_file(path, "w") as f:
            json.dump(obj, cast(TextIO, f), indent=4)
    else:
        with open(path, "w") as f:
            json.dump(obj, f, indent=4)


def load_json(path: str) -> Any:
    """Load a JSON object from a file."""

    if is_cloud_path(path):
        with open_file(path, "r") as f:
            return json.load(f)
    else:
        with open(path, "r") as f:
            return json.load(f)


def get_latest_checkpoint_step(checkpoint_base_path: str) -> int:
    """
    Get the latest global step from checkpoint directory by reading latest_ckpt_global_step.txt.

    Args:
        checkpoint_base_path: Base path where checkpoints are stored

    Returns:
        int: Latest global step, or 0 if no checkpoint found
    """
    latest_file_path = os.path.join(checkpoint_base_path, "latest_ckpt_global_step.txt")

    if not exists(latest_file_path):
        return 0

    try:
        content = read_text(latest_file_path).strip()
        return int(content)
    except (ValueError, IOError) as e:
        logger.warning(f"Failed to read latest checkpoint step from {latest_file_path}: {e}")
        return 0


def list_checkpoint_dirs(checkpoint_base_path: str) -> list[str]:
    """
    List all checkpoint directories in the base path.

    Args:
        checkpoint_base_path: Base path where checkpoints are stored

    Returns:
        list[str]: List of checkpoint directory names
    """
    if not exists(checkpoint_base_path):
        return []

    try:
        all_items = list_dir(checkpoint_base_path)
        # Filter for directories that match the global_step_* pattern
        checkpoint_dirs = []
        for item in all_items:
            # Get just the basename for pattern matching
            basename = os.path.basename(item)
            if basename.startswith("global_step_") and isdir(item):
                checkpoint_dirs.append(basename)

        return sorted(checkpoint_dirs)
    except Exception as e:
        logger.warning(f"Failed to list checkpoint directories from {checkpoint_base_path}: {e}")
        return []


def cleanup_old_checkpoints(checkpoint_base_path: str, max_checkpoints: int, current_step: int) -> None:
    """
    Clean up old checkpoints, keeping only the most recent ones.

    Args:
        checkpoint_base_path: Base path where checkpoints are stored
        max_checkpoints: Maximum number of checkpoints to keep
        current_step: Current global step (for logging)
    """
    if max_checkpoints <= 0:
        return

    checkpoint_dirs = list_checkpoint_dirs(checkpoint_base_path)

    if len(checkpoint_dirs) <= max_checkpoints:
        return

    # Sort by step number (extract number from global_step_N)
    def extract_step(dirname):
        try:
            return int(dirname.split("global_step_")[1])
        except (IndexError, ValueError):
            return 0

    checkpoint_dirs.sort(key=extract_step)

    # Remove oldest checkpoints
    dirs_to_remove = checkpoint_dirs[:-max_checkpoints]

    for dir_name in dirs_to_remove:
        full_path = os.path.join(checkpoint_base_path, dir_name)
        try:
            remove(full_path)
            step_num = extract_step(dir_name)
            logger.info(f"Cleaned up old checkpoint: global_step_{step_num}")
        except Exception as e:
            logger.warning(f"Failed to remove old checkpoint {full_path}: {e}")


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
                copy_tree(temp_dir, output_path)
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
            copy_tree(input_path, temp_dir)
            logger.info(f"Downloaded directory contents from {input_path}")
            yield temp_dir
    else:
        # For local paths, use directly (but check it exists)
        if not exists(input_path):
            raise FileNotFoundError(f"Path does not exist: {input_path}")
        yield input_path
