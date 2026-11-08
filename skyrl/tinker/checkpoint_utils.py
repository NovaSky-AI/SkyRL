"""Utilities for managing Tinker checkpoint artifacts."""

import shutil
from typing import Any

from skyrl.tinker import types


def checkpoint_file_path(
    checkpoints_base: Any, unique_id: str, checkpoint_id: str, checkpoint_type: types.CheckpointType
) -> Any:
    """Construct the file path for a checkpoint artifact."""
    checkpoint_dir = checkpoints_base / unique_id
    if checkpoint_type == types.CheckpointType.SAMPLER:
        checkpoint_dir = checkpoint_dir / "sampler_weights"
    return checkpoint_dir / f"{checkpoint_id}.tar.gz"


def delete_checkpoint_file(checkpoint_path: Any) -> None:
    """Remove a checkpoint artifact from disk or cloud storage.

    Only removes the checkpoint artifact itself; leaves the enclosing directories in
    place. The run may still be active, and every save path recreates its directory
    on demand (parents=True / makedirs), so pruning them here is pointless churn that
    couples a per-checkpoint delete to run-wide directory state.
    """
    if checkpoint_path.is_dir():
        if hasattr(checkpoint_path, "rmtree"):
            checkpoint_path.rmtree()
        else:
            shutil.rmtree(checkpoint_path)
    else:
        try:
            checkpoint_path.unlink()
        except FileNotFoundError:
            return
