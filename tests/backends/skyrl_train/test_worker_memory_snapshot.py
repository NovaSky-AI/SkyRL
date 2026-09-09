from unittest.mock import Mock, patch

import pytest
import torch

from skyrl.backends.skyrl_train.workers.worker import Worker, _is_cuda_oom


@pytest.mark.parametrize(
    "error",
    [
        torch.OutOfMemoryError("CUDA out of memory"),
        RuntimeError("cuDNN error: CUDA out of memory"),
    ],
)
def test_cuda_oom_detection(error: Exception) -> None:
    assert _is_cuda_oom(error)


def test_non_cuda_memory_error_is_not_cuda_oom() -> None:
    assert not _is_cuda_oom(MemoryError("host allocation failed"))


def test_oom_snapshot_does_not_synchronize_ranks() -> None:
    worker = object.__new__(Worker)
    worker.record_memory = True
    worker.save_memory_snapshot = Mock()

    worker.save_memory_snapshot_on_oom("forward_backward", torch.OutOfMemoryError("CUDA out of memory"))

    worker.save_memory_snapshot.assert_called_once_with(
        "forward_backward_oom",
        synchronize_ranks=False,
    )


def test_snapshot_failure_does_not_mask_cuda_oom() -> None:
    worker = object.__new__(Worker)
    worker.record_memory = True
    worker.save_memory_snapshot = Mock(side_effect=RuntimeError("snapshot failed"))

    with patch("skyrl.backends.skyrl_train.workers.worker.logger.exception") as log_exception:
        worker.save_memory_snapshot_on_oom("forward_backward", torch.OutOfMemoryError("CUDA out of memory"))

    log_exception.assert_called_once_with("Failed to save CUDA memory snapshot after OOM")
