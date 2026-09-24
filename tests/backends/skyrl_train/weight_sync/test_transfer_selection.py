"""Backend resolution at SkyRL's native trainer/receiver boundary."""

import pytest

from skyrl.backends.skyrl_train.weight_sync import (
    get_transfer_strategy,
    get_vllm_receive_backend,
)


@pytest.mark.parametrize(
    "configured_backend,colocate_all,logical_backend,receive_backend",
    [
        ("nccl", False, "nccl", "skyrl_nccl"),
        ("nccl", True, "ipc", "skyrl_ipc"),
        ("delta", False, "delta", "delta"),
        ("delta", True, "delta", "delta"),
        ("sharded_rdt", False, "sharded_rdt", "sharded_rdt"),
        ("sharded_rdt", True, "sharded_rdt", "sharded_rdt"),
        ("rdt", False, "sharded_rdt", "sharded_rdt"),
    ],
)
def test_backend_selection_matches_the_native_trainer_and_receiver(
    configured_backend,
    colocate_all,
    logical_backend,
    receive_backend,
):
    """Replacement for the deleted transfer-strategy-class selection test."""

    assert get_transfer_strategy(configured_backend, colocate_all) == logical_backend
    assert get_vllm_receive_backend(configured_backend, colocate_all) == receive_backend
