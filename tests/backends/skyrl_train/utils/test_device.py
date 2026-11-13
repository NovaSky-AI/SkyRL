import pytest

from skyrl.backends.skyrl_train.utils import device
from skyrl.backends.skyrl_train.utils.device import (
    accelerator_resource_name,
    neuron_driver_core,
    neuron_worker_core,
)


def test_neuron_driver_core_is_last_logical_core():
    """The driver's in-process inference engine owns the last logical NeuronCore (63).

    FSDP workers take ``0 .. world_size-1`` (the NCCL collective backend requires each
    worker's core to equal its local rank), so the driver pins the last core to stay
    disjoint from every worker.
    """
    assert neuron_driver_core() == 63


def test_neuron_worker_core_equals_rank():
    """Each single-node worker claims core ``rank`` (its local rank)."""
    assert neuron_worker_core(0, num_nodes=1) == 0
    assert neuron_worker_core(1, num_nodes=1) == 1
    assert neuron_worker_core(7, num_nodes=1) == 7


def test_neuron_worker_core_ignores_ray_assigned_core(monkeypatch):
    """Ray rewrites NEURON_RT_VISIBLE_CORES per worker; the mapping must not read it.

    Ray hands worker ``rank`` its own Ray-assigned core (in reverse rank order) before
    ``__init__`` runs. If ``neuron_worker_core`` derived the core from that value it would
    collapse every worker onto the same core. It must instead return the fixed ``rank``.
    """
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "3")
    assert neuron_worker_core(0, num_nodes=1) == 0
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "0,1,2,3")
    assert neuron_worker_core(3, num_nodes=1) == 3


def test_neuron_worker_core_rejects_multi_node():
    """Multi-node per-node core pinning is not wired; reject it explicitly."""
    with pytest.raises(AssertionError, match="single-node"):
        neuron_worker_core(0, num_nodes=2)


def test_accelerator_resource_name_is_neuron_cores_on_neuron(monkeypatch):
    """The Ray placement-group bundle uses the neuron_cores resource on neuron."""
    monkeypatch.setattr(device, "_backend", "neuron")
    assert accelerator_resource_name() == "neuron_cores"


def test_accelerator_resource_name_is_gpu_on_cuda(monkeypatch):
    """CUDA keeps the existing GPU resource key."""
    monkeypatch.setattr(device, "_backend", "cuda")
    assert accelerator_resource_name() == "GPU"
