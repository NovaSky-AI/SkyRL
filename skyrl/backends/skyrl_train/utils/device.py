"""Backend-aware accelerator helpers (CUDA / TorchNeuron).

The FSDP trainer was written for CUDA only.  TorchNeuron (Trainium2) registers the
``neuron`` private-use device and runs ``model.generate()``, but has none of the
``torch.cuda`` memory / RNG / P2P APIs.  These helpers centralise the dispatch so the
trainer can run on either backend.

On neuron SkyRL runs a single-node process (``num_nodes == 1``): the device id is
always 0, memory info comes from ``torch_neuronx``, P2P is unavailable, and RNG is
CPU-only.  Each FSDP worker claims its own disjoint NeuronCore (see
:func:`neuron_worker_core`).  torch.distributed collectives use NCCL on CUDA and the
CPU ``gloo`` backend everywhere else, so :func:`collective_device` returns CPU on
neuron.
"""

import torch

_backend: str | None = None


def _detect_backend() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # torch-neuronx registers `neuron` as a private-use device; on a CUDA/CPU torch
    # `torch.device("neuron")` raises, so the resolution itself is the probe.
    try:
        torch.device("neuron")
        return "neuron"
    except Exception:
        return "cpu"


def backend() -> str:
    """Return ``"cuda"``, ``"neuron"`` or ``"cpu"`` for the active accelerator."""
    global _backend
    if _backend is None:
        _backend = _detect_backend()
    return _backend


def is_cuda() -> bool:
    return backend() == "cuda"


def is_neuron() -> bool:
    return backend() == "neuron"


def _neuronx():
    import torch_neuronx

    return torch_neuronx


def current_device() -> torch.device:
    """Active accelerator device, as an unambiguous ``torch.device`` for ``.to(...)``."""
    if is_cuda():
        return torch.device("cuda", torch.cuda.current_device())
    if is_neuron():
        return torch.device("neuron")
    return torch.device("cpu")


def current_device_id() -> int:
    if is_cuda():
        return torch.cuda.current_device()
    return 0


def set_device(rank: int) -> None:
    if is_cuda():
        torch.cuda.set_device(rank)


# trn2.48xlarge exposes 64 logical NeuronCores (16 devices x 4 logical cores under
# ``logical-neuroncore-config: 2``). FSDP workers claim ``0 .. world_size-1`` (the NCCL
# collective backend requires each worker's core to equal its local rank), so the driver
# engine takes the *last* core, which is disjoint for every worker count that still
# leaves room for a driver process.
_NEURON_LOGICAL_CORE_COUNT = 64


def neuron_driver_core() -> int:
    """NeuronCore id for the driver's in-process inference engine.

    The FSDP workers claim logical NeuronCores ``0 .. world_size-1`` (see
    :func:`neuron_worker_core`).  The engine is a separate Ray process on the driver
    (``cholec_entrypoint``) actor and cannot share a core with any worker, so it takes
    the *last* logical core.  The value is a constant for the same reason
    :func:`neuron_worker_core` returns a constant: the driver actor is a Ray worker
    spawned by the raylet, so it does not inherit the launch script's shell export and
    must pin its core itself (Ray's accelerator manager would otherwise leave the
    variable unset and ``.to("neuron")`` spreads across the default visible-core set).
    """
    return _NEURON_LOGICAL_CORE_COUNT - 1


def neuron_worker_core(rank: int, num_nodes: int = 1) -> int:
    """NeuronCore id for an FSDP worker actor of ``rank``.

    torch-neuronx's native ``neuron`` collective backend (the NCCL/NRT graph it builds
    for sharded-DTensor all-gather / reduce-scatter) requires each process's visible
    core to *equal its local rank*: ``_neuron_runtime_setup`` sets the local device
    start index to the local rank, and the NCCL topology layer aborts
    (``getNumOfVcoresPerDevice`` assertion) when the actual core is offset from it.
    A single-node FSDP world therefore pins worker ``rank`` to logical NeuronCore
    ``rank`` -- ``0 .. world_size-1``.  The driver takes the last core
    (:func:`neuron_driver_core`), so it never collides with a worker.

    The offset must not be read from ``NEURON_RT_VISIBLE_CORES`` inside the worker.
    Ray's ``NeuronAcceleratorManager`` rewrites that variable in each worker process --
    before the actor's ``__init__`` runs -- to the worker's *Ray-assigned* core (one of
    ``0..world_size-1``, handed out in reverse rank order).  Reading it there would
    collapse every worker onto the same core.  The worker instead pins the constant
    ``rank`` and lets the backend's hostname rendezvous recompute the local rank.

    Single-node only: this mapping assumes every worker shares the driver's node and
    therefore the same core namespace.  On multi-node the scheduler pins each node's
    ``NEURON_RT_VISIBLE_CORES`` independently, so a global ``rank`` mapping is not valid
    and is rejected via ``num_nodes``.
    """
    assert num_nodes == 1, (
        "neuron_worker_core only supports single-node (num_nodes == 1); "
        "multi-node neuron is not wired for per-node core pinning"
    )
    return rank


def device_count() -> int:
    if is_cuda():
        return torch.cuda.device_count()
    return 1


def collective_device() -> torch.device:
    """Device tensors must live on to run ``torch.distributed`` collectives.

    CUDA uses NCCL; neuron and cpu use the CPU-only ``gloo`` backend, so collectives
    run on CPU there.
    """
    if is_cuda():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def synchronize() -> None:
    if is_cuda():
        torch.cuda.synchronize()
    elif is_neuron():
        _neuronx().synchronize()


def empty_cache() -> None:
    if is_cuda():
        torch.cuda.empty_cache()
    elif is_neuron():
        _neuronx().empty_cache()


def manual_seed_all(seed: int) -> None:
    if is_cuda():
        torch.cuda.manual_seed_all(seed)


def memory_allocated() -> int:
    if is_cuda():
        return torch.cuda.memory_allocated()
    if is_neuron():
        return _neuronx().memory_allocated()
    return 0


def memory_reserved() -> int:
    if is_cuda():
        return torch.cuda.memory_reserved()
    return 0


def mem_get_info():
    if is_cuda():
        return torch.cuda.mem_get_info()
    return (0, 0)


def can_device_access_peer(i: int, j: int) -> bool:
    if is_cuda():
        return torch.cuda.can_device_access_peer(i, j)
    return False


def accelerator_resource_name() -> str:
    """Ray resource key for the active accelerator: ``GPU`` on CUDA, ``neuron_cores`` on neuron."""
    return "neuron_cores" if is_neuron() else "GPU"


def ray_actor_options(num_accelerators: float, resources: dict | None = None) -> dict:
    """Ray actor ``.options()`` kwargs to reserve ``num_accelerators`` accelerators.

    CUDA ties ``num_cpus`` to ``num_gpus`` (as the trainer already does) and passes
    the caller's ``resources`` through unchanged.  Neuron has no ``num_gpus`` sugar:
    it reserves the ``neuron_cores`` custom resource, requests a single CPU, and
    forces ``num_gpus=0`` (overriding any ``@ray.remote(num_gpus=1)`` class default).
    Any caller-supplied resources are preserved and merged with the ``neuron_cores``
    request rather than dropped.
    """
    if is_neuron():
        resources = dict(resources or {})
        resources["neuron_cores"] = num_accelerators
        return {
            "num_cpus": 1,
            "num_gpus": 0,
            "resources": resources,
        }
    return {
        "num_cpus": num_accelerators,
        "num_gpus": num_accelerators,
        "resources": resources,
    }
