"""Backend-aware accelerator helpers (CUDA / TorchNeuron).

The FSDP trainer was written for CUDA only.  TorchNeuron (Trainium2) registers the
``neuron`` private-use device and runs ``model.generate()``, but has none of the
``torch.cuda`` memory / RNG / P2P APIs.  These helpers centralise the dispatch so the
trainer can run on either backend.

On neuron SkyRL runs a single-core (``world_size == 1``) process: the device id is
always 0, memory info comes from ``torch_neuronx``, P2P is unavailable, and RNG is
CPU-only.  torch.distributed collectives use NCCL on CUDA and the CPU ``gloo`` backend
everywhere else, so :func:`collective_device` returns CPU on neuron.
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


def is_accelerator() -> bool:
    return is_cuda() or is_neuron()


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


def ray_actor_options(num_accelerators: float, resources: dict | None = None) -> dict:
    """Ray actor ``.options()`` kwargs to reserve ``num_accelerators`` accelerators.

    CUDA ties ``num_cpus`` to ``num_gpus`` (as the trainer already does) and passes
    the caller's ``resources`` through unchanged.  Neuron has no ``num_gpus`` sugar:
    it reserves the ``neuron_cores`` custom resource, requests a single CPU, and
    forces ``num_gpus=0`` (overriding any ``@ray.remote(num_gpus=1)`` class default).
    """
    if is_neuron():
        return {
            "num_cpus": 1,
            "num_gpus": 0,
            "resources": {"neuron_cores": num_accelerators},
        }
    return {
        "num_cpus": num_accelerators,
        "num_gpus": num_accelerators,
        "resources": resources,
    }
