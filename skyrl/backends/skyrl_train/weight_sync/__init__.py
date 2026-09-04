"""Weight synchronization for distributed RL training.

SkyRL drives weight sync through vLLM's trainer-send abstraction: each training
worker builds a ``WeightSource`` over its live model (``sources.py``) and hands
it to a ``TrainerWeightTransferEngine`` (``trainer_engines.py``) whose
``send_weights()`` owns the whole round trip. Four backends:

===============  ==========================================================
``nccl``         broadcast from trainer rank 0; non-colocated
``ipc``          CUDA IPC handles; colocated (``placement.colocate_all``)
``delta``        compressed checkpoint deltas via disk / ``gs://`` / ``s3://``
``sharded_rdt``  the inference workers pull slices over NIXL
===============  ==========================================================

This package imports no vLLM at module scope: half the CPU suite runs without the
(Linux-only, optional) wheel. The vLLM-facing modules -- ``sources``,
``trainer_engines``, ``skyrl_engines``, ``delta_trainer`` -- are imported at their
call sites instead.
"""

from .base import LoraLoadRequest

#: Logical backend name -> the name its *receive*-side engine is registered
#: under in vLLM's ``WeightTransferEngineFactory``.
#:
#: NCCL and IPC take new names because SkyRL subclasses vLLM's engines and
#: ``register_engine`` refuses an already-registered name. The trainer-side
#: factory has a separate registry, so the send side keeps vLLM's own names.
_VLLM_RECEIVE_BACKENDS = {
    "nccl": "skyrl_nccl",
    "ipc": "skyrl_ipc",
    "delta": "delta",
    "sharded_rdt": "sharded_rdt",
}


def get_transfer_strategy(weight_sync_backend: str, colocate_all: bool) -> str:
    """Resolve the logical weight-sync backend from config + placement.

    A configured ``nccl`` means CUDA IPC when training and inference share GPUs,
    and broadcast otherwise.
    """
    if weight_sync_backend in ("sharded_rdt", "rdt"):
        return "sharded_rdt"
    if weight_sync_backend == "delta":
        return "delta"
    if weight_sync_backend == "nccl" and colocate_all:
        return "ipc"
    return "nccl"


def get_vllm_receive_backend(weight_sync_backend: str, colocate_all: bool) -> str:
    """The ``WeightTransferConfig.backend`` the inference servers must be built with."""
    return _VLLM_RECEIVE_BACKENDS[get_transfer_strategy(weight_sync_backend, colocate_all)]


__all__ = [
    "LoraLoadRequest",
    "get_transfer_strategy",
    "get_vllm_receive_backend",
]
