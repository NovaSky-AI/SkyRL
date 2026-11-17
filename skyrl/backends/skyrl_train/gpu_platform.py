"""GPU-vendor decisions for the skyrl_train backend.

PyTorch on AMD ROCm presents the GPU as an ordinary ``cuda`` device: the device type is ``cuda``, the
namespace is ``torch.cuda``, and the ``nccl`` process-group backend is served by RCCL. SkyRL therefore
has no separate AMD backend. This module holds the few places where the two vendors need different
handling. Every function here does nothing on a torch build without ROCm.
"""

import os
from typing import Mapping, Optional

import torch

HIP_VISIBLE_DEVICES = "HIP_VISIBLE_DEVICES"
CUDA_VISIBLE_DEVICES = "CUDA_VISIBLE_DEVICES"
RAY_NOSET_HIP_VISIBLE_DEVICES = "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES"


def is_rocm() -> bool:
    """Returns True when this process's torch is a ROCm build.

    Reads a build attribute, so it is safe on a driver without a GPU and does not initialise one.
    """
    return getattr(torch.version, "hip", None) is not None


def ray_masks_amd_gpus(env: Optional[Mapping[str, str]] = None) -> bool:
    """Returns True when Ray writes a per-process ``HIP_VISIBLE_DEVICES`` for GPU actors on an AMD node.

    That is Ray's default. Ray skips the write only when ``RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES`` is
    ``1`` or ``true``; the CUDA variant of that variable has no effect on an AMD node.

    Args:
        env: Environment to read. Defaults to this process's environment.
    """
    env = os.environ if env is None else env
    return env.get(RAY_NOSET_HIP_VISIBLE_DEVICES, "").lower() not in ("1", "true")


def check_driver_device_masks(env: Optional[Mapping[str, str]] = None) -> None:
    """On ROCm, refuses a GPU restriction that Ray would silently ignore.

    Call in the process that starts Ray, before it does. Ray detects and assigns AMD GPUs through
    ``HIP_VISIBLE_DEVICES`` only, so a ``CUDA_VISIBLE_DEVICES`` set alone restricts nothing there.

    Args:
        env: Environment to check. Defaults to this process's environment.

    Raises:
        RuntimeError: ``CUDA_VISIBLE_DEVICES`` is set, ``HIP_VISIBLE_DEVICES`` is not, and torch is a
            ROCm build.
    """
    env = os.environ if env is None else env
    cuda = env.get(CUDA_VISIBLE_DEVICES)
    if cuda and not env.get(HIP_VISIBLE_DEVICES) and is_rocm():
        raise RuntimeError(
            f"{CUDA_VISIBLE_DEVICES}={cuda!r} is set without {HIP_VISIBLE_DEVICES}. On AMD GPUs Ray detects "
            f"and assigns devices through {HIP_VISIBLE_DEVICES} only, so this restriction would be ignored. "
            f"Unset {CUDA_VISIBLE_DEVICES}."
        )


_ROCM_UNSUPPORTED = {
    "sharded_rdt": (
        "weight_sync_backend='sharded_rdt' is not supported: SkyRL creates Ray's NIXL agent from the 'nixl' "
        "package, which ROCm builds ship under another name, and shares buffers with a sidecar pinned by "
        "CUDA_VISIBLE_DEVICES. Use 'nccl' or 'delta'."
    ),
    "fp8": (
        "FP8 training (fp8, fp8_param, or per-module recipes from te_precision_config_file) and FP8 weight "
        "sync are not supported: SkyRL picks FP8 recipes and the block-scale contract from NVIDIA compute "
        "capability, which means nothing on an AMD GPU."
    ),
    "mp_executor": (
        "generator.inference_engine.distributed_executor_backend='mp' is not supported with the Megatron "
        "backend: with it SkyRL confines each inference engine to its own GPUs through CUDA_VISIBLE_DEVICES, "
        "so the trainer and the engine's workers see different GPUs, and weight sync over RCCL to a multi-GPU "
        "engine never connects. Use the default 'ray'."
    ),
    "nccl_multi_gpu_engine": (
        "weight_sync_backend='nccl' cannot reach a non-colocated inference engine that spans several GPUs "
        "(tensor_parallel_size * pipeline_parallel_size > 1) while Ray masks GPUs per process: the trainer then "
        "sees one GPU and the engine's workers see several, and RCCL never finishes connecting them. Turn Ray's "
        "masking off for the whole run by exporting RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1 before "
        "launching (SkyRL's AMD image sets it), or keep the engine on one GPU."
    ),
}


def rocm_unsupported(feature: str) -> str:
    """Returns the error message for a feature that SkyRL rejects on ROCm.

    Args:
        feature: One of ``"sharded_rdt"``, ``"fp8"``, ``"mp_executor"``, ``"nccl_multi_gpu_engine"``.
    """
    return f"Not supported on AMD ROCm: {_ROCM_UNSUPPORTED[feature]}"
