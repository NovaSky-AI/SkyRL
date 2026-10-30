"""Accelerator shim for the FSDP training path.

SkyRL's FSDP workers were written against ``torch.cuda``. This module is the one
place that knows which accelerator the trainer runs on, so the workers can call
``device.synchronize()`` instead of ``torch.cuda.synchronize()``.

The accelerator is chosen by ``SKYRL_DEVICE``:

==========  ============================================  ==================
value       stack                                         c10d backend
==========  ============================================  ==================
``cuda``    stock PyTorch (default; upstream behaviour)   ``cpu:gloo,cuda:nccl``
``tpu``     TorchTPU (``torch_tpu``, device ``"tpu"``)    ``cpu:gloo,tpu:tpu_dist``
==========  ============================================  ==================

With ``SKYRL_DEVICE`` unset every function here does exactly what the original
``torch.cuda`` call did, so GPU runs are unchanged.

TorchTPU notes (torch_tpu 0.2.x):

* One process per chip. A chip can be opened by one process only, so policy,
  ref and the rollout engine cannot share a chip (no ``colocate_all``).
* ``torch.accelerator.*`` is the supported API for sync and memory stats;
  ``torch.tpu.synchronize`` is deprecated.
* Multi-process launch needs the topology env (``TORCH_TPU_TOPOLOGY``,
  ``TORCH_TPU_SLICEBUILDER_ADDRESSES``, ...) set before the backend initialises.
  :func:`configure_process_env` sets it from ``SKYRL_TPU_GROUP_ENV_<group>``
  values exported by the launch script.
"""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from functools import lru_cache
from typing import Any, Optional

import torch

_CUDA = "cuda"
_TPU = "tpu"


@lru_cache(maxsize=None)
def device_type() -> str:
    """``"cuda"`` (default) or ``"tpu"``, from ``SKYRL_DEVICE``."""
    value = os.environ.get("SKYRL_DEVICE", _CUDA).strip().lower()
    if value not in (_CUDA, _TPU):
        raise ValueError(f"SKYRL_DEVICE must be 'cuda' or 'tpu', got {value!r}")
    return value


def is_tpu() -> bool:
    return device_type() == _TPU


def is_cuda() -> bool:
    return device_type() == _CUDA


_BACKEND_LOADED = False


def _ensure_backend_loaded() -> None:
    """Register TorchTPU's ``"tpu"`` device in this process (idempotent; no-op off TPU).

    TPU runs set ``TORCH_DEVICE_BACKEND_AUTOLOAD=0``: with autoload on, every process that
    imports torch, such as DataLoader workers, opens a TPU chip and collides with the training
    workers (``/dev/vfio/<n> ... already opened``). Only processes that own a chip call this,
    which runs TorchTPU's autoload hook (``torch_tpu._loader:load``) explicitly. A plain
    ``import torch_tpu`` does not register the device name.
    """
    global _BACKEND_LOADED
    if _BACKEND_LOADED or not is_tpu():
        return
    try:
        import torch_tpu  # noqa: F401
    except ImportError as e:  # pragma: no cover - depends on the host
        raise RuntimeError("SKYRL_DEVICE=tpu but torch_tpu is not installed (pip install torch_tpu)") from e
    # `import torch_tpu` adds `torch.tpu` but leaves the backend unnamed ("privateuseone");
    # only the loader registers "tpu".
    if torch._C._get_privateuse1_backend_name() != "tpu":
        from torch_tpu._loader import load

        load()
    _BACKEND_LOADED = True


def is_available() -> bool:
    if is_tpu():
        _ensure_backend_loaded()
        acc = getattr(torch, "accelerator", None)
        return bool(acc is not None and acc.is_available())
    return torch.cuda.is_available()


def device_count() -> int:
    if is_tpu():
        # One process per chip: a TPU process sees exactly its own device.
        return 1
    return torch.cuda.device_count()


def current_device() -> Any:
    """What ``tensor.to(...)`` and ``to_empty(device=...)`` should receive."""
    if is_tpu():
        _ensure_backend_loaded()
        return torch.device(_TPU)
    return torch.cuda.current_device()


def torch_device() -> torch.device:
    """A ``torch.device`` for this rank's accelerator."""
    if is_tpu():
        return current_device()
    return torch.device(_CUDA, torch.cuda.current_device())


def set_device(local_rank: int) -> None:
    if is_tpu():
        # Device selection is done by the topology env before init, not per call.
        _ensure_backend_loaded()
        return
    torch.cuda.set_device(local_rank)


def synchronize() -> None:
    if is_tpu():
        torch.accelerator.synchronize()
        return
    torch.cuda.synchronize()


def empty_cache() -> None:
    if is_tpu():
        acc = getattr(torch, "accelerator", None)
        fn = getattr(acc, "empty_cache", None) if acc is not None else None
        if fn is not None:
            fn()
        return
    torch.cuda.empty_cache()


def manual_seed_all(seed: int) -> None:
    if is_tpu():
        # torch.manual_seed seeds every registered device generator, including tpu.
        torch.manual_seed(seed)
        return
    torch.cuda.manual_seed_all(seed)


def get_rng_state() -> Optional[torch.Tensor]:
    if is_tpu():
        return None
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return torch.cuda.get_rng_state()
    return None


def set_rng_state(state: torch.Tensor) -> None:
    if is_tpu() or state is None:
        return
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.cuda.set_rng_state(state)


def mem_get_info() -> tuple[int, int]:
    """(free, total) bytes. TorchTPU exposes ``torch.accelerator.get_memory_info``."""
    if is_tpu():
        acc = torch.accelerator
        fn = getattr(acc, "get_memory_info", None) or getattr(acc, "mem_get_info", None)
        if fn is not None:
            try:
                info = fn()
                if isinstance(info, (tuple, list)) and len(info) == 2:
                    return int(info[0]), int(info[1])
                if isinstance(info, dict):
                    total = int(info.get("bytes_limit") or info.get("total") or 0)
                    used = int(info.get("bytes_in_use") or info.get("used") or 0)
                    return max(total - used, 0), total
            except Exception:
                pass
        return 0, 0
    return torch.cuda.mem_get_info()


def memory_allocated() -> int:
    if is_tpu():
        fn = getattr(torch.accelerator, "memory_allocated", None)
        return int(fn()) if fn is not None else 0
    return torch.cuda.memory_allocated()


def memory_reserved() -> int:
    if is_tpu():
        fn = getattr(torch.accelerator, "memory_reserved", None)
        return int(fn()) if fn is not None else 0
    return torch.cuda.memory_reserved()


def dist_backend() -> str:
    """Backend string for ``torch.distributed.init_process_group``.

    ``SKYRL_DIST_BACKEND`` overrides it, for a TorchTPU release that names its
    c10d backend differently.
    """
    override = os.environ.get("SKYRL_DIST_BACKEND")
    if override:
        return override
    if is_tpu():
        return "cpu:gloo,tpu:tpu_dist"
    return "cpu:gloo,cuda:nccl"


def autocast(dtype: torch.dtype = torch.bfloat16):
    """bf16 autocast on CUDA. On TPU the weights are already bf16 and the MXU
    computes bf16 x bf16 -> fp32 by default, so autocast is skipped unless
    ``SKYRL_TPU_AUTOCAST=1`` (TorchTPU's autocast support is not documented)."""
    if is_tpu():
        if os.environ.get("SKYRL_TPU_AUTOCAST") == "1":
            return torch.autocast(device_type=_TPU, dtype=dtype)
        return nullcontext()
    return torch.autocast(dtype=dtype, device_type=_CUDA)


def supports_cuda_memory_tools() -> bool:
    """Memory history, allocator settings, IPC, NUMA by PCI bus: CUDA only."""
    return is_cuda() and torch.cuda.is_available()


_FSDP_PATCHED = False


def patch_fsdp_move_states_for_tpu() -> None:
    """Let FSDP2 move CPU parameters onto ``tpu`` (idempotent; no-op off TPU).

    ``fully_shard`` moves each CPU parameter with ``tensor.data = tensor.to(device)``
    (``torch/distributed/fsdp/_fully_shard/_fsdp_init.py:_move_states_to_device``).
    TorchTPU tensors are not shallow-copy compatible with CPU tensors, so that
    assignment raises ``Attempted to call variable.set_data(tensor), but variable and
    tensor have incompatible tensor type``. FSDP2 already
    swaps tensors instead of assigning ``.data`` for wrapper subclasses; this routes
    every incompatible tensor through that same ``torch.utils.swap_tensors`` path.
    Compatible tensors (all of them on CUDA) take the original code path unchanged.
    """
    global _FSDP_PATCHED
    if _FSDP_PATCHED or not is_tpu():
        return
    import itertools

    import torch.nn as nn
    from torch.distributed.fsdp._fully_shard import _fsdp_init

    original = _fsdp_init._move_states_to_device

    def _move_states_to_device(params, buffers, device):
        rest_params, rest_buffers = [], []
        for tensor, bucket in itertools.chain(((p, rest_params) for p in params), ((b, rest_buffers) for b in buffers)):
            if tensor.device == device or tensor.device.type == "meta" or type(tensor).__name__ == "DTensor":
                bucket.append(tensor)
                continue
            with torch.no_grad():
                moved = tensor.to(device)
            if torch._has_compatible_shallow_copy_type(tensor, moved):
                bucket.append(tensor)
                continue
            if isinstance(tensor, nn.Parameter):
                moved = nn.Parameter(moved, requires_grad=tensor.requires_grad)
            torch.utils.swap_tensors(tensor, moved)
        original(rest_params, rest_buffers, device)

    _fsdp_init._move_states_to_device = _move_states_to_device
    _FSDP_PATCHED = True


_MROPE_PATCHED = False


def _apply_interleaved_mrope_cloned(self, freqs, mrope_section):
    # Same result as transformers' version, without in-place writes: TorchTPU 0.2 aborts on its
    # strided in-place assignment between views of one tensor (inversion.cc:112).
    d = freqs.shape[-1]
    col = torch.arange(d, device=freqs.device)
    use_h = (col % 3 == 1) & (col < mrope_section[1] * 3)
    use_w = (col % 3 == 2) & (col < mrope_section[2] * 3)
    return torch.where(use_h, freqs[1], torch.where(use_w, freqs[2], freqs[0]))


def patch_qwen35_mrope_for_tpu() -> None:
    """Make Qwen3.5's ``apply_interleaved_mrope`` alias-free on TPU (idempotent; no-op off TPU).

    It assigns into ``freqs[0]`` (a view) from other slices of ``freqs``. TorchTPU 0.2 aborts
    on in-place operations between views of one tensor (``inversion.cc:112``). The replacement
    uses ``torch.where``: same values, no in-place writes.
    """
    global _MROPE_PATCHED
    if _MROPE_PATCHED or not is_tpu():
        return
    import inspect

    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    except ImportError:
        return
    for _, cls in inspect.getmembers(m, inspect.isclass):
        if "apply_interleaved_mrope" in cls.__dict__:
            cls.apply_interleaved_mrope = _apply_interleaved_mrope_cloned
    # Newer transformers do the same interleave in the text rotary embedding.
    text_rope = getattr(m, "Qwen3_5TextRotaryEmbedding", None)
    if text_rope is not None and "recomposition_frequencies" in text_rope.__dict__:
        text_rope.recomposition_frequencies = _text_mrope_recomposition
    _MROPE_PATCHED = True


def _text_mrope_recomposition(self, freq):
    # ``Qwen3_5TextRotaryEmbedding.recomposition_frequencies`` without the in-place strided write
    # into a view of ``freq``: the same T/H/W interleave, built with ``torch.where``.
    d = freq.shape[-1]
    col = torch.arange(d, device=freq.device)
    use_h = (col % 3 == 1) & (col < self.mrope_section[1] * 3)
    use_w = (col % 3 == 2) & (col < self.mrope_section[2] * 3)
    thw = torch.where(use_h, freq[1], torch.where(use_w, freq[2], freq[0]))
    return torch.cat((thw, thw), dim=-1)


_VISION_ROPE_PATCHED = False


def patch_qwen35_vision_rope_for_tpu() -> None:
    """Keep Qwen3.5's vision RoPE ``inv_freq`` on its input's device in every worker (idempotent; TPU only).

    ``Qwen3_5VisionRotaryEmbedding.inv_freq`` is a non-persistent buffer that FSDP can leave on
    CPU, and its forward multiplies it with on-device ``position_ids`` without moving it, which
    fails with a ``tpu:0`` / ``cpu`` device mismatch. The patch runs in each worker process,
    since a patch applied in the driver does not reach the Ray workers that run the model.
    """
    global _VISION_ROPE_PATCHED
    if _VISION_ROPE_PATCHED or not is_tpu():
        return
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    except ImportError:
        return
    cls = getattr(m, "Qwen3_5VisionRotaryEmbedding", None)
    if cls is not None:
        original_forward = cls.forward

        # Wraps the installed transformers' own forward, so its signature and math are kept.
        def _forward(self, *args, **kwargs):
            ref = next((a for a in (*args, *kwargs.values()) if isinstance(a, torch.Tensor)), None)
            if ref is not None and self.inv_freq.device != ref.device:
                self.inv_freq = self.inv_freq.to(ref.device)
            return original_forward(self, *args, **kwargs)

        cls.forward = _forward
    _VISION_ROPE_PATCHED = True


def configure_process_env(*, rank: int, local_rank: int, world_size: int, group: Optional[str] = None) -> None:
    """Set this worker's TorchTPU launch env. Must run before the first TPU op.

    The launcher exports, per actor group (``policy``, ``ref``, ``critic``), a
    JSON object ``SKYRL_TPU_GROUP_ENV_<GROUP>`` of the form::

        {"env": {"TORCH_TPU_TOPOLOGY": "2,2,1", ...},       # shared by the group
         "per_rank": [{"TPU_VISIBLE_DEVICES": "0"}, ...]}   # one entry per rank

    which is what ``python -m torch.tpu.distributed`` prints for that chip
    subset, plus the per-rank device selection. Without it, the process keeps
    whatever env it inherited (enough for a group that owns the whole VM).
    """
    if not is_tpu():
        return
    _ensure_backend_loaded()
    patch_fsdp_move_states_for_tpu()
    patch_qwen35_mrope_for_tpu()
    patch_qwen35_vision_rope_for_tpu()
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ.setdefault("NODE_RANK", "0")
    if group is None:
        return
    raw = os.environ.get(f"SKYRL_TPU_GROUP_ENV_{group.upper()}")
    if not raw:
        return
    spec = json.loads(raw)
    for key, value in (spec.get("env") or {}).items():
        os.environ[key] = str(value)
    per_rank = spec.get("per_rank") or []
    if rank < len(per_rank):
        for key, value in per_rank[rank].items():
            os.environ[key] = str(value)
