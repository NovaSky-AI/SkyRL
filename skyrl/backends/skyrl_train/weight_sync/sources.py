"""``WeightSource`` implementations over SkyRL's training backends.

A ``WeightSource`` is vLLM's trainer-side weight contract
(``vllm.distributed.weight_transfer.base``): ``metadata()`` and ``__iter__``,
which must agree element for element. Chunking is vLLM's: the packed producers
cut bounded-memory chunks out of a fixed buffer and consume the source lazily,
so a source's only obligation is to be a lazy generator that does not retain.

Sharded RDT needs two more channels (per-rank ownership under PP/EP, and a group
index) that vLLM has no concept of; they live in
``sharded_rdt/sharded_rdt_base.GroupedWeightSource``.

Imports vLLM at module scope, so import this lazily from anything that must work
without the wheel.
"""

from typing import Any, Iterator, List, Optional, Tuple

import torch
from vllm.distributed.weight_transfer.base import (
    ParamMeta,
    WeightSource,
    materialize_full_tensor,
)

__all__ = [
    "FsdpWeightSource",
    "MegatronWeightSource",
    "ParamMeta",
    "WeightSource",
    "materialize_full_tensor",
]


class FsdpWeightSource(WeightSource):
    """``WeightSource`` over an FSDP2-sharded HF model.

    ``metadata()`` reads ``state_dict()`` shapes only: an FSDP2 ``DTensor``'s
    ``.shape`` is already the global shape, so declaring the stream costs no
    collective. Iteration all-gathers each parameter (``full_tensor()``), which
    IS a collective, so every trainer rank must iterate the same source in the
    same order.

    Args:
        model: the inner HF module (``self.model.model`` on the worker), whose
            ``state_dict()`` keys are the names vLLM expects.
        dtype: inference dtype. Both channels use it.
        weight_prefix: prepended to every name (``"language_model."`` when
            syncing a CausalLM backbone into a vLLM multimodal namespace).
    """

    def __init__(self, model: torch.nn.Module, dtype: torch.dtype, weight_prefix: str = "") -> None:
        self._model = model
        self._dtype = dtype
        self._prefix = weight_prefix or ""

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    @property
    def weight_prefix(self) -> str:
        return self._prefix

    def metadata(self) -> List[ParamMeta]:
        sd = self._model.state_dict()
        return [ParamMeta(f"{self._prefix}{key}", self._dtype, tuple(param.shape)) for key, param in sd.items()]

    def __iter__(self) -> Iterator[Tuple[str, torch.Tensor]]:
        # The caller selects this rank's CUDA device before iterating; a worker
        # thread does not inherit it (see Worker._weight_sync_thread).
        device = torch.cuda.current_device() if torch.cuda.is_available() else None
        for key, param in self._model.state_dict().items():
            if device is not None:
                param = param.to(device, non_blocking=True)
            full = materialize_full_tensor(param).to(self._dtype).detach().contiguous()
            yield f"{self._prefix}{key}", full


class MegatronWeightSource(WeightSource):
    """``WeightSource`` over a Megatron model, via Megatron-Bridge.

    ``export_hf_weights(conversion_tasks=None)`` streams the whole model in
    HF-canonical order, gathering each parameter across TP / PP / EP internally
    and giving all ranks full tensors. It is a generator, so its collectives run
    as the consumer pulls and a lazy consumer never holds more than the tensors
    it is working on.

    Exporting in ONE call is required: the bridge's ``_accumulate_grouped_export``
    needs every task sharing a ``group_key`` present in the same call, or the
    expert weights are silently never yielded.

    There is no shape-only export, so ``metadata()`` runs a dry export and
    caches. Engines call it every round; the cost is one-time.
    """

    def __init__(self, bridge: Any, module: Any, dtype: torch.dtype) -> None:
        self._bridge = bridge
        self._module = module
        self._dtype = dtype
        self._meta: Optional[List[ParamMeta]] = None

    def _export(self) -> Iterator[Tuple[str, torch.Tensor]]:
        return self._bridge.export_hf_weights(self._module, show_progress=False, conversion_tasks=None)

    def metadata(self) -> List[ParamMeta]:
        if self._meta is None:
            meta: List[ParamMeta] = []
            for name, tensor in self._export():
                meta.append(ParamMeta(name, self._dtype, tuple(tensor.shape)))
                del tensor
            self._meta = meta
        return self._meta

    def __iter__(self) -> Iterator[Tuple[str, torch.Tensor]]:
        # See FsdpWeightSource.__iter__ on the device.
        device = torch.cuda.current_device() if torch.cuda.is_available() else None
        for name, tensor in self._export():
            full = tensor.to(device=device, dtype=self._dtype, non_blocking=True).detach().contiguous()
            yield name, full
