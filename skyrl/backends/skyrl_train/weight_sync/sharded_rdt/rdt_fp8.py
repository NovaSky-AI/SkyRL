"""MXFP8 over sharded RDT: serialize a grouped source into vLLM's MXFP8 checkpoint tensors.

With ``fp8_weight_sync_mode=mxfp8`` the trainer quantizes, as on the NCCL MXFP8 wire: every weight
the model's ``ModelFp8Spec`` marks quantizable becomes E4M3 codes plus an E8M0 ``weight_scale``
(``mx_cast_to_fp8``, the same bytes as the NCCL wire); everything else passes through unchanged.

Expansion is driven by metadata (shape only), so names this rank does not hold expand to ``None``
weight and scale entries. Each scale directly follows its weight, so ``layerwise_groups`` over
the expanded names gives the inner source's groups with the scales added.
"""

from __future__ import annotations

from collections.abc import Collection, Iterator
from functools import cached_property
from typing import List

import torch

from skyrl.backends.skyrl_train.weight_sync.fp8 import (
    MXFP8,
    MXFP8_GROUP_SIZE,
    SerializedFp8Config,
    mx_cast_to_fp8,
    scale_name_for_weight,
)
from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.sharded_rdt_base import (
    GroupedWeightSource,
    ParamMeta,
)

MXFP8_CODE_DTYPE = torch.float8_e4m3fn
MXFP8_SCALE_DTYPE = torch.uint8


class SerializedMxfp8GroupedSource(GroupedWeightSource):
    """Wrap a ``GroupedWeightSource`` so it yields MXFP8 checkpoint tensors.

    Ownership, groups and order are the inner source's; a held weight's scale is held with it.
    """

    def __init__(self, inner: GroupedWeightSource, config: SerializedFp8Config) -> None:
        if config.wire_format != MXFP8:
            raise ValueError(
                f"sharded_rdt carries only the MXFP8 wire, got wire_format={config.wire_format!r}; "
                "use the nccl or ipc backend for blockwise FP8"
            )
        self._inner = inner
        self._config = config
        self._spec = config.require_spec()
        self._scale_of: dict[str, str] = {}

    @property
    def inner(self) -> GroupedWeightSource:
        return self._inner

    def _quantizes(self, name: str, shape: tuple) -> bool:
        if name.removesuffix(".weight") in self._config.user_provided_exclude_list:
            return False
        if self._spec.stacked_experts(name) is not None:
            # Stacked expert tensors need the receiver's fused-MoE loader, which the RDT bake does not drive.
            raise NotImplementedError(f"sharded_rdt MXFP8 cannot serialize stacked MoE expert tensors ({name!r})")
        return len(shape) == 2 and self._spec.should_quantize(name, shape, MXFP8)

    @cached_property
    def metadata(self) -> List[ParamMeta]:
        meta: List[ParamMeta] = []
        for m in self._inner.metadata:
            shape = tuple(m.shape)
            if self._quantizes(m.name, shape):
                scale = scale_name_for_weight(m.name, MXFP8)
                self._scale_of[m.name] = scale
                meta.append(ParamMeta(m.name, MXFP8_CODE_DTYPE, shape))
                meta.append(ParamMeta(scale, MXFP8_SCALE_DTYPE, (shape[0], shape[1] // MXFP8_GROUP_SIZE)))
            else:
                meta.append(m)
        return meta

    def held_names(self) -> "Collection[str] | None":
        held = self._inner.held_names()
        if held is None:
            return None
        self.metadata  # noqa: B018 -- populates _scale_of
        out: list[str] = []
        for name in held:
            out.append(name)
            scale = self._scale_of.get(name)
            if scale is not None:
                out.append(scale)
        return out

    def _expand(self, names: list, tensors: list) -> tuple[list, list]:
        out_names: list = []
        out_tensors: list = []
        for name, tensor in zip(names, tensors):
            scale_name = self._scale_of.get(name)
            if scale_name is None:
                out_names.append(name)
                out_tensors.append(tensor)
                continue
            codes, scales = (None, None) if tensor is None else mx_cast_to_fp8(tensor)
            out_names += [name, scale_name]
            out_tensors += [codes, scales]
        return out_names, out_tensors

    def iter_groups(self) -> Iterator[tuple[list[str], list[torch.Tensor]]]:
        self.metadata  # noqa: B018 -- populates _scale_of
        for names, tensors in self._inner.iter_groups():
            out = self._expand(list(names), list(tensors))
            # Drop the inner group's BF16 tensors before the next group is gathered.
            names = tensors = None
            yield out

    def iter_full(self) -> Iterator[tuple[str, torch.Tensor]]:
        for names, tensors in self.iter_groups():
            yield from zip(names, tensors)
