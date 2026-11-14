"""``expert_nvfp4`` over sharded RDT: serialize a grouped source into ModelOpt NVFP4 expert tensors.

Each routed-expert projection becomes the four ModelOpt tensors of ``weight_sync/nvfp4.py``, quantized by the
same ``Nvfp4ExpertSerializer`` as the push backends, so the bytes match. Everything else passes through.
Experts always go out per expert: ``expert_nvfp4_stacked_sync`` is a push-backend layout.

Gate and up of an expert share one global scale, so both must be in the same gather group and held by the
same rank. Names this rank does not hold expand to ``None`` entries, declared by ``Nvfp4ExpertSpecSerializer``.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Iterator
from functools import cached_property
from typing import List

import torch

from skyrl.backends.skyrl_train.weight_sync.nvfp4 import (
    _EXPERT_RE,
    Nvfp4ExpertSerializer,
    Nvfp4ExpertSpecSerializer,
    Nvfp4WeightQuantSettings,
)
from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.sharded_rdt_base import (
    GroupedWeightSource,
    ParamMeta,
)

_SUFFIXES = ("weight", "weight_scale", "weight_scale_2", "input_scale")


class SerializedNvfp4GroupedSource(GroupedWeightSource):
    """Wrap a ``GroupedWeightSource`` so it yields ModelOpt NVFP4 routed-expert tensors.

    Ownership, groups and order are the inner source's; each quantized projection's four tensors are held
    (or ``None``) together.
    """

    def __init__(
        self, inner: GroupedWeightSource, settings: Nvfp4WeightQuantSettings, bf16_layers: Iterable[int] = ()
    ) -> None:
        self._inner = inner
        self._settings = settings
        self._bf16_layers = frozenset(bf16_layers)
        self._expanded: dict[str, list[str]] = {}

    @property
    def inner(self) -> GroupedWeightSource:
        return self._inner

    def _quantizes(self, name: str) -> bool:
        m = _EXPERT_RE.match(name)
        return m is not None and int(m.group("layer")) not in self._bf16_layers

    @cached_property
    def metadata(self) -> List[ParamMeta]:
        inner_meta = {m.name: m for m in self._inner.metadata}
        for name in inner_meta:
            if self._quantizes(name):
                base = name[: -len(".weight")]
                self._expanded[name] = [f"{base}.{suffix}" for suffix in _SUFFIXES]
        if not self._expanded:
            raise ValueError("expert_nvfp4 over sharded_rdt found no per-expert routed-expert weights to quantize")
        spec = Nvfp4ExpertSpecSerializer(self._settings, self._bf16_layers)
        meta: List[ParamMeta] = []
        for m in inner_meta.values():
            for name, s in spec(m.name, m):
                meta.append(s if s is m else ParamMeta(name, s.dtype, tuple(s.shape)))
        spec.finish()
        return meta

    def held_names(self) -> "Collection[str] | None":
        held = self._inner.held_names()
        if held is None:
            return None
        self.metadata  # noqa: B018 -- populates _expanded
        out: list[str] = []
        for name in held:
            out += self._expanded.get(name, [name])
        return out

    def _expand(self, names: list, tensors: list) -> tuple[list, list]:
        meta = self._inner_meta
        real = Nvfp4ExpertSerializer(self._settings, self._bf16_layers)
        spec = Nvfp4ExpertSpecSerializer(self._settings, self._bf16_layers)
        out_names: list = []
        out_tensors: list = []
        for name, tensor in zip(names, tensors):
            if tensor is None:
                for n, _ in spec(name, meta[name]):
                    out_names.append(n)
                    out_tensors.append(None)
            else:
                for n, t in real(name, tensor):
                    out_names.append(n)
                    out_tensors.append(t)
        # Raises on a gate/up pair split across groups or ranks.
        real.finish()
        spec.finish()
        return out_names, out_tensors

    @cached_property
    def _inner_meta(self) -> dict[str, ParamMeta]:
        return {m.name: m for m in self._inner.metadata}

    def iter_groups(self) -> Iterator[tuple[list[str], list[torch.Tensor]]]:
        self.metadata  # noqa: B018 -- validates and populates _expanded
        for names, tensors in self._inner.iter_groups():
            out = self._expand(list(names), list(tensors))
            # Drop the inner group's BF16 tensors before the next group is gathered.
            names = tensors = None
            yield out

    def iter_full(self) -> Iterator[tuple[str, torch.Tensor]]:
        for names, tensors in self.iter_groups():
            yield from zip(names, tensors)
