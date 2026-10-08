# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""What sharded RDT needs on top of vLLM's trainer-side weight-transfer ABCs.

Those ABCs come from the wheel. This module holds the two extra ``WeightSource``
channels a *pull* backend needs and vLLM has no concept of, plus
``layerwise_groups``, which makes a group index mean the same thing on every rank
and every consumer:

* ``held_names()`` -- per-rank **ownership** under pipeline / expert parallelism.
  A consumer routes each parameter to a rank that holds it. Not a chunking
  concern, and not optional: the default (hold everything) is correct at
  pp=1/ep=1 and wrong above it.
* ``groups()`` / ``iter_groups()`` -- the coordination index the producer's free
  barrier counts (``_inflight`` is keyed by group), plus batching: gathering per
  group rather than per tensor turns ~37k generator resumes into ~95 on a
  per-expert MoE model.

They stay here rather than in ``weight_sync/sources.py``, which is vLLM's
contract verbatim, so a future upstream chunking change costs nothing. If vLLM
grows ownership or group channels, :class:`GroupedWeightSource` collapses into
them.
"""

from abc import abstractmethod
from collections.abc import Collection, Iterator

import torch
from vllm.distributed.weight_transfer.base import (
    ParamMeta,
    TrainerInitInfo,
    TrainerWeightTransferEngine,
    VLLMWeightSyncClient,
    WeightSource,
    materialize_full_tensor,
)

__all__ = [
    "GroupedWeightSource",
    "ParamMeta",
    "TrainerInitInfo",
    "TrainerWeightTransferEngine",
    "VLLMWeightSyncClient",
    "WeightSource",
    "layerwise_groups",
    "materialize_full_tensor",
]


def _stack_key(name: str) -> "tuple[str, int] | None":
    """``(prefix, index)`` of the OUTERMOST integer segment, or None if there is
    none.

    Outermost is what keeps a MoE layer whole:
    ``model.layers.3.mlp.experts.7.w1`` keys on the layer, not the expert.
    """
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part.isdigit():
            return ".".join(parts[:i]), int(part)
    return None


def layerwise_groups(names: list[str]) -> list[list[str]]:
    """Partition flat parameter names into one group per decoder layer, keyed on
    the outermost index segment of each name.

    This defines what a *group index* means for `GroupedWeightSource.groups` and
    `GroupedWeightSource.iter_groups`: index *g* names the same group on every
    trainer rank and every consumer, because it is derived from one rank's
    `metadata()` order.

    Keying on the index rather than a literal prefix needs no per-architecture
    naming table: ``model.layers.0.``, ``model.language_model.layers.0.``,
    ``transformer.h.0.``, ``backbone.layers.0.`` and a vision tower's
    ``visual.blocks.0.`` all partition alike. Matching one fixed prefix does not,
    and its failure is silent — every name lands in a single group holding the
    whole model, which defeats the per-layer bound below.

    Un-indexed names split by POSITION relative to the first indexed one: the pre
    block (embeddings) and the post block (the final norm, `lm_head`, and any
    inter-stack projector). Post lands last however early it arrived, which is
    what a pipeline-parallel source needs — Megatron-Bridge streams the last
    stage's output block *before* its layers.

    Stacks come out in first-appearance order of their prefix and ascending index
    within it, whatever order the source yielded them, so a source can normalize
    an arbitrary export order by flattening this partition.

    Backends that gather and free per group (sharded RDT) also use it as the unit
    of transfer, which bounds their buffer sizes: without it a whole model becomes
    one chunk.
    """
    pre: list[str] = []
    post: list[str] = []
    stacks: dict[tuple[str, int], list[str]] = {}
    order: list[tuple[str, int]] = []
    for name in names:
        key = _stack_key(name)
        if key is None:
            (post if order else pre).append(name)
            continue
        if key not in stacks:
            stacks[key] = []
            order.append(key)
        stacks[key].append(name)

    prefix_rank: dict[str, int] = {}
    for key in order:
        prefix_rank.setdefault(key[0], len(prefix_rank))
    order.sort(key=lambda key: (prefix_rank[key[0]], key[1]))

    groups: list[list[str]] = [pre] if pre else []
    groups += [stacks[key] for key in order]
    if post:
        groups.append(post)
    return groups


class GroupedWeightSource(WeightSource):
    """A ``WeightSource`` with the ownership and group channels sharded RDT pulls over.

    Adds two channels to vLLM's ``metadata()`` + ``__iter__`` contract:

    * `held_names()` — which parameters this rank holds, for producers that are
      split so each rank holds only part of the model. Defaults to all.
    * `iter_groups()` — the same stream batched per gather group (see
      `layerwise_groups`). Defaults to batching `__iter__`; override to
      materialize a whole group in one step.
    """

    @abstractmethod
    def metadata(self) -> list[ParamMeta]:
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[str, torch.Tensor]]:
        raise NotImplementedError

    def held_names(self) -> "Collection[str] | None":
        """The parameters this rank holds, or None for all of them.

        This is the whole ownership contract. Override it when producers are
        split so each holds only part of the model — pipeline parallelism (a rank
        holds some layers), expert parallelism (a rank holds some experts), or
        any combination, including layouts that fit neither. A consumer routes
        each name to a rank that holds it, so per-name is the granularity that
        matters; the engine derives everything else from this.

        Three requirements come with overriding it:

        * `metadata()` must still describe the WHOLE model on every rank. The
          group partition, the iteration checks and the consumers' pull plans are
          all built from one rank's metadata, so a rank reporting only its own
          share would leave the rest silently un-transferred. The engine
          cross-checks this across ranks at init.
        * Every name must be held by at least one rank, or it can never be
          served. The engine raises at init naming the first orphan.
        * Iteration must cover exactly `groups()` in metadata order, yielding a
          real tensor for each held name and `None` for the rest -- a group's
          gather is a collective among the ranks holding part of it, so the name
          must appear to keep the order check aligned even when the data does not.

        Returns:
            The held parameter names, or None to hold every one.
        """
        return None

    def groups(self) -> list[list[str]]:
        """This rank's gather groups, in metadata order: `layerwise_groups` over
        `metadata()`, restricted to the groups holding at least one held name.

        A group with nothing held here is not iterated at all — its gather is a
        collective among the ranks that do hold part of it.
        """
        groups = layerwise_groups([m.name for m in self.metadata()])
        held = self.held_names()
        if held is None:
            return groups
        held = set(held)
        return [g for g in groups if any(n in held for n in g)]

    def iter_groups(self) -> Iterator[tuple[list[str], list[torch.Tensor]]]:
        """Yield one `(names, tensors)` batch per group from `groups()`.

        The default drives `__iter__` and batches its output, checking as it goes
        that the names arrive in metadata order — ranks sharing a parameter
        materialize it with a collective, so a rank that iterates out of order
        deadlocks its peers rather than returning wrong data.

        Override when a backend can produce a whole group at once; it must yield
        the same batches in the same order as this default.
        """
        it = iter(self)
        for group in self.groups():
            names: list[str] = []
            tensors: list[torch.Tensor] = []
            for expected in group:
                name, tensor = next(it)
                if name != expected:
                    raise RuntimeError(
                        f"WeightSource yielded {name!r} but expected "
                        f"{expected!r}; iteration order must match metadata()."
                    )
                names.append(name)
                tensors.append(tensor)
            yield names, tensors
