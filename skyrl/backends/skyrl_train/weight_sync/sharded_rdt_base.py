# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Trainer-side weight-transfer base ABCs, vendored from the vLLM RDT fork.

These classes live in ``vllm/distributed/weight_transfer/base.py`` in the
``vllm-rdt-weight-sync`` fork, but are NOT present in the pinned
``vllm==0.23.0`` wheel (0.23.0 ships only the *worker*-side
``WeightTransferEngine`` / ``WeightTransferInitInfo`` / ``WeightTransferUpdateInfo``,
which the consumer engine still imports from the installed vLLM). The
sidecar trainer engine (``sharded_rdt_trainer.py``) needs the trainer-side ABCs,
so they are copied here VERBATIM.

REMOVAL: once SkyRL's pinned vLLM carries these in
``vllm.distributed.weight_transfer.base``, delete this module and repoint
``sharded_rdt_trainer.py``'s import back to that path.

Two consequences of that split, both load-bearing:

* ``layerwise_groups``, ``WeightSource.groups()`` and ``WeightSource.iter_groups()``
  live HERE (as in the fork's ``base.py``), because they define what a group index
  means for a ``WeightSource``. ``sharded_rdt_common`` re-exports
  ``layerwise_groups`` for callers that already import it from there.
* ``WeightTransferEngine.defers_processing`` / ``drain_pending()`` are declared on
  the fork's ABC, but the *worker*-side ABC still comes from the pinned wheel,
  which has neither. So ``sharded_rdt_engine`` declares both on the concrete
  engine class, and ``inference_servers/layerwise_reload.py`` probes them with
  ``getattr`` — which works against either ABC.
"""

from abc import ABC, abstractmethod
from collections.abc import Collection, Iterator
from dataclasses import dataclass, field
from typing import (
    Any,
    ClassVar,
    Generic,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import torch
from typing_extensions import Self

TTrainerInitInfo = TypeVar("TTrainerInitInfo", bound="TrainerInitInfo")

# A trainer supplies its parameters as a `WeightSource` (defined below): a
# re-iterable stream of materialized `(name, tensor)` pairs plus a `metadata()`
# channel. The built-in `ModuleSource` uses `materialize_full_tensor`.


def _stack_key(name: str) -> tuple[str, int] | None:
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

    This defines what a *group index* means for `WeightSource.groups` and
    `WeightSource.iter_groups`: index *g* names the same group on every trainer
    rank and every consumer, because it is derived from one rank's `metadata()`
    order.

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


def materialize_full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return a full, locally-materialized tensor ready to send.

    FSDP shards (DTensors) expose `full_tensor()`, a collective all-gather;
    regular tensors do not and are returned unchanged. Trainer engines call
    this at send time so the (potentially expensive) gather happens exactly
    once — reading `.shape`/`.dtype` for metadata does not trigger it.
    """
    full_tensor = getattr(tensor, "full_tensor", None)
    return full_tensor() if callable(full_tensor) else tensor


@dataclass(frozen=True)
class ParamMeta:
    """Name / wire dtype / full (HF) shape for one output parameter."""

    name: str
    dtype: torch.dtype
    shape: tuple[int, ...]


class WeightSource(ABC):
    """A re-iterable source of the trainer's weights, handed to a trainer engine.

    Two channels:

    * `metadata()` — `(name, wire dtype, full shape)` for every parameter,
      *without* transferring. Cheap when shapes are known locally (FSDP
      `DTensor` global shape); may be expensive on first call for backends that
      must materialize to learn shapes (e.g. a Megatron-Bridge export), in which
      case it should cache.
    * iteration — yields fully-materialized `(name, tensor)` pairs, one at a
      time. Materializing is typically a collective (FSDP `full_tensor()`, a
      Megatron export), so the ranks that share a parameter must iterate it in
      the same order in lockstep, or they deadlock.
    * `held_names()` — which parameters this rank holds, for producers that are
      split so each rank holds only part of the model. Defaults to all.
    * `iter_groups()` — the same stream batched per gather group (see
      `layerwise_groups`). Defaults to batching `__iter__`; override to
      materialize a whole group in one step.

    `iter(source)` must yield a *fresh* pass each round. Backends with custom
    producer logic (Megatron export, RDT plans, MoE re-fusing) subclass this.
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
          all built from one rank's metadata, so a rank that reported only its own
          share would leave the rest of the model silently un-transferred. The
          sharded-RDT engine cross-checks this across ranks at init.
        * Every name must be held by at least one rank, or it can never be
          served. The engine raises at init naming the first orphan.
        * Iteration must cover exactly `groups()` in metadata order, yielding a
          real tensor for each held name and `None` for the rest. A group's
          gather is a collective among the ranks that hold part of it, so the
          name must still appear (to keep the order check aligned) while the
          data is absent.

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

        Override when a backend can produce a whole group at once. Materializing
        is usually a collective, and driving it per group instead of per tensor
        turns ~37k generator resumes into ~95 on a per-expert MoE model (worth
        ~0.9s per sync there). An override must yield the same batches in the same
        order as this default.
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


class ModuleSource(WeightSource):
    """`WeightSource` over `module.named_parameters()` — the common case.

    Handles both plain dense modules and FSDP-sharded ones with no special
    casing: iteration all-gathers each `DTensor` via `full_tensor()` (a
    collective) and passes regular tensors through. `metadata()` reads the
    *global* `.shape` / `.dtype`, so it never triggers a gather.
    """

    def __init__(self, module: torch.nn.Module) -> None:
        self._module = module

    def metadata(self) -> list[ParamMeta]:
        return [ParamMeta(name, p.dtype, tuple(p.shape)) for name, p in self._module.named_parameters()]

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor]]:
        for name, param in self._module.named_parameters():
            yield name, materialize_full_tensor(param)


@dataclass
class TrainerInitInfo:
    """Base trainer-side init info: which trainer rank drives the transfer.

    `rank` is this trainer process's rank, provided **explicitly** by the
    caller — the engine does not read it from a global process group, which is
    ambiguous once several groups (FSDP / TP / PP / EP) exist. Rank 0 is always
    the sender: only it opens the endpoint and drives the inference-side RPCs,
    while every rank still runs the trainer-side collectives. Backend subclasses
    add their own (positional) fields; `rank` is keyword-only so that ordering
    never conflicts.

    Every concrete subclass sets a class-level `backend` string (the same key it
    registers under in `WeightTransferTrainerFactory`). The factory reads it to
    dispatch, so callers pass only the init info/ It is a `ClassVar`
    (a fixed per-backend constant), so it is not an ``__init__`` field.
    """

    backend: ClassVar[str]

    rank: int = field(kw_only=True)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "backend", None):
            raise TypeError(
                f"{cls.__name__} must set a class-level `backend` string "
                "(the WeightTransferTrainerFactory registry key)."
            )

    @property
    def is_sender(self) -> bool:
        return self.rank == 0


@runtime_checkable
class VLLMWeightSyncClient(Protocol):
    """Trainer-side stub for the inference engine's weight-sync control plane.

    Mirrors the weight-sync methods that the inference engine exposes
    (`EngineClient` / the HTTP RLHF routes / Ray actors). A
    `TrainerWeightTransferEngine` drives the full handshake through this
    protocol so trainer code never has to know the transport.

    All methods are synchronous and accept plain dicts (matching what the
    inference side already accepts). Concurrency that some backends need
    (e.g. NCCL must run `update_weights` concurrently with the trainer-side
    broadcast) is the engine's responsibility, not the client's, so the
    protocol stays a flat four-method surface that any wrapper can implement.

    The protocol is structural (PEP 544), so user implementations need only
    define these four methods — no import or subclassing required.
    """

    def init_weight_transfer_engine(self, init_info: dict[str, Any]) -> None: ...

    def start_weight_update(self) -> None: ...

    def update_weights(self, update_info: dict[str, Any]) -> None: ...

    def finish_weight_update(self) -> None: ...


class TrainerWeightTransferEngine(ABC, Generic[TTrainerInitInfo]):
    """Trainer-side weight transfer engine.

    Symmetric to `WeightTransferEngine` but lives in the training process.
    Constructed via the `trainer_init` factory classmethod; carries any
    backend-specific state (NCCL communicators, IPC device info, transfer
    plans) on `self`. Full-resync backends (NCCL, IPC) take a `WeightSource` at
    `trainer_init` and replay it each round via the no-argument
    `send_weights()`. Backends that push per-round deltas instead (e.g. sparse
    patches) leave `source` as `None` and take their payload as a `send_weights`
    argument.

    Unlike the worker engine, the trainer side does not take a
    `WeightTransferConfig`: the backend is selected from the init info's
    `backend` `ClassVar` (so callers pass only the init info), and the static
    wire params (packed, buffer sizes) ride the backend-specific
    `TrainerInitInfo`, which the sender also propagates to the worker at the init
    handshake.

    Multi-rank trainers: `trainer_init` and `send_weights` are
    called on *every* trainer rank. Rank 0 is the sender, resolved once at
    `trainer_init` into `is_sender`. Non-sender ranks still run every
    collective (iterating the source, metadata export, IPC handle all-gather) so
    the group stays aligned, but each engine explicitly guards the control-plane
    RPCs and the transmit on `self.is_sender`, so only the sender touches the
    client.

    Subclasses should define:
        init_info_cls: Type of backend-specific trainer init info
    """

    # Subclasses should override this class attribute
    init_info_cls: type[TTrainerInitInfo]

    def __init__(
        self,
        *,
        client: "VLLMWeightSyncClient",
        source: "WeightSource | None" = None,
        is_sender: bool = True,
    ) -> None:
        self.is_sender = is_sender
        # The real client is held on every rank; each engine only *calls* it when
        # `is_sender`, so non-sender ranks never touch the wire.
        self.client = client
        self.source = source

    @classmethod
    @abstractmethod
    def trainer_init(
        cls,
        init_info: TTrainerInitInfo,
        *,
        client: "VLLMWeightSyncClient",
        source: "WeightSource | None" = None,
    ) -> Self:
        """Rendezvous with the inference side and return a ready instance.

        Called on every trainer rank. The sender drives the full handshake via
        `client` (build the worker-side init info, call
        `client.init_weight_transfer_engine`, open the trainer-side endpoint);
        non-sender ranks skip the rendezvous and the RPC.
        """
        raise NotImplementedError

    @abstractmethod
    def send_weights(self) -> None:
        """Push weights to inference workers and drive the full update round
        trip: `start_weight_update`, `update_weights` (run concurrently with the
        trainer-side broadcast when the backend requires it), then
        `finish_weight_update`. Called on every trainer rank.
        """
        raise NotImplementedError

    def shutdown(self) -> None:
        """Tear down communicators / process groups. Default no-op."""
