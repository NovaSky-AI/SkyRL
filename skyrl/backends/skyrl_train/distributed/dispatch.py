"""Defines dispatch and collect logic for distributed training"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

import ray
import torch
from ray import ObjectRef
from ray.actor import ActorHandle

from skyrl.backends.skyrl_train.training_batch import (
    MICROBATCH_SIZES_KEY,
    TrainingInputBatch,
    TrainingOutputBatch,
    pad_training_input_batch,
)
from skyrl.train.dataset.bin_packing import balanced_dp_assignment


class OrderedRefs(list):
    """A list of object refs that remembers how the driver permuted the samples.

    ``sample_order[i]`` is the batch index of the i-th sample in the concatenation of the
    per-rank chunks (DP-rank order). ``None`` means the chunks follow the batch order.
    """

    sample_order: Optional[List[int]] = None


def split_for_dp(
    data: TrainingInputBatch, dp_size: int, balance_max_tokens: int = 0
) -> Tuple[List[TrainingInputBatch], Optional[List[int]]]:
    """Split ``data`` into one chunk per DP rank.

    Default: ``dp_size`` chunks with the same number of samples, in batch order.

    With ``balance_max_tokens > 0``: token-balanced chunks from :func:`balanced_dp_assignment`.
    Each chunk lists its micro-batches in its metadata (``MICROBATCH_SIZES_KEY``). The second
    return value is the sample order of the concatenated chunks (``None`` for the default split).
    """
    assignment = None
    # Rows with sub-sequences are already laid out per rank by the collator.
    if balance_max_tokens > 0 and data.get("sub_seq_lengths") is None and data.get("attention_mask") is not None:
        token_counts = data["attention_mask"].sum(dim=1).cpu().tolist()
        assignment = balanced_dp_assignment(token_counts, dp_size, balance_max_tokens)
    if assignment is None:
        assert len(data) % dp_size == 0, "data batch size must be divisible by dp_size, got {} and {}".format(
            len(data), dp_size
        )
        return data.chunk(len(data) // dp_size), None

    chunks, sample_order = [], []
    for microbatches in assignment:
        indices = [i for microbatch in microbatches for i in microbatch]
        index = torch.tensor(indices, dtype=torch.long)
        chunk = data.__class__({key: (None if value is None else value[index]) for key, value in data.items()})
        chunk.metadata = dict(data.metadata or {})
        chunk.metadata[MICROBATCH_SIZES_KEY] = [len(microbatch) for microbatch in microbatches]
        chunks.append(chunk)
        sample_order.extend(indices)
    return chunks, sample_order


def restore_sample_order(items: List[Any], sample_order: Optional[List[int]]) -> List[Any]:
    """Undo the driver's permutation on a per-sample list (no-op without one, or on an empty list)."""
    if sample_order is None or not items:
        return items
    if len(items) != len(sample_order):
        raise ValueError(f"Got {len(items)} per-sample outputs for {len(sample_order)} dispatched samples")
    restored: List[Any] = [None] * len(items)
    for item, original_index in zip(items, sample_order):
        restored[original_index] = item
    return restored


@dataclass
class MeshRank:
    """Represents a rank in the device mesh.

    This is a tuple of (DP, SP, TP, PP) ranks.
    """

    dp: int
    sp: int
    tp: int
    pp: int

    world_size: int
    dp_size: int
    pp_size: int

    def is_collection_dp_rank(self) -> bool:
        """Check if this rank is a DP rank to collect from

        This is the rank with (SP=0, TP=0, PP=pp_size-1)

        Note: double check this for ETP > 1 (but this is not a typically used case)
        """
        return self.tp == 0 and self.pp == self.pp_size - 1 and self.sp == 0

    def __str__(self) -> str:
        return f"MeshRank(dp={self.dp}, sp={self.sp}, tp={self.tp}, pp={self.pp}, world_size={self.world_size}, dp_size={self.dp_size}, pp_size={self.pp_size})"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class ActorInfo:
    """Actor information for distributed training.

    This includes the actor handle and the rank in the device mesh.
    """

    handle: ActorHandle
    rank: MeshRank


class Dispatch(ABC):
    """Base class for dispatch types

    Dispatch types are responsible for:
    - dispatching method calls to actors handling data sharding if necessary
    - validating arguments for dispatch
    """

    @classmethod
    @abstractmethod
    def dispatch(cls, actor_infos: List[ActorInfo], method: str, *args, **kwargs) -> List[ObjectRef]:
        """Dispatches method calls to the actors with data sharding if necessary."""
        pass

    @classmethod
    @abstractmethod
    def validate_dispatch_args(cls, *args, **kwargs) -> Tuple[Tuple, Dict[str, Any]]:
        """Validate and process arguments for dispatch.

        Returns:
            Tuple of (args, kwargs) to be passed to dispatch
        """
        pass


class MeshDispatch(Dispatch):
    """Mesh dispatch type to dispatch data to a group of actors along the device mesh.

    Supports DP (Data Parallel), SP (Sequence Parallel), TP (Tensor Parallel) and PP (Pipeline Parallel) parallelism.
    The actor method should accept a single argument - the data batch.

    For data dispatch:

    * The input data is chunked into `dp_size` equal chunks, where `dp_size` is the size of data parallelism.
    * Each actor with the same DP rank processes the same data chunk in parallel.

    Example: For a world size of 8, with DP size=2, SP size=2, TP size=2, PP size=1:

    * Data dispatch: The data is chunked into 2 chunks. All actors with DP rank 0 process the first chunk,
      and all actors with DP rank 1 process the second chunk.
    """

    @classmethod
    def dispatch(
        cls,
        actor_infos: List[ActorInfo],
        method: str,
        data: TrainingInputBatch,
        dp_balance_max_tokens: int = 0,
        **kwargs,
    ) -> List[ObjectRef]:
        """``dp_balance_max_tokens > 0`` selects the token-balanced split (see :func:`split_for_dp`);
        the returned list then carries the sample order (:class:`OrderedRefs`)."""
        assert len(actor_infos) > 0, "actor_infos must be a non-empty list"
        object_refs = OrderedRefs()
        dp_size = actor_infos[0].rank.dp_size
        data_chunks, object_refs.sample_order = split_for_dp(data, dp_size, dp_balance_max_tokens)

        # Put each unique chunk in object store ONCE to avoid redundant serialization
        # when the same chunk is sent to multiple workers (e.g., SP/TP replicas)
        chunk_refs: List[ObjectRef] = [ray.put(chunk) for chunk in data_chunks]

        for actor_info in actor_infos:
            # Pass ObjectRef instead of data - workers will fetch from object store
            chunk_ref = chunk_refs[actor_info.rank.dp]
            object_refs.append(getattr(actor_info.handle, method).remote(chunk_ref, **kwargs))
        return object_refs

    @classmethod
    def stage_chunks(
        cls,
        dp_size: int,
        data: TrainingInputBatch,
        mini_batch_boundaries: List[Tuple[int, int]],
        dp_balance_max_tokens: int = 0,
    ) -> List[List[ObjectRef]]:
        """Pre-stage mini-batch chunks into the object store.

        Each mini-batch is defined by a ``(start, end)`` index pair from mini_batch_boundaries.
        Mini-batches are individually padded so that their size is divisible by dp_size, using dummy
        entries with ``loss_mask=0`` that do not affect the loss.

        Args:
            dp_size: Number of data-parallel ranks.
            data: Full TrainingInputBatch to slice from.
            mini_batch_boundaries: List of ``(start, end)`` index pairs.  The i-th mini-batch is
                data[mini_batch_boundaries[i][0]:mini_batch_boundaries[i][1]].

            dp_balance_max_tokens: When > 0, split each mini-batch with the token-balanced split
                (see :func:`split_for_dp`).

        Returns:
            ``result[i][dp_rank]`` - ObjectRef for mini-batch *i*, DP rank *dp_rank*. Each
            ``result[i]`` is an :class:`OrderedRefs` that carries the sample order of mini-batch *i*.
        """
        all_chunk_refs: List[List[ObjectRef]] = []
        for start, end in mini_batch_boundaries:
            mini_batch = data[start:end]
            mb_size = end - start

            # Pad to make divisible by dp_size. Will only be non-zero for step-wise training.
            pad_size = (-mb_size) % dp_size
            if pad_size > 0:
                mini_batch = pad_training_input_batch(mini_batch, pad_size)

            mini_batch_size = len(mini_batch)
            assert (
                mini_batch_size % dp_size == 0
            ), f"mini_batch_size % dp_size != 0, got {mini_batch_size} and {dp_size}"
            chunks, sample_order = split_for_dp(mini_batch, dp_size, dp_balance_max_tokens)
            refs = OrderedRefs(ray.put(chunk) for chunk in chunks)
            refs.sample_order = sample_order
            all_chunk_refs.append(refs)
        return all_chunk_refs

    @classmethod
    def dispatch_from_staged(
        cls,
        actor_infos: List[ActorInfo],
        method: str,
        chunk_refs: List[ObjectRef],
        **kwargs: Any,
    ) -> List[ObjectRef]:
        """
        Dispatch pre-staged per-DP chunks to workers.

        Each worker receives only its own chunk (already in the object
        store), avoiding unnecessary deserialization overhead.

        Args:
            actor_infos: List of actor info objects
            method: Name of method to call on workers (receives a single data chunk)
            chunk_refs: Pre-staged ObjectRefs, one per DP rank (from ``stage_chunks``)
            **kwargs: Additional keyword arguments to pass to the method

        Returns:
            List of ObjectRefs for worker results
        """
        assert len(actor_infos) > 0, "actor_infos must be a non-empty list"
        object_refs = []
        for actor_info in actor_infos:
            chunk_ref = chunk_refs[actor_info.rank.dp]
            object_refs.append(getattr(actor_info.handle, method).remote(chunk_ref, **kwargs))
        return object_refs

    @classmethod
    def validate_dispatch_args(cls, *args, **kwargs) -> Tuple[Tuple, Dict[str, Any]]:
        # Extract data from either positional arg or kwarg
        if args:
            data = args[0]
            remaining_kwargs = kwargs
        elif "data" in kwargs:
            data = kwargs.pop("data")
            remaining_kwargs = kwargs
        else:
            raise ValueError("MeshDispatch requires 'data' as first positional argument or keyword argument")

        if not isinstance(data, TrainingInputBatch):
            raise ValueError(f"For MeshDispatch, `data` entry should be a `TrainingInputBatch`, got {type(data)}")
        # Pass through data as positional arg, and any other kwargs (e.g., loss_fn, loss_fn_config)
        return (data,), remaining_kwargs


class PassThroughDispatch(Dispatch):
    """PassThrough dispatch type to dispatch data to a group of actors without any sharding.

    This is useful for cases where we want to run the same method on all the actors.
    Supports methods with any number of arguments.
    """

    @classmethod
    def dispatch(cls, actor_infos: List[ActorInfo], method: str, *args, **kwargs) -> List[ObjectRef]:
        return [getattr(actor_info.handle, method).remote(*args, **kwargs) for actor_info in actor_infos]

    @classmethod
    def validate_dispatch_args(cls, *args, **kwargs) -> Tuple[Tuple, Dict[str, Any]]:
        # no validation needed just pass everything
        return args, kwargs


class DispatchRegistry:
    _registry: Dict[str, Type[Dispatch]] = {"mesh": MeshDispatch, "pass_through": PassThroughDispatch}

    @classmethod
    def register(cls, name: str, dispatch_class: Type[Dispatch]) -> None:
        """Register a new dispatch type."""
        assert issubclass(dispatch_class, Dispatch)
        cls._registry[name] = dispatch_class

    @classmethod
    def get(cls, name: str) -> Type[Dispatch]:
        """Get a registered dispatch type."""
        if name not in cls._registry:
            raise KeyError(f"Dispatch type '{name}' not registered")
        return cls._registry[name]

    @classmethod
    def list_registered(cls) -> Dict[str, Type[Dispatch]]:
        """List all registered dispatch types."""
        return cls._registry


def register_dispatch_type(name: str, dispatch_class: Type) -> None:
    DispatchRegistry.register(name, dispatch_class)


def concatenate_outputs_after_mesh_dispatch(
    actor_infos: List[ActorInfo], data_batches: List[TrainingOutputBatch]
) -> TrainingOutputBatch:
    """Concatenate data batches from different ranks after mesh dispatch.

    - Data is collected only from the primary DP rank.
    - The collected chunks are concatenated in order of DP rank to reconstruct the full data.
    """
    assert len(actor_infos) == len(data_batches), "`actor_infos` and `data_batches` must have the same length"
    shards = []
    # collect in-order
    dp_rank_to_shard = {}
    for actor_info, data_batch in zip(actor_infos, data_batches):
        if actor_info.rank.is_collection_dp_rank():
            dp_rank = actor_info.rank.dp
            dp_rank_to_shard[dp_rank] = data_batch
    for i in range(actor_infos[0].rank.dp_size):
        shards.append(dp_rank_to_shard[i])
    return TrainingOutputBatch.cat(shards)


@dataclass(frozen=True)
class WorkerOutput:
    """Unified worker output for ``forward`` and ``forward_backward``.

    All worker-side outputs (RL / SFT, inference / loss-with-backward) flow
    through this dataclass at the dispatch boundary so callers can program
    against a uniform API.

    Attributes:
        loss_fn_output_type: Tag describing the schema of each entry in
            ``loss_fn_outputs`` (e.g. ``"scalar"`` for per-token scalar arrays).
        loss_fn_outputs: Per-sample list of dicts. Each entry contains keys
            specific to the worker role (e.g. ``logprobs`` for policy/ref,
            ``values`` for critic, plus optional ``elementwise_loss`` on the
            SFT path).
        metrics: Scalar metrics (loss, lr, response_length, ...). Already
            all-reduced across DP ranks by the worker.
    """

    loss_fn_output_type: str = "scalar"
    loss_fn_outputs: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def cat(
        cls,
        actor_infos: List[ActorInfo],
        shards: List["WorkerOutput"],
        sample_order: Optional[List[int]] = None,
    ) -> "WorkerOutput":
        """Concatenate per-DP-rank shards in DP-rank order.

        ``sample_order`` is the permutation of a token-balanced dispatch (see
        :class:`OrderedRefs`); with it, ``loss_fn_outputs`` return to the batch order.

        Only collects from collection DP ranks (matches
        :meth:`MeshRank.is_collection_dp_rank`). ``loss_fn_outputs`` are
        concatenated; ``metrics`` are taken from the first DP shard (they are
        already all-reduced across DP within the worker).
        """
        assert len(actor_infos) == len(shards), "`actor_infos` and `shards` must have the same length"
        dp_rank_to_shard: Dict[int, "WorkerOutput"] = {}
        for ai, s in zip(actor_infos, shards):
            if ai.rank.is_collection_dp_rank():
                dp_rank_to_shard[ai.rank.dp] = s
        if not dp_rank_to_shard:
            # Unreachable in practice: any actor group has at least one collection
            # DP rank. Default ``loss_fn_output_type="scalar"`` is fine here.
            return cls()
        ordered = [dp_rank_to_shard[i] for i in range(actor_infos[0].rank.dp_size)]
        return cls(
            loss_fn_output_type=ordered[0].loss_fn_output_type,
            loss_fn_outputs=restore_sample_order([x for s in ordered for x in s.loss_fn_outputs], sample_order),
            # metrics are already all-reduced across DP within each worker, so
            # taking rank 0's dict (rather than re-aggregating) is correct.
            metrics=dict(ordered[0].metrics),
        )


def loss_fn_outputs_to_tensor(
    outputs: List[Dict[str, Any]],
    key: str = "logprobs",
    pad_value: float = 0.0,
    dtype=torch.float32,
    device=None,
) -> torch.Tensor:
    """Re-stack per-sample loss_fn_outputs into a right-padded ``[B, T_max]`` tensor.

    Args:
        outputs: Per-sample list of dicts (as in :attr:`WorkerOutput.loss_fn_outputs`).
        key: Field to extract from each dict (e.g. ``"logprobs"`` for policy/ref,
            ``"values"`` for critic).
        pad_value: Padding value for right-padding shorter sequences.
        dtype: Target dtype for the output tensor.
        device: Optional target device.

    Returns:
        ``torch.Tensor[B, T_max]`` right-padded with ``pad_value``.
    """
    seqs = [torch.tensor(o[key], dtype=dtype, device=device) for o in outputs]
    return torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_value)
