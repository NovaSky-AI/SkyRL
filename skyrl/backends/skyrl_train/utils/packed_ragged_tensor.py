"""CSR containers for token metadata whose per-token row length varies.

Some per-token metadata has a variable number of entries per token: sample support holds the
token IDs the sampler could have drawn, from one ID under a sharp top-p to ``top_k`` under a
flat one. A dense ``[tokens, width]`` array padded to the widest row wastes most of its
memory, wire bytes, and scorer work on padding. These containers store such metadata as CSR
instead: flat member buffers plus offsets, with no padding.

- ``RaggedRows`` is the host (NumPy) form for one span of tokens. Generators, the wire codec
  and token-metadata traces build, slice and join it per trajectory.
- ``PackedRaggedTensor`` is the trainer (torch) form for a whole batch: the ``TensorBatch``
  field, with one segment of rows per trajectory. ``PackedRaggedTensor.from_rows`` packs one
  ``RaggedRows`` per trajectory into it: one concatenation per column, which torch then
  shares without a further copy.

Both hold the same member columns: a tuple of flat buffers with one entry per member, each in
its own dtype (for example int32 token IDs plus float32 logprobs), all addressed by the same
offsets. Every operation moves all columns together, so a new per-member value is a new
column rather than a new side channel. A subclass, or for the packed form a column-index
convention, gives the columns their meaning.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import TypeVar

import numpy as np
import torch

from skyrl.backends.skyrl_train.utils.packed_tensor import (
    CU_SEQLENS_DTYPE,
    cu_seqlens_from_lengths,
    lengths_from_offsets,
    row_index_from_offsets,
)

_Block = TypeVar("_Block", "RaggedRows", "PackedRaggedTensor")


def _offsets_from_lengths(row_lengths: Sequence[int] | np.ndarray) -> np.ndarray:
    """Return the ``[rows + 1]`` int32 exclusive prefix sum of ``row_lengths``."""
    lengths = np.asarray(row_lengths, dtype=np.int32)
    offsets = np.zeros(lengths.size + 1, dtype=np.int32)
    np.cumsum(lengths, dtype=np.int32, out=offsets[1:])
    return offsets


def _populated_blocks(blocks: Sequence[_Block], kind: str) -> list[_Block]:
    """The blocks whose member columns join; they must share one column schema.

    A block with no members contributes only rows, so it joins blocks of any column schema.
    """
    populated = [block for block in blocks if block.num_members] or list(blocks[:1])
    schemas = {block.column_dtypes for block in populated}
    if len(schemas) > 1:
        raise ValueError(f"{kind} with members must share member columns, got {sorted(schemas, key=str)}")
    return populated


@dataclass(frozen=True)
class RaggedRows:
    """Host-side CSR rows for a span of tokens, one row per token.

    Row ``i`` holds members ``offsets[i]:offsets[i + 1]`` of every column. ``offsets`` is the
    ``[rows + 1]`` int32 prefix sum of the row lengths, and an empty row (a token with no
    metadata) is two equal offsets. For example, rows ``[7, 8], [], [9]`` with a logprob column
    are ``columns=([7, 8, 9], [-0.1, -2.4, 0.0])`` and ``offsets=[0, 2, 2, 3]``.

    It stands in for a dense ``[tokens, ...]`` array: ``len`` counts token rows and a contiguous
    slice selects them, so token-metadata plumbing (``TokenMetadataTrace``) builds, slices,
    pads, and joins either form the same way. Subclasses name and validate the columns and add
    per-block fields such as ``top_k``; slicing, ``empty_like``, and ``concat`` keep them.
    """

    columns: tuple[np.ndarray, ...]
    offsets: np.ndarray

    def __post_init__(self) -> None:
        if not self.columns or any(column.ndim != 1 for column in self.columns):
            raise ValueError(
                f"ragged rows need at least one flat member column, got shapes {[c.shape for c in self.columns]}"
            )
        if any(column.size != self.num_members for column in self.columns):
            raise ValueError(f"ragged member columns must align, got sizes {[c.size for c in self.columns]}")
        if self.offsets.dtype != np.int32 or self.offsets.ndim != 1 or self.offsets.size == 0:
            raise ValueError("ragged row offsets must be a nonempty int32 array")
        if self.offsets[0] != 0 or self.offsets[-1] != self.num_members or np.any(self.row_lengths < 0):
            raise ValueError("ragged row offsets must ascend from 0 and span the members")

    @classmethod
    def from_lengths(
        cls, columns: tuple[np.ndarray, ...], row_lengths: Sequence[int] | np.ndarray, **schema
    ) -> "RaggedRows":
        """Build rows from consecutive ``row_lengths`` runs of the members."""
        return cls(columns, _offsets_from_lengths(row_lengths), **schema)

    @property
    def num_rows(self) -> int:
        return self.offsets.size - 1

    @property
    def num_members(self) -> int:
        return self.columns[0].size

    @property
    def row_lengths(self) -> np.ndarray:
        return np.diff(self.offsets)

    @property
    def column_dtypes(self) -> tuple[np.dtype, ...]:
        return tuple(column.dtype for column in self.columns)

    def __len__(self) -> int:
        return self.num_rows

    def __getitem__(self, rows: slice) -> "RaggedRows":
        start, stop, step = rows.indices(self.num_rows)
        if step != 1:
            raise ValueError("ragged rows support only contiguous row slices")
        offsets = self.offsets[start : max(start, stop) + 1]
        return replace(
            self,
            columns=tuple(column[offsets[0] : offsets[-1]] for column in self.columns),
            offsets=offsets - offsets[0],
        )

    def empty_like(self, num_rows: int) -> "RaggedRows":
        """``num_rows`` rows with no members, in this block's schema."""
        return replace(
            self,
            columns=tuple(column[:0] for column in self.columns),
            offsets=np.zeros(num_rows + 1, dtype=np.int32),
        )

    @classmethod
    def concat(cls, blocks: Sequence["RaggedRows"]) -> "RaggedRows":
        """Join blocks in order, keeping the schema of the first block with members.

        Blocks with members must share their column dtypes; a block with none contributes only rows.
        """
        if not blocks:
            raise ValueError("ragged row concatenation needs at least one block")
        populated = _populated_blocks(blocks, "ragged row blocks")
        return replace(
            populated[0],
            columns=tuple(np.concatenate(parts) for parts in zip(*(block.columns for block in populated))),
            offsets=_offsets_from_lengths(np.concatenate([block.row_lengths for block in blocks])),
        )


class PackedRaggedTensor:
    """The trainer-side batch of ragged token rows: member columns plus two levels of offsets.

    It is ``PackedTensor`` with ragged rows. Batch entries (trajectories) are segments of rows,
    and rows are runs of members:

    - ``row_offsets`` (``[rows + 1]``) partitions the members into token rows, like
      ``RaggedRows.offsets`` across the whole batch;
    - ``cu_seqlens`` (``[batch + 1]``) partitions the rows into batch entries. It means exactly
      what it does on ``PackedTensor``, so token alignment treats both forms alike.

    Both offset levels are int32 (``CU_SEQLENS_DTYPE``). Indexing and the batch operations
    ``TensorBatch`` needs (slicing, gathering, repeat, ``cat``, padding segments, device moves)
    address batch entries, rebuild both offset levels, and move every column together; ``row``
    reads one token row. ``TensorBatch`` pickles the columns and row offsets out of band. The
    packed form has no subclass, so callers address columns by index (e.g. ``SampleSupportColumn``).
    """

    def __init__(self, columns: Sequence[torch.Tensor], row_offsets: torch.Tensor, cu_seqlens: torch.Tensor):
        columns = tuple(columns)
        if not columns or any(column.ndim != 1 for column in columns):
            raise ValueError(
                "packed ragged tensors need at least one flat member column, "
                f"got shapes {[tuple(column.shape) for column in columns]}"
            )
        num_members = columns[0].shape[0]
        if any(column.shape[0] != num_members for column in columns):
            raise ValueError(
                f"packed ragged member columns must align, got sizes {[column.shape[0] for column in columns]}"
            )
        device = columns[0].device
        if any(tensor.device != device for tensor in (*columns, row_offsets, cu_seqlens)):
            raise ValueError("packed ragged member columns and offsets must share a device")
        for name, offsets in (("row_offsets", row_offsets), ("cu_seqlens", cu_seqlens)):
            if offsets.ndim != 1 or offsets.numel() < 1:
                raise ValueError(f"{name} must be a non-empty 1-D offsets array, got shape {tuple(offsets.shape)}")
            if offsets.dtype != CU_SEQLENS_DTYPE:
                raise ValueError(f"{name} must be {CU_SEQLENS_DTYPE}, got {offsets.dtype}")
        if cu_seqlens.numel() < 2:
            raise ValueError(f"cu_seqlens must hold at least two offsets, got shape {tuple(cu_seqlens.shape)}")
        if int(row_offsets[0]) != 0 or int(row_offsets[-1]) != num_members:
            raise ValueError(
                f"row_offsets must run from 0 to the {num_members} packed members, "
                f"got {int(row_offsets[0])} to {int(row_offsets[-1])}"
            )
        num_rows = row_offsets.numel() - 1
        if int(cu_seqlens[0]) != 0 or int(cu_seqlens[-1]) != num_rows:
            raise ValueError(
                f"cu_seqlens must run from 0 to the {num_rows} packed rows, "
                f"got {int(cu_seqlens[0])} to {int(cu_seqlens[-1])}"
            )
        self.columns = columns
        self.row_offsets = row_offsets
        self.cu_seqlens = cu_seqlens

    @classmethod
    def from_rows(cls, segments: Sequence[RaggedRows]) -> "PackedRaggedTensor":
        """Pack host CSR rows, one segment per batch entry."""
        joined = type(segments[0]).concat(segments)
        return cls(
            [torch.from_numpy(column) for column in joined.columns],
            torch.from_numpy(joined.offsets),
            cu_seqlens_from_lengths([len(segment) for segment in segments]),
        )

    def _map_columns(
        self, fn: Callable[[torch.Tensor], torch.Tensor], row_offsets: torch.Tensor, cu_seqlens: torch.Tensor
    ) -> "PackedRaggedTensor":
        return PackedRaggedTensor([fn(column) for column in self.columns], row_offsets, cu_seqlens)

    def _map_buffers(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> "PackedRaggedTensor":
        """Apply a device/layout transform to every buffer, offsets included."""
        return self._map_columns(fn, fn(self.row_offsets), fn(self.cu_seqlens))

    @property
    def num_rows(self) -> int:
        return self.row_offsets.numel() - 1

    @property
    def num_members(self) -> int:
        return self.columns[0].shape[0]

    @property
    def row_lengths(self) -> torch.Tensor:
        """Return the ``[rows]`` member count of every row, across the whole batch."""
        return lengths_from_offsets(self.row_offsets)

    @property
    def sequence_lengths(self) -> torch.Tensor:
        """Return the ``[batch]`` ROW count of each batch entry, as ``PackedTensor`` does."""
        return lengths_from_offsets(self.cu_seqlens)

    @property
    def device(self) -> torch.device:
        return self.columns[0].device

    @property
    def column_dtypes(self) -> tuple[torch.dtype, ...]:
        return tuple(column.dtype for column in self.columns)

    def __len__(self) -> int:
        return self.cu_seqlens.numel() - 1

    def __getitem__(self, index) -> "PackedRaggedTensor":
        if isinstance(index, slice):
            if index.step in (None, 1):
                start, stop, _ = index.indices(len(self))
                stop = max(start, stop)
                row_bounds = self.cu_seqlens[start : stop + 1]
                member_offsets = self.row_offsets[int(row_bounds[0]) : int(row_bounds[-1]) + 1]
                members = slice(int(member_offsets[0]), int(member_offsets[-1]))
                return self._map_columns(
                    lambda column: column[members], member_offsets - member_offsets[0], row_bounds - row_bounds[0]
                )
            return self._gather(range(*index.indices(len(self))))
        if isinstance(index, torch.Tensor):
            if index.ndim == 0:
                return self.segment(int(index))
            return self._gather(index.tolist())
        if isinstance(index, (list, tuple, range)):
            return self._gather(index)
        return self.segment(index)

    def segment(self, index: int) -> "PackedRaggedTensor":
        """Return one batch entry as a single-entry ragged batch."""
        position = index + len(self) if index < 0 else index
        if not 0 <= position < len(self):
            raise IndexError(f"segment {index} is out of range for a packed batch of {len(self)}")
        return self[position : position + 1]

    def row(self, row_index: int, column: int = 0) -> torch.Tensor:
        """Return one token row of a member column as a view, addressed across the whole batch."""
        position = row_index + self.num_rows if row_index < 0 else row_index
        if not 0 <= position < self.num_rows:
            raise IndexError(f"row {row_index} is out of range for {self.num_rows} packed rows")
        return self.columns[column][int(self.row_offsets[position]) : int(self.row_offsets[position + 1])]

    def _gather(self, indices: Sequence[int]) -> "PackedRaggedTensor":
        """Select batch entries in order and rebuild both offset levels."""
        selected = torch.as_tensor(list(indices), dtype=torch.long, device=self.device)
        row_counts = self.sequence_lengths.to(torch.long)[selected]
        row_index = row_index_from_offsets(self.cu_seqlens[:-1].to(torch.long)[selected], row_counts)
        row_lengths = self.row_lengths.to(torch.long)[row_index]
        member_index = row_index_from_offsets(self.row_offsets[:-1][row_index], row_lengths)
        return self._map_columns(
            lambda column: column.index_select(0, member_index),
            cu_seqlens_from_lengths(row_lengths, device=self.device),
            cu_seqlens_from_lengths(row_counts, device=self.device),
        )

    def to(self, device=None, non_blocking: bool = False) -> "PackedRaggedTensor":
        """Move every buffer to ``device``; member columns keep their own dtypes."""
        return self._map_buffers(lambda tensor: tensor.to(device=device, non_blocking=non_blocking))

    def contiguous(self) -> "PackedRaggedTensor":
        return self._map_buffers(torch.Tensor.contiguous)

    def pin_memory(self) -> "PackedRaggedTensor":
        return self._map_buffers(torch.Tensor.pin_memory)

    def repeat(self, repeats: int) -> "PackedRaggedTensor":
        return self._gather(list(range(len(self))) * repeats)

    def repeat_interleave(self, repeats: int) -> "PackedRaggedTensor":
        return self._gather([index for index in range(len(self)) for _ in range(repeats)])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PackedRaggedTensor) or len(self.columns) != len(other.columns):
            return False
        return (
            all(torch.equal(mine, theirs) for mine, theirs in zip(self.columns, other.columns))
            and torch.equal(self.row_offsets, other.row_offsets)
            and torch.equal(self.cu_seqlens, other.cu_seqlens)
        )

    def __repr__(self) -> str:
        return (
            f"PackedRaggedTensor(batch={len(self)}, rows={self.num_rows}, "
            f"members={self.num_members}, columns={list(self.column_dtypes)})"
        )

    @staticmethod
    def cat(batches: Sequence["PackedRaggedTensor"]) -> "PackedRaggedTensor":
        """Join batches in order; member columns follow the same rule as ``RaggedRows.concat``."""
        if not batches:
            raise ValueError("cannot cat an empty list of packed ragged batches")
        device = batches[0].device
        populated = _populated_blocks(batches, "packed ragged batches")
        return PackedRaggedTensor(
            [torch.cat(parts, dim=0) for parts in zip(*(batch.columns for batch in populated))],
            cu_seqlens_from_lengths(torch.cat([batch.row_lengths for batch in batches]), device=device),
            cu_seqlens_from_lengths(torch.cat([batch.sequence_lengths for batch in batches]), device=device),
        )


def packed_ragged_padding_segments(
    reference: PackedRaggedTensor,
    *,
    segment_lengths: Sequence[int],
) -> PackedRaggedTensor:
    """Return padding segments of empty rows, in the reference's column schema."""
    return PackedRaggedTensor(
        [column[:0] for column in reference.columns],
        torch.zeros(sum(segment_lengths) + 1, dtype=CU_SEQLENS_DTYPE, device=reference.device),
        cu_seqlens_from_lengths(segment_lengths, device=reference.device),
    )
