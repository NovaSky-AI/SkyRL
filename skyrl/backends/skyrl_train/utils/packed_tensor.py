"""One packed buffer plus segment offsets for ragged token-aligned batch fields."""

from collections.abc import Sequence

import torch

# Megatron's own cu_seqlens dtype; a packed global batch stays far inside int32.
CU_SEQLENS_DTYPE = torch.int32


def cu_seqlens_from_lengths(
    sequence_lengths: Sequence[int] | torch.Tensor,
    *,
    device: torch.device | str | int | None = None,
) -> torch.Tensor:
    """Return the ``[batch + 1]`` exclusive prefix sum of ``sequence_lengths``."""
    lengths = torch.as_tensor(sequence_lengths, dtype=CU_SEQLENS_DTYPE, device=device)
    if lengths.ndim != 1:
        raise ValueError(f"sequence lengths must be 1-D, got shape {lengths.shape}")
    if lengths.numel() and int(lengths.min()) < 0:
        raise ValueError(f"sequence lengths must be non-negative, got {lengths.tolist()}")
    offsets = torch.zeros(lengths.numel() + 1, dtype=CU_SEQLENS_DTYPE, device=lengths.device)
    # torch.cumsum promotes to int64; accumulate into the target dtype instead.
    torch.cumsum(lengths, dim=0, out=offsets[1:])
    return offsets


def lengths_from_offsets(cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Return the ``[batch]`` segment lengths that ``cu_seqlens`` encodes."""
    return cu_seqlens[1:] - cu_seqlens[:-1]


def row_index_from_offsets(
    starts: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Return row indices that lay the requested segments back to back."""
    starts = starts.to(torch.long)
    lengths = lengths.to(torch.long)
    total_rows = int(lengths.sum())
    # output_size lets repeat_interleave skip its own device-side sum of `lengths`.
    destination_starts = torch.repeat_interleave(
        cu_seqlens_from_lengths(lengths, device=lengths.device)[:-1].to(torch.long),
        lengths,
        output_size=total_rows,
    )
    within_segment = torch.arange(total_rows, device=lengths.device) - destination_starts
    return torch.repeat_interleave(starts, lengths, output_size=total_rows) + within_segment


class PackedTensor:
    """A ragged batch of token-aligned rows held as one buffer plus ``cu_seqlens``.

    ``values`` is ``[sum(sequence_lengths), *row_shape]`` in canonical batch order and
    ``cu_seqlens`` is the ``[batch + 1]`` exclusive prefix sum of the segment lengths.
    Indexing and batch operations address segments rather than individual rows.
    """

    def __init__(self, values: torch.Tensor, cu_seqlens: torch.Tensor):
        if values.ndim < 1:
            raise ValueError("packed values must have a token-row dimension")
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError(f"cu_seqlens must hold at least two offsets, got shape {cu_seqlens.shape}")
        if cu_seqlens.dtype != CU_SEQLENS_DTYPE:
            raise ValueError(f"cu_seqlens must be {CU_SEQLENS_DTYPE}, got {cu_seqlens.dtype}")
        if cu_seqlens.device != values.device:
            raise ValueError(
                f"packed values and cu_seqlens must share a device, got {values.device} and {cu_seqlens.device}"
            )
        if int(cu_seqlens[0]) != 0 or int(cu_seqlens[-1]) != values.shape[0]:
            raise ValueError(
                f"cu_seqlens must run from 0 to the {values.shape[0]} packed rows, "
                f"got {int(cu_seqlens[0])} to {int(cu_seqlens[-1])}"
            )
        self.values = values
        self.cu_seqlens = cu_seqlens

    @classmethod
    def from_segments(cls, segments: Sequence[torch.Tensor]) -> "PackedTensor":
        """Concatenate per-batch-entry row blocks into one packed buffer."""
        if not segments:
            raise ValueError("cannot pack an empty list of segments")
        cu_seqlens = cu_seqlens_from_lengths([segment.shape[0] for segment in segments], device=segments[0].device)
        return cls(torch.cat(segments, dim=0), cu_seqlens)

    @property
    def sequence_lengths(self) -> torch.Tensor:
        return lengths_from_offsets(self.cu_seqlens)

    @property
    def row_shape(self) -> torch.Size:
        return self.values.shape[1:]

    @property
    def device(self) -> torch.device:
        return self.values.device

    @property
    def dtype(self) -> torch.dtype:
        return self.values.dtype

    def __len__(self) -> int:
        return self.cu_seqlens.numel() - 1

    def __getitem__(self, index) -> "torch.Tensor | PackedTensor":
        if isinstance(index, slice):
            if index.step in (None, 1):
                start, stop, _ = index.indices(len(self))
                stop = max(start, stop)
                offsets = self.cu_seqlens[start : stop + 1]
                return PackedTensor(self.values[int(offsets[0]) : int(offsets[-1])], offsets - offsets[0])
            return self._gather(range(*index.indices(len(self))))
        if isinstance(index, torch.Tensor):
            if index.ndim == 0:
                return self.segment(int(index))
            return self._gather(index.tolist())
        if isinstance(index, (list, tuple, range)):
            return self._gather(index)
        return self.segment(index)

    def segment(self, index: int) -> torch.Tensor:
        """Return one batch entry's row block as a view."""
        position = index + len(self) if index < 0 else index
        if not 0 <= position < len(self):
            raise IndexError(f"segment {index} is out of range for a packed batch of {len(self)}")
        return self.values[int(self.cu_seqlens[position]) : int(self.cu_seqlens[position + 1])]

    def _gather(self, indices: Sequence[int]) -> "PackedTensor":
        """Select segments in the requested order into a freshly allocated buffer."""
        selected = torch.as_tensor(list(indices), dtype=torch.long, device=self.values.device)
        selected_starts = self.cu_seqlens[:-1].to(torch.long)[selected]
        selected_lengths = self.sequence_lengths.to(torch.long)[selected]
        row_index = row_index_from_offsets(selected_starts, selected_lengths)
        cu_seqlens = cu_seqlens_from_lengths(selected_lengths, device=self.values.device)
        return PackedTensor(self.values.index_select(0, row_index), cu_seqlens)

    def to(self, device=None, dtype=None, non_blocking: bool = False) -> "PackedTensor":
        return PackedTensor(
            self.values.to(device=device, dtype=dtype, non_blocking=non_blocking),
            self.cu_seqlens.to(device=device, non_blocking=non_blocking),
        )

    def contiguous(self) -> "PackedTensor":
        return PackedTensor(self.values.contiguous(), self.cu_seqlens.contiguous())

    def pin_memory(self) -> "PackedTensor":
        return PackedTensor(self.values.pin_memory(), self.cu_seqlens.pin_memory())

    def repeat(self, repeats: int) -> "PackedTensor":
        return self._gather(list(range(len(self))) * repeats)

    def repeat_interleave(self, repeats: int) -> "PackedTensor":
        return self._gather([index for index in range(len(self)) for _ in range(repeats)])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PackedTensor):
            return False
        return torch.equal(self.values, other.values) and torch.equal(self.cu_seqlens, other.cu_seqlens)

    def __repr__(self) -> str:
        return f"PackedTensor(batch={len(self)}, values={tuple(self.values.shape)}, dtype={self.values.dtype})"

    @staticmethod
    def cat(batches: Sequence["PackedTensor"]) -> "PackedTensor":
        if not batches:
            raise ValueError("cannot cat an empty list of packed batches")
        lengths = torch.cat([batch.sequence_lengths for batch in batches])
        return PackedTensor(
            torch.cat([batch.values for batch in batches], dim=0),
            cu_seqlens_from_lengths(lengths, device=batches[0].device),
        )


def packed_padding_segments(
    reference: PackedTensor,
    *,
    segment_lengths: Sequence[int],
    fill: torch.Tensor | int | float | bool,
) -> PackedTensor:
    """Return ``fill``-valued segments in ``reference``'s row shape, dtype and device.

    ``fill`` broadcasts over the row shape, so it may be a scalar or one whole row.
    """
    values = torch.empty(
        (sum(segment_lengths), *reference.row_shape),
        dtype=reference.dtype,
        device=reference.device,
    )
    values[...] = fill
    return PackedTensor(values, cu_seqlens_from_lengths(segment_lengths, device=reference.device))
