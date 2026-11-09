"""Where each real token of a microbatch sits in one rank's model input."""

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from skyrl.backends.skyrl_train.distributed.megatron.packing_utils import (
    get_cp_chunk_token_align,
    get_packed_seq_align_size,
    get_unpacked_seq_align_size,
)
from skyrl.backends.skyrl_train.utils.packed_ragged_tensor import RaggedRows
from skyrl.backends.skyrl_train.utils.packed_tensor import cu_seqlens_from_lengths

# Megatron is imported lazily inside functions so that non-Megatron backends can
# import this module for its layout and trace.

# Marks a layout position that holds no real token.
NO_TOKEN = -1


@dataclass(frozen=True)
class TokenLayout:
    """Where each real token of a ``[batch, seq]`` microbatch sits in one rank's model input.

    Real tokens are numbered by ordinal, segment by segment: one segment per row, or one per
    sub-sequence under controller packing. Packed, segments are padded and laid end to end
    (``cu_seqlens_padded``), and a CP rank holds a front and a mirrored back chunk of each one;
    unpacked, each row's real tokens start at column 0 of a ``[batch, aligned_len]`` block. Every
    token-aligned field reaches the model's layout through :meth:`place`.
    """

    # Flat ``[batch * seq]`` position of every real token, in ordinal order.
    source: torch.Tensor
    # ``[segments + 1]`` int64 ordinal offset of each segment.
    cu_lengths: torch.Tensor
    batch_shape: tuple[int, int]
    # The ordinal at each of this rank's positions, NO_TOKEN at padding: ``[1, T // CP]`` packed,
    # ``[batch, aligned_len]`` unpacked.
    ordinals: torch.Tensor
    # Packed only: each padded segment's offset, as ``PackedSeqParams`` carries them.
    cu_seqlens_padded: Optional[torch.Tensor] = None
    cp_size: int = 1
    cp_rank: int = 0

    @classmethod
    def build(
        cls,
        attention_mask: torch.Tensor,
        *,
        packed: bool,
        tp_size: int,
        cp_size: int = 1,
        cp_rank: int = 0,
        sub_seq_lengths: Optional[list[list[int]]] = None,
        fp8_enabled: bool = False,
        fp8_recipe: Optional[str] = None,
        seq_align: int = 1,
    ) -> "TokenLayout":
        """Lay out ``attention_mask``'s real tokens for one rank.

        Packed, every segment is padded to the TP / CP / FP8 alignment (``get_packed_seq_align_size``).
        ``sub_seq_lengths`` (packed only) names the sub-sequences of each row, at the collator's
        aligned row offsets; otherwise a row's attended tokens are its one segment. ``seq_align``
        (unpacked only) additionally rounds the width up to a multiple of it, combined with the TP / FP8
        alignment by lcm.
        """
        mask = attention_mask.to(torch.bool)
        batch, seq = mask.shape
        align = (
            get_packed_seq_align_size(tp_size, cp_size, fp8_enabled=fp8_enabled, fp8_recipe=fp8_recipe)
            if packed
            else None
        )
        if packed and sub_seq_lengths is not None:
            if cp_size > 1 and get_cp_chunk_token_align() != 1:
                # The controller's collator lays sub-sequences out without the model's chunk alignment.
                raise NotImplementedError(
                    "Mini-batch packing (sub_seq_lengths) under context parallelism does not support models "
                    "that align CP chunks to token blocks (e.g. Qwen4Exp)."
                )
            source, lengths = _sub_sequence_tokens(sub_seq_lengths, batch, seq, align, mask.device)
        else:
            source = mask.reshape(-1).nonzero().squeeze(1)
            lengths = mask.sum(dim=1).tolist()
        cu_lengths = cu_seqlens_from_lengths(lengths, device=mask.device).to(torch.long)

        if not packed:
            if cp_size != 1:
                raise ValueError("context parallelism requires packed sequences")
            longest = max(lengths, default=0)
            align = math.lcm(get_unpacked_seq_align_size(tp_size, fp8_enabled, fp8_recipe), seq_align)
            width = longest + (-longest % align)
            columns = torch.arange(width, device=mask.device)[None, :]
            row_lengths = cu_lengths.diff()[:, None]
            ordinals = torch.where(columns < row_lengths, cu_lengths[:-1, None] + columns, NO_TOKEN)
            return cls(source=source, cu_lengths=cu_lengths, batch_shape=(batch, seq), ordinals=ordinals)

        padded = torch.as_tensor(lengths, dtype=torch.long)
        padded = padded + (-padded % align)
        cu_seqlens_padded = cu_seqlens_from_lengths(padded.tolist(), device=mask.device)
        return cls(
            source=source,
            cu_lengths=cu_lengths,
            batch_shape=(batch, seq),
            ordinals=_packed_ordinals(cu_lengths, cu_seqlens_padded, cp_size, cp_rank).unsqueeze(0),
            cu_seqlens_padded=cu_seqlens_padded,
            cp_size=cp_size,
            cp_rank=cp_rank,
        )

    @property
    def packed(self) -> bool:
        return self.cu_seqlens_padded is not None

    @property
    def sequence_lengths(self) -> list[int]:
        return self.cu_lengths.diff().tolist()

    @property
    def padding_mask(self) -> torch.Tensor:
        """True at this rank's positions that hold no real token."""
        return self.ordinals == NO_TOKEN

    def next_token_mask(self) -> torch.Tensor:
        """True at positions whose token is followed by another real token of its segment."""
        return self._positions(next_token=True) != NO_TOKEN

    def full(self) -> "TokenLayout":
        """This packed layout before CP sharding."""
        if not self.packed:
            raise ValueError("only a packed layout is CP-sharded")
        if self.cp_size == 1:
            return self
        ordinals = _packed_ordinals(self.cu_lengths, self.cu_seqlens_padded, 1, 0).unsqueeze(0)
        return dataclasses.replace(self, ordinals=ordinals, cp_size=1, cp_rank=0)

    def place(self, values: torch.Tensor, fill, *, next_token: bool = False) -> torch.Tensor:
        """Place ``[batch, seq, ...]`` ``values`` in this layout, ``fill`` where no token is.

        ``next_token`` places at each position the value of the token after it, as next-token
        targets are; a segment's last token and padding get ``fill``.
        """
        if values.shape[:2] != self.batch_shape:
            raise ValueError(f"values of shape {tuple(values.shape)} do not match the batch {self.batch_shape}")
        flat = values.reshape(-1, *values.shape[2:])
        return self._fill(self._positions(next_token), lambda ordinals: flat[self.source[ordinals]], flat, fill)

    def place_ordinals(self, values: torch.Tensor, fill, *, next_token: bool = False) -> torch.Tensor:
        """Place per-ordinal ``[num_tokens, ...]`` ``values`` (one row per real token) in this layout."""
        if values.shape[0] != self.source.numel():
            raise ValueError(f"{values.shape[0]} token rows for a layout of {self.source.numel()} tokens")
        return self._fill(self._positions(next_token), lambda ordinals: values[ordinals], values, fill)

    def to_batch(self, values: torch.Tensor, fill) -> torch.Tensor:
        """Scatter per-ordinal ``values`` back to canonical ``[batch, seq, ...]`` positions."""
        out = torch.full(
            (self.batch_shape[0] * self.batch_shape[1], *values.shape[1:]),
            fill,
            dtype=values.dtype,
            device=values.device,
        )
        out[self.source] = values
        return out.view(*self.batch_shape, *values.shape[1:])

    def packed_scatter(self) -> "PackedScatter":
        """Where this rank's packed next-token outputs land in canonical ``[batch, seq - 1]``.

        Computed here, on the host, so scattering on the device indexes with integers and needs no sync.
        """
        if not self.packed:
            raise ValueError("scattering packed outputs requires a packed layout")
        full = self.full()
        predicting = full.next_token_mask().squeeze(0)
        sources = self.source[full.ordinals.squeeze(0)[predicting]]
        seq = self.batch_shape[1]
        return PackedScatter(
            positions=predicting.nonzero().squeeze(1),
            rows=sources // seq,
            cols=sources % seq,
            batch_shape=self.batch_shape,
            cu_seqlens_padded=self.cu_seqlens_padded,
            cp_size=self.cp_size,
        )

    def recover(self) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """Unpacked: each token's flat layout row, its flat canonical row, and per-row counts."""
        if self.packed:
            raise ValueError("only an unpacked layout is restored row by row")
        return self.ordinals.reshape(-1).ne(NO_TOKEN).nonzero().squeeze(1), self.source, self.sequence_lengths

    def _positions(self, next_token: bool) -> torch.Tensor:
        if not next_token:
            return self.ordinals
        ends = self.cu_lengths[1:]
        segment_end = ends[torch.searchsorted(ends, self.ordinals.clamp(min=0), right=True).clamp(max=ends.numel() - 1)]
        successor = self.ordinals + 1
        return torch.where((self.ordinals != NO_TOKEN) & (successor < segment_end), successor, NO_TOKEN)

    @staticmethod
    def _fill(positions: torch.Tensor, gather, like: torch.Tensor, fill) -> torch.Tensor:
        out = torch.full((*positions.shape, *like.shape[1:]), fill, dtype=like.dtype, device=like.device)
        valid = positions != NO_TOKEN
        out[valid] = gather(positions[valid])
        return out


@dataclass(frozen=True)
class PackedScatter:
    """Scatters one rank's packed next-token outputs ``[1, T // CP]`` to canonical ``[batch, seq - 1]``;
    position ``t`` of the result predicts token ``t + 1``. Built by ``TokenLayout.packed_scatter``."""

    # Positions of the CP-gathered packed stream whose token is followed by another real token of its
    # segment, and the canonical row and column each one predicts at.
    positions: torch.Tensor
    rows: torch.Tensor
    cols: torch.Tensor
    batch_shape: tuple[int, int]
    cu_seqlens_padded: torch.Tensor
    cp_size: int

    def scatter(self, model_values: torch.Tensor, fill) -> torch.Tensor:
        """Gather the other CP ranks' shares, then place each predicting position at its canonical one."""
        if model_values.ndim != 2 or model_values.shape[0] != 1:
            raise ValueError(f"Expected packed model values with shape [1, tokens], got {tuple(model_values.shape)}")
        values = model_values.squeeze(0)
        if self.cp_size > 1:
            import megatron.core.parallel_state as mpu

            from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
                allgather_cp_sharded_packed_tensor,
            )

            values = allgather_cp_sharded_packed_tensor(
                values, self.cu_seqlens_padded, mpu.get_context_parallel_group()
            )
        batch, seq = self.batch_shape
        out = torch.full((batch, seq - 1), fill, dtype=values.dtype, device=values.device)
        out[self.rows, self.cols] = values[self.positions]
        return out


def _sub_sequence_tokens(
    sub_seq_lengths: list[list[int]], batch: int, seq: int, sequence_align: int, device: torch.device
) -> tuple[torch.Tensor, list[int]]:
    """Flat positions and lengths of controller-packed sub-sequences, at the collator's row offsets."""
    if len(sub_seq_lengths) != batch:
        raise ValueError(f"sub_seq_lengths has {len(sub_seq_lengths)} rows but batch size is {batch}")
    starts: list[int] = []
    lengths: list[int] = []
    for row, row_lengths in enumerate(sub_seq_lengths):
        row_lengths = [int(length) for length in row_lengths]
        offset = 0
        for length in row_lengths:
            if offset + length > seq:
                raise ValueError("sub_seq_lengths extends beyond the packed batch row")
            starts.append(row * seq + offset)
            lengths.append(length)
            # Each sub-sequence starts at the previous one's end rounded up to the alignment.
            offset += length + (-length % sequence_align)
    return _concatenated_ranges(torch.tensor(starts, dtype=torch.long), lengths).to(device), lengths


def _concatenated_ranges(starts: torch.Tensor, lengths: Sequence[int]) -> torch.Tensor:
    """``cat([arange(start, start + length) ...])`` without a Python loop over tensors."""
    counts = torch.as_tensor(lengths, dtype=torch.long, device=starts.device)
    offsets = counts.cumsum(0) - counts
    return torch.repeat_interleave(starts - offsets, counts) + torch.arange(int(counts.sum()), device=starts.device)


def _packed_ordinals(
    cu_lengths: torch.Tensor, cu_seqlens_padded: torch.Tensor, cp_size: int, cp_rank: int
) -> torch.Tensor:
    """The ordinal at each of a CP rank's packed positions, NO_TOKEN at padding."""
    cu_padded = cu_seqlens_padded.to(torch.long)
    padded = cu_padded.diff()
    if cp_size > 1 and bool((padded % (2 * cp_size)).any()):
        raise ValueError(f"packed segment lengths {padded.tolist()} are not divisible by 2 * CP={2 * cp_size}")
    local = padded // cp_size
    segment = torch.repeat_interleave(torch.arange(padded.numel(), device=padded.device), local)
    offset = torch.arange(int(local.sum()), device=padded.device) - (cu_padded[:-1] // cp_size)[segment]
    if cp_size > 1:
        # Each rank holds chunk ``cp_rank`` and its mirror ``2 * CP - 1 - cp_rank`` of every segment.
        half = (local // 2)[segment]
        back = padded[segment] - half * (cp_rank + 1) + (offset - half)
        offset = torch.where(offset < half, half * cp_rank + offset, back)
    lengths = cu_lengths.diff()
    return torch.where(offset < lengths[segment], cu_lengths[:-1][segment] + offset, NO_TOKEN)


def _widen_dtype(current: np.dtype, incoming: np.dtype) -> np.dtype:
    """Return the wider of two dtypes when one losslessly contains the other.

    Chunks of one trace may arrive compacted to different widths (uint8, int16,
    int32) because each producer picks the smallest dtype for its own values.
    Widening between them is lossless; a change of kind (integer to float) or a
    pair with no common member (uint8 and int8) is a schema error.
    """
    if incoming == current:
        return current
    promoted = np.promote_types(current, incoming)
    same_family = np.issubdtype(promoted, np.integer) == np.issubdtype(current, np.integer)
    if not same_family or promoted not in (current, incoming):
        raise ValueError(f"token metadata dtype changed from {current} to {incoming}")
    return promoted


class TokenMetadataTrace:
    """Accumulate token-aligned rows: arrays whose first dimension is tokens, or ``RaggedRows``.

    Every chunk shares one schema -- a trailing row shape, or one ``RaggedRows`` type. An
    array's dtype may widen across chunks, and the finalized rows use the widest dtype seen;
    ragged chunks join under ``RaggedRows.concat``'s column rule.
    """

    def __init__(self) -> None:
        self._chunks: list[np.ndarray | RaggedRows] = []
        self._schema: tuple[int, ...] | type[RaggedRows] | None = None
        self._dtype: np.dtype | None = None
        self._num_rows = 0
        self._finalized = False

    @property
    def num_rows(self) -> int:
        return self._num_rows

    @property
    def dtype(self) -> np.dtype | None:
        """Widest array dtype appended so far; None before the first append or for ragged rows."""
        return self._dtype

    def append(self, rows: np.ndarray | RaggedRows, *, expected_rows: int) -> None:
        if self._finalized:
            raise RuntimeError("token metadata trace is already finalized")
        if isinstance(expected_rows, bool) or not isinstance(expected_rows, int) or expected_rows < 0:
            raise ValueError(f"expected_rows must be a non-negative integer, got {expected_rows!r}")
        if isinstance(rows, np.ndarray):
            if rows.ndim < 1:
                raise ValueError("token metadata must have a token-row dimension")
            if not rows.flags.c_contiguous:
                raise ValueError("token metadata rows must be contiguous")
            schema = rows.shape[1:]
        elif isinstance(rows, RaggedRows):
            schema = type(rows)
        else:
            raise TypeError("token metadata rows must be a NumPy array or RaggedRows")
        if len(rows) != expected_rows:
            raise ValueError(f"token metadata has {len(rows)} rows, expected {expected_rows}")

        if self._schema is None:
            self._schema = schema
        elif schema != self._schema:
            raise ValueError(f"token metadata schema changed from {self._schema} to {schema}")
        if isinstance(rows, np.ndarray):
            self._dtype = rows.dtype if self._dtype is None else _widen_dtype(self._dtype, rows.dtype)

        self._chunks.append(rows)
        self._num_rows += expected_rows

    def append_padding(self, count: int, *, fill: int = -1) -> None:
        """Append ``count`` padding rows in the established schema: ``fill`` rows, or empty ragged rows."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"padding count must be a non-negative integer, got {count!r}")
        if count == 0:
            return
        if not self._chunks:
            raise ValueError("cannot pad token metadata before any rows are captured")

        template = self._chunks[0]
        if isinstance(template, RaggedRows):
            padding = template.empty_like(count)
        else:
            padding = np.full((count, *template.shape[1:]), fill, dtype=self._dtype, order="C")
        self.append(padding, expected_rows=count)

    def finalize(self, *, expected_rows: int) -> np.ndarray | RaggedRows:
        if self._finalized:
            raise RuntimeError("token metadata trace is already finalized")
        if self._num_rows != expected_rows:
            raise ValueError(f"token metadata trace has {self._num_rows} rows, expected {expected_rows}")
        if not self._chunks:
            raise ValueError("token metadata trace has no chunks")

        self._finalized = True
        if len(self._chunks) == 1:
            return self._chunks[0]
        if isinstance(self._chunks[0], RaggedRows):
            return type(self._chunks[0]).concat(self._chunks)
        return np.concatenate(self._chunks, axis=0, dtype=self._dtype)
