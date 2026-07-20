"""Token-aligned metadata layout transforms shared by training features."""

from dataclasses import dataclass

import numpy as np
import torch

from skyrl.backends.skyrl_train.distributed.megatron.packing_utils import (
    get_packed_seq_align_size,
    get_unpacked_seq_align_size,
)


def _new_metadata_tensor(
    source: torch.Tensor,
    shape: tuple[int, ...],
    padding_value: torch.Tensor | bool | int,
) -> torch.Tensor:
    output = torch.empty(shape, dtype=source.dtype, device=source.device)
    output[...] = padding_value
    return output


@dataclass(frozen=True)
class TokenMetadataLayout:
    """One shared description of Megatron's token padding and CP sharding."""

    attention_mask: torch.Tensor
    sequence_lengths: list[int]
    aligned_sequence_length: int
    padded_sequence_lengths: list[int] | None = None
    # Retained to reconstruct CP-sharded packed outputs in canonical batch order.
    cu_seqlens_padded: torch.Tensor | None = None
    context_parallel_size: int = 1
    context_parallel_rank: int = 0


def build_token_metadata_layout(
    attention_mask: torch.Tensor,
    device: torch.device,
    *,
    packed: bool,
    fp8_enabled: bool,
) -> TokenMetadataLayout:
    """Compute the shared layout once for all replayed token metadata."""
    import megatron.core.parallel_state as mpu

    aligned_attention_mask = attention_mask.to(device=device, dtype=torch.bool)
    sequence_lengths_tensor = aligned_attention_mask.sum(dim=1, dtype=torch.int32)
    sequence_lengths = sequence_lengths_tensor.tolist()
    tp_size = mpu.get_tensor_model_parallel_world_size()

    if not packed:
        align_size = get_unpacked_seq_align_size(tp_size, fp8_enabled=fp8_enabled)
        max_sequence_length = max(sequence_lengths)
        aligned_sequence_length = max_sequence_length + (-max_sequence_length % align_size)
        return TokenMetadataLayout(
            attention_mask=aligned_attention_mask,
            sequence_lengths=sequence_lengths,
            aligned_sequence_length=aligned_sequence_length,
        )

    cp_size = mpu.get_context_parallel_world_size()
    align_size = get_packed_seq_align_size(tp_size, cp_size, fp8_enabled=fp8_enabled)
    padded_sequence_lengths_tensor = sequence_lengths_tensor + (-sequence_lengths_tensor % align_size)
    padded_sequence_lengths = padded_sequence_lengths_tensor.tolist()
    cu_seqlens_padded = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=device),
            padded_sequence_lengths_tensor.cumsum(dim=0),
        )
    )
    return TokenMetadataLayout(
        attention_mask=aligned_attention_mask,
        sequence_lengths=sequence_lengths,
        aligned_sequence_length=sum(padded_sequence_lengths),
        padded_sequence_lengths=padded_sequence_lengths,
        cu_seqlens_padded=cu_seqlens_padded,
        context_parallel_size=cp_size,
        context_parallel_rank=mpu.get_context_parallel_rank() if cp_size > 1 else 0,
    )


def align_token_metadata(
    metadata: torch.Tensor,
    layout: TokenMetadataLayout,
    padding_value: torch.Tensor | bool | int,
    *,
    next_token: bool = False,
) -> torch.Tensor:
    """Apply padding, optional next-token shifting, and CP sharding."""
    if metadata.device != layout.attention_mask.device:
        raise ValueError("Token-aligned metadata and attention_mask must be on the same device")
    if metadata.shape[:2] != layout.attention_mask.shape:
        raise ValueError(
            f"Token-aligned metadata shape {metadata.shape[:2]} does not match "
            f"attention_mask shape {layout.attention_mask.shape}"
        )

    if layout.padded_sequence_lengths is None:
        if next_token:
            raise ValueError("next-token metadata alignment is only used for packed sequences")
        aligned = _new_metadata_tensor(
            metadata,
            (metadata.shape[0], layout.aligned_sequence_length, *metadata.shape[2:]),
            padding_value,
        )
        for row_index, sequence_length in enumerate(layout.sequence_lengths):
            aligned[row_index, :sequence_length] = metadata[row_index, layout.attention_mask[row_index]]
        return aligned

    packed = _new_metadata_tensor(
        metadata,
        (layout.aligned_sequence_length, *metadata.shape[2:]),
        padding_value,
    )
    offset = 0
    for row_index, (sequence_length, padded_length) in enumerate(
        zip(layout.sequence_lengths, layout.padded_sequence_lengths, strict=True)
    ):
        packed[offset : offset + sequence_length] = metadata[row_index, layout.attention_mask[row_index]]
        # Match Megatron's [seq0, pad0, seq1, pad1, ...] microbatch layout.
        offset += padded_length

    if next_token:
        # Each packed logit predicts the next token within its own padded sequence.
        shifted = _new_metadata_tensor(metadata, packed.shape, padding_value)
        offset = 0
        for padded_length in layout.padded_sequence_lengths:
            shifted[offset : offset + padded_length - 1] = packed[offset + 1 : offset + padded_length]
            offset += padded_length
        packed = shifted

    if layout.context_parallel_size > 1:
        out = _new_metadata_tensor(
            metadata,
            (packed.shape[0] // layout.context_parallel_size, *packed.shape[1:]),
            padding_value,
        )
        src_offset = 0
        dst_offset = 0
        for padded_length in layout.padded_sequence_lengths:
            # CP uses matching front/back chunks of each padded sequence.
            length_per_cp = padded_length // layout.context_parallel_size
            half = length_per_cp // 2
            front_start = src_offset + half * layout.context_parallel_rank
            back_start = src_offset + padded_length - half * (layout.context_parallel_rank + 1)
            out[dst_offset : dst_offset + half] = packed[front_start : front_start + half]
            out[dst_offset + half : dst_offset + length_per_cp] = packed[back_start : back_start + half]
            src_offset += padded_length
            dst_offset += length_per_cp
        packed = out

    return packed.unsqueeze(0)


def scatter_packed_token_values_to_batch(
    model_values: torch.Tensor,
    layout: TokenMetadataLayout,
    padding_value: bool | int,
) -> torch.Tensor:
    """Scatter packed model outputs into canonical ``[batch, seq_len - 1]`` positions."""
    if layout.padded_sequence_lengths is None or layout.cu_seqlens_padded is None:
        raise ValueError("Scattering packed token values requires a packed metadata layout")
    if model_values.ndim != 2 or model_values.shape[0] != 1:
        raise ValueError(f"Expected packed model values with shape [1, tokens], got {model_values.shape}")

    values = model_values.squeeze(0)
    if layout.context_parallel_size > 1:
        import megatron.core.parallel_state as mpu

        from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
            allgather_cp_sharded_packed_tensor,
        )

        values = allgather_cp_sharded_packed_tensor(
            values,
            layout.cu_seqlens_padded,
            mpu.get_context_parallel_group(),
        )

    from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
        _packed_sequence_indices,
    )

    _, _, sequence_indices, sequence_offsets, _ = _packed_sequence_indices(
        layout.cu_seqlens_padded,
        values.shape[0],
        values.device,
    )
    valid_counts = torch.tensor(layout.sequence_lengths, dtype=torch.long, device=values.device) - 1
    packed_mask = sequence_offsets < valid_counts[sequence_indices]

    attention_mask = layout.attention_mask
    token_ordinals = attention_mask.to(torch.long).cumsum(dim=1)
    output_mask = attention_mask[:, :-1] & (
        token_ordinals[:, :-1] < torch.tensor(layout.sequence_lengths, device=values.device).unsqueeze(1)
    )
    batch_values = _new_metadata_tensor(
        model_values,
        (attention_mask.shape[0], attention_mask.shape[1] - 1),
        padding_value,
    )
    batch_values[output_mask] = values[packed_mask]
    return batch_values


class TokenMetadataTrace:
    """Accumulate arrays whose first dimension is aligned to tokens."""

    def __init__(self) -> None:
        self._chunks: list[np.ndarray] = []
        self._schema: tuple[tuple[int, ...], np.dtype] | None = None
        self._num_rows = 0
        self._finalized = False

    @property
    def num_rows(self) -> int:
        return self._num_rows

    def append(self, rows: np.ndarray, *, expected_rows: int) -> None:
        if self._finalized:
            raise RuntimeError("token metadata trace is already finalized")
        if isinstance(expected_rows, bool) or not isinstance(expected_rows, int) or expected_rows < 0:
            raise ValueError(f"expected_rows must be a non-negative integer, got {expected_rows!r}")
        if not isinstance(rows, np.ndarray):
            raise TypeError("token metadata rows must be a NumPy array")
        if rows.ndim < 1:
            raise ValueError("token metadata must have a token-row dimension")
        if rows.shape[0] != expected_rows:
            raise ValueError(f"token metadata has {rows.shape[0]} rows, expected {expected_rows}")
        if not rows.flags.c_contiguous:
            raise ValueError("token metadata rows must be contiguous")

        schema = (rows.shape[1:], rows.dtype)
        if self._schema is None:
            self._schema = schema
        elif schema != self._schema:
            raise ValueError(f"token metadata schema changed from {self._schema} to {schema}")

        self._chunks.append(rows)
        self._num_rows += expected_rows

    def finalize(self, *, expected_rows: int) -> np.ndarray:
        if self._finalized:
            raise RuntimeError("token metadata trace is already finalized")
        if self._num_rows != expected_rows:
            raise ValueError(f"token metadata trace has {self._num_rows} rows, expected {expected_rows}")
        if not self._chunks:
            raise ValueError("token metadata trace has no chunks")

        self._finalized = True
        return self._chunks[0] if len(self._chunks) == 1 else np.concatenate(self._chunks, axis=0)
