"""Per-token bounded sampler support used to renormalize rollout logprobs.

Support travels as CSR rows from the inference engine to the trainer: one row of
vocabulary IDs per token, and an empty row for a token without captured support.
An optional second member column holds the sampler's processed logprob of each member,
as score centering needs.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import TypeAlias

import numpy as np
import torch

from skyrl.backends.skyrl_train.distributed.megatron.token_metadata import (
    TokenMetadataLayout,
    TokenMetadataTrace,
    align_packed_token_metadata,
)
from skyrl.backends.skyrl_train.utils.packed_ragged_tensor import (
    PackedRaggedTensor,
    RaggedRows,
)
from skyrl.backends.skyrl_train.utils.packed_tensor import PackedTensor

# Outer ``cu_seqlens`` hold one segment of rows per trajectory; inner offsets name each row's members.
PackedSampleSupport: TypeAlias = PackedRaggedTensor
SAMPLE_SUPPORT_DTYPE = np.dtype(np.int32)
SAMPLE_SUPPORT_TORCH_DTYPE = torch.int32
SAMPLE_SUPPORT_DTYPES = frozenset({SAMPLE_SUPPORT_DTYPE})
SAMPLE_SUPPORT_LOGPROBS_DTYPE = np.dtype(np.float32)
SAMPLE_SUPPORT_FIELD = "rollout_sample_support"
# Sentinel outside the valid packed-row range.
SAMPLE_SUPPORT_NO_ROW = -1
# Forward-output ``SampleSupportScores`` in canonical batch positions, when score centering is on.
SCORE_CENTERING_SCORES_KEY = "score_centering_scores"


class SampleSupportColumn(IntEnum):
    """Index of each member column in sample support's ``columns``."""

    TOKEN_IDS = 0
    # Optional: the sampler's processed logprob of each member.
    LOGPROBS = 1


_TOKEN_IDS_ONLY = (SAMPLE_SUPPORT_DTYPE,)
_WITH_LOGPROBS = (SAMPLE_SUPPORT_DTYPE, SAMPLE_SUPPORT_LOGPROBS_DTYPE)


@dataclass(frozen=True)
class CompactSampleSupport(RaggedRows):
    """CSR support rows for a span of tokens.

    One int32 token-ID column plus int32 offsets, and an optional finite float32 logprob
    column. ``top_k`` bounds every row's length.
    """

    top_k: int

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.column_dtypes not in (_TOKEN_IDS_ONLY, _WITH_LOGPROBS):
            raise ValueError(
                f"compact support holds int32 token IDs and optional float32 logprobs, got {self.column_dtypes}"
            )
        if self.top_k < 1 or np.any(self.row_lengths > self.top_k) or np.any(self.token_ids < 0):
            raise ValueError("compact support rows must hold at most a positive top_k nonnegative IDs")
        if self.logprobs is not None and not np.isfinite(self.logprobs).all():
            raise ValueError("compact support member logprobs must be finite")

    @property
    def token_ids(self) -> np.ndarray:
        return self.columns[SampleSupportColumn.TOKEN_IDS]

    @property
    def logprobs(self) -> np.ndarray | None:
        """Each member's sampler logprob, or None when only token IDs were captured."""
        return self.columns[SampleSupportColumn.LOGPROBS] if len(self.columns) > SampleSupportColumn.LOGPROBS else None

    @classmethod
    def from_token_ids(
        cls,
        token_ids: np.ndarray,
        row_lengths: Sequence[int] | np.ndarray,
        *,
        top_k: int,
        logprobs: np.ndarray | None = None,
    ) -> "CompactSampleSupport":
        """Build rows from consecutive ``row_lengths`` runs of ``token_ids`` and, if given, ``logprobs``."""
        columns = (token_ids,) if logprobs is None else (token_ids, logprobs)
        return cls.from_lengths(columns, row_lengths, top_k=top_k)

    @classmethod
    def from_masked_rows(
        cls,
        ids: np.ndarray,
        present: np.ndarray,
        logprobs: np.ndarray | None = None,
    ) -> "CompactSampleSupport":
        """Keep ``ids[present]`` row by row from ``[tokens, top_k]`` candidate grids.

        ``logprobs``, a matching grid, keeps the same cells.
        """
        if ids.ndim != 2 or present.shape != ids.shape:
            raise ValueError(f"support candidates must be matching [tokens, top_k] grids, got {ids.shape}")
        if logprobs is not None and logprobs.shape != ids.shape:
            raise ValueError(f"member logprobs must match the {ids.shape} candidate grid, got {logprobs.shape}")
        # Boolean indexing is row-major, so each row keeps its candidate order.
        return cls.from_token_ids(
            np.ascontiguousarray(ids[present], dtype=SAMPLE_SUPPORT_DTYPE),
            present.sum(axis=1),
            top_k=ids.shape[1],
            logprobs=(
                None
                if logprobs is None
                else np.ascontiguousarray(logprobs[present], dtype=SAMPLE_SUPPORT_LOGPROBS_DTYPE)
            ),
        )

    @classmethod
    def empty(cls, num_rows: int, top_k: int) -> "CompactSampleSupport":
        """``num_rows`` rows with no captured support."""
        return cls.from_token_ids(
            np.empty(0, dtype=SAMPLE_SUPPORT_DTYPE), np.zeros(num_rows, dtype=np.int64), top_k=top_k
        )

    @classmethod
    def concat(cls, blocks: Sequence["CompactSampleSupport"]) -> "CompactSampleSupport":
        if any(block.top_k != blocks[0].top_k for block in blocks):
            raise ValueError("sample-support blocks must share top_k")
        return super().concat(blocks)

    def contains(self, token_ids: np.ndarray) -> np.ndarray:
        """Per row, whether ``token_ids[row]`` is one of the row's members."""
        if token_ids.shape != (self.num_rows,):
            raise ValueError(f"expected {self.num_rows} token IDs, got shape {token_ids.shape}")
        rows = np.repeat(np.arange(self.num_rows), self.row_lengths)
        hits = rows[self.token_ids == token_ids[rows]]
        return np.bincount(hits, minlength=self.num_rows) > 0


def packed_support_logprobs(support: PackedSampleSupport) -> torch.Tensor | None:
    """The packed members' sampler logprobs, or None when the support carries only token IDs."""
    return (
        support.columns[SampleSupportColumn.LOGPROBS] if len(support.columns) > SampleSupportColumn.LOGPROBS else None
    )


def align_sample_support_row_ids(
    sample_support: PackedSampleSupport,
    layout: TokenMetadataLayout,
) -> torch.Tensor:
    """Map model positions to packed support rows.

    Support for response tokens occupies ``[prompt_len - 1, sequence_len - 1)`` because
    position ``t`` predicts token ``t + 1``. Derive IDs per micro-batch because slicing and
    padding rebase the packed row space.
    """
    segment_lengths = sample_support.sequence_lengths.to(torch.long)
    if segment_lengths.numel() != len(layout.sequence_lengths):
        raise ValueError(
            f"Sample support holds {segment_lengths.numel()} segments for "
            f"{len(layout.sequence_lengths)} trajectories"
        )
    trajectory_lengths = torch.as_tensor(
        layout.sequence_lengths,
        dtype=torch.long,
        device=segment_lengths.device,
    )
    # The first response token is predicted at prompt_len - 1.
    segment_starts = trajectory_lengths - segment_lengths - 1
    if segment_lengths.numel() and int(segment_starts.min()) < 0:
        raise ValueError(
            "A trajectory whose support covers all of its real tokens has no position that "
            f"predicts its first response token, got lengths {segment_lengths.tolist()} for "
            f"trajectories {trajectory_lengths.tolist()}"
        )
    row_ids = PackedTensor(
        torch.arange(int(sample_support.cu_seqlens[-1]), dtype=torch.long, device=sample_support.device),
        sample_support.cu_seqlens,
    )
    return align_packed_token_metadata(
        row_ids,
        layout,
        SAMPLE_SUPPORT_NO_ROW,
        segment_starts=segment_starts.tolist(),
    )


class SampleSupportTrace:
    """Accumulate sample support across incremental generation calls."""

    def __init__(self) -> None:
        self._metadata = TokenMetadataTrace()

    @property
    def num_rows(self) -> int:
        return self._metadata.num_rows

    def append(self, sample_support: CompactSampleSupport, *, expected_rows: int) -> None:
        self._metadata.append(sample_support, expected_rows=expected_rows)

    def append_padding(self, count: int) -> None:
        self._metadata.append_padding(count)

    def finalize(self, *, token_count: int, extra_rows: int) -> CompactSampleSupport:
        """Validate the trace length and discard ``extra_rows`` trailing rows."""
        if self.num_rows != token_count + extra_rows:
            raise ValueError(
                f"sample-support trace has {self.num_rows} rows for {token_count} tokens plus "
                f"{extra_rows} trailing rows"
            )
        return self._metadata.finalize(expected_rows=self.num_rows)[:token_count]
