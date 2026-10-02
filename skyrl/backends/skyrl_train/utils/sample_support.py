"""Per-token bounded sampler support used to renormalize rollout logprobs.

Rows contain top-k vocabulary IDs and use trailing ``SAMPLE_SUPPORT_PADDING``.
Tokens without captured support use an all-padding row.
"""

from typing import Mapping, TypeAlias

import numpy as np
import torch

from skyrl.backends.skyrl_train.distributed.megatron.token_metadata import (
    TokenMetadataLayout,
    TokenMetadataTrace,
    align_packed_token_metadata,
)
from skyrl.backends.skyrl_train.utils.packed_tensor import PackedTensor

SampleSupport: TypeAlias = np.ndarray
SAMPLE_SUPPORT_DTYPE = np.dtype(np.int32)
SAMPLE_SUPPORT_TORCH_DTYPE = torch.int32
SAMPLE_SUPPORT_DTYPES = frozenset({SAMPLE_SUPPORT_DTYPE})
SAMPLE_SUPPORT_PADDING = -1
SAMPLE_SUPPORT_FIELD = "rollout_sample_support"
# Sentinel outside the valid packed-row range.
SAMPLE_SUPPORT_NO_ROW = -1

# Sampler logprobs of the recorded support members, row-aligned with ``SAMPLE_SUPPORT_FIELD``:
# ``[tokens, top_k]`` float32, ``-inf`` wherever the support row holds padding. Captured for score
# centering (``trainer.algorithm.score_centering``), which needs the sampler's probability mass on
# its head, not only which ids survived the sampler's filter.
SampleSupportLogprobs: TypeAlias = np.ndarray
SAMPLE_SUPPORT_LOGPROBS_DTYPE = np.dtype(np.float32)
SAMPLE_SUPPORT_LOGPROBS_TORCH_DTYPE = torch.float32
SAMPLE_SUPPORT_LOGPROBS_DTYPES = frozenset({SAMPLE_SUPPORT_LOGPROBS_DTYPE})
SAMPLE_SUPPORT_LOGPROBS_PADDING = float("-inf")
SAMPLE_SUPPORT_LOGPROBS_FIELD = "rollout_sample_support_logprobs"


def sample_support_width(sampling_params: "Mapping[str, object]") -> int:
    """Row width of a captured support: the sampler's ``top_k`` when it truncates, else ``logprobs``.

    Sample-support replay needs the complete post-filter support, so it requires ``top_k > 1``. Score
    centering only needs the sampler's top-``logprobs`` head and works with an untruncated sampler
    (``top_k = -1``), where the head is the ``logprobs`` most probable tokens.
    """
    top_k = sampling_params.get("top_k")
    if isinstance(top_k, int) and not isinstance(top_k, bool) and top_k > 1:
        return top_k
    logprobs = sampling_params.get("logprobs")
    if isinstance(logprobs, int) and not isinstance(logprobs, bool) and logprobs > 1:
        return logprobs
    return 0


def validate_sample_support_logprobs(
    logprobs: SampleSupportLogprobs, sample_support: SampleSupport | None = None
) -> SampleSupportLogprobs:
    """Validate dtype, shape and padding of sampler logprobs against their support rows."""
    if not isinstance(logprobs, np.ndarray):
        raise TypeError("sample support logprobs must be a NumPy array")
    if logprobs.ndim != 2 or logprobs.dtype not in SAMPLE_SUPPORT_LOGPROBS_DTYPES:
        raise ValueError(
            "sample support logprobs must be a float32 [tokens, top_k] array, "
            f"got shape {logprobs.shape} and dtype {logprobs.dtype}"
        )
    if np.isnan(logprobs).any() or np.isposinf(logprobs).any():
        raise ValueError("sample support logprobs must be finite or -inf")
    if sample_support is not None:
        if sample_support.shape != logprobs.shape:
            raise ValueError(
                f"sample support logprobs shape {logprobs.shape} does not match support shape {sample_support.shape}"
            )
        padding = sample_support == SAMPLE_SUPPORT_PADDING
        if np.any(np.isfinite(logprobs) & padding):
            raise ValueError("sample support logprobs must be -inf on padding support entries")
        if np.any(~np.isfinite(logprobs) & ~padding):
            raise ValueError("sample support logprobs must be finite on captured support entries")
    return logprobs


def validate_sample_support(sample_support: SampleSupport) -> SampleSupport:
    """Validate vocabulary IDs and trailing padding."""
    if not isinstance(sample_support, np.ndarray):
        raise TypeError("sample support must be a NumPy array")
    if sample_support.ndim != 2 or not np.issubdtype(sample_support.dtype, np.integer):
        raise ValueError(
            "sample support must be an integer [tokens, top_k] array, "
            f"got shape {sample_support.shape} and dtype {sample_support.dtype}"
        )
    if int(sample_support.min(initial=0)) < SAMPLE_SUPPORT_PADDING:
        raise ValueError(f"sample support IDs must be {SAMPLE_SUPPORT_PADDING} padding or non-negative vocab IDs")
    if np.any((sample_support[:, :-1] == SAMPLE_SUPPORT_PADDING) & (sample_support[:, 1:] >= 0)):
        raise ValueError(f"sample support padding must be trailing {SAMPLE_SUPPORT_PADDING} values")
    return sample_support


def align_sample_support_row_ids(
    sample_support: PackedTensor,
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
        torch.arange(sample_support.values.shape[0], dtype=torch.long, device=sample_support.device),
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

    def append(self, sample_support: SampleSupport, *, expected_rows: int) -> None:
        self._metadata.append(validate_sample_support(sample_support), expected_rows=expected_rows)

    def append_padding(self, count: int) -> None:
        self._metadata.append_padding(count, fill=SAMPLE_SUPPORT_PADDING)

    def finalize(self, *, token_count: int, extra_rows: int) -> SampleSupport:
        """Validate the trace length and discard ``extra_rows`` trailing rows."""
        if self.num_rows != token_count + extra_rows:
            raise ValueError(
                f"sample-support trace has {self.num_rows} rows for {token_count} tokens plus "
                f"{extra_rows} trailing rows"
            )
        return self._metadata.finalize(expected_rows=self.num_rows)[:token_count]


class SampleSupportLogprobsTrace:
    """Accumulate sampler head logprobs across incremental generation calls, row-aligned with
    :class:`SampleSupportTrace`."""

    def __init__(self) -> None:
        self._metadata = TokenMetadataTrace()

    @property
    def num_rows(self) -> int:
        return self._metadata.num_rows

    def append(self, logprobs: SampleSupportLogprobs, *, expected_rows: int) -> None:
        self._metadata.append(validate_sample_support_logprobs(logprobs), expected_rows=expected_rows)

    def append_padding(self, count: int) -> None:
        self._metadata.append_padding(count, fill=SAMPLE_SUPPORT_LOGPROBS_PADDING)

    def finalize(self, *, token_count: int, extra_rows: int) -> SampleSupportLogprobs:
        """Validate the trace length and discard ``extra_rows`` trailing rows."""
        if self.num_rows != token_count + extra_rows:
            raise ValueError(
                f"sample-support logprobs trace has {self.num_rows} rows for {token_count} tokens plus "
                f"{extra_rows} trailing rows"
            )
        return self._metadata.finalize(expected_rows=self.num_rows)[:token_count]
