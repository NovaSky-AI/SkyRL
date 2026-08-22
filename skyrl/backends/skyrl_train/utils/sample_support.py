"""Per-token bounded sampler support used to renormalize rollout logprobs.

Rows contain top-k vocabulary IDs and use trailing ``SAMPLE_SUPPORT_PADDING``.
"""

from typing import TypeAlias

import numpy as np

from skyrl.backends.skyrl_train.distributed.megatron.token_metadata import (
    TokenMetadataTrace,
)

SampleSupport: TypeAlias = np.ndarray
SAMPLE_SUPPORT_DTYPE = np.dtype(np.int32)
SAMPLE_SUPPORT_DTYPES = frozenset({SAMPLE_SUPPORT_DTYPE})
SAMPLE_SUPPORT_PADDING = -1


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
