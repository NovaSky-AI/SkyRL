"""Per-token sampler support: the bounded top-k set vLLM actually sampled from.

A support row is ``[top-1, ..., top-k]`` vocab IDs for one generated token, so the
trainer can renormalize its logprobs over the same bounded set the rollout sampler
drew from instead of the full vocabulary. Rows are dense and right-padded with
``SAMPLE_SUPPORT_PADDING``; a token with no captured support (prompt tokens,
observation tokens, a synthetic EOS) is an all-padding row.
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
    """Check the two invariants the generic packed-array codec cannot: no
    negatives other than the padding sentinel, and padding only ever trailing."""
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
        """Concatenate the trace and cut it to ``token_count`` rows.

        ``extra_rows`` is the trailing row count the response never keeps (the final observation),
        so the total is checked exactly in both directions and an unexpected overshoot raises
        instead of being silently truncated away.
        """
        if self.num_rows != token_count + extra_rows:
            raise ValueError(
                f"sample-support trace has {self.num_rows} rows for {token_count} tokens plus "
                f"{extra_rows} trailing rows"
            )
        return self._metadata.finalize(expected_rows=self.num_rows)[:token_count]
