"""Shared builders for CSR sample-support fixtures."""

from collections.abc import Sequence

import numpy as np

from skyrl.backends.skyrl_train.utils.packed_ragged_tensor import PackedRaggedTensor
from skyrl.backends.skyrl_train.utils.packed_tensor import PackedTensor
from skyrl.backends.skyrl_train.utils.sample_support import (
    SAMPLE_SUPPORT_DTYPE,
    SAMPLE_SUPPORT_LOGPROBS_DTYPE,
    CompactSampleSupport,
)


def csr(rows, logprobs=None) -> CompactSampleSupport:
    """CSR support from a ``[tokens, top_k]`` grid whose negative entries are absent.

    ``logprobs``, a matching grid, supplies each kept member's sampler logprob.
    """
    rows = np.asarray(rows, dtype=SAMPLE_SUPPORT_DTYPE)
    if logprobs is not None:
        logprobs = np.asarray(logprobs, dtype=SAMPLE_SUPPORT_LOGPROBS_DTYPE)
    return CompactSampleSupport.from_masked_rows(rows, rows >= 0, logprobs)


def ragged_support(trajectories: PackedTensor | Sequence, logprobs: Sequence | None = None) -> PackedRaggedTensor:
    """Pack one ``[tokens, top_k]`` grid per trajectory; negative entries are absent.

    A ``PackedTensor`` of fixed-width rows supplies one grid per segment. ``logprobs``
    supplies one matching logprob grid per trajectory.
    """
    if isinstance(trajectories, PackedTensor):
        trajectories = [trajectories.segment(index) for index in range(len(trajectories))]
    if logprobs is None:
        logprobs = [None] * len(trajectories)
    return PackedRaggedTensor.from_rows([csr(rows, grid) for rows, grid in zip(trajectories, logprobs, strict=True)])


def dense(support: CompactSampleSupport, fill: int = -1) -> np.ndarray:
    """Expand CSR rows back into a ``[tokens, top_k]`` grid padded with ``fill``."""
    grid = np.full((support.num_rows, support.top_k), fill, dtype=SAMPLE_SUPPORT_DTYPE)
    columns = np.arange(support.token_ids.size) - np.repeat(support.offsets[:-1], support.row_lengths)
    grid[np.repeat(np.arange(support.num_rows), support.row_lengths), columns] = support.token_ids
    return grid
