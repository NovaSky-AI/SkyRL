"""Parity test for SkyRL's vendored DSA k-pool pooling math.

``_kpool_compress_keys`` is vendored from NVIDIA/Megatron-LM#7522 into
``mcore_ext/dsa_kpool.py`` because the pinned megatron-core does not have the pooled indexer.
GLM-5.3-Flash depends on it matching HF's ``Glm5NextTextIndexer.compress_keys`` exactly, so a
transcription error would otherwise only surface as a quality drop in a training run. The HF
formulation is transcribed independently below rather than imported, so a change on either
side shows up.

Pure tensor math, so it runs on CPU and lives outside ``gpu/``; the selection-behaviour half
needs the CUDA kernels and stays in
``tests/backends/skyrl_train/gpu/gpu_ci/patches/megatron/mcore_ext/test_dsa_kpool.py``.
"""

import pytest
import torch

pytest.importorskip("megatron.core", reason="requires the megatron extra")

# Runs in the CPU megatron job (`-m megatron`); without the marker that job deselects it.
pytestmark = pytest.mark.megatron

POOL_SIZE = 4
HEAD_DIM = 128


def _hf_reference_pool(k: torch.Tensor, gate_score: torch.Tensor, ape: torch.Tensor, pool_size: int):
    """Independent transcription of HF ``Glm5NextTextIndexer`` pooling.

    HF works in [batch, seq, dim] and does::

        logits        = grouped_gate_scores.float() + ape.float()[None, None]
        probabilities = logits.softmax(dim=2)          # over the pool's slots
        pool_keys     = (probabilities * grouped_keys).sum(dim=2)

    megatron-core works in [tokens, batch, dim] and only compresses complete pools.
    """
    tokens, batch, dim = k.shape
    n_pools = tokens // pool_size
    trimmed = n_pools * pool_size
    grouped_k = k[:trimmed].reshape(n_pools, pool_size, batch, dim)
    grouped_gate = gate_score[:trimmed].reshape(n_pools, pool_size, batch, dim)

    logits = grouped_gate.float() + ape.float()[None, :, None, :]
    probs = logits.softmax(dim=1)
    return (probs * grouped_k.float()).sum(dim=1)


@pytest.mark.parametrize("seqlen", [64, 256, 1024])
def test_kpool_compress_keys_matches_hf_reference(seqlen):
    """The softmax-weighted pooling itself, against HF's formulation."""
    from skyrl.backends.skyrl_train.patches.megatron.mcore_ext.dsa_kpool import (
        _kpool_compress_keys,
    )

    gen = torch.Generator(device="cpu").manual_seed(0)
    k = torch.randn(seqlen, 1, HEAD_DIM, dtype=torch.bfloat16, generator=gen)
    gate = torch.randn(seqlen, 1, HEAD_DIM, dtype=torch.bfloat16, generator=gen)
    ape = torch.randn(POOL_SIZE, HEAD_DIM, dtype=torch.float32, generator=gen)

    got = _kpool_compress_keys(k, gate, ape, POOL_SIZE)
    want = _hf_reference_pool(k, gate, ape, POOL_SIZE)

    assert got.shape[0] == seqlen // POOL_SIZE, f"expected {seqlen // POOL_SIZE} pools, got {got.shape[0]}"
    assert got.shape[1:] == k.shape[1:]
    # megatron accumulates in fp32 and returns bf16; compare at bf16 resolution.
    torch.testing.assert_close(got.float(), want.float(), rtol=2e-2, atol=2e-2)
