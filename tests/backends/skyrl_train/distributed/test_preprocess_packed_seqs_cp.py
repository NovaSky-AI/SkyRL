"""Test that preprocess_packed_seqs handles short sequences with CP > 1.

Regression test for the CP=2 tensor shape crash:
  RuntimeError: The expanded size of the tensor (4) must match the existing
  size (2) at non-singleton dimension 0.

Run with:
  uv run --isolated --extra dev -- pytest -s tests/backends/skyrl_train/distributed/test_preprocess_packed_seqs_cp.py
"""

import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

# ---------------------------------------------------------------------------
# Inject lightweight megatron stubs into sys.modules so that megatron_utils
# can be imported without megatron-core installed (CPU CI).
# The only real class the function under test constructs is PackedSeqParams;
# everything else (mpu, DDP, Float16Module, …) is mocked per-test or unused.
# ---------------------------------------------------------------------------


@dataclass
class _PackedSeqParams:
    qkv_format: str = ""
    cu_seqlens_q: Any = None
    max_seqlen_q: Any = None
    cu_seqlens_kv: Any = None
    max_seqlen_kv: Any = None
    cu_seqlens_q_padded: Any = None
    cu_seqlens_kv_padded: Any = None


_MEGATRON_MODULES = [
    "megatron",
    "megatron.core",
    "megatron.core.parallel_state",
    "megatron.core.distributed",
    "megatron.core.optimizer",
    "megatron.core.packed_seq_params",
    "megatron.core.transformer",
    "megatron.core.transformer.module",
    "megatron.core.utils",
]

_mock_modules: dict[str, ModuleType] = {}
for _name in _MEGATRON_MODULES:
    _mock_modules[_name] = ModuleType(_name)

_mock_modules["megatron.core"].parallel_state = _mock_modules["megatron.core.parallel_state"]
_mock_modules["megatron.core.packed_seq_params"].PackedSeqParams = _PackedSeqParams
_mock_modules["megatron.core.distributed"].DistributedDataParallel = MagicMock
_mock_modules["megatron.core.optimizer"].ChainedOptimizer = MagicMock
_mock_modules["megatron.core.transformer.module"].Float16Module = MagicMock
_mock_modules["megatron.core.utils"].get_attr_wrapped_model = MagicMock()
sys.modules.update(_mock_modules)


class TestPreprocessPackedSeqsShortSequencesCP:
    """preprocess_packed_seqs must not crash when sequences are shorter than align_size."""

    @pytest.mark.parametrize(
        "tp_size,cp_size,real_tokens",
        [
            (4, 2, 2),  # Production crash: align=16, 2-token masked seq
            (4, 2, 1),  # Even shorter
            (2, 2, 2),  # Smaller TP, still needs padding
            (1, 2, 2),  # TP=1, CP=2: align=4, 2-token seq (borderline)
        ],
    )
    def test_short_seq_no_crash(self, tp_size, cp_size, real_tokens):
        """Short sequences padded to align_size should not cause index errors."""
        from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
            preprocess_packed_seqs,
        )

        seq_len = 32  # padded dim of input_ids
        batch_size = 2

        # Build input_ids and attention_mask.
        # Seq 0: short (simulates masked/failed instance)
        # Seq 1: full length
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

        # Short sequence: only `real_tokens` valid tokens at the front
        input_ids[0, :real_tokens] = torch.arange(1, real_tokens + 1)
        attention_mask[0, :real_tokens] = True

        # Normal sequence: all tokens valid
        input_ids[1, :] = torch.arange(1, seq_len + 1)
        attention_mask[1, :] = True

        for cp_rank in range(cp_size):
            with patch("skyrl.backends.skyrl_train.distributed.megatron.megatron_utils.mpu") as mock_mpu:
                mock_mpu.get_tensor_model_parallel_world_size.return_value = tp_size
                mock_mpu.get_context_parallel_world_size.return_value = cp_size
                mock_mpu.get_context_parallel_rank.return_value = cp_rank

                # This used to raise RuntimeError for short sequences
                result_ids, packed_params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=True)

                assert result_ids.shape[0] == 1  # unsqueezed
                assert packed_params.qkv_format == "thd"
