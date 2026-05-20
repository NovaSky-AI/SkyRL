"""Test preprocess_packed_seqs with the new ``sub_seq_lengths`` argument.

Run with:
  uv run --extra dev --extra megatron -- pytest \
      tests/backends/skyrl_train/distributed/test_preprocess_packed_seqs_multiseq.py
"""

import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import torch


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


def _mock_mpu(tp_size: int = 1, cp_size: int = 1, cp_rank: int = 0):
    """Create a mock mpu module with the given world sizes."""
    mock = MagicMock()
    mock.get_tensor_model_parallel_world_size.return_value = tp_size
    mock.get_context_parallel_world_size.return_value = cp_size
    mock.get_context_parallel_rank.return_value = cp_rank
    return mock


class TestSubSeqLengths:
    """preprocess_packed_seqs with sub_seq_lengths enumerates every sub-seq."""

    def test_one_subseq_per_row_matches_legacy_path(self):
        """When sub_seq_lengths has one length per row, output matches the
        attention_mask-inferred path."""
        from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
            preprocess_packed_seqs,
        )

        seq_len = 16
        batch_size = 2

        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

        # Row 0: 5 valid tokens left-aligned. Row 1: 8 valid tokens left-aligned.
        input_ids[0, :5] = torch.arange(1, 6)
        attention_mask[0, :5] = True
        input_ids[1, :8] = torch.arange(10, 18)
        attention_mask[1, :8] = True

        with patch(
            "skyrl.backends.skyrl_train.distributed.megatron.megatron_utils.mpu",
            _mock_mpu(tp_size=1, cp_size=1),
        ):
            # Legacy path: each row's attention_mask sum is its sub-seq length.
            legacy_packed, legacy_params = preprocess_packed_seqs(
                input_ids,
                attention_mask,
                pre_process=True,
            )
            # New path: same outcome when sub_seq_lengths matches per-row sums.
            new_packed, new_params = preprocess_packed_seqs(
                input_ids,
                attention_mask,
                pre_process=True,
                sub_seq_lengths=[[5], [8]],
            )

        assert torch.equal(legacy_params.cu_seqlens_q, new_params.cu_seqlens_q)
        assert torch.equal(legacy_packed, new_packed)
        # In the legacy path we infer per-row sub-seqs via attention_mask;
        # the new path is given sub-seqs directly. With one sub-seq per row
        # the *new* path needs to read the row left-aligned (offset 0) for
        # both rows. The new path reads input_ids[r, 0 : seqlen], which here
        # IS the same data as input_ids[r, attention_mask[r]] because we
        # left-aligned valid tokens.

    def test_multiseq_row_emits_extra_cu_seqlens_entries(self):
        """A row with two sub-seqs produces three cu_seqlens entries (0, s0, s0+s1)."""
        from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
            preprocess_packed_seqs,
        )

        seq_len = 16
        batch_size = 1

        # Bin row contains two sub-seqs of length 3 and 4 concatenated.
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        input_ids[0, :3] = torch.tensor([11, 12, 13])
        input_ids[0, 3:7] = torch.tensor([21, 22, 23, 24])
        attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        attention_mask[0, :7] = True

        with patch(
            "skyrl.backends.skyrl_train.distributed.megatron.megatron_utils.mpu",
            _mock_mpu(tp_size=1, cp_size=1),
        ):
            packed, params = preprocess_packed_seqs(
                input_ids,
                attention_mask,
                pre_process=True,
                sub_seq_lengths=[[3, 4]],
            )

        # cu_seqlens enumerates both sub-seqs: [0, 3, 7].
        assert params.cu_seqlens_q.tolist() == [0, 3, 7]
        # Packed buffer holds both sub-seqs back to back.
        assert packed.shape == (1, 7)
        assert packed[0].tolist() == [11, 12, 13, 21, 22, 23, 24]

    def test_multiseq_with_tp_alignment(self):
        """Each sub-seq is independently padded to a multiple of tp_size.

        The intra-row offsets read by preprocess must match the
        collator's row layout, which advances ``row_offset += round_up(s,
        align_size)`` between sub-seqs. So with sub-seqs of length 3 and
        5 and tp_size=4, the collator places sub-seq 1 at row column 4
        (after a 1-token TP-alignment pad gap), NOT row column 3.
        """
        from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
            preprocess_packed_seqs,
        )

        seq_len = 32
        batch_size = 1

        # Two sub-seqs of length 3 and 5; tp_size=4 should pad each to 4 and 8.
        # Row layout mirrors what SFTTrainerWithPacking.collate_batch produces:
        # row[0:3]   = sub-seq 0 tokens
        # row[3]     = TP-alignment pad (zero)
        # row[4:9]   = sub-seq 1 tokens
        # row[9:12]  = TP-alignment pad (zero)
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        input_ids[0, :3] = torch.tensor([1, 2, 3])
        input_ids[0, 4:9] = torch.tensor([10, 11, 12, 13, 14])
        attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        attention_mask[0, :3] = True
        attention_mask[0, 4:9] = True

        with patch(
            "skyrl.backends.skyrl_train.distributed.megatron.megatron_utils.mpu",
            _mock_mpu(tp_size=4, cp_size=1),
        ):
            packed, params = preprocess_packed_seqs(
                input_ids,
                attention_mask,
                pre_process=True,
                sub_seq_lengths=[[3, 5]],
            )

        # cu_seqlens (un-padded) tracks the real token starts: 0, 3, 8.
        # cu_seqlens_padded reflects tp-aligned starts: 0, 4 (=ceil(3/4)*4), 12 (=4+8).
        # cu_seqlens_q == cu_seqlens_padded in qkv_format="thd".
        assert params.cu_seqlens_q.tolist() == [0, 4, 12]
        # Packed buffer has tp-aligned slots: sub-seq 0 occupies tokens 0..3,
        # padded to 0..4. Sub-seq 1 occupies tokens 4..9 (padded to 4..12).
        assert packed.shape == (1, 12)
        assert packed[0, :3].tolist() == [1, 2, 3]
        # Position 3 is pad (zero).
        assert packed[0, 3].item() == 0
        assert packed[0, 4:9].tolist() == [10, 11, 12, 13, 14]

    def test_multiple_bin_rows(self):
        """Two bin rows, each with two sub-seqs, produce 4+1 cu_seqlens entries."""
        from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
            preprocess_packed_seqs,
        )

        seq_len = 16
        batch_size = 2

        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        input_ids[0, :2] = torch.tensor([1, 2])
        input_ids[0, 2:5] = torch.tensor([3, 4, 5])
        input_ids[1, :4] = torch.tensor([10, 11, 12, 13])
        input_ids[1, 4:6] = torch.tensor([20, 21])
        attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        attention_mask[0, :5] = True
        attention_mask[1, :6] = True

        with patch(
            "skyrl.backends.skyrl_train.distributed.megatron.megatron_utils.mpu",
            _mock_mpu(tp_size=1, cp_size=1),
        ):
            packed, params = preprocess_packed_seqs(
                input_ids,
                attention_mask,
                pre_process=True,
                sub_seq_lengths=[[2, 3], [4, 2]],
            )

        # Four sub-seqs total: cu_seqlens [0, 2, 5, 9, 11].
        assert params.cu_seqlens_q.tolist() == [0, 2, 5, 9, 11]
        assert packed.shape == (1, 11)
        assert packed[0].tolist() == [1, 2, 3, 4, 5, 10, 11, 12, 13, 20, 21]
