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

import pytest
import torch

from skyrl.backends.skyrl_train.distributed.megatron.packing_utils import (
    get_packed_seq_align_size,
)


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
    "megatron.core.transformer.moe",
    "megatron.core.transformer.moe.moe_utils",
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
_mock_modules["megatron.core.transformer.moe.moe_utils"].clear_aux_losses_tracker = MagicMock()
_mock_modules["megatron.core.transformer.moe.moe_utils"].get_moe_layer_wise_logging_tracker = MagicMock()
_mock_modules["megatron.core.transformer.moe.moe_utils"].reduce_aux_losses_tracker_across_ranks = MagicMock()
_mock_modules["megatron.core.utils"].get_attr_wrapped_model = MagicMock()
_mock_modules["megatron.core.utils"].unwrap_model = MagicMock()


@pytest.fixture(scope="module", autouse=True)
def _stub_megatron_modules():
    """Install the mock ``megatron`` modules for this module's tests only.

    The stubs are injected into ``sys.modules`` at module setup and removed at
    teardown so they do not leak into other test files in the same pytest
    session. Only the megatron entries are touched: evicting everything this
    module imported (e.g. ``vllm``) would force a re-import whose module-level
    side effects are not idempotent.
    """
    saved = {_name: sys.modules.get(_name) for _name in _MEGATRON_MODULES}
    sys.modules.update(_mock_modules)
    try:
        yield
    finally:
        for _name in _MEGATRON_MODULES:
            if saved[_name] is None:
                sys.modules.pop(_name, None)
            else:
                sys.modules[_name] = saved[_name]


def _mock_mpu(tp_size: int = 1, cp_size: int = 1, cp_rank: int = 0):
    """Create a mock mpu module with the given world sizes."""
    mock = MagicMock()
    mock.get_tensor_model_parallel_world_size.return_value = tp_size
    mock.get_context_parallel_world_size.return_value = cp_size
    mock.get_context_parallel_rank.return_value = cp_rank
    return mock


def _get_align_size(tp_size: int, cp_size: int, fp8_enabled: bool = False) -> int:
    return get_packed_seq_align_size(tp_size, cp_size, fp8_enabled=fp8_enabled)


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

    def test_multiseq_row_emits_padded_cu_seqlens_entries(self):
        """A row with two sub-seqs produces one padded cu_seqlens segment per sub-seq."""
        from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
            preprocess_packed_seqs,
        )

        seq_len = 48
        batch_size = 1

        # Bin row contains two sub-seqs of length 3 and 4 concatenated.
        align_size = _get_align_size(tp_size=1, cp_size=1, fp8_enabled=True)
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        input_ids[0, :3] = torch.tensor([11, 12, 13])
        input_ids[0, align_size : align_size + 4] = torch.tensor([21, 22, 23, 24])
        attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        attention_mask[0, :3] = True
        attention_mask[0, align_size : align_size + 4] = True

        with patch(
            "skyrl.backends.skyrl_train.distributed.megatron.megatron_utils.mpu",
            _mock_mpu(tp_size=1, cp_size=1),
        ):
            packed, params = preprocess_packed_seqs(
                input_ids,
                attention_mask,
                pre_process=True,
                sub_seq_lengths=[[3, 4]],
                fp8_enabled=True,
            )

        assert params.cu_seqlens_q.tolist() == [0, 16, 32]
        assert packed.shape == (1, 32)
        assert packed[0, :3].tolist() == [11, 12, 13]
        assert packed[0, 16:20].tolist() == [21, 22, 23, 24]

    def test_multiseq_with_tp_alignment(self):
        """Each sub-seq is independently padded to a multiple of tp_size.

        The intra-row offsets read by preprocess must match the
        collator's row layout, which advances ``row_offset += round_up(s,
        align_size)`` between sub-seqs. So with sub-seqs of length 3 and
        5 and tp_size=4, the collator places sub-seq 1 at row column
        ``align_size`` after the FP8/TP alignment gap, not row column 3.
        """
        from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
            preprocess_packed_seqs,
        )

        batch_size = 1

        # Each sequence uses a global footprint that leaves TP-local inputs
        # aligned to 128 tokens.
        align_size = _get_align_size(tp_size=4, cp_size=1, fp8_enabled=True)
        seq_len = 2 * align_size
        # Row layout mirrors what PackedDataCollator produces:
        # row[0:3]                         = sub-seq 0 tokens
        # row[3:align_size]                = alignment pad (zero)
        # row[align_size:align_size + 5]   = sub-seq 1 tokens
        # row[align_size + 5:2*align_size] = alignment pad (zero)
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        input_ids[0, :3] = torch.tensor([1, 2, 3])
        input_ids[0, align_size : align_size + 5] = torch.tensor([10, 11, 12, 13, 14])
        attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        attention_mask[0, :3] = True
        attention_mask[0, align_size : align_size + 5] = True

        with patch(
            "skyrl.backends.skyrl_train.distributed.megatron.megatron_utils.mpu",
            _mock_mpu(tp_size=4, cp_size=1),
        ):
            packed, params = preprocess_packed_seqs(
                input_ids,
                attention_mask,
                pre_process=True,
                sub_seq_lengths=[[3, 5]],
                fp8_enabled=True,
            )

        assert params.cu_seqlens_q.tolist() == [0, align_size, 2 * align_size]
        assert packed.shape == (1, 2 * align_size)
        assert packed[0, :3].tolist() == [1, 2, 3]
        assert packed[0, 3:align_size].tolist() == [0] * (align_size - 3)
        assert packed[0, align_size : align_size + 5].tolist() == [10, 11, 12, 13, 14]

    def test_multiple_bin_rows(self):
        """Two bin rows, each with two sub-seqs, produce 4+1 cu_seqlens entries."""
        from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
            preprocess_packed_seqs,
        )

        seq_len = 48
        batch_size = 2

        align_size = _get_align_size(tp_size=1, cp_size=1, fp8_enabled=True)
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        input_ids[0, :2] = torch.tensor([1, 2])
        input_ids[0, align_size : align_size + 3] = torch.tensor([3, 4, 5])
        input_ids[1, :4] = torch.tensor([10, 11, 12, 13])
        input_ids[1, align_size : align_size + 2] = torch.tensor([20, 21])
        attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        attention_mask[0, :2] = True
        attention_mask[0, align_size : align_size + 3] = True
        attention_mask[1, :4] = True
        attention_mask[1, align_size : align_size + 2] = True

        with patch(
            "skyrl.backends.skyrl_train.distributed.megatron.megatron_utils.mpu",
            _mock_mpu(tp_size=1, cp_size=1),
        ):
            packed, params = preprocess_packed_seqs(
                input_ids,
                attention_mask,
                pre_process=True,
                sub_seq_lengths=[[2, 3], [4, 2]],
                fp8_enabled=True,
            )

        assert params.cu_seqlens_q.tolist() == [0, 16, 32, 48, 64]
        assert packed.shape == (1, 64)
        assert packed[0, :2].tolist() == [1, 2]
        assert packed[0, 16:19].tolist() == [3, 4, 5]
        assert packed[0, 32:36].tolist() == [10, 11, 12, 13]
        assert packed[0, 48:50].tolist() == [20, 21]


class TestMultiSeqCPLayout:
    """preprocess CP sharding layout with CP > 1.

    The CP zigzag in preprocess_packed_seqs is pure index manipulation, so it
    can be exercised on CPU by mocking the context-parallel rank/world-size.
    We analytically reassemble each CP rank's shard in the test to verify the
    sharded buffers recover the full row layout.
    """

    @staticmethod
    def _build_batch(tp_size, cp_size, sub_seq_lengths, seq_len=64, fp8_enabled=True):
        """Build (input_ids, attention_mask) whose row layout matches the
        PackedDataCollator: each sub-seq padded to align_size,
        laid out back-to-back from column 0. Real tokens get unique nonzero
        ids so reassembly is exactly verifiable.
        """
        align_size = _get_align_size(tp_size, cp_size, fp8_enabled=fp8_enabled)

        def round_up(x, m):
            return ((x + m - 1) // m) * m

        batch_size = len(sub_seq_lengths)
        seq_len = max(
            seq_len,
            max(sum(round_up(length, align_size) for length in lens) for lens in sub_seq_lengths),
        )
        input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        next_tok = 1
        for r, lens in enumerate(sub_seq_lengths):
            offset = 0
            for length in lens:
                ids = torch.arange(next_tok, next_tok + length, dtype=torch.long)
                next_tok += length
                input_ids[r, offset : offset + length] = ids
                attention_mask[r, offset : offset + length] = True
                offset += round_up(length, align_size)
        return input_ids, attention_mask

    def _run_roundtrip(self, tp_size, cp_size, sub_seq_lengths, fp8_enabled=True):
        from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
            preprocess_packed_seqs,
        )

        input_ids, attention_mask = self._build_batch(tp_size, cp_size, sub_seq_lengths, fp8_enabled=fp8_enabled)
        batch_size, seq_len = input_ids.shape

        # ---- ground truth: the full (un-sharded) padded THD layout ----
        # Under the identity model, analytically reassembling each row's
        # sub-seqs (each padded to align_size) back-to-back from column 0 with
        # the rest zero should recover ``input_ids`` cast to float.
        # NOTE: we deliberately do NOT use the cp_size==1 path as ground truth
        # here -- the cp==1 and cp>1 paths use different divisors, so the two
        # assume different intra-row sub-seq offsets and are not directly
        # comparable. The analytic layout below is align_size-consistent and a
        # stronger target.
        gt = input_ids.to(torch.float32)

        # ---- CP > 1: preprocess each rank, then analytically reassemble ----
        per_rank_out = []
        params_cp = None
        for cp_rank in range(cp_size):
            with patch(
                "skyrl.backends.skyrl_train.distributed.megatron.megatron_utils.mpu",
                _mock_mpu(tp_size=tp_size, cp_size=cp_size, cp_rank=cp_rank),
            ):
                packed_r, params_r = preprocess_packed_seqs(
                    input_ids,
                    attention_mask,
                    pre_process=True,
                    sub_seq_lengths=sub_seq_lengths,
                    fp8_enabled=fp8_enabled,
                )
            per_rank_out.append(packed_r.to(torch.float32))
            params_cp = params_r  # cu_seqlens are global, identical across ranks

        # Every rank's local buffer must be the same length (== total/cp).
        for r in range(1, cp_size):
            assert per_rank_out[r].shape == per_rank_out[0].shape
        if fp8_enabled:
            assert per_rank_out[0].shape[1] % 16 == 0

        recovered = self._reassemble_cp_shards(per_rank_out, params_cp, batch_size, seq_len, cp_size, sub_seq_lengths)

        return gt, recovered

    @staticmethod
    def _reassemble_cp_shards(per_rank_out, params_cp, batch_size, seq_len, cp_size, sub_seq_lengths):
        cu_padded = params_cp.cu_seqlens_q_padded.tolist()
        output = torch.zeros((batch_size, seq_len), dtype=per_rank_out[0].dtype)

        flat_sub_idx_of_row_start = []
        running = 0
        for row_lens in sub_seq_lengths:
            flat_sub_idx_of_row_start.append(running)
            running += len(row_lens)

        for row_idx, row_lens in enumerate(sub_seq_lengths):
            row_first_sub = flat_sub_idx_of_row_start[row_idx]
            row_last_sub_exclusive = row_first_sub + len(row_lens)
            row_global_start = cu_padded[row_first_sub]
            for seg in range(row_first_sub, row_last_sub_exclusive):
                local_segment_len = (cu_padded[seg + 1] - cu_padded[seg]) // cp_size
                half_seqlen = local_segment_len // 2
                padded_segment_len = local_segment_len * cp_size
                packed_start_idx = cu_padded[seg] // cp_size
                dest_off = cu_padded[seg] - row_global_start

                tmp = torch.empty(padded_segment_len, dtype=per_rank_out[0].dtype)
                for cp_rank in range(cp_size):
                    rank_tokens = per_rank_out[cp_rank][0]
                    front = rank_tokens[packed_start_idx : packed_start_idx + half_seqlen]
                    back = rank_tokens[packed_start_idx + half_seqlen : packed_start_idx + local_segment_len]
                    tmp[cp_rank * half_seqlen : (cp_rank + 1) * half_seqlen] = front
                    tmp[
                        padded_segment_len - (cp_rank + 1) * half_seqlen : padded_segment_len - cp_rank * half_seqlen
                    ] = back
                output[row_idx, dest_off : dest_off + padded_segment_len] = tmp

        return output

    def test_roundtrip_recovers_full_layout_cp2(self):
        # tp=1, cp=2 -> align_size=32. Sub-seq lengths chosen so each row has
        # multiple sub-seqs of differing lengths.
        gt, recovered = self._run_roundtrip(tp_size=1, cp_size=2, sub_seq_lengths=[[6, 5], [7, 3]])
        assert torch.equal(recovered, gt), (
            "CP=2 multi-subseq un-zigzag did not recover the full padded THD layout.\n"
            f"gt=\n{gt}\nrecovered=\n{recovered}"
        )

    def test_bf16_roundtrip_cp2_uses_layout_alignment_only(self):
        gt, recovered = self._run_roundtrip(
            tp_size=1,
            cp_size=2,
            sub_seq_lengths=[[6, 5], [7, 3]],
            fp8_enabled=False,
        )
        assert recovered[0, 8:13].tolist() == [7.0, 8, 9, 10, 11]
        assert recovered[0, 13:32].tolist() == [0.0] * 19
        assert torch.equal(recovered, gt)

    def test_roundtrip_recovers_full_layout_tp2_cp2(self):
        # tp=2, cp=2 -> align_size=32.
        gt, recovered = self._run_roundtrip(tp_size=2, cp_size=2, sub_seq_lengths=[[5, 9], [3, 6]])
        assert torch.equal(recovered, gt)

    def test_roundtrip_cp4(self):
        # tp=1, cp=4 -> align_size=64. Exercises >2 CP ranks so the per-rank
        # zigzag chunk assignment (ranks 0..3 hold chunks {0,7},{1,6},{2,5},{3,4})
        # is genuinely tested, not just the symmetric cp=2 case.
        gt, recovered = self._run_roundtrip(tp_size=1, cp_size=4, sub_seq_lengths=[[10, 6], [13, 3]])
        assert torch.equal(recovered, gt)

    def test_roundtrip_single_subseq_per_row_cp2(self):
        # Degenerate multi-subseq (one sub-seq per row) must also round-trip
        # and recover the full padded layout.
        gt, recovered = self._run_roundtrip(tp_size=1, cp_size=2, sub_seq_lengths=[[7], [10]])
        assert torch.equal(recovered, gt)

    def test_roundtrip_recovers_original_tokens_cp2(self):
        # Strongest check: hard-coded expected token ids at known slots, so a
        # wrong un-zigzag that happens to be self-consistent with a wrong
        # preprocess would still be caught.
        tp_size, cp_size = 1, 2
        sub_seq_lengths = [[6, 5], [7, 3]]
        _, recovered = self._run_roundtrip(tp_size, cp_size, sub_seq_lengths)

        # align_size=32: row0 starts sub-seqs at offsets 0 and 32; row1 does
        # the same with its own token ids.
        assert recovered[0, 0:6].tolist() == [1.0, 2, 3, 4, 5, 6]
        assert recovered[0, 6:32].tolist() == [0.0] * 26
        assert recovered[0, 32:37].tolist() == [7.0, 8, 9, 10, 11]
        assert recovered[1, 0:7].tolist() == [12.0, 13, 14, 15, 16, 17, 18]
        assert recovered[1, 7:32].tolist() == [0.0] * 25
        assert recovered[1, 32:35].tolist() == [19.0, 20, 21]
