"""Collator -> preprocess -> postprocess round-trip tests for packed SFT.

These tests close the specific gap called out in Reviewer-1's verdict:
the 34 packing-related tests that existed prior to this file did not
exercise the *collator output -> preprocess -> postprocess* round-trip
with multi-subseq rows under both ``mbs > 1`` and ``tp_size > 1``. The
two correctness bugs that the reviewer found live exactly at the
interfaces between those steps:

  Bug #1 (postprocess) — when ``cu_seqlens_q_padded`` enumerates one
    entry per *sub-seq* (not per row), iterating over batch rows
    using ``cu_padded_cpu[i]`` reads from the wrong THD offset for
    every row ``i > 0`` whose predecessors contained > 1 sub-seq.

  Bug #2 (preprocess) — the controller-side collator advances
    ``row_offset += round_up(s, align_size)`` between sub-seqs in the
    same row. ``preprocess_packed_seqs`` was advancing by the un-padded
    length, so for ``tp_size > 1`` and multi-subseq rows it read
    sub-seq ``i > 0`` starting inside the TP-alignment pad gap of
    sub-seq ``i - 1``.

The configuration here (``mbs=2``, ``tp_size=4``, two multi-subseq rows
with sub-seq lengths ``[7, 5]`` and ``[3, 11]``) hits both bugs at once.

Run with:
  uv run --extra dev --extra megatron -- pytest \
      tests/train/test_packing_round_trip.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import torch


# ---------------------------------------------------------------------------
# Mock the megatron-core surface that the preprocess/postprocess utils import
# at the top of the module. Same recipe as
# tests/backends/skyrl_train/distributed/test_preprocess_packed_seqs_multiseq.py
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


def _mock_mpu(tp_size: int = 1, cp_size: int = 1, cp_rank: int = 0):
    mock = MagicMock()
    mock.get_tensor_model_parallel_world_size.return_value = tp_size
    mock.get_context_parallel_world_size.return_value = cp_size
    mock.get_context_parallel_rank.return_value = cp_rank
    return mock


def _build_collator_layout(
    sub_seq_token_lists: list[list[list[int]]],
    *,
    align_size: int,
    pad_token_id: int = 0,
):
    """Construct the (sequences, attention_mask) tensor pair that the
    controller-side collator produces for the given per-row sub-seq token
    lists. Mirrors ``SFTTrainerWithPacking.collate_batch`` exactly:
    write each sub-seq's tokens at ``row_offset``, advance row_offset by
    ``_round_up(len(subseq), align_size)`` between sub-seqs in the same
    row, and set ``attention_mask=1`` only at the valid (non-pad) slots.
    """

    def _round_up(x: int, m: int) -> int:
        return ((x + m - 1) // m) * m

    batch_size = len(sub_seq_token_lists)
    sub_seq_lengths: list[list[int]] = [[len(s) for s in row] for row in sub_seq_token_lists]
    bin_packed_lengths: list[int] = [sum(_round_up(s, align_size) for s in row) for row in sub_seq_lengths]
    max_packed_len = max(bin_packed_lengths) if bin_packed_lengths else 0

    sequences = torch.full((batch_size, max_packed_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_packed_len), dtype=torch.bool)
    for r, row in enumerate(sub_seq_token_lists):
        row_offset = 0
        for tokens in row:
            s = len(tokens)
            sequences[r, row_offset : row_offset + s] = torch.tensor(tokens)
            attention_mask[r, row_offset : row_offset + s] = True
            row_offset += _round_up(s, align_size)
    return sequences, attention_mask, sub_seq_lengths


class TestRoundTrip:
    """End-to-end: collator output -> preprocess -> [identity model] -> postprocess."""

    def test_mbs2_tp4_multisubseq_rows_recover_correctly(self):
        """The exact configuration that would have caught both bugs.

        - ``mbs = 2`` so the THD offset stride matters per micro-batch row.
        - ``tp_size = 4`` so the TP-alignment padding inside rows kicks in.
        - Multi-subseq rows ``[(7, 5)]`` and ``[(3, 11)]`` to expose offset
          mismatches in both directions (sub-seq 0 length 7 is NOT a multiple
          of 4, sub-seq 1 length 5 is also not; row 1's sub-seq 0 length 3 is
          shorter than its sub-seq 1 length 11).
        """
        from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
            postprocess_packed_seqs,
            preprocess_packed_seqs,
        )

        tp_size = 4
        # Distinct unique tokens per sub-seq so we can spot off-by-one bugs.
        row_0_sub_0 = [101, 102, 103, 104, 105, 106, 107]  # len 7
        row_0_sub_1 = [201, 202, 203, 204, 205]  # len 5
        row_1_sub_0 = [301, 302, 303]  # len 3
        row_1_sub_1 = [401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411]  # len 11

        sequences, attention_mask, sub_seq_lengths = _build_collator_layout(
            [[row_0_sub_0, row_0_sub_1], [row_1_sub_0, row_1_sub_1]],
            align_size=tp_size,
        )

        # Sanity: collator layout matches what SFTTrainerWithPacking does.
        #   row 0: [101..107] + [pad] + [201..205] + [pad, pad, pad]  => 16 slots
        #   row 1: [301..303] + [pad] + [401..411] + [pad]            => 16 slots
        assert sequences.shape == (2, 16)
        assert sequences[0, :7].tolist() == row_0_sub_0
        assert sequences[0, 7].item() == 0
        assert sequences[0, 8:13].tolist() == row_0_sub_1
        assert sequences[1, :3].tolist() == row_1_sub_0
        assert sequences[1, 3].item() == 0
        assert sequences[1, 4:15].tolist() == row_1_sub_1
        # attention_mask is True ONLY at valid slots (NOT at TP-alignment gaps).
        assert attention_mask[0, 7].item() is False
        assert attention_mask[0].sum().item() == 7 + 5
        assert attention_mask[1].sum().item() == 3 + 11

        # ------------------------------------------------------------------
        # preprocess: pack the row tensors into a THD slab.
        # ------------------------------------------------------------------
        with patch(
            "skyrl.backends.skyrl_train.distributed.megatron.megatron_utils.mpu",
            _mock_mpu(tp_size=tp_size, cp_size=1),
        ):
            packed, params = preprocess_packed_seqs(
                sequences,
                attention_mask,
                pre_process=True,
                sub_seq_lengths=sub_seq_lengths,
            )

        # cu_seqlens_q (== cu_seqlens_q_padded for THD) enumerates 4 sub-seqs:
        #   row 0: 0 -> ceil(7/4)*4=8 -> 8+ceil(5/4)*4=16
        #   row 1: 16 -> 16+ceil(3/4)*4=20 -> 20+ceil(11/4)*4=32
        assert params.cu_seqlens_q.tolist() == [0, 8, 16, 20, 32]
        # Packed slab is 32 tokens. Verify each sub-seq's *valid* tokens were
        # read from the *correct intra-row offset* — i.e. preprocess respected
        # the TP-alignment gap that the collator inserted.
        assert packed.shape == (1, 32)
        assert packed[0, :7].tolist() == row_0_sub_0  # row 0 sub 0
        assert packed[0, 7].item() == 0  # TP-alignment pad
        assert packed[0, 8:13].tolist() == row_0_sub_1  # row 0 sub 1
        assert packed[0, 13:16].tolist() == [0, 0, 0]  # TP-alignment pad
        assert packed[0, 16:19].tolist() == row_1_sub_0  # row 1 sub 0
        assert packed[0, 19].item() == 0  # TP-alignment pad
        assert packed[0, 20:31].tolist() == row_1_sub_1  # row 1 sub 1
        assert packed[0, 31].item() == 0  # TP-alignment pad

        # ------------------------------------------------------------------
        # "Forward pass" — use the packed slab itself as the model output
        # (identity model). Shape becomes [1, T, hidden_dim] with hidden_dim=1.
        # ------------------------------------------------------------------
        output = packed.unsqueeze(-1).float()
        assert output.shape == (1, 32, 1)

        # ------------------------------------------------------------------
        # postprocess: scatter the THD slab back into [B, S, ...] layout.
        # ------------------------------------------------------------------
        with patch(
            "skyrl.backends.skyrl_train.distributed.megatron.megatron_utils.mpu",
            _mock_mpu(tp_size=tp_size, cp_size=1),
        ):
            recovered = postprocess_packed_seqs(
                output,
                params,
                attention_mask,
                batch_size=2,
                seq_len=sequences.shape[1],
                post_process=True,
                sub_seq_lengths=sub_seq_lengths,
            )

        recovered = recovered.squeeze(-1)
        # Both rows should round-trip to their original collator layout
        # (including TP-alignment pads). Without the Bug #1 fix, row 1 would
        # read from THD offset 8 instead of 16, mixing row 0's sub-seq 1
        # tokens into row 1's recovered output.
        assert recovered[0].tolist() == sequences[0].float().tolist()
        assert recovered[1].tolist() == sequences[1].float().tolist()

    def test_mbs2_tp1_singlesubseq_rows_match_legacy_path(self):
        """No regression in the legacy path: when each row has 1 sub-seq and
        tp_size=1, the multi-subseq postprocess output equals the legacy
        ``attention_mask``-based postprocess output.
        """
        from skyrl.backends.skyrl_train.distributed.megatron.megatron_utils import (
            postprocess_packed_seqs,
            preprocess_packed_seqs,
        )

        sequences, attention_mask, sub_seq_lengths = _build_collator_layout(
            [[[1, 2, 3, 4, 5]], [[10, 11, 12]]], align_size=1
        )

        with patch(
            "skyrl.backends.skyrl_train.distributed.megatron.megatron_utils.mpu",
            _mock_mpu(tp_size=1, cp_size=1),
        ):
            packed_multi, params_multi = preprocess_packed_seqs(
                sequences,
                attention_mask,
                pre_process=True,
                sub_seq_lengths=sub_seq_lengths,
            )
            packed_legacy, params_legacy = preprocess_packed_seqs(sequences, attention_mask, pre_process=True)

        assert torch.equal(packed_multi, packed_legacy)
        assert torch.equal(params_multi.cu_seqlens_q, params_legacy.cu_seqlens_q)

        output_multi = packed_multi.unsqueeze(-1).float()
        output_legacy = packed_legacy.unsqueeze(-1).float()

        with patch(
            "skyrl.backends.skyrl_train.distributed.megatron.megatron_utils.mpu",
            _mock_mpu(tp_size=1, cp_size=1),
        ):
            recovered_multi = postprocess_packed_seqs(
                output_multi,
                params_multi,
                attention_mask,
                batch_size=2,
                seq_len=sequences.shape[1],
                post_process=True,
                sub_seq_lengths=sub_seq_lengths,
            )
            recovered_legacy = postprocess_packed_seqs(
                output_legacy,
                params_legacy,
                attention_mask,
                batch_size=2,
                seq_len=sequences.shape[1],
                post_process=True,
            )

        # Both recoveries must put the same *valid token* values at the same
        # mask=True positions. The two paths differ in the destination layout
        # (multi-subseq path writes the whole padded row slab; legacy path
        # writes only attention_mask=True positions) but at attention_mask=True
        # positions the values must match.
        valid_multi = recovered_multi.squeeze(-1)[attention_mask]
        valid_legacy = recovered_legacy.squeeze(-1)[attention_mask]
        assert torch.equal(valid_multi, valid_legacy)

    def test_mbs2_tp4_multisubseq_loss_mask_is_zero_at_alignment_pad(self):
        """The collator zeros ``loss_mask`` at every TP-alignment pad slot
        inside the row. Verify that the recovered logits at those slots are
        irrelevant: the loss formula uses ``logits * loss_mask`` so any
        garbage at the alignment pads is masked out.

        Concretely: postprocess writes the *full padded row* slab back into
        ``output_new[i, :row_len]``, including TP-alignment pad tokens
        (which carry pad_token_id=0). The downstream loss path gathers
        logprobs by token id and then multiplies by ``loss_mask``. As long
        as ``loss_mask`` is 0 at the alignment pad positions, garbage there
        cannot contaminate the loss.

        This is a *contract* assertion against the collator + loss path,
        not a tensor-equality check.
        """
        # Build a minimal trainer instance and run its collate_batch.
        from unittest.mock import MagicMock

        from skyrl.train.config import MegatronConfig
        from skyrl.train.config.sft_config import (
            SFTConfig,
            SFTPlacementConfig,
            build_skyrl_config_for_sft,
        )
        from skyrl.train.sft_trainer_with_packing import SFTTrainerWithPacking

        # num_gpus must be divisible by TP*PP*CP. With TP=4 we need 4 GPUs.
        # dp_size = 4/(4*1*1) = 1, so all bins live on a single DP shard.
        cfg = SFTConfig(
            strategy="megatron",
            max_length=128,
            batch_size=4,
            micro_train_batch_size_per_gpu=1,
            use_sample_packing=True,
            use_minibatch_packing=True,
            packed_micro_batch_size_per_gpu=2,
            placement=SFTPlacementConfig(num_nodes=1, num_gpus_per_node=4),
            megatron_config=MegatronConfig(
                tensor_model_parallel_size=4,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                expert_model_parallel_size=1,
            ),
        )
        skyrl_cfg = build_skyrl_config_for_sft(cfg)
        trainer = SFTTrainerWithPacking(cfg, skyrl_cfg=skyrl_cfg)
        tok = MagicMock()
        tok.pad_token_id = 0
        trainer.tokenizer = tok

        # 4 examples sized to force 2 bin rows each holding 2 sub-seqs.
        examples = [
            {"input_ids": [1] * 7, "attention_mask": [1] * 7, "num_actions": 6, "loss_mask": [1] * 6},
            {"input_ids": [2] * 5, "attention_mask": [1] * 5, "num_actions": 4, "loss_mask": [1] * 4},
            {"input_ids": [3] * 3, "attention_mask": [1] * 3, "num_actions": 2, "loss_mask": [1] * 2},
            {"input_ids": [4] * 11, "attention_mask": [1] * 11, "num_actions": 10, "loss_mask": [1] * 10},
        ]
        batch = trainer.collate_batch(examples, batch_size=4)

        # The collator MUST zero loss_mask at every TP-alignment pad position
        # inside each row. We verify this directly: at any row column where
        # attention_mask=0 (i.e. an alignment pad), loss_mask must be 0.
        for r in range(batch.batch_size):
            attn = batch["attention_mask"][r]
            lm = batch["loss_mask"][r]
            row_len = lm.shape[0]
            # Within the loss_mask span (one shorter than attention_mask),
            # every position with attn=0 must have loss_mask=0.
            zero_attn_positions = (attn[:row_len] == 0).nonzero(as_tuple=True)[0]
            assert (lm[zero_attn_positions] == 0).all(), (
                f"row {r}: loss_mask is non-zero at TP-alignment pad slots "
                f"{zero_attn_positions.tolist()}: {lm[zero_attn_positions].tolist()}"
            )
