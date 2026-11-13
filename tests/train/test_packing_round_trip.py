"""Collator -> worker layout tests for packed SFT.

The controller-side collator writes each sub-sequence of a bin row at its aligned row offset; the
worker's ``TokenLayout`` reads them back from those offsets into the packed THD stream.

Run with:
  uv run --isolated --extra dev -- pytest tests/train/test_packing_round_trip.py
"""

from unittest.mock import MagicMock

import pytest

from skyrl.backends.skyrl_train.distributed.megatron.token_metadata import TokenLayout
from skyrl.train.dataset.collators import PackedDataCollator


class TestCollatedPackedRows:
    """Collator output -> packed layout checks."""

    @pytest.mark.parametrize("tp_size,cp_size", [(4, 1), (1, 2), (2, 2)])
    def test_every_collated_sub_sequence_is_one_packed_segment(self, tp_size, cp_size):
        # No length is an alignment multiple, so a misread offset shifts tokens across a segment seam.
        lengths = [7, 5, 3, 11]
        examples = [
            {
                "input_ids": list(range(100 * (k + 1), 100 * (k + 1) + n)),
                "attention_mask": [1] * n,
                "num_actions": n - 1,
                "loss_mask": [1] * (n - 1),
            }
            for k, n in enumerate(lengths)
        ]
        collator = PackedDataCollator(
            MagicMock(pad_token_id=0),
            max_tokens_per_microbatch=16,
            tp_size=tp_size,
            pp_size=1,
            cp_size=cp_size,
            dp_size=1,
            batch_size=len(examples),
            micro_train_batch_size_per_gpu=1,
        )
        batch = collator(examples, batch_size=len(examples))
        sub_seq_lengths = [t.tolist() for t in batch["sub_seq_lengths"]]
        assert max(len(row) for row in sub_seq_lengths) > 1

        layout = TokenLayout.build(
            batch["attention_mask"], packed=True, tp_size=tp_size, cp_size=cp_size, sub_seq_lengths=sub_seq_lengths
        ).full()
        stream = layout.place(batch["sequences"], 0).squeeze(0)
        cu = layout.cu_seqlens_padded.tolist()
        segments = [stream[start:end] for start, end in zip(cu[:-1], cu[1:])]

        # Each segment is one example's tokens, then only padding.
        assert sorted(segment[segment != 0].tolist() for segment in segments) == [e["input_ids"] for e in examples]
        for segment in segments:
            assert not segment[int(segment.count_nonzero()) :].any()

    def test_mbs2_tp4_multisubseq_loss_mask_is_zero_at_alignment_pad(self):
        """The collator zeros ``loss_mask`` at every TP-alignment pad slot
        inside the row. The packed-logprob scatter leaves alignment pads at
        zero, and the loss formula masks those positions out.

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
        from skyrl.train.sft_trainer import SFTTrainer

        # num_gpus must be divisible by TP*PP*CP. With TP=4 we need 4 GPUs.
        # dp_size = 4/(4*1*1) = 1, so all bins live on a single DP shard.
        cfg = SFTConfig(
            strategy="megatron",
            max_length=128,
            batch_size=4,
            micro_train_batch_size_per_gpu=1,
            remove_microbatch_padding=True,
            use_sequence_packing=True,
            max_tokens_per_microbatch=256,
            placement=SFTPlacementConfig(num_nodes=1, num_gpus_per_node=4),
            megatron_config=MegatronConfig(
                tensor_model_parallel_size=4,
                pipeline_model_parallel_size=1,
                context_parallel_size=1,
                expert_model_parallel_size=1,
            ),
        )
        skyrl_cfg = build_skyrl_config_for_sft(cfg)
        trainer = SFTTrainer(cfg, skyrl_cfg=skyrl_cfg)
        tok = MagicMock()
        tok.pad_token_id = 0
        trainer.collator = trainer._build_collator(tok)

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
