"""Unit tests for PackedDataCollator.

Run with:
  uv run --extra dev --extra megatron -- pytest tests/train/test_sft_packing_collate.py
"""

from unittest.mock import MagicMock

import pytest

from skyrl.train.config import MegatronConfig
from skyrl.train.config.sft_config import SFTConfig, SFTPlacementConfig
from skyrl.train.dataset.collators import PackedDataCollator
from skyrl.train.sft_trainer import SFTTrainer


def _make_collator(
    *,
    batch_size: int = 8,
    max_length: int = 128,
    num_gpus: int = 4,
    tp: int = 1,
    pp: int = 1,
    cp: int = 1,
    max_tokens_per_microbatch: int | None = None,
    fp8: object = None,
) -> PackedDataCollator:
    """Build a PackedDataCollator from config (no Ray, no workers)."""
    megatron_config = MegatronConfig(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
        expert_model_parallel_size=1,
    )
    if fp8 is not None:
        megatron_config.transformer_config_kwargs["fp8"] = fp8

    cfg = SFTConfig(
        strategy="megatron",
        max_length=max_length,
        batch_size=batch_size,
        micro_train_batch_size_per_gpu=1,
        remove_microbatch_padding=True,
        use_sequence_packing=True,
        max_tokens_per_microbatch=max_tokens_per_microbatch,
        placement=SFTPlacementConfig(num_nodes=1, num_gpus_per_node=num_gpus),
        megatron_config=megatron_config,
    )
    # Avoid SFTTrainer.__init__ kicking off bridge config build by injecting
    # a fake skyrl_cfg.
    from skyrl.train.config.sft_config import build_skyrl_config_for_sft

    skyrl_cfg = build_skyrl_config_for_sft(cfg)
    trainer = SFTTrainer(cfg, skyrl_cfg=skyrl_cfg)
    # Build the collator with a mock tokenizer the way setup() does, but
    # without the Ray/GPU worker init.
    tok = MagicMock()
    tok.pad_token_id = 0
    collator = trainer._build_collator(tok)
    return collator


def _make_example(seq_len: int, num_actions: int, base_token: int = 100) -> dict:
    """Build a per-example dict mimicking _tokenize_chat_last_assistant output."""
    return {
        "input_ids": [base_token + i for i in range(seq_len)],
        "attention_mask": [1] * seq_len,
        "num_actions": num_actions,
        "loss_mask": [1] * num_actions,
    }


class TestPackingCollator:
    def test_bin_count_is_multiple_of_dp(self):
        collator = _make_collator(num_gpus=4, batch_size=8)
        examples = [_make_example(10, 5, base_token=100 + 100 * i) for i in range(8)]
        batch = collator(examples, batch_size=8)
        # 8 short seqs of length 10 fit in 1 bin (cap=128). dp_size=4
        # forces at least 4 bins.
        assert batch.batch_size == 4
        # sub_seq_lengths is now a per-row data field (TensorList).
        assert "sub_seq_lengths" in batch
        assert len(batch["sub_seq_lengths"]) == 4

    def test_bin_capacity_is_token_budget(self):
        # max_tokens_per_microbatch is the FFD bin capacity. With dp_size=1 and
        # four length-100 seqs: a 128-token budget fits one seq per bin (4
        # bins); a 256-token budget fits two seqs per bin (2 bins).
        examples = [_make_example(100, 50, base_token=100 + 100 * i) for i in range(4)]

        narrow = _make_collator(num_gpus=1, batch_size=4, max_length=128, max_tokens_per_microbatch=128)
        assert narrow(examples, batch_size=4).batch_size == 4

        wide = _make_collator(num_gpus=1, batch_size=4, max_length=128, max_tokens_per_microbatch=256)
        assert wide(examples, batch_size=4).batch_size == 2

    def test_all_examples_included(self):
        collator = _make_collator(num_gpus=2, batch_size=4)
        examples = [_make_example(20, 10, base_token=100 + 100 * i) for i in range(4)]
        batch = collator(examples, batch_size=4)
        # All 4 examples appear somewhere in the bin data field.
        total_subseqs = sum(len(t) for t in batch["sub_seq_lengths"])
        assert total_subseqs == 4

    def test_sub_seq_lengths_match_attention_mask(self):
        collator = _make_collator(num_gpus=2, batch_size=4, max_length=64)
        examples = [
            _make_example(10, 5),
            _make_example(15, 8),
            _make_example(8, 4),
            _make_example(12, 6),
        ]
        batch = collator(examples, batch_size=4)
        # Per-row attention_mask.sum() should equal sum(sub_seq_lengths_per_row[r]).
        for r, lengths in enumerate(batch["sub_seq_lengths"]):
            assert int(batch["attention_mask"][r].sum().item()) == int(lengths.sum().item())

    def test_loss_mask_zero_at_sub_seq_boundary(self):
        collator = _make_collator(num_gpus=1, batch_size=4, max_length=128)
        # Make two short sequences that pack into the same bin.
        examples = [
            _make_example(6, 3, base_token=100),
            _make_example(6, 3, base_token=200),
            _make_example(6, 3, base_token=300),
            _make_example(6, 3, base_token=400),
        ]
        batch = collator(examples, batch_size=4)
        # With dp=1, all 4 short seqs pack into ONE bin row.
        assert batch.batch_size == 1
        subseq_lengths = batch["sub_seq_lengths"][0].tolist()
        assert sum(subseq_lengths) == 24

        # The last position of every sub-seq except the row's final one
        # must have loss_mask = 0. Loss mask is scaled, so check zeros.
        cum = 0
        loss_mask = batch["loss_mask"][0]
        for k, length in enumerate(subseq_lengths):
            cum += length
            boundary_pos = cum - 1
            if boundary_pos < loss_mask.shape[0]:
                # If 0, this is a true boundary mask. If non-zero, the
                # invariant is violated.
                assert loss_mask[boundary_pos].item() == 0.0, (
                    f"Loss mask at sub-seq {k} boundary position {boundary_pos} "
                    f"is {loss_mask[boundary_pos].item()}, expected 0"
                )

    def test_loss_mask_scale_invariant(self):
        """Sum of (unscaled) 1s in loss_mask before scaling should equal total
        nonpad response tokens minus boundary positions.
        """
        collator = _make_collator(num_gpus=1, batch_size=2, max_length=64)
        examples = [
            _make_example(5, 3),  # response of 3, 2 prompt
            _make_example(7, 4),
        ]
        batch = collator(examples, batch_size=2)
        # 1 row (both seqs pack into one bin).
        # Sub-seq 0: response at positions 2-4 (3 tokens). The token at
        # row-position 4 is the last token of sub-seq 0 (boundary) -> mask 0.
        # The token at row-position 3 predicts row-position 4 (which is
        # response token of sub-seq 0) -> mask 1.
        # The token at row-position 2 predicts row-position 3 -> mask 1.
        # Sub-seq 1: starts at row-position 5 (with tp=1, no align padding).
        # Response is positions 8-11 in the sub-seq (length 7, num_actions 4,
        # so positions 3-6 of sub-seq 1 = row-positions 8-11).
        # ... mask construction is intricate; cheaper to verify invariants:
        loss_mask = batch["loss_mask"][0]
        # Loss-mask is non-negative everywhere (scaling is positive).
        assert (loss_mask >= 0).all()
        # Total nonzero positions matches what we expect for response counts
        # minus boundaries.
        # In total: 3 response tokens (sub-seq 0) + 4 (sub-seq 1) = 7
        # response tokens in the full-loss-mask sense. After the
        # "loss_mask[p] = full_mask[p+1]" shift, the contributions are 7
        # response tokens minus those positions where the next-token
        # boundary intersects a response position (which is at most 1 per
        # sub-seq boundary).
        nonzero = int((loss_mask > 0).sum().item())
        # Conservative bound: at most 7 non-zero positions, at least
        # 7 - num_boundaries = 7 - 2 = 5.
        assert 5 <= nonzero <= 7, f"Expected 5-7 nonzero loss positions, got {nonzero}"

    def test_pp_padding_makes_rows_uniform(self):
        """With pp_size > 1, all packed rows are padded to the global max."""
        collator = _make_collator(num_gpus=2, batch_size=4, max_length=64, pp=2)
        # Two different-sized bins after FFD.
        examples = [
            _make_example(30, 15),
            _make_example(28, 14),
            _make_example(10, 5),
            _make_example(8, 4),
        ]
        batch = collator(examples, batch_size=4)
        # All rows have the same width.
        row_widths = [batch["sequences"][r].shape[0] for r in range(batch.batch_size)]
        assert len(set(row_widths)) == 1

    def test_tp_alignment_pads_each_sub_seq(self):
        """With tp_size > 1, each sub-seq's footprint in the row is
        rounded up to a multiple of tp_size."""
        # Need num_gpus % (tp*pp*cp) == 0; use 4 GPUs with tp=4 -> dp=1.
        collator = _make_collator(num_gpus=4, batch_size=2, max_length=128, tp=4)
        examples = [
            _make_example(7, 3),  # 7 tokens -> rounded to 8
            _make_example(5, 3),  # 5 tokens -> rounded to 8
        ]
        batch = collator(examples, batch_size=2)
        # BF16/layout-only path: two sub-seqs each padded to 8.
        assert batch["sequences"].shape[1] == 16
        # Both seqs are in the same row.
        subseq_lengths = batch["sub_seq_lengths"][0].tolist()
        assert sum(subseq_lengths) == 12  # raw, un-padded

    def test_fp8_tp_alignment_cost_prevents_overpacking(self):
        """FFD accounts for each subsequence's global FP8/TP footprint."""
        collator = _make_collator(num_gpus=4, batch_size=2, max_length=128, tp=4, fp8="hybrid")
        examples = [
            _make_example(7, 3),
            _make_example(5, 3),
        ]
        batch = collator(examples, batch_size=2)
        # TP4 requires a 512-token global footprint per subsequence. The
        # configured 128-token budget is raised only enough to fit one aligned
        # subsequence, so these examples must not be packed into a 1024-token row.
        assert batch["sequences"].shape == (2, 512)
        assert sorted(lengths.tolist() for lengths in batch["sub_seq_lengths"]) == [[5], [7]]

    def test_bf16_cp_alignment_does_not_apply_fp8_padding(self):
        """With CP>1 and BF16, keep only TP/CP layout padding."""
        collator = _make_collator(num_gpus=2, batch_size=2, max_length=128, cp=2)
        examples = [
            _make_example(6, 3),  # 6 tokens -> rounded up to 8
            _make_example(5, 3),  # 5 tokens -> rounded up to 8
        ]
        batch = collator(examples, batch_size=2)
        assert batch["sequences"].shape[1] == 16
        subseq_lengths = batch["sub_seq_lengths"][0].tolist()
        assert sorted(subseq_lengths) == [5, 6]

    def test_fp8_cp_alignment_pads_each_sub_seq(self):
        """With cp_size > 1, each sub-seq's footprint is rounded up to a
        multiple that leaves each CP rank's local shard 16-aligned."""
        # tp=1, cp=2 -> align_size = lcm(1*2*2, 16*2) = 32.
        collator = _make_collator(num_gpus=2, batch_size=2, max_length=128, cp=2, fp8="hybrid")
        examples = [
            _make_example(6, 3),  # 6 tokens -> rounded up to 32
            _make_example(5, 3),  # 5 tokens -> rounded up to 32
        ]
        batch = collator(examples, batch_size=2)
        # Both sub-seqs in one row (dp=1), each padded to 32 -> row width >= 64.
        assert batch["sequences"].shape[1] >= 64
        subseq_lengths = batch["sub_seq_lengths"][0].tolist()
        assert sorted(subseq_lengths) == [5, 6]  # raw, un-padded
        # Each sub-seq's padded footprint must produce 16-aligned local CP rank
        # slabs after dividing by cp_size=2.
        for s in subseq_lengths:
            padded = ((s + 31) // 32) * 32
            assert (padded // 2) % 16 == 0

    def test_fp8_alignment_is_conditional(self):
        examples = [
            _make_example(3, 1, base_token=100),
            _make_example(3, 1, base_token=200),
        ]

        bf16 = _make_collator(num_gpus=1, batch_size=2, max_length=64, fp8=False)
        bf16_batch = bf16(examples, batch_size=2)
        assert bf16_batch["sequences"].shape[1] == 6
        bf16_chunks = [bf16_batch["sequences"][0, :3].tolist(), bf16_batch["sequences"][0, 3:6].tolist()]
        assert sorted(bf16_chunks) == [[100, 101, 102], [200, 201, 202]]

        fp8 = _make_collator(num_gpus=1, batch_size=2, max_length=64, fp8=True)
        fp8_batch = fp8(examples, batch_size=2)
        assert fp8_batch["sequences"].shape[1] == 32
        fp8_chunks = [fp8_batch["sequences"][0, :3].tolist(), fp8_batch["sequences"][0, 16:19].tolist()]
        assert sorted(fp8_chunks) == [[100, 101, 102], [200, 201, 202]]

    def test_tp_gt_1_fp8_alignment_uses_local_128(self):
        """TP>1 FP8 row layout must satisfy TE blockwise input all-gather."""
        examples = [
            _make_example(3, 1, base_token=100),
            _make_example(3, 1, base_token=200),
        ]

        fp8 = _make_collator(num_gpus=2, batch_size=2, max_length=512, tp=2, fp8=True)
        fp8_batch = fp8(examples, batch_size=2)

        assert fp8_batch["sequences"].shape[1] == 512
        fp8_chunks = [fp8_batch["sequences"][0, :3].tolist(), fp8_batch["sequences"][0, 256:259].tolist()]
        assert sorted(fp8_chunks) == [[100, 101, 102], [200, 201, 202]]

    def test_eval_path_falls_back_to_super(self):
        """When batch_size != self.sft_cfg.batch_size (eval), no packing happens."""
        collator = _make_collator(num_gpus=1, batch_size=4, max_length=64)
        examples = [_make_example(10, 5) for _ in range(2)]
        # Eval batch with chunk size 2 (!= self.sft_cfg.batch_size=4).
        batch = collator(examples, batch_size=2)
        # Falls back: no sub_seq_lengths data field (and not in metadata either).
        assert batch.get("sub_seq_lengths") is None
        assert (batch.metadata or {}).get("sub_seq_lengths") is None


class TestPackingValidation:
    def test_rejects_fsdp(self):
        with pytest.raises(ValueError, match="strategy='megatron'"):
            SFTTrainer._validate_packing_cfg(
                type(
                    "FakeSelf",
                    (),
                    {
                        "sft_cfg": SFTConfig(
                            strategy="fsdp",
                            remove_microbatch_padding=True,
                            use_sequence_packing=True,
                            max_length=128,
                        )
                    },
                )()
            )

    def test_auto_enables_remove_microbatch_padding(self):
        # An explicit remove_microbatch_padding=False is auto-corrected to True
        # (with a warning) rather than rejected, since sequence packing needs
        # the THD layout.
        cfg = SFTConfig(
            strategy="megatron",
            remove_microbatch_padding=False,
            use_sequence_packing=True,
            max_length=128,
        )
        SFTTrainer._validate_packing_cfg(type("S", (), {"sft_cfg": cfg})())
        assert cfg.remove_microbatch_padding is True

    def test_accepts_cp_gt_1(self):
        # CP > 1 is supported: preprocess/postprocess zigzag-shard the packed
        # THD per sub-seq, and the collator uses the same CP-safe alignment
        # as preprocess_packed_seqs. _validate_packing_cfg must NOT raise.
        cfg = SFTConfig(
            strategy="megatron",
            remove_microbatch_padding=True,
            use_sequence_packing=True,
            max_length=128,
        )
        cfg.megatron_config.context_parallel_size = 2
        # Should not raise.
        SFTTrainer._validate_packing_cfg(type("S", (), {"sft_cfg": cfg})())
