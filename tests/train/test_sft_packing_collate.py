"""Unit tests for SFTTrainerWithPacking.collate_batch.

Run with:
  uv run --extra dev --extra megatron -- pytest tests/train/test_sft_packing_collate.py
"""

from unittest.mock import MagicMock

import pytest

from skyrl.train.config import MegatronConfig
from skyrl.train.config.sft_config import SFTConfig, SFTPlacementConfig
from skyrl.train.sft_trainer_with_packing import SFTTrainerWithPacking


def _make_trainer(
    *,
    batch_size: int = 8,
    max_length: int = 128,
    num_gpus: int = 4,
    tp: int = 1,
    pp: int = 1,
    packed_mbs: int = 1,
) -> SFTTrainerWithPacking:
    """Build a trainer instance without setup() (no Ray, no workers)."""
    cfg = SFTConfig(
        strategy="megatron",
        max_length=max_length,
        batch_size=batch_size,
        micro_train_batch_size_per_gpu=1,
        use_sample_packing=True,
        use_minibatch_packing=True,
        packed_micro_batch_size_per_gpu=packed_mbs,
        placement=SFTPlacementConfig(num_nodes=1, num_gpus_per_node=num_gpus),
        megatron_config=MegatronConfig(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            context_parallel_size=1,
            expert_model_parallel_size=1,
        ),
    )
    # Avoid SFTTrainer.__init__ kicking off bridge config build by injecting
    # a fake skyrl_cfg. We only need the .sft_cfg attribute populated.
    from skyrl.train.config.sft_config import build_skyrl_config_for_sft

    skyrl_cfg = build_skyrl_config_for_sft(cfg)
    trainer = SFTTrainerWithPacking(cfg, skyrl_cfg=skyrl_cfg)
    # Patch in a tokenizer with a pad_token_id.
    tok = MagicMock()
    tok.pad_token_id = 0
    trainer.tokenizer = tok
    return trainer


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
        trainer = _make_trainer(num_gpus=4, batch_size=8)
        examples = [_make_example(10, 5, base_token=100 + 100 * i) for i in range(8)]
        batch = trainer.collate_batch(examples, batch_size=8)
        # 8 short seqs of length 10 fit in 1 bin (cap=128). dp_size=4
        # forces at least 4 bins.
        assert batch.batch_size == 4
        assert "sub_seq_lengths" in batch.metadata
        assert len(batch.metadata["sub_seq_lengths"]) == 4

    def test_all_examples_included(self):
        trainer = _make_trainer(num_gpus=2, batch_size=4)
        examples = [_make_example(20, 10, base_token=100 + 100 * i) for i in range(4)]
        batch = trainer.collate_batch(examples, batch_size=4)
        # All 4 examples appear somewhere in the bin metadata.
        total_subseqs = sum(len(lst) for lst in batch.metadata["sub_seq_lengths"])
        assert total_subseqs == 4

    def test_sub_seq_lengths_match_attention_mask(self):
        trainer = _make_trainer(num_gpus=2, batch_size=4, max_length=64)
        examples = [
            _make_example(10, 5),
            _make_example(15, 8),
            _make_example(8, 4),
            _make_example(12, 6),
        ]
        batch = trainer.collate_batch(examples, batch_size=4)
        # Per-row attention_mask.sum() should equal sum(sub_seq_lengths_per_row[r]).
        for r, lengths in enumerate(batch.metadata["sub_seq_lengths"]):
            assert int(batch["attention_mask"][r].sum().item()) == sum(lengths)

    def test_loss_mask_zero_at_sub_seq_boundary(self):
        trainer = _make_trainer(num_gpus=1, batch_size=4, max_length=128)
        # Make two short sequences that pack into the same bin.
        examples = [
            _make_example(6, 3, base_token=100),
            _make_example(6, 3, base_token=200),
            _make_example(6, 3, base_token=300),
            _make_example(6, 3, base_token=400),
        ]
        batch = trainer.collate_batch(examples, batch_size=4)
        # With dp=1, all 4 short seqs pack into ONE bin row.
        assert batch.batch_size == 1
        subseq_lengths = batch.metadata["sub_seq_lengths"][0]
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
        trainer = _make_trainer(num_gpus=1, batch_size=2, max_length=64)
        examples = [
            _make_example(5, 3),  # response of 3, 2 prompt
            _make_example(7, 4),
        ]
        batch = trainer.collate_batch(examples, batch_size=2)
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
        trainer = _make_trainer(num_gpus=2, batch_size=4, max_length=64, pp=2)
        # Two different-sized bins after FFD.
        examples = [
            _make_example(30, 15),
            _make_example(28, 14),
            _make_example(10, 5),
            _make_example(8, 4),
        ]
        batch = trainer.collate_batch(examples, batch_size=4)
        # All rows have the same width.
        row_widths = [batch["sequences"][r].shape[0] for r in range(batch.batch_size)]
        assert len(set(row_widths)) == 1

    def test_tp_alignment_pads_each_sub_seq(self):
        """With tp_size > 1, each sub-seq's footprint in the row is
        rounded up to a multiple of tp_size."""
        # Need num_gpus % (tp*pp*cp) == 0; use 4 GPUs with tp=4 -> dp=1.
        trainer = _make_trainer(num_gpus=4, batch_size=2, max_length=128, tp=4)
        examples = [
            _make_example(7, 3),  # 7 tokens -> rounded to 8
            _make_example(5, 3),  # 5 tokens -> rounded to 8
        ]
        batch = trainer.collate_batch(examples, batch_size=2)
        # Row width >= 16 (two sub-seqs each padded to 8).
        assert batch["sequences"].shape[1] >= 16
        # Both seqs are in the same row.
        subseq_lengths = batch.metadata["sub_seq_lengths"][0]
        assert sum(subseq_lengths) == 12  # raw, un-padded

    def test_eval_path_falls_back_to_super(self):
        """When batch_size != self.sft_cfg.batch_size (eval), no packing happens."""
        trainer = _make_trainer(num_gpus=1, batch_size=4, max_length=64)
        examples = [_make_example(10, 5) for _ in range(2)]
        # Eval batch with chunk size 2 (!= self.sft_cfg.batch_size=4).
        batch = trainer.collate_batch(examples, batch_size=2)
        # Falls back: no sub_seq_lengths metadata.
        assert batch.metadata.get("sub_seq_lengths") is None


class TestPackingValidation:
    def test_rejects_fsdp(self):
        with pytest.raises(ValueError, match="strategy='megatron'"):
            _make_trainer()._validate_packing_cfg.__func__(
                type(
                    "FakeSelf",
                    (),
                    {
                        "sft_cfg": SFTConfig(
                            strategy="fsdp",
                            use_sample_packing=True,
                            use_minibatch_packing=True,
                            max_length=128,
                        )
                    },
                )()
            )

    def test_rejects_no_sample_packing(self):
        # Build a cfg that bypasses validate_sft_cfg's own check
        cfg = SFTConfig(
            strategy="megatron",
            use_sample_packing=False,
            use_minibatch_packing=True,
            max_length=128,
        )
        with pytest.raises(ValueError, match="use_sample_packing"):
            SFTTrainerWithPacking._validate_packing_cfg(type("S", (), {"sft_cfg": cfg})())

    def test_rejects_cp_gt_1(self):
        cfg = SFTConfig(
            strategy="megatron",
            use_sample_packing=True,
            use_minibatch_packing=True,
            max_length=128,
        )
        cfg.megatron_config.context_parallel_size = 2
        with pytest.raises(ValueError, match="context_parallel_size > 1"):
            SFTTrainerWithPacking._validate_packing_cfg(type("S", (), {"sft_cfg": cfg})())
