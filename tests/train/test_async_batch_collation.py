"""CPU test: async batch collation (double-buffering) is byte-identical to serial.

The SFT training loop optionally collates the NEXT step's batch on a background
thread while the current step's forward/backward runs on the GPU
(``SFTConfig.async_batch_collation``). The collated-ahead batch MUST be
byte-identical to what the synchronous path would have produced for that same
step — including across epoch boundaries, where the data reshuffles in place.

Strategy: run the real ``SFTTrainer.train()`` loop twice over the same dummy
dataset/seed — once with async batch collation OFF (the baseline) and once ON —
capturing a deep snapshot of every ``batch`` handed to ``train_step``. Then
assert the two batch sequences are tensor-equal step-for-step over a run that
spans TWO epoch boundaries. The examples have distinct token ids so a stale
pre-shuffle batch would differ from the baseline and fail the comparison.

Both collator paths are covered:
  * ``DefaultCollator`` (FSDP default left-pad path).
  * ``PackedDataCollator`` (Megatron FFD packing) — constructed directly,
    CPU-only, no Megatron runtime needed since the collate is pure numpy/torch.

Also unit-tests the ``AsyncBatchCollator`` invariants (mismatch assertion,
exception propagation, single-slot guard).

Run:
  uv run --isolated --extra dev --extra fsdp pytest tests/train/test_async_batch_collation.py -v
"""

import copy
from unittest.mock import MagicMock

import pytest
import torch

from skyrl.train.config.sft_config import SFTConfig, SFTPlacementConfig
from skyrl.train.dataset.collators import DefaultCollator, PackedDataCollator
from skyrl.train.sft_trainer import SFTTrainer
from skyrl.train.utils.async_batch_collator import AsyncBatchCollator

# ---------------------------------------------------------------------------
# Helpers: dummy config, dataset, and a trainer wired for a CPU-only train()
# ---------------------------------------------------------------------------


def _build_test_sft_config(num_steps: int, batch_size: int) -> SFTConfig:
    cfg = SFTConfig()
    cfg.strategy = "fsdp"
    cfg.model.path = "unused"
    cfg.placement = SFTPlacementConfig(num_nodes=1, num_gpus_per_node=1)
    cfg.dataset_name = "unused-monkeypatched"
    cfg.dataset_split = "train"
    cfg.eval_dataset_name = ""  # no eval path
    cfg.eval_interval = 0
    cfg.eval_before_train = False
    cfg.num_steps = num_steps
    cfg.num_epochs = None
    cfg.batch_size = batch_size
    cfg.micro_train_batch_size_per_gpu = 1
    cfg.max_length = 64
    cfg.remove_microbatch_padding = False
    cfg.logger = "console"
    cfg.ckpt_path = ""  # no checkpointing
    cfg.ckpt_interval = 0
    cfg.hf_save_interval = 0
    cfg.enable_ray_gpu_monitor = False
    cfg.seed = 1234
    return cfg


def _distinct_tokenized(n: int) -> list[dict]:
    """``n`` examples with DISTINCT token ids and varying lengths.

    Distinct ids + varying lengths mean different orderings produce different
    collated batches, so a stale-shuffle collate-ahead bug cannot accidentally match
    the synchronous baseline.
    """
    examples = []
    for i in range(n):
        length = 6 + (i % 4)  # vary length 6..9 so packing/padding differs by order
        base = (i + 1) * 1000
        input_ids = [base + j for j in range(length)]
        num_actions = 1 + (i % 3)  # 1..3 response tokens
        examples.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * length,
                "num_actions": num_actions,
                "loss_mask": [1] * num_actions,
            }
        )
    return examples


def _make_trainer(cfg: SFTConfig, collator) -> SFTTrainer:
    # Pass a mock bridge config so we don't call ``build_skyrl_config_for_sft``,
    # which (via ``SkyRLTrainConfig.__post_init__``) eagerly imports the vllm
    # inference-server stack — unneeded for a CPU-only collate test.
    skyrl_cfg = MagicMock()
    trainer = SFTTrainer(cfg, skyrl_cfg=skyrl_cfg)

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    trainer.tokenizer = tokenizer
    trainer.collator = collator
    trainer.tracker = MagicMock()

    # Mock the worker dispatch: forward_backward / optim_step are the only GPU
    # touch points in train_step.
    step_output = MagicMock()
    step_output.metrics = {"loss": 0.5, "final_loss": 0.5}
    dispatch_mock = MagicMock()
    dispatch_mock.forward_backward = MagicMock(return_value=step_output)
    dispatch_mock.optim_step = MagicMock(return_value=1.0)
    dispatch_mock.dp_size = MagicMock(return_value=1)
    trainer.dispatch = dispatch_mock
    return trainer


def _snapshot_batch(batch) -> dict:
    """Deep, order-preserving snapshot of the tensor payload of a batch."""
    snap = {}
    for key in ("sequences", "attention_mask", "loss_mask"):
        if key in batch:
            snap[key] = batch[key].clone()
    # PackedDataCollator carries sub_seq_lengths (a TensorList of 1-D tensors).
    if "sub_seq_lengths" in batch:
        snap["sub_seq_lengths"] = [t.clone() for t in batch["sub_seq_lengths"]]
    return snap


def _capture_batches(trainer: SFTTrainer, monkeypatch) -> list[dict]:
    """Run train() and capture a snapshot of every batch passed to train_step."""
    captured: list[dict] = []
    real_train_step = trainer.train_step

    def _spy_train_step(batch, step):
        captured.append({"step": step, "snap": _snapshot_batch(batch)})
        return real_train_step(batch, step)

    monkeypatch.setattr(trainer, "train_step", _spy_train_step)
    monkeypatch.setattr(trainer, "load_checkpoint", lambda: 0)
    trainer.train()
    return captured


def _assert_batch_sequences_equal(serial: list[dict], collated: list[dict]):
    assert len(serial) == len(collated), f"step count differs: serial={len(serial)} collated={len(collated)}"
    for s, p in zip(serial, collated):
        assert s["step"] == p["step"], f"step mismatch: {s['step']} vs {p['step']}"
        s_snap, p_snap = s["snap"], p["snap"]
        assert s_snap.keys() == p_snap.keys(), f"step {s['step']} key mismatch: {s_snap.keys()} vs {p_snap.keys()}"
        for key in s_snap:
            sv, pv = s_snap[key], p_snap[key]
            if key == "sub_seq_lengths":
                assert len(sv) == len(pv), f"step {s['step']} sub_seq_lengths len differs"
                for a, b in zip(sv, pv):
                    assert torch.equal(a, b), f"step {s['step']} sub_seq_lengths differ"
            else:
                assert sv.shape == pv.shape, f"step {s['step']} {key} shape differs: {sv.shape} vs {pv.shape}"
                assert torch.equal(sv, pv), f"step {s['step']} {key} not byte-identical"


# ---------------------------------------------------------------------------
# Integration: collate-ahead ON == serial baseline across epoch boundaries
# ---------------------------------------------------------------------------


def _run_pair(monkeypatch, collator_factory, n_examples, batch_size, num_steps):
    """Run the loop serially then with collate-ahead; return (serial, collated) captures.

    Each run gets its OWN tokenized list (the loop shuffles in place) and its
    OWN collator so there is no cross-run state bleed.
    """
    base = _distinct_tokenized(n_examples)

    def _run(collate_ahead_on: bool):
        cfg = _build_test_sft_config(num_steps=num_steps, batch_size=batch_size)
        cfg.async_batch_collation = collate_ahead_on
        tokenized = copy.deepcopy(base)
        trainer = _make_trainer(cfg, collator_factory(cfg))
        monkeypatch.setattr(trainer, "load_dataset", lambda: tokenized)
        monkeypatch.setattr(trainer, "load_eval_dataset", lambda: None)
        return _capture_batches(trainer, monkeypatch)

    serial = _run(collate_ahead_on=False)
    collated = _run(collate_ahead_on=True)
    return serial, collated


def test_async_collation_matches_serial_default_collator(monkeypatch):
    """DefaultCollator: collated-ahead batches are byte-identical across 2 epoch boundaries.

    n_examples=6, batch_size=2 -> steps_per_epoch=3; num_steps=7 spans the
    boundaries after step 3 and step 6 (3 epochs touched).
    """
    n_examples, batch_size, num_steps = 6, 2, 7

    def factory(cfg):
        return DefaultCollator(MagicMock(pad_token_id=0), micro_train_batch_size_per_gpu=1)

    serial, collated = _run_pair(monkeypatch, factory, n_examples, batch_size, num_steps)
    assert len(serial) == num_steps, f"expected {num_steps} steps, got {len(serial)}"
    # Confirm the run actually crossed >=2 epoch boundaries (steps_per_epoch=3).
    assert num_steps // (n_examples // batch_size) >= 2
    _assert_batch_sequences_equal(serial, collated)


def test_async_collation_matches_serial_packed_collator(monkeypatch):
    """PackedDataCollator (FFD packing): collated-ahead batches byte-identical across boundaries.

    Constructed CPU-only (dp_size=1, tp/pp/cp=1) — the collate is pure
    numpy/torch and needs no Megatron runtime.
    """
    n_examples, batch_size, num_steps = 6, 2, 7

    def factory(cfg):
        return PackedDataCollator(
            tokenizer=MagicMock(pad_token_id=0),
            max_tokens_per_microbatch=64,
            tp_size=1,
            pp_size=1,
            cp_size=1,
            dp_size=1,
            batch_size=batch_size,
            micro_train_batch_size_per_gpu=1,
        )

    serial, collated = _run_pair(monkeypatch, factory, n_examples, batch_size, num_steps)
    assert len(serial) == num_steps
    # sub_seq_lengths must be present on the packed path (proves packing fired).
    assert all("sub_seq_lengths" in c["snap"] for c in collated)
    _assert_batch_sequences_equal(serial, collated)


def test_async_collation_matches_serial_uneven_wraparound(monkeypatch):
    """Wrap-around + uneven epoch: dataset not divisible by batch_size.

    n_examples=5, batch_size=2 -> the slice wraps around the dataset end inside
    an epoch and epoch boundaries land mid-list. Exercises the
    ``end_idx > len(tokenized)`` branch under collate-ahead.
    """
    n_examples, batch_size, num_steps = 5, 2, 9

    def factory(cfg):
        return DefaultCollator(MagicMock(pad_token_id=0), micro_train_batch_size_per_gpu=1)

    serial, collated = _run_pair(monkeypatch, factory, n_examples, batch_size, num_steps)
    assert len(serial) == num_steps
    _assert_batch_sequences_equal(serial, collated)


# ---------------------------------------------------------------------------
# Unit tests for the AsyncBatchCollator invariants
# ---------------------------------------------------------------------------


def test_async_collator_basic_submit_get():
    ac = AsyncBatchCollator(lambda step: f"batch-{step}")
    try:
        assert not ac.has_pending()
        ac.submit(5)
        assert ac.has_pending()
        assert ac.pending_step() == 5
        assert ac.get(5) == "batch-5"
        assert not ac.has_pending()
    finally:
        ac.shutdown()


def test_async_collator_step_mismatch_raises():
    ac = AsyncBatchCollator(lambda step: step)
    try:
        ac.submit(3)
        with pytest.raises(AssertionError, match="!= expected step"):
            ac.get(4)  # asking for a different step than was submitted
    finally:
        ac.shutdown()


def test_async_collator_double_submit_raises():
    ac = AsyncBatchCollator(lambda step: step)
    try:
        ac.submit(1)
        with pytest.raises(AssertionError, match="slot already occupied"):
            ac.submit(2)  # single-slot invariant
    finally:
        ac.shutdown()


def test_async_collator_worker_exception_propagates():
    def _boom(step):
        raise RuntimeError("producer failed")

    ac = AsyncBatchCollator(_boom)
    try:
        ac.submit(1)
        with pytest.raises(RuntimeError, match="producer failed"):
            ac.get(1)
    finally:
        ac.shutdown()


def test_async_collator_clear_drains_in_flight():
    ac = AsyncBatchCollator(lambda step: step)
    try:
        ac.submit(7)
        ac.clear()
        assert not ac.has_pending()
        assert ac.pending_step() is None
        # A fresh submit after clear works normally.
        ac.submit(8)
        assert ac.get(8) == 8
    finally:
        ac.shutdown()
