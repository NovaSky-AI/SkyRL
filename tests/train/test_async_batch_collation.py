"""CPU tests for async SFT batch collation.

The integration tests compare serial and collate-ahead batches across epoch
reshuffles for both default padding and packed collation. Unit tests cover the
single-slot ordering and error-propagation invariants.
"""

import copy
from unittest.mock import MagicMock

import pytest
import torch

from skyrl.train.config.sft_config import SFTConfig, SFTPlacementConfig
from skyrl.train.dataset.collators import DefaultCollator, PackedDataCollator
from skyrl.train.sft_trainer import SFTTrainer
from skyrl.train.utils.async_batch_collator import AsyncBatchCollator

# Helpers: dummy config, dataset, and a CPU-only trainer.


def _build_test_sft_config(num_steps: int, batch_size: int) -> SFTConfig:
    cfg = SFTConfig()
    cfg.strategy = "fsdp"
    cfg.model.path = "unused"
    cfg.placement = SFTPlacementConfig(num_nodes=1, num_gpus_per_node=1)
    cfg.dataset_name = "unused-monkeypatched"
    cfg.dataset_split = "train"
    cfg.eval_datasets = None  # no eval path
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
    """Examples whose order changes are visible in collated tensors."""
    examples = []
    for i in range(n):
        length = 6 + (i % 4)
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
    # Avoid importing the vLLM inference-server stack for a CPU-only collate test.
    skyrl_cfg = MagicMock()
    trainer = SFTTrainer(cfg, skyrl_cfg=skyrl_cfg)

    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    trainer.tokenizer = tokenizer
    trainer.collator = collator
    trainer.tracker = MagicMock()

    # Mock the GPU touch points in train_step.
    step_output = MagicMock()
    step_output.metrics = {"loss": 0.5, "final_loss": 0.5}
    dispatch_mock = MagicMock()
    dispatch_mock.forward_backward = MagicMock(return_value=step_output)
    dispatch_mock.optim_step = MagicMock(return_value=1.0)
    dispatch_mock.dp_size = MagicMock(return_value=1)
    trainer.dispatch = dispatch_mock
    return trainer


def _snapshot_batch(batch) -> dict:
    """Snapshot the tensor payload needed for step-by-step comparison."""
    snap = {}
    for key in ("sequences", "attention_mask", "loss_mask"):
        if key in batch:
            snap[key] = batch[key].clone()
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


# Integration: collate-ahead ON == serial baseline across epoch boundaries.


def _run_pair(monkeypatch, collator_factory, n_examples, batch_size, num_steps):
    """Run serial and collate-ahead loops with isolated mutable state."""
    base = _distinct_tokenized(n_examples)

    def _run(collate_ahead_on: bool):
        cfg = _build_test_sft_config(num_steps=num_steps, batch_size=batch_size)
        cfg.async_batch_collation = collate_ahead_on
        tokenized = copy.deepcopy(base)
        trainer = _make_trainer(cfg, collator_factory(cfg))
        monkeypatch.setattr(trainer, "load_dataset", lambda: (tokenized, [len(tokenized)]))
        monkeypatch.setattr(trainer, "load_eval_datasets", lambda: None)
        return _capture_batches(trainer, monkeypatch)

    serial = _run(collate_ahead_on=False)
    collated = _run(collate_ahead_on=True)
    return serial, collated


def test_async_collation_matches_serial_default_collator(monkeypatch):
    """DefaultCollator matches serial across two epoch boundaries."""
    n_examples, batch_size, num_steps = 6, 2, 7

    def factory(cfg):
        return DefaultCollator(MagicMock(pad_token_id=0), micro_train_batch_size_per_gpu=1)

    serial, collated = _run_pair(monkeypatch, factory, n_examples, batch_size, num_steps)
    assert len(serial) == num_steps, f"expected {num_steps} steps, got {len(serial)}"
    assert num_steps // (n_examples // batch_size) >= 2
    _assert_batch_sequences_equal(serial, collated)


def test_async_collation_matches_serial_packed_collator(monkeypatch):
    """PackedDataCollator matches serial across epoch boundaries."""
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
    assert all("sub_seq_lengths" in c["snap"] for c in collated)
    _assert_batch_sequences_equal(serial, collated)


def test_async_collation_matches_serial_with_partial_tail(monkeypatch):
    """Collate-ahead matches serial across padded partial tail batches."""
    n_examples, batch_size, num_steps = 5, 2, 9

    def factory(cfg):
        return DefaultCollator(MagicMock(pad_token_id=0), micro_train_batch_size_per_gpu=1)

    serial, collated = _run_pair(monkeypatch, factory, n_examples, batch_size, num_steps)
    assert len(serial) == num_steps
    _assert_batch_sequences_equal(serial, collated)


def test_checkpoint_state_excludes_collated_ahead_batch(monkeypatch):
    """A step-N checkpoint resumes at the already-collated step-N+1 batch."""
    cfg = _build_test_sft_config(num_steps=2, batch_size=2)
    cfg.async_batch_collation = True
    cfg.ckpt_interval = 1
    data = _distinct_tokenized(6)
    trainer = _make_trainer(cfg, DefaultCollator(MagicMock(pad_token_id=0), micro_train_batch_size_per_gpu=1))
    monkeypatch.setattr(trainer, "load_dataset", lambda: (copy.deepcopy(data), [len(data)]))
    monkeypatch.setattr(trainer, "load_eval_datasets", lambda: None)
    monkeypatch.setattr(trainer, "load_checkpoint", lambda: 0)

    batches = []
    real_train_step = trainer.train_step

    def _capture_train_step(batch, step):
        batches.append(_snapshot_batch(batch))
        return real_train_step(batch, step)

    checkpoint_states = []

    def _capture_checkpoint_state():
        checkpoint_states.append(copy.deepcopy(trainer._checkpoint_dataloader_state))
        return "unused"

    monkeypatch.setattr(trainer, "train_step", _capture_train_step)
    monkeypatch.setattr(trainer, "save_checkpoint", _capture_checkpoint_state)
    trainer.train()

    assert checkpoint_states[0] is not None
    resumed = _make_trainer(
        cfg,
        DefaultCollator(MagicMock(pad_token_id=0), micro_train_batch_size_per_gpu=1),
    )
    resumed.train_dataloader = resumed.build_train_dataloader(copy.deepcopy(data))
    resumed.train_dataloader.load_state_dict(checkpoint_states[0])
    resumed_batch = next(iter(resumed.train_dataloader))
    _assert_batch_sequences_equal(
        [{"step": 2, "snap": batches[1]}],
        [{"step": 2, "snap": _snapshot_batch(resumed_batch)}],
    )


# Unit tests for the AsyncBatchCollator invariants.


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
            ac.get(4)
    finally:
        ac.shutdown()


def test_async_collator_double_submit_raises():
    ac = AsyncBatchCollator(lambda step: step)
    try:
        ac.submit(1)
        with pytest.raises(AssertionError, match="slot already occupied"):
            ac.submit(2)
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
        ac.submit(8)
        assert ac.get(8) == 8
    finally:
        ac.shutdown()
