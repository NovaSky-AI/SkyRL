"""CPU tests for the driver-side MFU helpers.

uv run --isolated --extra dev --extra skyrl-train pytest tests/train/utils/test_mfu.py
"""

import pytest
import torch
from loguru import logger

from skyrl.backends.skyrl_train.training_batch import TensorList, TrainingInputBatch
from skyrl.train.config import MegatronConfig
from skyrl.train.utils.mfu import (
    MFUTracker,
    SeqlenStats,
    mfu_metrics,
    mfu_support,
    resolve_peak_tflops,
    seqlen_stats,
)


def _padded_batch(lengths, width=8, pad_size=0):
    mask = torch.zeros(len(lengths), width, dtype=torch.long)
    for i, n in enumerate(lengths):
        mask[i, width - n :] = 1  # left padding, as the RL batches use
    batch = TrainingInputBatch({"sequences": torch.zeros_like(mask), "attention_mask": mask})
    batch.metadata = {"pad_size": pad_size}
    return batch


def test_seqlen_stats_padded_rows():
    stats = seqlen_stats(_padded_batch([3, 8, 5]))
    assert stats == SeqlenStats(num_seqs=3, seqlen_sum=16, seqlen_squared_sum=9 + 64 + 25)


def test_seqlen_stats_excludes_pad_rows():
    # The last row is a copy appended to fill the DP split; it isn't real data.
    stats = seqlen_stats(_padded_batch([3, 8, 3], pad_size=1))
    assert stats == SeqlenStats(num_seqs=2, seqlen_sum=11, seqlen_squared_sum=9 + 64)


def test_seqlen_stats_packed_rows_use_documents():
    # Two packed rows of 8 tokens holding documents [2, 6] and [5]: attention runs per document.
    batch = _padded_batch([8, 5])
    batch["sub_seq_lengths"] = TensorList([torch.tensor([2, 6]), torch.tensor([5])])
    stats = seqlen_stats(batch)
    assert stats == SeqlenStats(num_seqs=3, seqlen_sum=13, seqlen_squared_sum=4 + 36 + 25)


def test_seqlen_stats_packed_rows_exclude_pad_rows():
    # The last packed row is a copy appended to fill the DP split; its documents aren't real data.
    batch = _padded_batch([8, 5, 8], pad_size=1)
    batch["sub_seq_lengths"] = TensorList([torch.tensor([2, 6]), torch.tensor([5]), torch.tensor([2, 6])])
    stats = seqlen_stats(batch)
    assert stats == SeqlenStats(num_seqs=3, seqlen_sum=13, seqlen_squared_sum=4 + 36 + 25)


def test_resolve_peak_tflops():
    assert resolve_peak_tflops("NVIDIA B200") == 2250.0
    assert resolve_peak_tflops("NVIDIA GB200") == 2500.0
    assert resolve_peak_tflops("NVIDIA H100 80GB HBM3") == 989.0
    assert resolve_peak_tflops("NVIDIA H100 PCIe") == 756.0
    assert resolve_peak_tflops("NVIDIA H100 NVL") == 835.0
    assert resolve_peak_tflops("NVIDIA A100-SXM4-80GB") == 312.0
    assert resolve_peak_tflops("NVIDIA L4") is None


@pytest.mark.parametrize(
    "strategy, fp8, lora_rank, log_flops, log_utilization, warns",
    [
        ("megatron", None, 0, True, True, False),
        ("megatron", "e4m3", 0, True, False, True),
        ("megatron", None, 16, False, False, True),
        ("fsdp", None, 0, False, False, False),
    ],
)
def test_mfu_support(strategy, fp8, lora_rank, log_flops, log_utilization, warns):
    support = mfu_support(strategy, MegatronConfig(fp8=fp8), lora_rank)
    assert (support.log_flops, support.log_utilization, support.warning is not None) == (
        log_flops,
        log_utilization,
        warns,
    )


def test_mfu_metrics():
    stats = SeqlenStats(num_seqs=4, seqlen_sum=1000, seqlen_squared_sum=250_000)
    flops = {"flops": 8e15, "hfu_flops": 1e16}
    # Two passes of 8 PFLOP over 4 s of worker compute on 8 GPUs = 0.5 PFLOP/s per GPU; the 5 s
    # driver step gives 0.4.
    metrics = mfu_metrics(
        flops,
        stats,
        compute_seconds=4.0,
        step_seconds=5.0,
        num_gpus=8,
        peak_tflops=1000.0,
        prefix="trainer",
        passes=2,
    )
    assert metrics["trainer/model_tflops_per_gpu"] == pytest.approx(500.0)
    assert metrics["trainer/flops_per_token"] == pytest.approx(8e12)
    assert metrics["trainer/mfu"] == pytest.approx(0.5)
    assert metrics["trainer/mfu_e2e"] == pytest.approx(0.4)
    assert metrics["trainer/hfu_est"] == pytest.approx(0.625)


def test_mfu_metrics_without_peak_skips_utilization():
    stats = SeqlenStats(num_seqs=1, seqlen_sum=10, seqlen_squared_sum=100)
    metrics = mfu_metrics({"flops": 1e12, "hfu_flops": 2e12}, stats, 1.0, 1.0, 1, None, "train")
    assert set(metrics) == {"train/model_tflops_per_gpu", "train/flops_per_token"}


def test_mfu_metrics_without_step_time_skips_e2e():
    stats = SeqlenStats(num_seqs=1, seqlen_sum=10, seqlen_squared_sum=100)
    metrics = mfu_metrics({"flops": 1e12}, stats, 1.0, 0.0, 1, 1000.0, "train")
    assert set(metrics) == {"train/model_tflops_per_gpu", "train/flops_per_token", "train/mfu"}


class _FakeDispatch:
    def __init__(self, flops, compute_seconds=2.0):
        self.flops = flops
        self.compute_seconds = compute_seconds
        self.calls = []
        self.pops = 0

    def get_device_name(self, model):
        return "NVIDIA H100 80GB HBM3"

    def model_flops(self, model, num_seqs, seqlen_sum, seqlen_squared_sum):
        self.calls.append((num_seqs, seqlen_sum, seqlen_squared_sum))
        return self.flops

    def pop_train_compute_seconds(self, model):
        self.pops += 1
        return self.compute_seconds


def test_tracker_fp8_logs_throughput_only():
    tracker = MFUTracker(mfu_support("megatron", MegatronConfig(fp8="e4m3"), 0), num_gpus=1, prefix="trainer")
    metrics = tracker.step_metrics(_FakeDispatch({"flops": 1e15, "hfu_flops": 1.3e15}), _padded_batch([4, 4]), 2.0)
    assert set(metrics) == {"trainer/model_tflops_per_gpu", "trainer/flops_per_token"}


def test_tracker_lora_logs_nothing():
    dispatch = _FakeDispatch({"flops": 1e15})
    tracker = MFUTracker(mfu_support("megatron", MegatronConfig(), 8), num_gpus=1, prefix="trainer")
    assert tracker.step_metrics(dispatch, _padded_batch([4]), 1.0) == {}
    assert dispatch.calls == []


def test_tracker_passes_batch_geometry():
    dispatch = _FakeDispatch({"flops": 989e12}, compute_seconds=2.0)
    tracker = MFUTracker(mfu_support("megatron", MegatronConfig(), 0), num_gpus=1, prefix="trainer")
    metrics = tracker.step_metrics(dispatch, _padded_batch([3, 5]), step_seconds=4.0)
    assert dispatch.calls == [(2, 8, 34)]
    assert dispatch.pops == 1
    assert metrics["trainer/mfu"] == pytest.approx(0.5)
    assert metrics["trainer/mfu_e2e"] == pytest.approx(0.25)


def test_tracker_without_compute_time_logs_nothing():
    # A backend that doesn't time its workers (pop returns None) gets no FLOPs metrics.
    dispatch = _FakeDispatch({"flops": 989e12}, compute_seconds=None)
    tracker = MFUTracker(mfu_support("megatron", MegatronConfig(), 0), num_gpus=1, prefix="trainer")
    assert tracker.step_metrics(dispatch, _padded_batch([3, 5]), step_seconds=4.0) == {}


def test_sft_validation_warns_for_fp8():
    from skyrl.train.config.sft_config import SFTConfig, validate_sft_cfg

    cfg = SFTConfig(
        model=SFTConfig().model,
        megatron_config=MegatronConfig(fp8="e4m3"),
    )
    cfg.placement.num_gpus_per_node = 1
    cfg.megatron_config.tensor_model_parallel_size = 1
    cfg.megatron_config.pipeline_model_parallel_size = 1
    messages = []
    handler = logger.add(lambda m: messages.append(m), level="WARNING")
    try:
        validate_sft_cfg(cfg)
    finally:
        logger.remove(handler)
    assert sum("not logged for FP8" in m for m in messages) == 1
