"""Model FLOPs utilization (MFU) for policy training.

The backend worker supplies the model FLOPs for a batch (see ``WorkerDispatch.model_flops``) and the
time it spent in ``forward_backward`` and ``optim_step`` (see
``WorkerDispatch.pop_train_compute_seconds``); this module measures the batch geometry on the driver,
resolves the GPU's peak, and turns FLOPs and time into metrics. MFU counts the model's forward +
backward only, so activation recompute lowers it; ``hfu_est`` adds the recomputed forward back in
when recompute is full.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from loguru import logger

from skyrl.backends.skyrl_train.distributed.megatron.quantization_utils import (
    is_fp8_enabled,
)

# Dense (non-sparse) BF16 tensor-core peak in TFLOP/s, matched by device-name prefix in order, so a
# specific variant ("NVIDIA H100 PCIe") precedes its family ("NVIDIA H100"). MFU is not logged for
# GPUs not listed here.
PEAK_BF16_DENSE_TFLOPS = (
    ("NVIDIA GB200", 2500.0),
    ("NVIDIA B200", 2250.0),
    ("NVIDIA H200", 989.0),
    ("NVIDIA H100 PCIe", 756.0),
    ("NVIDIA H100 NVL", 835.0),
    ("NVIDIA H100", 989.0),
    ("NVIDIA A100", 312.0),
)


@dataclass
class SeqlenStats:
    """Per-document sequence statistics of a training batch.

    Core attention scales with ``sum(s^2)`` over the documents actually attended, not with the
    padded row length, so these are taken over real documents.
    """

    num_seqs: int
    seqlen_sum: int
    seqlen_squared_sum: int


def seqlen_stats(batch: Any) -> Optional[SeqlenStats]:
    """Returns ``batch``'s per-document length statistics, or None if it has no documents.

    Uses ``sub_seq_lengths`` when the SFT packing collator produced packed rows, and the attention
    mask's per-row sums otherwise. Rows appended as padding (``metadata["pad_size"]``) are excluded,
    matching how the token-throughput metrics count tokens.
    """
    pad_size = (batch.metadata or {}).get("pad_size", 0) if hasattr(batch, "metadata") else 0
    sub_seq_lengths = batch.get("sub_seq_lengths")
    if sub_seq_lengths is not None:
        rows = list(sub_seq_lengths)
        rows = rows[: len(rows) - pad_size] if pad_size else rows
        lengths = [int(s) for row in rows for s in row.tolist()]
    else:
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            return None
        if pad_size:
            attention_mask = attention_mask[: attention_mask.shape[0] - pad_size]
        lengths = [int(s) for s in attention_mask.sum(dim=1).tolist()]
    lengths = [s for s in lengths if s > 0]
    if not lengths:
        return None
    return SeqlenStats(
        num_seqs=len(lengths),
        seqlen_sum=sum(lengths),
        seqlen_squared_sum=sum(s * s for s in lengths),
    )


def resolve_peak_tflops(device_name: str) -> Optional[float]:
    """Returns the peak dense BF16 TFLOP/s for ``device_name``, or None for an unlisted GPU."""
    for prefix, peak in PEAK_BF16_DENSE_TFLOPS:
        if device_name.startswith(prefix):
            return peak
    return None


@dataclass
class MFUSupport:
    """Describes which FLOPs metrics a run's configuration supports."""

    log_flops: bool
    """Log ``model_tflops_per_gpu`` and ``flops_per_token``."""
    log_utilization: bool
    """Also log ``mfu``, ``mfu_e2e`` and ``hfu_est``, which divide by a single precision's peak."""
    warning: Optional[str] = None
    """Why something is left out, for the config-validation warning."""


def mfu_support(strategy: str, megatron_config: Any, lora_rank: int) -> MFUSupport:
    """Returns which FLOPs metrics a run's configuration supports.

    Args:
        strategy: Training backend, e.g. ``"megatron"``.
        megatron_config: The policy's Megatron config; only ``transformer_config_kwargs`` is read.
        lora_rank: LoRA rank of the policy; 0 for full fine-tuning.

    Returns:
        The supported metrics, with a warning for the config-validation log when some are left out.
    """
    if strategy != "megatron":
        return MFUSupport(False, False)
    if lora_rank > 0:
        return MFUSupport(
            False,
            False,
            "MFU is not logged for LoRA runs: the FLOPs formula assumes full fine-tuning, which "
            "overstates LoRA's backward (the frozen base computes no weight gradients).",
        )
    if is_fp8_enabled((megatron_config.transformer_config_kwargs or {}).get("fp8")):
        return MFUSupport(
            True,
            False,
            "MFU is not logged for FP8 training: FP8 GEMMs run at the FP8 peak while attention, "
            "logits and other kernels run at the BF16 peak, so a single-peak MFU is ill-defined. "
            "model_tflops_per_gpu and flops_per_token are still logged.",
        )
    return MFUSupport(True, True)


def mfu_metrics(
    flops: Dict[str, float],
    stats: SeqlenStats,
    compute_seconds: float,
    step_seconds: float,
    num_gpus: int,
    peak_tflops: Optional[float],
    prefix: str,
    passes: int = 1,
) -> Dict[str, float]:
    """Returns the throughput and utilization metrics for one training step.

    Args:
        flops: ``{"flops": ..., "hfu_flops": ...}`` for one forward + backward pass over the batch,
            as returned by ``WorkerDispatch.model_flops``. ``hfu_flops`` is present only when
            activation recompute is full.
        stats: Geometry of the batch.
        compute_seconds: Time the slowest worker spent in ``forward_backward`` and ``optim_step``.
        step_seconds: Driver wall-clock time of the training step, including data staging and
            dispatch overhead.
        num_gpus: GPUs the work ran on.
        peak_tflops: Per-GPU peak, or None to skip the utilization ratios (unknown GPU, FP8).
        prefix: Metric namespace, e.g. ``"trainer"`` or ``"train"``.
        passes: Passes over the batch within the step (RL's ``update_epochs_per_batch``).
    """
    if compute_seconds <= 0 or num_gpus <= 0 or not flops.get("flops"):
        return {}

    def tflops_per_gpu(step_flops: float, seconds: float) -> float:
        return passes * step_flops / seconds / num_gpus / 1e12

    metrics = {
        f"{prefix}/model_tflops_per_gpu": tflops_per_gpu(flops["flops"], compute_seconds),
        f"{prefix}/flops_per_token": flops["flops"] / stats.seqlen_sum,
    }
    if peak_tflops:
        metrics[f"{prefix}/mfu"] = tflops_per_gpu(flops["flops"], compute_seconds) / peak_tflops
        if step_seconds > 0:
            metrics[f"{prefix}/mfu_e2e"] = tflops_per_gpu(flops["flops"], step_seconds) / peak_tflops
        if flops.get("hfu_flops"):
            metrics[f"{prefix}/hfu_est"] = tflops_per_gpu(flops["hfu_flops"], compute_seconds) / peak_tflops
    return metrics


class MFUTracker:
    """Computes per-step FLOPs metrics for the policy model.

    The peak is resolved from the first worker's device name on first use, since the driver may
    run without a GPU.
    """

    def __init__(self, support: MFUSupport, num_gpus: int, prefix: str):
        self.support = support
        self.num_gpus = num_gpus
        self.prefix = prefix
        self._peak_tflops: Optional[float] = None
        self._resolved = False

    def _resolve_peak(self, dispatch: Any) -> None:
        self._resolved = True
        if not self.support.log_utilization:
            return
        device_name = dispatch.get_device_name("policy")
        self._peak_tflops = resolve_peak_tflops(device_name)
        if self._peak_tflops is None:
            logger.warning(
                f"[mfu] no peak TFLOP/s known for GPU {device_name!r}; MFU is not logged. "
                "model_tflops_per_gpu is still logged."
            )
        else:
            logger.info(f"[mfu] GPU {device_name!r}, peak {self._peak_tflops:.0f} dense BF16 TFLOP/s per GPU")

    def step_metrics(self, dispatch: Any, batch: Any, step_seconds: float, passes: int = 1) -> Dict[str, float]:
        """Returns the FLOPs metrics for a step of ``passes`` forward + backward passes over ``batch``.

        Reads and resets the policy workers' compute-time counters, so call it once per step.

        Args:
            dispatch: The ``WorkerDispatch`` that ran the step.
            batch: The step's training batch.
            step_seconds: Driver wall-clock time of the step, used for ``mfu_e2e``.
            passes: Passes over the batch within the step.
        """
        if not self.support.log_flops:
            return {}
        compute_seconds = dispatch.pop_train_compute_seconds("policy")
        stats = seqlen_stats(batch)
        if stats is None or not compute_seconds:
            return {}
        if not self._resolved:
            self._resolve_peak(dispatch)
        flops = dispatch.model_flops("policy", stats.num_seqs, stats.seqlen_sum, stats.seqlen_squared_sum)
        if not flops:
            return {}
        return mfu_metrics(
            flops,
            stats,
            compute_seconds,
            step_seconds,
            self.num_gpus,
            self._peak_tflops,
            self.prefix,
            passes=passes,
        )
