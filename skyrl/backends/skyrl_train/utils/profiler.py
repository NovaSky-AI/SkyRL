import os

import torch
import torch.distributed
from loguru import logger

from skyrl.backends.skyrl_train.utils.io.io import is_cloud_path

# Map config activity strings to torch ProfilerActivity members.
_ACTIVITY_MAP = {
    "cpu": torch.profiler.ProfilerActivity.CPU,
    "cuda": torch.profiler.ProfilerActivity.CUDA,
}


def build_profiler_from_policy_cfg(trainer_cfg):
    """Construct a :class:`Profiler` from ``policy.torch_profiler_config``, or None.

    Returns ``None`` when profiling is disabled, so callers can simply assign the
    result to ``self.profiler``. The trace ``save_path`` defaults to
    ``{ckpt_path}/profiler_traces`` (mirrors how memory snapshots default under
    ``ckpt_path``).
    """
    cfg = trainer_cfg.policy.torch_profiler_config
    if not cfg.enable:
        return None
    default_save_path = os.path.join(trainer_cfg.ckpt_path, "profiler_traces")
    return Profiler(cfg, default_save_path=default_save_path)


class Profiler:
    """A configurable ``torch.profiler`` wrapper driven by the training loop.

    The trainer brackets the loop: ``start()`` once before it, ``step()`` once
    per global step, ``stop()`` once after. Which steps are actually recorded is
    decided by ``torch.profiler.schedule(skip_first, wait, warmup, active,
    repeat)`` -- this is the "profile N steps, every M, repeating K times" knob,
    so nothing about the window is hardcoded.

    Traces are written by ``torch.profiler.tensorboard_trace_handler`` (one
    ``*.pt.trace.json`` per active window, per rank), which is the
    Kineto/Holistic-Trace-Analysis-friendly format -- no manual export.

    Every method is exception-isolated: a profiler fault disables profiling for
    the rest of the run rather than crashing training.

    ``config`` fields (see :class:`skyrl.train.config.config.TorchProfilerConfig`):
        enable, ranks, save_path,
        skip_first, wait, warmup, active, repeat,
        activities, record_shapes, profile_memory, with_stack, with_flops,
        with_modules, export_type.
    """

    def __init__(self, config, default_save_path: str = None):
        self.enable = config.enable
        self.prof = None
        # Per-window self-device-time kernel summary, refreshed by on_trace_ready
        # at the close of each active window (the per-kernel attribution denominator).
        # Exposed read-only via get_kernel_summary().
        self._last_pairs: list = []
        self._window_count: int = 0
        if not config.enable:
            return
        self.config = config
        self.save_path = config.save_path or default_save_path or "./profiler_traces"
        # torch.profiler writes traces with the local filesystem only. ``save_path``
        # commonly defaults to ``{ckpt_path}/profiler_traces``, and ckpt_path can be a
        # cloud URI (s3://, gs://) -- which torch can't write to, so the trace would be
        # silently lost. Fall back to a local dir so profiling still produces output.
        if is_cloud_path(self.save_path):
            logger.warning(
                f"[Profiler] cloud save_path {self.save_path!r} is not writable by torch.profiler; "
                f"falling back to local './profiler_traces'."
            )
            self.save_path = "./profiler_traces"
        self.ranks = list(config.ranks)
        self.export_type = getattr(config, "export_type", "chrome_trace")
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if self.rank not in self.ranks:
            return

        try:
            activities = [_ACTIVITY_MAP[a.lower()] for a in getattr(config, "activities", ["cpu", "cuda"])]
            schedule = torch.profiler.schedule(
                skip_first=getattr(config, "skip_first", 0),
                wait=getattr(config, "wait", 0),
                warmup=getattr(config, "warmup", 0),
                active=getattr(config, "active", 1),
                repeat=getattr(config, "repeat", 1),
            )
            logger.info(
                f"[Profiler] init rank {self.rank}: schedule(skip_first={getattr(config, 'skip_first', 0)}, "
                f"wait={getattr(config, 'wait', 0)}, warmup={getattr(config, 'warmup', 0)}, "
                f"active={getattr(config, 'active', 1)}, repeat={getattr(config, 'repeat', 1)}) "
                f"-> traces under {self.save_path}"
            )
            self.prof = torch.profiler.profile(
                activities=activities,
                schedule=schedule,
                on_trace_ready=self._on_trace_ready,
                record_shapes=getattr(config, "record_shapes", True),
                profile_memory=getattr(config, "profile_memory", False),
                with_stack=getattr(config, "with_stack", True),
                with_flops=getattr(config, "with_flops", False),
                with_modules=getattr(config, "with_modules", False),
            )
        except Exception as e:
            logger.warning(f"[Profiler] init failed on rank {self.rank}; profiling disabled: {e}")
            self.enable = False
            self.prof = None

    def _on_trace_ready(self, prof) -> None:
        """Fires at the close of each active window. Writes the trace AND stashes
        a pickle-safe per-kernel self-device-time summary for the just-closed
        window (the per-kernel attribution denominator -- exact, no cross-stream
        overlap double-counting). ``rank{N}`` in the worker_name keeps the rank
        parseable by HTA and avoids cross-rank filename collisions in a shared
        ``save_path``. Best-effort: a fault here must never crash the worker."""
        os.makedirs(self.save_path, exist_ok=True)
        worker_name = f"rank{self.rank}"
        if self.export_type == "stacks":
            # Flamegraph-style self-CUDA-time stacks (requires with_stack=True).
            out = os.path.join(self.save_path, f"{worker_name}_stacks.txt")
            prof.export_stacks(out, "self_cuda_time_total")
            logger.info(f"[Profiler] rank {self.rank}: exported stacks -> {out}")
        else:
            torch.profiler.tensorboard_trace_handler(self.save_path, worker_name=worker_name)(prof)
            logger.info(f"[Profiler] rank {self.rank}: exported chrome trace under {self.save_path}")

        try:
            # ``self_device_time_total`` is torch 2.11's field (the older
            # ``self_cuda_time_total`` was removed). Microseconds, self time.
            self._last_pairs = [(str(e.key), float(e.self_device_time_total)) for e in prof.key_averages()]
            self._window_count += 1
        except Exception as e:
            logger.warning(f"[Profiler] rank {self.rank}: kernel-summary capture failed: {e}")

    def get_kernel_summary(self):
        """Return the last closed window's self-device-time kernel summary, or None.

        Shape (pickle-safe, no tensors)::

            {"window_count": int, "pairs": [(kernel_name, self_device_us), ...]}

        ``None`` when profiling is disabled or no profiler was constructed on
        this rank. ``pairs`` is empty until the first active window closes.

        NOTE: SkyRL's own trainers never read this -- the high-level
        ``*.pt.trace.json`` files are the deliverable. This (and the
        ``_on_trace_ready`` capture that feeds it) is a deliberately-provided
        low-level API for downstream consumers that want per-kernel self-time
        attribution without re-parsing the trace. Reached via
        ``Worker.dump_profiler_summary`` -> ``WorkerDispatch.dump_profiler_summary``.
        """
        if not self.enable or self.prof is None:
            return None
        return {"window_count": self._window_count, "pairs": list(self._last_pairs)}

    def check(self) -> bool:
        return self.prof is not None and self.enable

    def _disable(self, where: str, err: Exception) -> None:
        logger.warning(f"[Profiler] {where} failed on rank {getattr(self, 'rank', '?')}; profiling disabled: {err}")
        self.enable = False
        self.prof = None

    def start(self) -> None:
        if self.check():
            try:
                logger.info(f"[Profiler] started for rank {self.rank}")
                self.prof.start()
            except Exception as e:
                self._disable("start", e)

    def step(self) -> None:
        if self.check():
            try:
                self.prof.step()
            except Exception as e:
                self._disable("step", e)

    def stop(self) -> None:
        if self.check():
            try:
                logger.info(f"[Profiler] stopped for rank {self.rank}")
                self.prof.stop()
            except Exception as e:
                self._disable("stop", e)


class CudaTimer:
    def __init__(self, device):
        self.device = device

        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start_event.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_event.record()
        torch.cuda.synchronize(self.device)
        self.elapsed_time = self.start_event.elapsed_time(self.end_event)  # Calculate the elapsed time
