import os

import torch
import torch.distributed
from loguru import logger

# Config string -> torch.profiler activity.
_ACTIVITY_MAP = {
    "cpu": torch.profiler.ProfilerActivity.CPU,
    "cuda": torch.profiler.ProfilerActivity.CUDA,
}


def build_profiler_from_policy_cfg(trainer_cfg):
    """Build the policy profiler, or return None when disabled."""
    cfg = trainer_cfg.policy.torch_profiler_config
    if not cfg.enable:
        return None
    return Profiler(cfg)


class Profiler:
    """Thin ``torch.profiler`` wrapper driven by trainer start/step/stop calls."""

    def __init__(self, config):
        self.enable = config.enable
        self.prof = None
        # Last closed-window kernel self time, exposed via get_kernel_summary().
        self._last_pairs: list = []
        self._window_count: int = 0
        if not config.enable:
            return
        self.config = config
        # Validated at startup by TorchProfilerConfig.
        self.save_path = config.save_path
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
        """Write a trace and cache the last-window kernel self-time summary."""
        os.makedirs(self.save_path, exist_ok=True)
        worker_name = f"rank{self.rank}"
        if self.export_type == "stacks":
            out = os.path.join(self.save_path, f"{worker_name}_stacks.txt")
            prof.export_stacks(out, "self_cuda_time_total")
            logger.info(f"[Profiler] rank {self.rank}: exported stacks -> {out}")
        else:
            torch.profiler.tensorboard_trace_handler(self.save_path, worker_name=worker_name)(prof)
            logger.info(f"[Profiler] rank {self.rank}: exported chrome trace under {self.save_path}")

        try:
            # Microseconds, self time.
            self._last_pairs = [(str(e.key), float(e.self_device_time_total)) for e in prof.key_averages()]
            self._window_count += 1
        except Exception as e:
            logger.warning(f"[Profiler] rank {self.rank}: kernel-summary capture failed: {e}")

    def get_kernel_summary(self):
        """Return ``{"window_count": int, "pairs": [(name, self_us), ...]}`` or None."""
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
