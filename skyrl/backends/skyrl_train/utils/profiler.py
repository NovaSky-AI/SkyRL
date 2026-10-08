import os
import shutil
import tempfile

import torch
import torch.distributed
from loguru import logger

from skyrl.backends.skyrl_train.utils.io.io import is_cloud_path, upload_directory

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
        # Set when a trace upload fails; surfaced to the caller via `last_error`.
        self.last_error = None
        self.remote_dir = None
        self._temp_dir = None
        if not config.enable:
            return
        self.config = config
        # Validated at startup by TorchProfilerConfig.
        self.save_path = config.save_path
        self.ranks = list(config.ranks)
        self.export_type = getattr(config, "export_type", "chrome_trace")
        self.use_gzip = getattr(config, "use_gzip", False)
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        if self.rank not in self.ranks:
            return

        # torch.profiler writes through the C++ Kineto writer, which only takes a
        # local path. For a cloud destination we record traces into a temp dir and
        # upload each window as it closes (see _flush_remote).
        if is_cloud_path(self.save_path):
            self.remote_dir = self.save_path
            self._temp_dir = tempfile.mkdtemp(prefix="skyrl_profiler_")
            self.save_path = self._temp_dir

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
        # Deterministic per-window names: unique without a timestamp, predictable
        # for users, and they keep `stacks` windows from overwriting each other.
        stem = f"rank{self.rank}_w{self._window_count}"
        if self.export_type == "stacks":
            out = os.path.join(self.save_path, f"{stem}_stacks.txt")
            prof.export_stacks(out, "self_cuda_time_total")
            logger.info(f"[Profiler] rank {self.rank}: exported stacks -> {out}")
        else:
            name = f"{stem}.pt.trace.json" + (".gz" if self.use_gzip else "")
            prof.export_chrome_trace(os.path.join(self.save_path, name))
            logger.info(f"[Profiler] rank {self.rank}: exported chrome trace -> {name}")

        try:
            # Microseconds, self time.
            self._last_pairs = [(str(e.key), float(e.self_device_time_total)) for e in prof.key_averages()]
        except Exception as e:
            logger.warning(f"[Profiler] rank {self.rank}: kernel-summary capture failed: {e}")
        self._window_count += 1

        if self.remote_dir:
            self._flush_remote()

    def _flush_remote(self) -> None:
        """Upload closed windows to the cloud destination and drop the local copies.

        Uploads the whole staging dir rather than the one new file, so a window
        whose upload failed earlier goes out with the next one. Uploads overwrite
        by key, so retrying is idempotent. A failure never raises: it would
        otherwise surface inside prof.step() on the training path.
        """
        try:
            names = os.listdir(self.save_path)
        except OSError as e:
            self.last_error = f"trace staging dir unreadable: {e}"
            return
        if not names:
            return
        try:
            upload_directory(self.save_path, self.remote_dir)
        except Exception as e:
            # Keep the files; the next window retries them.
            self.last_error = f"trace upload failed: {e}"
            logger.warning(f"[Profiler] rank {self.rank}: {self.last_error}")
            return
        self.last_error = None
        for name in names:
            try:
                os.remove(os.path.join(self.save_path, name))
            except OSError:
                pass

    def close(self) -> None:
        """Release the staging dir. Call after stop(); safe to call twice."""
        if self._temp_dir is not None:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

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
