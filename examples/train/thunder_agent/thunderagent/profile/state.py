"""Profile state and metrics."""

import atexit
import csv
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import IO, List, Optional


logger = logging.getLogger(__name__)

_CSV_LOCK = Lock()
_CSV_HANDLE: Optional[IO[str]] = None
_CSV_WRITER: Optional[csv.writer] = None
_CSV_PATH: Optional[Path] = None


def _close_profile_csv() -> None:
    """Close the shared profile CSV handle at process shutdown."""
    global _CSV_HANDLE, _CSV_WRITER, _CSV_PATH
    if _CSV_HANDLE is None:
        return
    try:
        _CSV_HANDLE.flush()
        _CSV_HANDLE.close()
    finally:
        _CSV_HANDLE = None
        _CSV_WRITER = None
        _CSV_PATH = None


atexit.register(_close_profile_csv)


def get_profile_csv_dir() -> Path:
    """Get profile CSV directory from config."""
    from ..config import get_config

    return Path(get_config().profile_dir)


@dataclass
class StepMetrics:
    """Timing metrics for a single step/request."""

    program_id: str
    step_id: int
    prefill_time: float = 0.0  # Time from request received to first token (seconds)
    decode_time: float = 0.0  # Time from first token to last token (seconds)
    pause_time: float = 0.0  # Time paused (seconds, 0 for now)
    tool_call_time: float = 0.0  # Time from last token to next request (seconds)
    kv_hit_rate: Optional[float] = None  # KV cache hit rate (0.0-1.0), None if unavailable
    completed_at: float = 0.0  # Unix timestamp when step completed


@dataclass
class ProfileState:
    """Profile state for a program, tracking timing across requests."""

    program_id: str = ""

    # Per-step metrics history
    step_metrics: List[StepMetrics] = field(default_factory=list)

    # Current request timing state
    request_arrive_time: Optional[float] = None  # When request arrived (before pause check)
    request_start_time: Optional[float] = None  # When request actually starts (after pause)
    first_token_time: Optional[float] = None  # When first token was received
    last_token_time: Optional[float] = None  # When last token was received
    last_request_end_time: Optional[float] = None  # When last request ended (for tool_call calc)

    # Current step metrics being built
    _current_prefill: float = 0.0
    _current_decode: float = 0.0
    _current_tool_call: float = 0.0
    _current_pause: float = 0.0

    # CSV writer
    _csv_initialized: bool = False

    def _ensure_csv(self) -> None:
        """Ensure CSV file and header exist."""
        if self._csv_initialized:
            return
        csv_dir = get_profile_csv_dir()
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / "step_profiles.csv"
        with _CSV_LOCK:
            self._ensure_shared_csv_writer(csv_path)
        self._csv_initialized = True

    def _ensure_shared_csv_writer(self, csv_path: Path) -> None:
        """Create or refresh the shared CSV writer for this process."""
        global _CSV_HANDLE, _CSV_WRITER, _CSV_PATH
        if _CSV_HANDLE is not None and _CSV_WRITER is not None and _CSV_PATH == csv_path:
            return

        _close_profile_csv()

        file_exists = csv_path.exists()
        handle = open(csv_path, "a", newline="", buffering=1)
        writer = csv.writer(handle)
        if not file_exists or csv_path.stat().st_size == 0:
            writer.writerow(
                [
                    "program_id",
                    "step_id",
                    "prefill_s",
                    "decode_s",
                    "pause_s",
                    "tool_call_s",
                    "kv_hit_rate",
                    "completed_at",
                ]
            )
            handle.flush()
        _CSV_HANDLE = handle
        _CSV_WRITER = writer
        _CSV_PATH = csv_path

    def _write_step_to_csv(self, metrics: StepMetrics) -> None:
        """Write a step's metrics to CSV."""
        self._ensure_csv()
        csv_path = get_profile_csv_dir() / "step_profiles.csv"
        with _CSV_LOCK:
            self._ensure_shared_csv_writer(csv_path)
            if _CSV_HANDLE is None or _CSV_WRITER is None:
                raise RuntimeError(f"Profile CSV writer unavailable for {csv_path}")
            kv_hit_val = "" if metrics.kv_hit_rate is None else round(metrics.kv_hit_rate, 4)
            try:
                _CSV_WRITER.writerow(
                    [
                        metrics.program_id,
                        metrics.step_id,
                        round(metrics.prefill_time, 4),
                        round(metrics.decode_time, 4),
                        round(metrics.pause_time, 4),
                        round(metrics.tool_call_time, 4),
                        kv_hit_val,
                        round(metrics.completed_at, 4),
                    ]
                )
                _CSV_HANDLE.flush()
            except OSError:
                logger.exception("Failed to write ThunderAgent step profile row to %s", csv_path)
                _close_profile_csv()
                raise

    def on_request_arrive(self) -> None:
        """Called when a new request arrives (BEFORE pause check).

        This captures the true time between requests (tool_call_time).
        """
        now = time.time()

        # Calculate tool_call_time: time from last request end to this request arriving
        if self.last_request_end_time is not None:
            self._current_tool_call = now - self.last_request_end_time
            if self._current_tool_call < 0:
                self._current_tool_call = 0.0
        else:
            self._current_tool_call = 0.0

        self.request_arrive_time = now
        self._current_pause = 0.0

    def on_request_start(self) -> None:
        """Called when request actually starts (AFTER any pause).

        This captures pause time if there was any.
        """
        now = time.time()

        # Calculate pause time: time from request arrive to request start
        if self.request_arrive_time is not None:
            self._current_pause = now - self.request_arrive_time
            if self._current_pause < 0:
                self._current_pause = 0.0
        else:
            self._current_pause = 0.0

        self.request_start_time = now
        self.first_token_time = None
        self.last_token_time = None
        self._current_prefill = 0.0
        self._current_decode = 0.0

    def on_first_token(self) -> None:
        """Called when first token is received (streaming)."""
        if self.first_token_time is None:
            self.first_token_time = time.time()
            # Calculate prefill time
            if self.request_start_time is not None:
                self._current_prefill = self.first_token_time - self.request_start_time

    def on_token(self) -> None:
        """Called on each token (to track last token time)."""
        self.last_token_time = time.time()

    def on_request_end(self, prompt_tokens: int = 0, cached_tokens: int = 0) -> None:
        """Called when request completes.

        Args:
            prompt_tokens: Number of prompt tokens from usage.prompt_tokens
            cached_tokens: Number of cached tokens from usage.prompt_tokens_details.cached_tokens
                          (0 if not available or null)
        """
        now = time.time()

        # Calculate decode time
        if self.first_token_time is not None:
            end_time = self.last_token_time if self.last_token_time else now
            self._current_decode = end_time - self.first_token_time
            if self._current_decode < 0:
                self._current_decode = 0.0

        self.last_request_end_time = now

        # Calculate KV cache hit rate: cached_tokens / prompt_tokens
        kv_hit_rate: Optional[float] = None
        if prompt_tokens > 0:
            kv_hit_rate = cached_tokens / prompt_tokens

        # Create step metrics
        step_id = len(self.step_metrics) + 1
        metrics = StepMetrics(
            program_id=self.program_id,
            step_id=step_id,
            prefill_time=self._current_prefill,
            decode_time=self._current_decode,
            pause_time=self._current_pause,
            tool_call_time=self._current_tool_call,
            kv_hit_rate=kv_hit_rate,
            completed_at=now,
        )
        self.step_metrics.append(metrics)

        # Write to CSV
        self._write_step_to_csv(metrics)

    def get_averages(self) -> dict:
        """Get average metrics across all steps (in seconds)."""
        n = len(self.step_metrics)
        if n == 0:
            return {
                "avg_prefill_s": 0.0,
                "avg_decode_s": 0.0,
                "avg_pause_s": 0.0,
                "avg_tool_call_s": 0.0,
                "avg_kv_hit_rate": None,
            }

        total_prefill = sum(m.prefill_time for m in self.step_metrics)
        total_decode = sum(m.decode_time for m in self.step_metrics)
        total_pause = sum(m.pause_time for m in self.step_metrics)
        # Tool call: N-1 for N steps (first step has no tool_call)
        total_tool_call = sum(m.tool_call_time for m in self.step_metrics[1:]) if n > 1 else 0.0

        # KV hit rate: average only over steps that have it
        kv_hit_rates = [m.kv_hit_rate for m in self.step_metrics if m.kv_hit_rate is not None]
        avg_kv_hit_rate = round(sum(kv_hit_rates) / len(kv_hit_rates), 4) if kv_hit_rates else None

        return {
            "avg_prefill_s": round(total_prefill / n, 4),
            "avg_decode_s": round(total_decode / n, 4),
            "avg_pause_s": round(total_pause / n, 4),
            "avg_tool_call_s": round(total_tool_call / max(1, n - 1), 4),
            "avg_kv_hit_rate": avg_kv_hit_rate,
        }

    def to_dict(self) -> dict:
        """Convert to dictionary for API response (only averages)."""
        avgs = self.get_averages()
        return {
            "step_count": len(self.step_metrics),
            **avgs,
        }
