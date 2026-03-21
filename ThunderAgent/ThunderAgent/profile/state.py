"""Profile state and metrics."""
import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


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
    decode_time: float = 0.0   # Time from first token to last token (seconds)
    pause_time: float = 0.0    # Time paused (seconds, 0 for now)
    tool_call_time: float = 0.0  # Time from last token to next request (seconds)
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
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
    request_start_time: Optional[float] = None   # When request actually starts (after pause)
    first_token_time: Optional[float] = None     # When first token was received
    last_token_time: Optional[float] = None      # When last token was received
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
        # Write header if file doesn't exist
        if not csv_path.exists():
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "program_id",
                    "step_id",
                    "prefill_s",
                    "decode_s",
                    "pause_s",
                    "tool_call_s",
                    "prompt_tokens",
                    "completion_tokens",
                    "cached_tokens",
                    "kv_hit_rate",
                    "completed_at",
                ])
        self._csv_initialized = True
    
    def _write_step_to_csv(self, metrics: StepMetrics) -> None:
        """Write a step's metrics to CSV."""
        self._ensure_csv()
        csv_path = get_profile_csv_dir() / "step_profiles.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            # kv_hit_rate: write empty string if None, otherwise round to 4 decimals
            kv_hit_val = "" if metrics.kv_hit_rate is None else round(metrics.kv_hit_rate, 4)
            prompt_tokens_val = "" if metrics.prompt_tokens is None else metrics.prompt_tokens
            completion_tokens_val = "" if metrics.completion_tokens is None else metrics.completion_tokens
            cached_tokens_val = "" if metrics.cached_tokens is None else metrics.cached_tokens
            writer.writerow([
                metrics.program_id,
                metrics.step_id,
                round(metrics.prefill_time, 4),
                round(metrics.decode_time, 4),
                round(metrics.pause_time, 4),
                round(metrics.tool_call_time, 4),
                prompt_tokens_val,
                completion_tokens_val,
                cached_tokens_val,
                kv_hit_val,
                round(metrics.completed_at, 4),
            ])
    
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
    
    def on_request_end(
        self,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        cached_tokens: Optional[int] = None,
    ) -> None:
        """Called when request completes.
        
        Args:
            prompt_tokens: Number of prompt tokens from usage.prompt_tokens
            completion_tokens: Number of completion tokens from usage.completion_tokens
            cached_tokens: Number of cached tokens from usage.prompt_tokens_details.cached_tokens
                          (None if not available or null)
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
        if prompt_tokens is not None and cached_tokens is not None and prompt_tokens > 0:
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
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
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
                "avg_prompt_tokens": None,
                "avg_completion_tokens": None,
                "avg_cached_tokens": None,
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
        prompt_tokens = [m.prompt_tokens for m in self.step_metrics if m.prompt_tokens is not None]
        completion_tokens = [m.completion_tokens for m in self.step_metrics if m.completion_tokens is not None]
        cached_tokens = [m.cached_tokens for m in self.step_metrics if m.cached_tokens is not None]
        avg_prompt_tokens = round(sum(prompt_tokens) / len(prompt_tokens), 2) if prompt_tokens else None
        avg_completion_tokens = (
            round(sum(completion_tokens) / len(completion_tokens), 2) if completion_tokens else None
        )
        avg_cached_tokens = round(sum(cached_tokens) / len(cached_tokens), 2) if cached_tokens else None
        
        return {
            "avg_prefill_s": round(total_prefill / n, 4),
            "avg_decode_s": round(total_decode / n, 4),
            "avg_pause_s": round(total_pause / n, 4),
            "avg_tool_call_s": round(total_tool_call / max(1, n - 1), 4),
            "avg_prompt_tokens": avg_prompt_tokens,
            "avg_completion_tokens": avg_completion_tokens,
            "avg_cached_tokens": avg_cached_tokens,
            "avg_kv_hit_rate": avg_kv_hit_rate,
        }
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API response (only averages)."""
        avgs = self.get_averages()
        return {
            "step_count": len(self.step_metrics),
            **avgs,
        }
