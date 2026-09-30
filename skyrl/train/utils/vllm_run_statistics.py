"""Weighted run summaries from non-overlapping vLLM counter windows."""

from dataclasses import asdict

from skyrl.train.utils.vllm_window_statistics import WindowStatistics


class RunStatistics:
    """Accumulate raw denominators separately from logged step scalars."""

    def __init__(self):
        self.windows = {}
        self.active_seconds = {}
        self.incomplete_windows = {}
        self.resumed_history = False

    def add(self, scope: str, window: WindowStatistics):
        """Add an interval once; missing snapshots do not become zero events."""
        self.active_seconds[scope] = self.active_seconds.get(scope, 0.0) + window.duration_seconds
        if not window.valid:
            self.incomplete_windows[scope] = self.incomplete_windows.get(scope, 0) + 1
            return
        total = self.windows.setdefault(scope, WindowStatistics())
        total.duration_seconds += window.duration_seconds
        for name, value in window.deltas.items():
            total.deltas[name] = total.deltas.get(name, 0.0) + value

    def summary(self):
        """Return weighted scalar summaries suitable for tracker backends."""
        from skyrl.train.utils.vllm_metrics_scraper import VLLMMetricsScraper

        result = {"vllm_run/covers_resumed_history": self.resumed_history}
        for scope in self.active_seconds:
            total = self.windows.get(scope, WindowStatistics())
            prefix = f"vllm_run/{scope}/"
            metrics = VLLMMetricsScraper._derive(
                total.deltas, dict.fromkeys(total.deltas, 0), total.duration_seconds, prefix
            )
            result.update(
                {key: value for key, value in metrics.items() if "draft_num_" not in key and "_pos_" not in key}
            )
            result[prefix + "active_generation_seconds"] = self.active_seconds.get(scope, 0.0)
            result[prefix + "observed_active_seconds"] = total.duration_seconds
            result[prefix + "unobserved_active_seconds"] = max(
                self.active_seconds.get(scope, 0.0) - total.duration_seconds, 0.0
            )
            result[prefix + "incomplete_windows"] = self.incomplete_windows.get(scope, 0)
            for counter, public in (
                ("generation_tokens", "output_tokens_total"),
                ("prompt_tokens", "prompt_tokens_total"),
                ("num_preemptions", "preemptions_total"),
                ("kv_offload_store_bytes", "kv_offload_store_bytes_total"),
                ("kv_offload_load_bytes", "kv_offload_load_bytes_total"),
            ):
                value = total.deltas.get(f"ray_vllm_{counter}_total")
                if value is not None:
                    result[prefix + public] = value
        return result

    def state_dict(self):
        """Return checkpointable accumulators without live engine baselines."""
        return {
            "windows": {scope: asdict(window) for scope, window in self.windows.items()},
            "active_seconds": dict(self.active_seconds),
            "incomplete_windows": dict(self.incomplete_windows),
        }

    def load_state_dict(self, state):
        """Restore accumulated history; new engines establish new baselines."""
        self.windows = {scope: WindowStatistics(**window) for scope, window in state.get("windows", {}).items()}
        self.active_seconds = dict(state.get("active_seconds", {}))
        self.incomplete_windows = dict(state.get("incomplete_windows", {}))
        self.resumed_history = bool(self.active_seconds)
