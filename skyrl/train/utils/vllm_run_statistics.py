"""Weighted summaries of observed, non-overlapping vLLM windows."""

from skyrl.train.utils.vllm_window_statistics import counter_deltas


class RunStatistics:
    """Sum raw counters and durations, keeping sync train and eval separate."""

    def __init__(self):
        self.windows = {}
        self.seconds = {}
        self.incomplete = set()

    def add(self, scope, previous, current, duration):
        """Omit a scope with a missing/reset window; never average step rates."""
        deltas = counter_deltas(previous, current)
        if deltas is None:
            self.incomplete.add(scope)
            return
        if scope not in self.windows:
            self.windows[scope] = deltas
        else:
            total = self.windows[scope]
            for name in deltas:
                if "_bucket::" in name and total.get(name.split("_bucket::", 1)[0] + "_count") == 0:
                    total.setdefault(name, 0.0)
            # Only counters observed over every window have full-run denominators.
            self.windows[scope] = {name: value + deltas[name] for name, value in total.items() if name in deltas}
        self.seconds[scope] = self.seconds.get(scope, 0.0) + duration

    def summary(self):
        """Derive rates, request-weighted means and merged histogram P90 values."""
        from skyrl.train.utils.vllm_metrics_scraper import VLLMMetricsScraper

        result = {}
        for scope, deltas in self.windows.items():
            if scope in self.incomplete:
                continue
            prefix = f"vllm_correct_aggregate/{scope}/"
            metrics = VLLMMetricsScraper._derive(deltas, dict.fromkeys(deltas, 0), self.seconds[scope], prefix)
            result.update(
                {key: value for key, value in metrics.items() if "draft_num_" not in key and "_pos_" not in key}
            )
            result[prefix + "measurement_seconds"] = self.seconds[scope]
            for counter, public in (
                ("generation_tokens", "output_tokens_total"),
                ("prompt_tokens", "prompt_tokens_total"),
                ("num_preemptions", "preemptions_total"),
                ("kv_offload_store_bytes", "kv_offload_store_bytes_total"),
                ("kv_offload_load_bytes", "kv_offload_load_bytes_total"),
            ):
                value = deltas.get(f"ray_vllm_{counter}_total")
                if value is not None:
                    result[prefix + public] = value
        return result
