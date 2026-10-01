"""Internal counter and histogram statistics for a vLLM observation window."""

import math
from dataclasses import dataclass, field
from typing import Dict, Optional


def histogram_quantile(q: float, buckets: Dict[float, float]) -> Optional[float]:
    """Estimate a quantile from cumulative classic-histogram bucket counts."""
    bounds = sorted(buckets)
    if len(bounds) < 2 or bounds[-1] != math.inf or buckets[math.inf] <= 0:
        return None
    counts = [buckets[b] for b in bounds]
    if any(not math.isfinite(c) or c < 0 for c in counts):
        return None
    if any(a > b for a, b in zip(counts, counts[1:])):
        return None
    rank = q * counts[-1]
    lower, previous = 0.0, 0.0
    for i, upper in enumerate(bounds):
        count = counts[i]
        if count >= rank:
            if upper == math.inf:
                return bounds[i - 1]
            if i == 0 and upper <= 0:
                return upper
            return lower + (upper - lower) * (rank - previous) / (count - previous) if count > previous else upper
        lower, previous = upper, count
    return None


@dataclass
class WindowStatistics:
    """Keep denominators and histogram buckets out of the public scalar payload."""

    duration_seconds: float = 0.0
    deltas: Dict[str, float] = field(default_factory=dict)
    valid: bool = True

    @classmethod
    def between(cls, previous, current, duration):
        """Difference cumulative snapshots; omit missing or decreasing counters."""
        result = cls(duration_seconds=max(duration or 0.0, 0.0), valid=previous is not None and current is not None)
        if not result.valid:
            return result
        for name, value in current.items():
            if not name.endswith(("_total", "_sum", "_count")) and "::" not in name:
                continue
            baseline = previous.get(name)
            if baseline is None and "_bucket::" in name:
                count = name.split("_bucket::", 1)[0] + "_count"
                if previous.get(count) == 0:
                    baseline = 0.0
            if baseline is None:
                continue
            delta = value - baseline
            if not math.isfinite(delta) or delta < 0:
                result.valid = False
                continue
            result.deltas[name] = delta
        return result

    def latency_metrics(self, prefix):
        """Reduce merged histogram deltas to latency means and P90 estimates."""
        if not self.valid:
            return {}
        out = {}
        for exported, public in (
            ("time_to_first_token_seconds", "ttft_seconds"),
            ("request_time_per_output_token_seconds", "request_tpot_seconds"),
        ):
            base = f"ray_vllm_{exported}"
            count = self.deltas.get(base + "_count", 0)
            total = self.deltas.get(base + "_sum")
            if count > 0 and total is not None:
                out[prefix + public + "_avg"] = total / count
            buckets = {
                float(name.split("::", 1)[1]): value
                for name, value in self.deltas.items()
                if name.startswith(base + "_bucket::")
            }
            value = histogram_quantile(0.9, buckets)
            if value is not None:
                out[f"{prefix}{public}_p90"] = value
        return out
