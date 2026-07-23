"""Scalar gauges published to Prometheus via ``ray.util.metrics``."""

from typing import Dict, Optional


class ScalarGauges:
    """Lazily creates a gauge per name on the first ``set``."""

    def __init__(self) -> None:
        self._gauges: Dict[str, object] = {}

    def set(self, name: str, value: float, description: Optional[str] = None) -> None:
        """Set gauge ``name``; ``description`` is used only when the gauge is first created."""
        gauge = self._gauges.get(name)
        if gauge is None:
            from ray.util.metrics import Gauge

            gauge = Gauge(name, description=description or name)
            self._gauges[name] = gauge
        gauge.set(float(value))
