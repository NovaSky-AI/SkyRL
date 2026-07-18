"""Best-effort scalar gauges published to Prometheus via ``ray.util.metrics``."""

from typing import Dict, Optional

from loguru import logger


class ScalarGauges:
    """Lazily creates a gauge per name on the first ``set``; no-ops when Ray metrics are unavailable."""

    def __init__(self) -> None:
        self._gauges: Dict[str, object] = {}
        self._enabled = True

    def set(self, name: str, value: float, description: Optional[str] = None) -> None:
        """Set gauge ``name``; ``description`` is used only when the gauge is first created."""
        if not self._enabled:
            return
        try:
            gauge = self._gauges.get(name)
            if gauge is None:
                from ray.util.metrics import Gauge

                gauge = Gauge(name, description=description or name)
                self._gauges[name] = gauge
            gauge.set(float(value))
        except Exception as e:
            logger.warning(f"ScalarGauges disabled ({e}); scalar training metrics will not be published.")
            self._enabled = False
