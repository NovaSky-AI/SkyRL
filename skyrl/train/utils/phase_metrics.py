"""Publishes training-loop state to Prometheus via ``ray.util.metrics``: the current macro-phase
(``TrainingPhaseGauge``) and scalar loop levels such as buffer depth (``ScalarGauges``).

Ray exports these to the same Prometheus that scrapes node GPU metrics, under the ``ray_`` prefix,
so loop phase joins GPU utilization on one wall-clock axis and survives a cluster restart, e.g.

    avg(ray_node_gpus_utilization) and on() (ray_skyrl_training_phase{phase="eval"} == 1)
"""

from contextlib import contextmanager
from typing import Dict, Optional

from loguru import logger

# Macro-phases of one async training step. Names match the paired Timer keys so the wandb timing/*
# keys and the Prometheus phase label share one vocabulary. "generating" is the default between
# blocks, when the trainer is not blocking and generation proceeds in the background.
PHASES = (
    "generating",
    "wait_for_generation_buffer",
    "convert_to_training_input",
    "run_training",
    "sync_weights",
    "eval",
    "save_checkpoints",
)


class TrainingPhaseGauge:
    """Sets ``skyrl_training_phase{phase=...}`` to 1.0 for the active phase and 0.0 for the rest.

    Best-effort: silently no-ops when disabled or Ray metrics are unavailable, so it is always safe
    to construct and call.
    """

    def __init__(self, enabled: bool = True) -> None:
        self._gauge = None
        self._current = "generating"
        if not enabled:
            return
        try:
            from ray.util.metrics import Gauge

            self._gauge = Gauge(
                "skyrl_training_phase",
                description="1.0 for the active training-loop macro-phase, 0.0 otherwise.",
                tag_keys=("phase",),
            )
            # Seed every series once so PromQL selectors never hit a missing series.
            for phase in PHASES:
                self._gauge.set(1.0 if phase == self._current else 0.0, tags={"phase": phase})
        except Exception as e:
            logger.warning(f"TrainingPhaseGauge disabled ({e}); the training-phase metric will not be published.")
            self._gauge = None

    def set_phase(self, name: str) -> None:
        if name not in PHASES:
            logger.warning(f"TrainingPhaseGauge: unknown phase {name!r}; keeping {self._current!r}.")
            return
        prev, self._current = self._current, name
        if self._gauge is None or prev == name:
            return
        try:
            self._gauge.set(0.0, tags={"phase": prev})
            self._gauge.set(1.0, tags={"phase": name})
        except Exception:
            pass

    @contextmanager
    def phase(self, name: str):
        """Mark ``name`` active for the duration of the block, restoring the prior phase on exit."""
        prev = self._current
        self.set_phase(name)
        try:
            yield
        finally:
            self.set_phase(prev)


class ScalarGauges:
    """Best-effort scalar gauges published to Prometheus via ``ray.util.metrics``.

    Lazily creates a gauge per name on first ``set`` (Ray exports it as ``ray_<name>``) and updates
    it on each call. Like ``TrainingPhaseGauge`` this no-ops when Ray metrics are unavailable.
    """

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
