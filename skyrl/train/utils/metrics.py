"""Publishes the training loop's current macro-phase to Prometheus via ``ray.util.metrics``.

Ray exports these to the same Prometheus that scrapes node GPU metrics, under the ``ray_`` prefix,
so loop phase joins GPU utilization on one wall-clock axis and survives a cluster restart, e.g.

    avg(ray_node_gpus_utilization) and on() (ray_skyrl_training_phase{phase="eval"} == 1)
"""

from contextlib import contextmanager
from typing import Dict

from loguru import logger

from skyrl.train.utils.utils import Timer

# Macro-phases of one async training step. Each name is the Timer key for that block, so the same
# phase shows up in wandb as timing/<name>. Generation is not a phase: it runs in the background
# across all of them, and its activity is tracked separately by the buffer-depth gauges.
PHASES = (
    "wait_for_generation_buffer",
    "convert_to_training_input",
    "run_training",
    "sync_weights",
    "eval",
    "save_checkpoints",
    "save_hf_model",
)


class TrainingPhaseGauge:
    """Sets ``skyrl_training_phase{phase=...}`` to 1.0 while the trainer is in that macro-phase.

    Exactly one phase is 1.0 at a time, or none between phases.
    """

    def __init__(self) -> None:
        from ray.util.metrics import Gauge

        self._gauge = Gauge(
            "skyrl_training_phase",
            description="1.0 while the trainer is in this macro-phase, 0.0 otherwise.",
            tag_keys=("phase",),
        )
        # Seed every series to 0.0 so PromQL selectors never hit a missing series.
        for phase in PHASES:
            self._gauge.set(0.0, tags={"phase": phase})

    @contextmanager
    def phase(self, name: str):
        """Mark ``name`` active for the duration of the block."""
        if name not in PHASES:
            logger.warning(f"TrainingPhaseGauge: unknown phase {name!r}; not publishing it.")
            yield
            return
        self._gauge.set(1.0, tags={"phase": name})
        try:
            yield
        finally:
            self._gauge.set(0.0, tags={"phase": name})

    @contextmanager
    def timed_phase(self, name: str, timings: Dict[str, float]):
        """Time the block as wandb ``timing/<name>`` and mark ``name`` the active phase for its duration."""
        with Timer(name, timings), self.phase(name):
            yield
