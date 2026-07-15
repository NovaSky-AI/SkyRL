"""Publishes the training loop's current macro-phase to Prometheus via ``ray.util.metrics``.

Ray exports metrics recorded through ``ray.util.metrics`` to the same Prometheus that scrapes
cluster-wide node metrics (GPU/CPU/disk). Emitting the training-loop phase there lets a query join
"what was the loop doing" against "how busy were the GPUs" in a single store and time base, e.g.

    avg(ray_node_gpus_utilization) by (Phase)

without correlating a separate experiment tracker by wall-clock. This is deliberately a Prometheus
gauge rather than a tracker/W&B scalar: Prometheus is the store that has cluster-wide GPU data and
survives a cluster restart, so it is the one place a post-hoc utilization breakdown can be computed.
"""

from contextlib import contextmanager

from loguru import logger

# Macro-phases of one async training step, plus the default "generating" state between blocks (the
# trainer is not blocking, so generation and staleness control proceed in the background).
PHASES = (
    "generating",
    "waiting_for_buffer",
    "converting",
    "training",
    "weight_sync",
    "eval",
    "checkpoint",
)


class TrainingPhaseGauge:
    """Sets a ``skyrl_training_phase`` gauge to 1.0 for the active phase and 0.0 for the rest.

    Exactly one phase is 1.0 at any time, so ``ray_skyrl_training_phase{phase="eval"} == 1`` marks the
    eval windows on the Prometheus timeline. Best-effort: if Ray metrics are unavailable the object
    silently no-ops, so it is always safe to construct and call (including in unit tests without Ray).
    """

    def __init__(self) -> None:
        self._gauge = None
        self._current = "generating"
        try:
            from ray.util.metrics import Gauge

            self._gauge = Gauge(
                "skyrl_training_phase",
                description="1.0 for the active training-loop macro-phase, 0.0 otherwise.",
                tag_keys=("phase",),
            )
            self._emit("generating")
        except Exception as e:
            logger.warning(f"TrainingPhaseGauge disabled ({e}); the training-phase metric will not be published.")
            self._gauge = None

    def _emit(self, active: str) -> None:
        if self._gauge is None:
            return
        try:
            for phase in PHASES:
                self._gauge.set(1.0 if phase == active else 0.0, tags={"phase": phase})
        except Exception:
            # Observability must never break training.
            pass

    def set_phase(self, name: str) -> None:
        self._current = name
        self._emit(name)

    @contextmanager
    def phase(self, name: str):
        """Mark ``name`` active for the duration of the block, restoring the prior phase on exit."""
        prev = self._current
        self.set_phase(name)
        try:
            yield
        finally:
            self.set_phase(prev)
