"""EarlyStopping callback for SkyRL SFT / RL trainers."""

from skyrl.train.utils.callbacks import (
    CallbackInput,
    TrainingCallback,
    TrainingControl,
)


class EarlyStopping(TrainingCallback):
    """Stop training when a monitored eval metric stops improving.

    Args:
        monitor: Key into ``callback_input.metrics`` to watch on ``on_eval_end``.
            Defaults to ``"eval_loss"``.
        patience: Number of consecutive non-improving eval rounds to tolerate
            before requesting a stop.
        min_delta: Minimum absolute improvement to count as progress.
        mode: ``"min"`` if lower is better (loss), ``"max"`` if higher is
            better (accuracy).
    """

    def __init__(
        self,
        monitor: str = "eval_loss",
        patience: int = 3,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        if mode not in {"min", "max"}:
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self._best: float = float("inf") if mode == "min" else float("-inf")
        self._stale: int = 0

    def _is_better(self, value: float) -> bool:
        if self.mode == "min":
            return value + self.min_delta < self._best
        return value - self.min_delta > self._best

    def on_eval_end(self, trainer, callback_input: CallbackInput, control: TrainingControl) -> None:
        metrics = callback_input.metrics or {}
        value = metrics.get(self.monitor)
        if value is None:
            return
        if self._is_better(float(value)):
            self._best = float(value)
            self._stale = 0
        else:
            self._stale += 1
            if self._stale >= self.patience:
                control.should_training_stop = True
