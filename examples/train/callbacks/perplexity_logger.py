"""PerplexityLogger callback — logs train perplexity alongside the trainer's
own metrics by writing into the same wandb step via ``trainer.tracker``.
"""

import math

from skyrl.train.utils.callbacks import (
    CallbackInput,
    TrainingCallback,
    TrainingControl,
)


class PerplexityLogger(TrainingCallback):
    """Log ``train/perplexity = exp(loss)`` on every step.

    Uses ``commit=False`` so the perplexity value is bundled into the same
    wandb step the trainer commits on its own. The ``min(loss, 20)`` cap
    keeps ``exp`` from overflowing on the first few unstable steps.
    """

    def on_step_end(self, trainer, callback_input: CallbackInput, control: TrainingControl) -> None:
        metrics = callback_input.metrics or {}
        loss = metrics.get("loss")
        if loss is None or math.isnan(loss):
            return
        trainer.tracker.log(
            {"train/perplexity": math.exp(min(loss, 20))},
            step=callback_input.global_step,
            commit=False,
        )
