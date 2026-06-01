"""Training callbacks for SkyRL trainers.

Callbacks let users inject custom behaviour at training events without
subclassing the trainer. The shape mirrors ``transformers.TrainerCallback`` so
users coming from HF / Lightning have an analogue they already know.

Every event method receives the same three arguments:

    def on_event(self, trainer, callback_input, control):
        ...

* ``trainer``         - the SFTTrainer or RayPPOTrainer. The stable callback
                        surface is ``trainer.cfg``, ``trainer.tracker``,
                        ``trainer.tokenizer``, ``trainer.dispatch``,
                        ``trainer.global_step``.
* ``callback_input``  - a :class:`CallbackInput` snapshot. Always-populated
                        fields are the loop counters; per-event fields
                        (``batch``, ``metrics``, ``logs``, ``ckpt_path``) are
                        populated only when relevant to the firing event.
* ``control``         - a mutable :class:`TrainingControl`. Callbacks set its
                        flags to request a save or an eval; the trainer honors
                        and resets them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch


@dataclass
class CallbackInput:
    """State passed to every callback event.

    The trainer rebuilds this before each event dispatch. Read-only from the
    callback's perspective. Per-event fields are ``None`` when not relevant
    to the firing event - callbacks should null-check the fields they use.
    """

    # Always populated
    global_step: int
    epoch: int
    total_steps: int
    steps_per_epoch: int

    # Step events
    batch: Optional["TrainingInputBatch"] = None

    # Step end / eval end
    metrics: Optional[Dict[str, Any]] = None

    # on_log only - the dict the trainer is about to commit. Callbacks may
    # mutate it in place to add extra fields.
    logs: Optional[Dict[str, Any]] = None

    # on_save only
    ckpt_path: Optional[str] = None


@dataclass
class TrainingControl:
    """Mutable flags callbacks can set to influence the trainer.

    The trainer reads these once per step — right after ``on_step_end`` — then
    honors and resets them. As a result, flags are only acted on for the
    *current* step when set during ``on_step_end`` (or earlier in the same
    step). Setting a flag from a later event in the same step (``on_eval_end``,
    ``on_save``, ``on_log``) takes effect on the *next* step's read, so prefer
    setting control flags from ``on_step_end``.
    """

    should_save: bool = False
    should_evaluate: bool = False

    def reset(self) -> None:
        self.should_save = False
        self.should_evaluate = False


class TrainingCallback:
    """Base class. Subclass and override the events you care about."""

    def on_train_start(self, trainer, callback_input: CallbackInput, control: TrainingControl) -> None:
        """Fires once before the training loop begins (after checkpoint resume)."""

    def on_train_end(self, trainer, callback_input: CallbackInput, control: TrainingControl) -> None:
        """Fires once after the training loop and all final saves/eval complete."""

    def on_epoch_start(self, trainer, callback_input: CallbackInput, control: TrainingControl) -> None:
        """Fires at the start of each epoch."""

    def on_epoch_end(self, trainer, callback_input: CallbackInput, control: TrainingControl) -> None:
        """Fires at the end of each epoch."""

    def on_step_start(self, trainer, callback_input: CallbackInput, control: TrainingControl) -> None:
        """Fires before each training step. ``callback_input.batch`` is populated."""

    def on_step_end(self, trainer, callback_input: CallbackInput, control: TrainingControl) -> None:
        """Fires after each training step. ``callback_input.batch`` and ``.metrics`` are populated."""

    def on_eval_start(self, trainer, callback_input: CallbackInput, control: TrainingControl) -> None:
        """Fires before an evaluation pass."""

    def on_eval_end(self, trainer, callback_input: CallbackInput, control: TrainingControl) -> None:
        """Fires after an evaluation pass. ``callback_input.metrics`` holds the eval metrics."""

    def on_save(self, trainer, callback_input: CallbackInput, control: TrainingControl) -> None:
        """Fires after a checkpoint is written. ``callback_input.ckpt_path`` is the folder path."""

    def on_log(self, trainer, callback_input: CallbackInput, control: TrainingControl) -> None:
        """Fires before metrics are committed to the tracker. Mutate ``callback_input.logs`` to add fields."""


class CallbackHandler(TrainingCallback):
    """Fan-out dispatcher. Itself a ``TrainingCallback`` (composite pattern).

    Trainers hold a single ``CallbackHandler`` and call event methods on it;
    the handler invokes each registered callback in registration order.
    """

    def __init__(self, callbacks: Optional[List[TrainingCallback]] = None):
        self.callbacks: List[TrainingCallback] = list(callbacks or [])

    def add(self, callback: TrainingCallback) -> None:
        self.callbacks.append(callback)

    def _dispatch(self, name: str, trainer, callback_input: CallbackInput, control: TrainingControl) -> None:
        for cb in self.callbacks:
            getattr(cb, name)(trainer, callback_input, control)

    def on_train_start(self, trainer, callback_input: CallbackInput, control: TrainingControl) -> None:
        self._dispatch("on_train_start", trainer, callback_input, control)

    def on_train_end(self, trainer, callback_input: CallbackInput, control: TrainingControl) -> None:
        self._dispatch("on_train_end", trainer, callback_input, control)

    def on_epoch_start(self, trainer, callback_input: CallbackInput, control: TrainingControl) -> None:
        self._dispatch("on_epoch_start", trainer, callback_input, control)

    def on_epoch_end(self, trainer, callback_input: CallbackInput, control: TrainingControl) -> None:
        self._dispatch("on_epoch_end", trainer, callback_input, control)

    def on_step_start(self, trainer, callback_input: CallbackInput, control: TrainingControl) -> None:
        self._dispatch("on_step_start", trainer, callback_input, control)

    def on_step_end(self, trainer, callback_input: CallbackInput, control: TrainingControl) -> None:
        self._dispatch("on_step_end", trainer, callback_input, control)

    def on_eval_start(self, trainer, callback_input: CallbackInput, control: TrainingControl) -> None:
        self._dispatch("on_eval_start", trainer, callback_input, control)

    def on_eval_end(self, trainer, callback_input: CallbackInput, control: TrainingControl) -> None:
        self._dispatch("on_eval_end", trainer, callback_input, control)

    def on_save(self, trainer, callback_input: CallbackInput, control: TrainingControl) -> None:
        self._dispatch("on_save", trainer, callback_input, control)

    def on_log(self, trainer, callback_input: CallbackInput, control: TrainingControl) -> None:
        self._dispatch("on_log", trainer, callback_input, control)
