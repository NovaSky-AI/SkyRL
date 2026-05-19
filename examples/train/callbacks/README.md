# Training callbacks

Demonstrates the `TrainingCallback` API by adding an `EarlyStopping` callback
to the SFT trainer. The same pattern works for the RL trainer
(`RayPPOTrainer` accepts a `callbacks=` constructor arg).

## Files

- `early_stopping.py` — example callback that sets
  `control.should_training_stop` when a monitored eval metric stops improving.
- `main_sft_with_callbacks.py` — custom entrypoint that constructs
  `SFTTrainer(..., callbacks=[EarlyStopping(...)])`.
- `run_sft_with_callbacks.sh` — launcher; mirrors `examples/train/sft/run_sft_fsdp.sh`
  but runs through the custom entrypoint.

## Run

```bash
bash examples/train/callbacks/run_sft_with_callbacks.sh
```

## Writing your own callback

Subclass `TrainingCallback` and override the events you care about. Every
event receives the same three arguments: `(trainer, callback_input, control)`.

```python
import math

from skyrl.train.utils.callbacks import TrainingCallback

class LogPerplexity(TrainingCallback):
    def on_step_end(self, trainer, callback_input, control):
        loss = (callback_input.metrics or {}).get("loss")
        if loss is None:
            return
        trainer.tracker.log(
            {"train/perplexity": math.exp(min(loss, 20))},
            step=callback_input.global_step,
            commit=False,
        )
```

`callback_input` carries the loop counters plus the per-event payload that
applies (`batch` on step events, `metrics` on step/eval end, `logs` on
`on_log`, `ckpt_path` on `on_save`). Anything else — `tokenizer`, `dispatch`,
`tracker`, `cfg` — is reached through `trainer.*`.

Set `control.should_save` / `should_evaluate` / `should_training_stop` to
request a save, an eval, or a training stop; the trainer honors and resets
those flags.
