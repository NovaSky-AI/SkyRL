# Training callbacks

Demonstrates the `TrainingCallback` API by adding a `PerplexityLogger`
callback to the SFT trainer. The same pattern works for the RL trainer
(`RayPPOTrainer` accepts a `callbacks=` constructor arg).

## Files

- `perplexity_logger.py` — example callback that logs `train/perplexity` on
  every step, piggy-backing on the trainer's own wandb step.
- `main_sft_with_callbacks.py` — custom entrypoint that constructs
  `SFTTrainer(..., callbacks=[PerplexityLogger()])`.
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
from skyrl.train.utils.callbacks import TrainingCallback

class LogGradNorm(TrainingCallback):
    def on_step_end(self, trainer, callback_input, control):
        gn = (callback_input.metrics or {}).get("grad_norm")
        if gn is None:
            return
        trainer.tracker.log(
            {"diag/grad_norm": gn},
            step=callback_input.global_step,
            commit=False,
        )
```

`callback_input` carries the loop counters plus the per-event payload that
applies (`batch` on step events, `metrics` on step/eval end, `logs` on
`on_log`, `ckpt_path` on `on_save`). Anything else — `tokenizer`, `dispatch`,
`tracker`, `cfg` — is reached through `trainer.*`.

Set `control.should_save` / `should_evaluate` to request a checkpoint save
or an eval pass at the end of the current step; the trainer honors and
resets those flags.
