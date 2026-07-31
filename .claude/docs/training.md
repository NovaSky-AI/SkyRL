# Training

## Entrypoints

- **`skyrl/train/entrypoints/main_base.py`** — Primary training entrypoint. Handles inference server setup, training loop, weight sync.
- **`skyrl/train/entrypoints/main_generate.py`** — Generation-only entrypoint (no training).

## Running Training

```bash
# Megatron GRPO on GSM8K
uv run --isolated --extra megatron -m skyrl.train.entrypoints.main_base \
  trainer.strategy=megatron \
  trainer.policy.model.path=<model> \
  environment.env_class=gsm8k \
  ...

# Use --env-file for secrets
uv run --isolated --extra megatron --env-file .env.test -m skyrl.train.entrypoints.main_base ...
```

## Config

- Configurations are implemented as dataclass. CLI parsing is via OmegaConf.
- Pass overrides as `key=value` args on the command line. Unlike Hydra, we do not support `+` overrides for new keys
- Main config object: `SkyRLTrainConfig` in `skyrl/train/config/`.
- **The dataclasses are the single source of truth for configuration.** Document a new or changed
  field with an attribute docstring directly below it (the PEP 224 `"""..."""` style already used
  throughout `config.py`) — not in a separate docs page. The `/docs/api-ref/skyrl/config` reference
  page is generated from these docstrings by `docs/generate-api-docs.py` via griffe, which reads
  attribute docstrings but *not* `field(metadata={"help": ...})`.
- Keep the first physical line of an attribute docstring a complete sentence: the generated
  reference's summary table shows only that first line, truncating at the newline.
- If a field's guidance is topic-specific (a backend, a training mode, a pitfall with a symptom),
  put the long-form explanation on the corresponding example/tutorial/troubleshooting page and keep
  the docstring itself focused on the field.

## Example Scripts

Located in `examples/train/<task>/`:
- `examples/train/gsm8k/` — GSM8K math training
- `examples/train/text_to_sql/` — SQL training
- Each has a `run_*.sh` script with preconfigured overrides.
