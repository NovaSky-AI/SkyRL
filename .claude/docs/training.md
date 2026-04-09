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

## Hydra Config

- Config is managed via Hydra (OmegaConf).
- Base config: `skyrl/train/config/ppo_base_config.yaml`.
- Pass overrides as `key=value` args on the command line.
- Main config object: `SkyRLTrainConfig` in `skyrl/train/config/`.

## Example Scripts

Located in `examples/train/<task>/`:
- `examples/train/gsm8k/` — GSM8K math training
- `examples/train/text_to_sql/` — SQL training
- Each has a `run_*.sh` script with preconfigured Hydra overrides.
