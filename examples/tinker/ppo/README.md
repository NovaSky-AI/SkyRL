# Tinker PPO Example

This example shows how to run a PPO-style RL loop against SkyRL's Tinker API server.

The setup mirrors [`examples/train/ppo/run_ppo.sh`](../../train/ppo/run_ppo.sh) where it makes sense:
- base model: `Qwen/Qwen2.5-1.5B-Instruct`
- dataset convention: `$HOME/data/gsm8k/{train,validation}.parquet`
- PPO-style loop with a policy model, a critic model, GAE, KL shaping, checkpointing, and eval

Two terminals are required.

## 1. Start the Tinker API server

```bash
bash examples/tinker/ppo/run_tinker_server.sh
```

The launcher already includes the PPO server-side defaults for placement,
inference-engine layout, and micro-batching. If you want to override them, pass
your own `BACKEND_CONFIG=...`, for example:

```bash
BACKEND_CONFIG='{
  "trainer.placement.policy_num_gpus_per_node": 2,
  "trainer.placement.ref_num_gpus_per_node": 2,
  "trainer.placement.critic_num_gpus_per_node": 2,
  "generator.inference_engine.num_engines": 2
}' bash examples/tinker/ppo/run_tinker_server.sh
```

That keeps execution details like placement, engine layout, and micro-batching on
the SkyRL server, while the client script keeps the PPO loop settings.

## 2. Run the PPO-style client loop

```bash
TINKER_API_KEY=tml-dummy uv run --extra tinker --with datasets --with torch \
  python examples/tinker/ppo/ppo_client.py
```

## Notes

- The client keeps the model fixed to the same default as `run_tinker_server.sh`
  so the CLI stays small.
- If your server is not on `http://localhost:8000`, pass `--base-url`.
- The actor uses the registered `ppo` (clipped-ratio) loss with
  `clip_low_threshold = 1 - eps_clip_low` and `clip_high_threshold = 1 + eps_clip_high`,
  matching SkyRL's `eps_clip_low/high = 0.2` defaults from `examples/train/ppo/run_ppo.sh`.
- KL is applied as **reward shaping** (semantically equivalent to SkyRL's
  `use_kl_in_reward=true`), not as a separate loss term. SkyRL's
  `use_kl_loss=true` path requires a backend-side KL term that the Tinker loss
  API does not currently expose, so this client implements the reward-shaping
  variant. The `KL_COEF = 1e-3` constant matches SkyRL's `kl_loss_coef` default.
- The critic (`ppo_critic`) is implemented end-to-end only on the SkyRL-Train
  (FSDP) backend (see `skyrl/backends/skyrl_train_backend.py`). The launcher
  in `run_tinker_server.sh` selects `--backend fsdp`, so this works out of the
  box. The JAX backend's `ppo_critic_loss` is a zero-loss stub and would
  silently no-op critic updates — do not switch backends without verifying this.
