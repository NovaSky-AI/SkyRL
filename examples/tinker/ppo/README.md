# Tinker PPO Example

This example shows how to run a PPO-style RL loop against SkyRL's Tinker API server.

Here is the setup:
- base model: `Qwen/Qwen2.5-1.5B-Instruct`
- critic model: `Qwen/Qwen2.5-1.5B-Instruct`
- dataset : `$HOME/data/gsm8k/{train,validation}.parquet`
- PPO-style loop with a policy model, a critic model, GAE, checkpointing, and eval

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

## Running on Modal (single container, server + client)

`modal_run.py` runs the full example end-to-end in a single Modal container:
it preps the GSM8K parquet files, starts `run_tinker_server.sh` in the
background, waits for `http://localhost:8000` to become ready, then runs
`ppo_client.py` against it. Four GPUs are requested to match the server-side
defaults for the colocated policy/critic/vLLM layout.

Prerequisites: [install Modal](https://modal.com/docs/guide) and run
`modal setup` once. If you want W&B logging, set `WANDB_API_KEY` (and
optionally `WANDB_PROJECT` / `WANDB_ENTITY` / `WANDB_RUN_NAME` / `WANDB_TAGS`)
in your shell — they are forwarded into the container.

```bash
# Full run (from the repo root)
modal run examples/tinker/ppo/modal_run.py
```

## Notes

- This example runs with KL loss disabled. The Tinker PPO path is a clipped-ratio
  PPO loop over rollout logprobs plus a critic trained with GAE; it does not
  build or use a separate reference model.
- The example client sends critic updates with the public `ppo` loss name because
  the upstream `tinker` SDK does not accept SkyRL's internal `ppo_critic` literal.
  On the SkyRL-Train (FSDP) backend, critic-role batches are still routed onto the
  dedicated `ppo_critic_loss` path in `skyrl/backends/skyrl_train/utils/ppo_utils.py`
  via `skyrl/backends/skyrl_train_backend.py`. The launcher in
  `run_tinker_server.sh` selects `--backend fsdp`, so this works out of the box.
- The JAX backend does not support critic training through this path and rejects
  `ppo_critic`, so do not switch backends without verifying critic-loss behavior.
