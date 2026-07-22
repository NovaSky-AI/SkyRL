# Faster Rollouts with MTP Speculative Decoding

This example shows how you can train a model's native **Multi-Token Prediction (MTP)** head with a decoupled draft loss, and reuse that head for **vLLM speculative decoding** to speed up rollout. The draft loss is autograd-decoupled — it trains only the MTP head and never perturbs the policy, so the RL dynamics are unchanged.

Requires the `megatron` backend and the `vllm` inference engine, plus an MTP-capable model.

`trainer.mtp` is the single high-level knob; it propagates to both the Megatron MTP heads + draft loss and vLLM's `speculative_config`:

```bash
trainer.mtp.enabled=true \
trainer.mtp.num_speculative_tokens=3 \      # vLLM draft depth
trainer.mtp.loss_weight=0.2 \               # weight of the draft loss
trainer.policy.megatron_config.mtp_loss_topk=256   # top-k soft CE (memory-efficient)
```

Three recipes are provided, all on a single 8×H100 node:

- **`run_megatron_dapo_mimo_7b_rl_specdecode.sh`** — DAPO on `dapo-math-17k` with `XiaomiMiMo/MiMo-7B-RL`.
- **`run_megatron_dapo_qwen3.5_9b_specdecode.sh`** — DAPO on `dapo-math-17k` with `Qwen/Qwen3.5-9B`.
- **`run_search_megatron_mimo_7b_specdecode.sh`** — multi-turn GRPO SearchR1 with `XiaomiMiMo/MiMo-7B-RL`.

See [the docs](https://docs.skyrl.ai/docs/examples/spec_decode) for the full walkthrough, configuration reference, and results (~2x rollout speedup with no effect on policy training).

## Quick start

```bash
# DAPO recipes
bash examples/train/algorithms/dapo/prepare_dapo_data.sh
export WANDB_API_KEY=<your_key_here>
bash examples/train/spec_decode/run_megatron_dapo_mimo_7b_rl_specdecode.sh

# SearchR1 recipe requires extra setup (dataset + retrieval server) —
# see examples/train/search/README.md, then:
bash examples/train/spec_decode/run_search_megatron_mimo_7b_specdecode.sh
```

## Monitoring acceptance

Acceptance determines your speedup. It's logged under `vllm/train/` — most importantly `draft_mean_acceptance_length` (tokens per target forward pass; this *is* the speedup factor) and the per-depth breakdown `draft_acceptance_rate_pos_{1..k}`. The draft loss is logged as `mtp_loss` on the training side. See [the docs](https://docs.skyrl.ai/docs/examples/spec_decode#monitoring-acceptance) for the full list.
