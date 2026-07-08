# DPPO through the Tinker API (Megatron backend)

Trains Qwen3.6-35B-A3B with LoRA on DAPO math data using the
[DPPO](https://arxiv.org/abs/2602.04879) policy loss, driven entirely through
SkyRL's Tinker API server backed by the Megatron strategy. This is the
Tinker-client counterpart of
`examples/train/megatron/run_megatron_dapo_qwen3.6_35b_a3b_lora.sh` with the
policy loss swapped from dual-clip PPO to DPPO, scaled to a single 8-GPU node
by default.

## Prerequisites

```bash
bash examples/train/algorithms/dapo/prepare_dapo_data.sh   # writes ~/data/dapo
```

## Run

```bash
# Terminal 1: Tinker API server (Megatron TP=4/EP=8, colocated vLLM TP=8, LoRA serving)
bash examples/tinker/dppo/run_tinker_server_megatron.sh

# Terminal 2: DPPO client
TINKER_API_KEY=tml-dummy uv run --isolated --extra tinker \
    --with torch --with ./skyrl-gym \
    python examples/tinker/dppo/dppo_client.py
```

For a quick smoke test, shrink the run:

```bash
TINKER_API_KEY=tml-dummy uv run --isolated --extra tinker \
    --with torch --with ./skyrl-gym \
    python examples/tinker/dppo/dppo_client.py \
    --max-train-steps 2 --train-batch-size 8 --n-samples-per-prompt 4 \
    --policy-mini-batch-size 8 --max-generate-length 1024 \
    --overlong-buffer-len 512 --eval-interval 0 --ckpt-interval 0 \
    --num-warmup-steps 0
```

## DPPO knobs

- `loss_fn="dppo"` with `loss_fn_config={"delta_low": ..., "delta_high": ...}`
  per `forward_backward` request (0.2 recommended for `binary_tv`, 0.05 for
  `binary_kl`).
- The divergence variant is server-side config:
  `"trainer.algorithm.dppo.dppo_type": "binary_tv" | "binary_kl"` in
  `--backend-config` (see the server script's `DPPO_TYPE` env var).
- The sampling logprobs are sent as `loss_fn_inputs["logprobs"]` and act as the
  behavior policy for both the importance ratio and the divergence mask.
- The `clip_ratio` metric reported back through `forward_backward` is the
  fraction of tokens masked out by the DPPO divergence mask.
