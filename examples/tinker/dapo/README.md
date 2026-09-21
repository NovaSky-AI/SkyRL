# Tinker DAPO Example

Reproduces the DAPO recipe for `Qwen/Qwen3-30B-A3B-Base` on the DAPO-Math-17k / AIME-2024 setup through
SkyRL's Tinker API server, mirroring [`examples/train/algorithms/dapo`](../../train/algorithms/dapo).

Two recipes share one client and one server launcher (Megatron backend):

| Recipe | Client flag | Server | Reference (native) |
|---|---|---|---|
| LoRA (rank 128, alpha 128) | `--lora-rank 128` (default) | `bash run_tinker_server.sh` | `run_dapo_qwen3_30b_a3b_lora_megatron_aime.sh` |
| Full fine-tuning | `--lora-rank 0` | `FULL_FT=1 bash run_tinker_server.sh` | `run_dapo_qwen3_30b_a3b_megatron_aime.sh` |

Both recipes run DAPO **without dynamic sampling**, and use **geometric sequence masking instead of TIS**
for off-policy correction (see [Off-policy correction](https://docs.skyrl.ai/docs/algorithms/off_policy_correction)).

## What lives where

The Tinker split puts the algorithm on the client and execution on the server:

- **`dapo_client.py`** (algorithm): GRPO group-normalized advantages, DAPO soft overlong punishment
  (buffer 4096, penalty 1.0), overlong filtering (loss weights zeroed for truncated responses),
  `token_mean_legacy` loss scaling, clip-higher epsilons (0.2 / 0.28), LR 1e-5 with 160-step linear warmup,
  512 prompts x 16 samples per step, 32-prompt minibatches, AIME reward, eval with 32 samples at top_p 0.7.
- **`run_tinker_server.sh`** (execution + knobs a Tinker client cannot send): Megatron TP4 / EP8, vLLM
  engine layout, `policy_loss_type=dual_clip` with `clip_ratio_c=10`, weight decay 0.1, grad clip 1.0,
  LoRA alpha, and `off_policy_correction.sequence_mask_metric=geometric` (0.99 / 1.01).

### Off-policy correction through Tinker

A Tinker datum carries one `logprobs` tensor. On the SkyRL server it fills both the PPO ratio denominator
and the "rollout" logprobs, so any train/inference-mismatch correction would compare identical tensors.
This example instead does what the native trainer does: after sampling, the client runs a `forward` pass
to obtain the training policy's logprobs (`logprobs`) and sends the vLLM sampling logprobs as
`rollout_logprobs`, a SkyRL extension of the datum. The geometric mask then measures the real mismatch.
Set `DAPO_RECOMPUTE_OLD_LOGPROBS=0` to skip the forward pass (the mask becomes a no-op).

## Hardware

| Purpose | GPUs |
|---|---|
| Full recipes (as configured here) | 1 node x 8 H100. Reference runs used 2 nodes; set `NUM_NODES=2 NUM_INFERENCE_ENGINES=2`. |
| Smoke test of the loop | 1 to 8 GPUs with a small model (below) |

## 1. Prepare data

```bash
bash examples/train/algorithms/dapo/prepare_dapo_data.sh   # writes ~/data/dapo/{dapo-math-17k-cleaned,aime-2024-cleaned}.parquet
```

## 2. Start the Tinker API server (GPU node)

```bash
bash examples/tinker/dapo/run_tinker_server.sh              # LoRA
FULL_FT=1 bash examples/tinker/dapo/run_tinker_server.sh    # full fine-tuning
```

Every setting is an environment variable with the reference value as default (see the script). Pass your own
`BACKEND_CONFIG='{...}'` to replace the whole dictionary. Extra arguments are forwarded to `skyrl.tinker.api`.

**Multi-node runs must pass `--checkpoints-base <shared path>`** (for example
`NUM_NODES=2 NUM_INFERENCE_ENGINES=2 bash examples/tinker/dapo/run_tinker_server.sh --checkpoints-base /mnt/shared/skyrl_checkpoints/dapo`).
The default `/tmp/skyrl_checkpoints` is node-local, and on the Tinker path it is on the critical path of every
sampling round: the LoRA sampler archive is written by the engine process and read by the vLLM engines on every
node for `load_lora_adapter`, and checkpoint staging happens next to it (see `_staging_root` in
`skyrl/backends/skyrl_train_backend.py`). Also set the client's `--output-dir` to shared storage so
`metrics.jsonl` survives a head-node restart.

Ray ships the code to worker nodes by uploading the launch directory (`working_dir`) via the uv runtime-env
hook. Launch from the repo root and keep the client's `.venv` ignored (SkyRL's `.gitignore` already lists it);
a venv inside an un-ignored directory makes the upload multi-GB and every actor launch fails.

## 3. Run the client

```bash
TINKER_API_KEY=tml-dummy uv run --extra tinker --extra skyrl-train \
  python examples/tinker/dapo/dapo_client.py --lora-rank 128     # or --lora-rank 0
```

The `skyrl-train` extra is required: the client imports `skyrl_gym` (the AIME verifier) and
`skyrl.backends.skyrl_train.utils.ppo_utils` (the loss-reduction helper).

Set `WANDB_API_KEY` (and optionally `WANDB_PROJECT` / `WANDB_RUN_NAME` / `WANDB_ENTITY`) for W&B logging;
metrics are also appended to `<output-dir>/metrics.jsonl`.

## Smoke test on a small model

Server (e.g. 4 GPUs, Qwen3-1.7B-Base, no expert parallelism):

```bash
BASE_MODEL=Qwen/Qwen3-1.7B-Base NUM_GPUS_PER_NODE=4 MEGATRON_TP=1 MEGATRON_EP=1 \
INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE=1 NUM_INFERENCE_ENGINES=4 MAX_RESPONSE_LENGTH=1024 \
LORA_ALPHA=32 \
bash examples/tinker/dapo/run_tinker_server.sh
```

`LORA_ALPHA` must be set to match the client's `--lora-rank`: megatron-bridge scales every adapter by
`alpha / rank`, so leaving the default `LORA_ALPHA=128` while sampling `--lora-rank 32` multiplies the
adapter update by 4 (the full recipe has alpha = rank = 128, i.e. scale 1). The symptom is entropy
collapse within ~50 steps -- `policy/entropy_loss` falling by an order of magnitude while
`reward/avg_pass_at_N` peaks and then drops back below its starting value.

Client, shrinking the recipe with `DAPO_*` overrides (keep `DAPO_MICRO_TRAIN_BATCH_SIZE` equal to the
server's `MICRO_TRAIN_BATCH_SIZE_PER_GPU`):

```bash
DAPO_TRAIN_BATCH_SIZE=16 DAPO_POLICY_MINI_BATCH_SIZE=4 DAPO_N_SAMPLES_PER_PROMPT=4 \
DAPO_EVAL_N_SAMPLES_PER_PROMPT=2 DAPO_MAX_GENERATE_LENGTH=1024 DAPO_OVERLONG_BUFFER_LEN=256 \
DAPO_NUM_WARMUP_STEPS=4 DAPO_EVAL_BATCH_SIZE=32 DAPO_EVAL_INTERVAL=2 \
TINKER_API_KEY=tml-dummy uv run --extra tinker --extra skyrl-train \
  python examples/tinker/dapo/dapo_client.py --model Qwen/Qwen3-1.7B-Base --lora-rank 32 --max-train-steps 4
```

## What to watch

- `eval/all/avg_score` against the reference W&B runs (LoRA: `fauf9scq`, full FT: `j9sv07vf` in
  `skyrl-train-dapo-aime`).
- `policy/rollout_train_logprobs_abs_diff_{mean,max}` from the server: the train/inference logprob gap.
  **Exactly 0 for every step means `rollout_logprobs` are not reaching the server** and the off-policy
  correction is a silent no-op (a healthy small run shows a mean around 1e-2).
- `policy/geo_sequence_mask_masked_ratio` from the server: the fraction of sequences the geometric mask
  drops, with `..._over_high_ratio` / `..._under_low_ratio` splitting it by direction. If it sits at 1.0,
  the 0.99/1.01 band is rejecting everything and needs widening — the off-policy-correction docs note MoE
  models often need a wider band, and both full recipes here are MoE.
- `reward/truncated_ratio` and `reward/overlong_penalized_ratio` from the client.

## Notes

- KL loss is disabled (as in the reference scripts; the Tinker backend does not support it).
- LoRA alpha cannot be sent through the Tinker SDK (the API server records 32). The launcher sets
  `trainer.policy.model.lora.alpha=128` in `backend_config`, which the SkyRL-Train backend now honors.
- Router replay (R3) for MoE models is not available through the Tinker datum path, so it is not enabled.
- The reference scripts use TIS (`use_tis=true`); this example deliberately does not.
- The launcher sets `megatron_config.lora_config.merge_lora=false`. On the Tinker path this is required,
  not an optimization: sampling addresses the policy by its Tinker `model_id`, and that name only exists
  on the inference engines when the LoRA adapter is registered through `load_lora_adapter`. With the
  Megatron default (`merge_lora=true`) the merged weights are served under the base model name and every
  sample request fails with `404 ... does not exist`.
