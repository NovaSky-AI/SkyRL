# Tinker DAPO Example

Reproduces the DAPO recipe for `Qwen/Qwen3-30B-A3B-Base` on the DAPO-Math-17k / AIME-2024 setup through
SkyRL's Tinker API server, mirroring [`examples/train/algorithms/dapo`](../../train/algorithms/dapo).

Two recipes share one client and one server launcher (Megatron backend):

| Recipe | Client flag | Learning rate | Server | Reference (native) |
|---|---|---|---|---|
| LoRA (rank 128, alpha 128) | `--lora-rank 128` (default) | 1e-5 | `bash run_tinker_server.sh` | `run_dapo_qwen3_30b_a3b_lora_megatron_aime.sh` |
| Full fine-tuning | `--lora-rank 0` | 1e-6 | `FULL_FT=1 bash run_tinker_server.sh` | `run_dapo_qwen3_30b_a3b_megatron_aime.sh` |

The client picks the learning rate from `--lora-rank` (the two reference scripts differ); `--learning-rate` or
`DAPO_POLICY_LEARNING_RATE` overrides it.

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

The forward pass is not free: on the 30B LoRA recipe on 16xH100 it takes about 5 minutes of a
35-minute step (generation about 14 minutes, training about 15 minutes). The native trainer runs the same
pass about a minute faster, and the Tinker training phase is correspondingly faster, so step-for-step wall
clock matched the native reference run (step 2: 2081 s vs 2084 s). Native step time itself grows over a run
as responses lengthen, so compare step for step, not against a mean.

## Hardware

2 nodes x 8 H100, the reference layout: Megatron TP4 / EP8 and two vLLM engines at TP8, one per node. The
launcher defaults to this; `NUM_NODES`, `NUM_GPUS_PER_NODE`, `NUM_INFERENCE_ENGINES` and
`INFERENCE_ENGINE_TENSOR_PARALLEL_SIZE` override it.

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

**Pass `--checkpoints-base <shared path>`** (for example
`bash examples/tinker/dapo/run_tinker_server.sh --checkpoints-base /mnt/shared/skyrl_checkpoints/dapo`): this is
a 2-node run and the default `/tmp/skyrl_checkpoints` is node-local. On the Tinker path it is on the critical path of every
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
metrics are also appended to `<output-dir>/metrics.jsonl`, which is the live source of truth for a running job
(train, checkpoint and eval payloads for one step share one W&B row, committed at the end of the step).

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
- **An eval score well above the reference early in the run is a red flag, not a success.** A full-FT run
  accidentally trained at the LoRA LR (10x the reference) scored +0.20 above the reference at step 5 while its
  entropy fell 4x faster than the LoRA run's; that is the leading edge of collapse, not a better recipe.
  Check the LR, alpha/rank, and clip settings before celebrating.

## Notes

- **`GPU_MEMORY_UTILIZATION` defaults to 0.6, not the reference scripts' 0.7.** On the 30B LoRA recipe on
  16xH100, the run at 0.7 completed step 1 and died in step-2 sampling with a genuine `torch.OutOfMemoryError`
  (356 MiB free; vLLM 60 GiB + trainer 18 GiB on one GPU). vLLM does not reserve a fixed amount: it takes
  `gpu_memory_utilization` of the free VRAM it sees when the engines are created. On the Tinker server the
  engines are created lazily on the first sampling call, after the model is built and offloaded but before any
  training step, when the trainer's post-offload residual was ~13 GiB. After the first `optim_step` that
  residual is ~25 GiB, so the KV-cache budget sized at engine init no longer fits. This is a capacity
  shortfall, not fragmentation (`use_expandable_segments` does not apply). The value therefore has to leave room
  for the post-first-step footprint rather than be copied from a native recipe; 0.6 was measured safe on 8xH100
  nodes with no measurable generation slowdown (generation time matched the native run's within 2%). Why the native trainer tolerates 0.7 for the same recipe is not established (the LoRA adapter store is
  not the cause: it lives in pinned host memory). Note the adapter store does cost one pinned-CPU mirror of
  params, grads and optimizer state per registered adapter.
- **The engine's SQLite database grows ~135 KB per trajectory** (1.3 GB after ~10k trajectories), i.e. tens of
  GB over a reference-length run. Its default location is inside the repo (`skyrl/tinker/tinker.db`, kept out
  of the Ray upload only by `.gitignore`). Pass `--database-url sqlite:////<path outside the repo>/tinker.db`
  through the launcher for long runs, and archive or rotate it between runs.
- **After a crash, a plain server restart is not enough.** Ray worker actors (`ray::MegatronPolicyWorkerBase`,
  vLLM `ray::RayWorkerProc.run`) can survive the API/engine processes and keep GPU memory, and the colocated
  placement group stays reserved, so a fresh 16-GPU request hangs. Kill the orphaned GPU processes on every
  node (match actor names against `nvidia-smi` compute apps) and `remove_placement_group` on the stale group,
  then confirm `ray status` shows 0/16 GPUs used before relaunching.
- KL loss is disabled (as in the reference scripts; the Tinker backend does not support it).
- LoRA alpha cannot be sent through the Tinker SDK (the API server records 32). The launcher sets
  `trainer.policy.model.lora.alpha=128` in `backend_config`, which the SkyRL-Train backend now honors.
  Keep `LORA_ALPHA` equal to the client's `--lora-rank`: megatron-bridge scales every adapter by `alpha / rank`,
  so a mismatch multiplies the adapter update (rank 32 with the default alpha 128 is a 4x update, and the
  symptom is entropy collapse within ~50 steps while `reward/avg_pass_at_N` peaks and then drops).
- Router replay (R3) for MoE models is not available through the Tinker datum path, so it is not enabled.
- The reference scripts use TIS (`use_tis=true`); this example deliberately does not.
- The launcher sets `megatron_config.lora_config.merge_lora=false`. On the Tinker path this is required,
  not an optimization: sampling addresses the policy by its Tinker `model_id`, and that name only exists
  on the inference engines when the LoRA adapter is registered through `load_lora_adapter`. With the
  Megatron default (`merge_lora=true`) the merged weights are served under the base model name and every
  sample request fails with `404 ... does not exist`.
