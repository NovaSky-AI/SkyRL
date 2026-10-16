# Math reasoning RL for Beam

These recipes train [Beam](https://huggingface.co/reflection/Beam-1.0-501B-A23B), Reflection's 501B-parameter
(23B active) mixture-of-experts model, with reinforcement learning in SkyRL. Every recipe runs synchronous DAPO on DAPO-Math-17k and evaluates
on AIME-2024 and AIME-2025. The [SkyRL blog post](https://blog.skyrl.ai/blog/day-0-beam) covers how Beam support was
built and validated.

## Configurations

| Configuration | Hardware | Script |
|---|---|---|
| LoRA, minimal setup | 2 nodes x 8 B200 | [`run_dapo_beam_lora_2node.sh`](run_dapo_beam_lora_2node.sh) |
| LoRA (reference run) | 4 nodes x 8 B200 | [`run_dapo_beam_lora_4node.sh`](run_dapo_beam_lora_4node.sh) |
| BF16 full fine-tuning | 8 nodes x 8 B200 | [`run_dapo_beam_fullft_8node.sh`](run_dapo_beam_fullft_8node.sh) |
| MXFP8 full fine-tuning | 8 nodes x 8 B200 | [`run_dapo_beam_fp8_fullft_8node.sh`](run_dapo_beam_fp8_fullft_8node.sh) |

All scripts train for 50 steps by default (`MAX_TRAINING_STEPS`) and evaluate every 5 steps. Other cluster shapes work
by adjusting the parallelism and memory settings in the scripts.

## Environment

Beam needs Beam-enabled builds of vLLM (`0.1.0+beam2`) and Megatron-Bridge (`0.7.0+beam1`). On this branch,
`pyproject.toml` installs both from wheels under `/opt/skyrl-wheels`.

The image `novaskyai/skyrl-train-ray-2.57.0-py3.12-cu13.0-megatron-beam` ships both wheels
and the FlashInfer cubins (`FLASHINFER_CUBIN_DIR=/opt/flashinfer-cubin/0.7.0.post1`). Run one container per node with
host networking and access to the GPUs and RDMA devices, then start Ray inside the containers:

```bash
# on every node (same mounts everywhere)
docker run -d --name skyrl-beam --gpus all --network host --ipc host --shm-size 64g \
  --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 \
  -v /path/to/SkyRL:/path/to/SkyRL -v /path/to/models:/path/to/models -v $HOME/data:$HOME/data \
  novaskyai/skyrl-train-ray-2.57.0-py3.12-cu13.0-megatron-beam sleep infinity

# head node
docker exec skyrl-beam ray start --head --port=6379 --num-gpus=8
# every other node
docker exec skyrl-beam ray start --address=<head-ip>:6379 --num-gpus=8
```

Then launch a recipe from the SkyRL checkout inside the head container (`docker exec -it skyrl-beam bash`,
`cd /path/to/SkyRL`).

The scripts launch through `uv run --isolated --extra megatron --extra fa4`. Start a Ray cluster across the nodes first and run the script on the head node; SkyRL places the trainer and the engines. The scripts use Flash Attention 4 and are meant to run on NVIDIA Blackwell GPUs.

## Model and data

Every Ray worker reads the model and the data from local paths, so download both on every node, or onto a shared
filesystem mounted at the same path everywhere:

```bash
# Beam checkpoint (~1 TB in bf16). Alternatively keep MODEL_PATH=reflection/Beam-1.0-501B-A23B
# and let each node populate its Hugging Face cache.
hf download reflection/Beam-1.0-501B-A23B --local-dir /path/to/models/Beam-1.0-501B-A23B

# DAPO-Math-17k, AIME-2024 and AIME-2025: writes dapo-math-17k-cleaned.parquet,
# aime-2024-cleaned.parquet and aime-2025-cleaned.parquet into DATA_DIR
DATA_DIR=$HOME/data/dapo bash examples/train/algorithms/dapo/prepare_dapo_data.sh
```

The scripts read `MODEL_PATH` (default `reflection/Beam-1.0-501B-A23B`) and `DATA_DIR` (default `$HOME/data/dapo`).

## LoRA

Trains rank-64 adapters (alpha 64) on all linear layers, with one adapter per expert, and keeps the base model frozen.
The learning rate is `1e-5`.

The 4-node reference script uses Megatron TP4/PP2/EP8 (DP4) and two vLLM engines, each TP8 x PP2. It samples
64 prompts x 8 responses per step, split into two mini-batches:

```bash
MODEL_PATH=/path/to/models/Beam-1.0-501B-A23B ENV_FILE=/path/to/.env \
  bash examples/train/models/reflection/beam/run_dapo_beam_lora_4node.sh
```

The 2-node script keeps the same Megatron layout (DP2) with one vLLM engine of TP8 x PP2. It halves the batch to
32 prompts x 8 responses and keeps the mini-batch size of 32 prompts, so each step is one optimizer update:

```bash
MODEL_PATH=/path/to/models/Beam-1.0-501B-A23B ENV_FILE=/path/to/.env \
  bash examples/train/models/reflection/beam/run_dapo_beam_lora_2node.sh
```

Both LoRA scripts turn on these throughput and memory optimizations:

- MoE LoRA slots in vLLM sized at the per-expert rank.
- Pinned host staging for the in-memory adapter sync.
- LoRA sequence-parallel input regathering.
- Fused LM-head log-probabilities with logit softcapping.

## BF16 full fine-tuning

Updates the full model with a learning rate of `1e-6`. Megatron uses TP4/PP4/EP8 with full recompute, and vLLM runs
four engines, each TP8 x PP2.

```bash
MODEL_PATH=/path/to/models/Beam-1.0-501B-A23B ENV_FILE=/path/to/.env \
  OPTIMIZER_STATE_DISK_DIR=/local/nvme/skyrl_optimizer_state \
  bash examples/train/models/reflection/beam/run_dapo_beam_fullft_8node.sh
```

The fp32 master weights and Adam moments (~6 TB in total) are kept on node-local NVMe and streamed through the GPU
one chunk at a time. Set `OPTIMIZER_STATE_DISK_DIR` to a fast local disk (NVMe RAID0) on every node; the optimizer
state takes about 0.6-0.7 TB per node. To use the CPU optimizer instead, set `OPTIMIZER_STATE_DISK_DIR=` (empty); it
needs about 1.7 TB of host memory per node.

## MXFP8 full fine-tuning

Trains with MXFP8 mixed precision in Megatron and generates with MXFP8 weights in vLLM. Layer norms and router
parameters keep their original precision.

- **Training:** MXFP8 GEMMs with bf16 primary weights. Megatron uses TP4/PP8/EP8.
- **Rollouts:** weights are synced to vLLM in MXFP8. vLLM runs eight single-node engines, each TP8 with EP8.
- **KV cache:** bf16 by default. `KV_CACHE_DTYPE=fp8_e4m3` decodes faster but raises the mean rollout/train
  log-probability difference by about 0.01.
- **Off-policy correction:** MXFP8 rollouts raise the mean log-probability difference to about 0.03 with R3, partly
  because Megatron and vLLM use different kernels (switching from FlashAttention 4 to FlashAttention 2 lowers it to
  about 0.02). The script enables geometric sequence masking, which drops sequences whose geometric-mean probability
  ratio falls outside [0.99, 1.01].

```bash
MODEL_PATH=/path/to/models/Beam-1.0-501B-A23B ENV_FILE=/path/to/.env \
  OPTIMIZER_STATE_DISK_DIR=/local/nvme/skyrl_optimizer_state \
  bash examples/train/models/reflection/beam/run_dapo_beam_fp8_fullft_8node.sh
```

FlashInfer autotunes the MXFP8 MoE kernels the first time they run, which can take several minutes, so the script
raises the inference-server health timeout.

## Training settings

The reference runs (4-node LoRA, BF16 and MXFP8 full fine-tuning) share these settings:

- **Algorithm:** GRPO with clip-higher, 50 training steps and 100 optimizer updates.
- **Batch:** 64 prompts x 8 responses per step, split into two mini-batches.
- **Token budget:** up to 2,048 prompt tokens and 8,192 response tokens.
- **Generation:** temperature `1.0`, top-p `1.0`, reasoning effort `minimal`.
- **Reward:** math correctness with the DAPO overlong-response penalty.
- **Routing:** Rollout Routing Replay (R3) enabled.
- **Sampling:** dynamic sampling disabled.
- **Evaluation:** AIME-2024 and AIME-2025 with 8 responses per problem, reported as average accuracy.

Beam's chat template takes a `reasoning_effort` (`minimal`, `low`, `medium`, `high`, `xhigh`, `max`) and defaults to
`max` when it is unset. The scripts pass it through `generator.chat_template_kwargs.reasoning_effort`, which requires
`generator.batched=false`. Training defaults to `minimal`, which keeps responses short and leaves room to improve
within the 8K response budget. Override it with, for example, `REASONING_EFFORT=low`.

## Reference results

With minimal reasoning effort and an 8,192-token response limit, AIME-2025 average accuracy rises from about 61% to
79% over 50 training steps. LoRA and BF16 full fine-tuning reach similar results over these short runs. R3 lowers the
mean rollout/train log-probability difference from about 0.02 to 0.008. Mean response length on the training set
grows from about 950 to 1,400 tokens.

MXFP8 increases generation throughput by 44% over BF16, with similar reward and evaluation accuracy over 50 training
steps.

## Checkpoints and resuming

Each script writes checkpoints to `CKPT_PATH` (default `$HOME/ckpts/<run name>`) and exports and evaluation dumps to
`EXPORT_PATH` (default `$HOME/exports/<run name>`).

- **LoRA** saves the adapters every 10 steps (`CKPT_INTERVAL`). To resume, rerun the same command with
  `RESUME_MODE=latest`.
- **Full fine-tuning (BF16 and MXFP8)** saves every 25 steps and resumes from the latest checkpoint in `CKPT_PATH` by
  default. A full checkpoint is about 7 TB (about 0.9 TB per node), so make sure each node has room for every
  checkpoint you keep.

To resume from a specific checkpoint, pass `RESUME_MODE=from_path` with the Hydra override
`trainer.resume_path=<checkpoint dir>`.

## Variables

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `reflection/Beam-1.0-501B-A23B` | Hugging Face repo id or a local copy of the checkpoint. A local copy must be at the same path on every node. |
| `DATA_DIR` | `$HOME/data/dapo` | Directory with the parquet files above, at the same path on every node. |
| `ENV_FILE` | unset | Env file passed to `uv run --env-file`, e.g. with `WANDB_API_KEY`, `HF_TOKEN` and your NCCL settings. Must exist at the same path on every node. |
| `LOGGER` | `wandb` | `trainer.logger`; `LOGGER=console` runs without W&B. |
| `MAX_TRAINING_STEPS` | `50` | Training steps. |
| `REASONING_EFFORT` | `minimal` | Beam's reasoning effort: `minimal`, `low`, `medium`, `high`, `xhigh` or `max`. |
| `EVAL_INTERVAL`, `EVAL_BEFORE_TRAIN` | `5`, `true` | Evaluate every N steps, and once before training. |
| `CKPT_INTERVAL` | `10` (LoRA), `25` (full FT) | Checkpoint every N steps; `0` turns saving off. |
| `RESUME_MODE` | `null` (LoRA), `latest` (full FT) | `latest` resumes from the newest checkpoint in `CKPT_PATH`; `from_path` with `trainer.resume_path=<dir>`. |
| `INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION` | `0.7` (LoRA), `0.6` (BF16 full FT), `0.55` (MXFP8) | vLLM's GPU memory fraction. |
| `PROJECT_NAME`, `RUN_NAME` | `skyrl-beam`, `beam_...` | W&B project and run names. |
| `CKPT_PATH`, `EXPORT_PATH`, `LOG_PATH` | `$HOME/ckpts/<run>`, `$HOME/exports/<run>`, `$HOME/skyrl-logs` | Output locations. |
| `OPTIMIZER_STATE_DISK_DIR` | `$HOME/skyrl_optimizer_state` | Full FT only: node-local NVMe directory for the optimizer state. Empty uses the CPU optimizer. |
| `OPTIMIZER_STATE_DISK_MOMENT_DTYPE` | `fp32` | Full FT only: on-disk dtype of the Adam moments; `bf16` reads and writes a third fewer bytes per step. |
| `LR` | `1e-6` | Full FT only: learning rate. |
| `KV_CACHE_DTYPE` | `auto` (bf16) | MXFP8 only: `fp8_e4m3` decodes faster but raises the rollout/train mismatch by about 0.01. |
| `SEQUENCE_MASK_METRIC` | `geometric` | MXFP8 only: off-policy sequence masking; empty turns it off. |
| `FP8_ROLLOUT` | `1` | MXFP8 only: `0` keeps bf16 rollouts while training in MXFP8. |
| `UV_EXTRA_ARGS` | unset | Extra `uv run` flags, e.g. `--with ray==<your cluster Ray version>` |

`uv run --env-file` does not override variables already set in your shell, so unset stale `WANDB_*` / `HF_TOKEN`
values first. Ray's uv integration replays the same `uv run` command, including `--env-file`, on every worker. Any
other Hydra override can be appended, e.g.
`bash examples/train/models/reflection/beam/run_dapo_beam_lora_4node.sh trainer.eval_interval=10`.

## Multi-node networking

Every recipe spans several nodes for both Megatron (pipeline and data parallel) and vLLM (PP2 engines in the BF16
recipes), so NCCL must use your cluster's high-speed fabric:

- Set `NCCL_NET`, `NCCL_IB_*` / `NCCL_SOCKET_IFNAME` and any NCCL net or tuner plugin settings for your cluster
  (InfiniBand, RoCE or EFA) in your `ENV_FILE`, so that every Ray worker picks them up.
- Every rank must see the same NCCL configuration: a tuner plugin that loads on some nodes and not others can hang the
  first collective.
- In Docker, pass the RDMA devices into the container (`--device /dev/infiniband --cap-add IPC_LOCK
  --ulimit memlock=-1`) so NCCL can use the fabric.

## Notes

- **Memory.** The vLLM `gpu_memory_utilization` values leave room for Megatron state that stays on the GPU around
  weight sync. If a weight sync stalls, check the vLLM worker logs for an out-of-memory error and lower
  `INFERENCE_ENGINE_GPU_MEMORY_UTILIZATION`.
- **Smoke test.** Before a long run: `MAX_TRAINING_STEPS=2 EVAL_BEFORE_TRAIN=false LOGGER=console`.
- **Eval metric names.** Metrics are grouped by each dataset's `data_source`: AIME-2024 (`aime-2024-cleaned.parquet`)
  logs as `eval/math_dapo/*` and AIME-2025 as `eval/aime-2025/*`. Per-sample results are written under
  `$EXPORT_PATH/dumped_evals/`.
