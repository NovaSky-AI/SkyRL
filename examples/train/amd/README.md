# AMD ROCm Tinker Example

This example is a starting point for running SkyRL's Tinker-compatible API on AMD GPUs with the FSDP backend and vLLM ROCm inference.

The runtime path is intentionally split from the image path:

- `docker/Dockerfile.amd` builds a SkyRL AMD image from `vllm/vllm-openai-rocm:v0.20.2`.
- This directory contains commands to run inside that image.

The Docker image bakes in Ray and the non-GPU SkyRL dependencies. It relies on the base vLLM ROCm image for ROCm builds of PyTorch, vLLM, and flash-attn.

## Build

From the repository root:

```bash
docker build -f docker/Dockerfile.amd -t skyrl-amd-rocm .
```

## Run The Container

Use the ROCm devices and host IPC. `--network host` is convenient for Ray and for reaching the Tinker API from another shell.

```bash
docker run --rm -it \
  --network host \
  --ipc=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  skyrl-amd-rocm
```

## Start The Tinker Server

Inside the container:

```bash
cd /workspace/SkyRL/examples/train/amd
bash run_tinker_server_amd.sh
```

The default server binds to `0.0.0.0:9000` and uses:

- `BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507`
- `BACKEND=fsdp`
- `POLICY_NUM_GPUS_PER_NODE=1`
- `INFERENCE_NUM_ENGINES=6`

This default targets an 8-GPU AMD node with one GPU for the FSDP policy worker,
six vLLM inference engines, and one GPU left for headroom. For smaller nodes or
faster local debugging, reduce the inference engine count:

```bash
INFERENCE_NUM_ENGINES=1 \
bash run_tinker_server_amd.sh
```

All server arguments can still be overridden through environment variables or by passing flags directly:

```bash
bash run_tinker_server_amd.sh --help
```

## Run The Client Smoke Test

In a second shell inside the container:

```bash
cd /workspace/SkyRL/examples/train/amd
TINKER_BASE_URL=http://localhost:9000 TINKER_API_KEY=tml-dummy \
python tinker_hello_world.py
```

The client is a fixed Tinker smoke test. It creates a rank-32 LoRA training
client, builds 16 tiny cross-entropy datums, samples once before training, runs
four `forward_backward` + `optim_step` iterations, syncs trained weights to the
sampler, samples once more, and prints `PASS` on success.

## Run The GRPO Client

For a fuller Tinker client, run the GRPO-style GSM8K example against the same
server:

```bash
cd /workspace/SkyRL/examples/train/amd
TINKER_BASE_URL=http://localhost:9000 TINKER_API_KEY=tml-dummy \
python grpo_client.py
```

The GRPO client samples groups of responses, computes group-relative advantages
from rule-based rewards, and trains a rank-32 LoRA policy with the public Tinker
`ppo` loss. SkyRL maps this to its standard PPO-style policy loss internally. If
`--data-dir` does not already contain
`train.parquet` and `validation.parquet`, the client prepares a small GSM8K
subset automatically under `/tmp/skyrl-tinker-grpo/gsm8k`. Prepared subsets are
shuffled before truncation, and each training step samples a fresh random prompt
batch from the loaded train pool.

The default client run uses:

- `--max-train-steps 5`
- `--num-prompts 64`
- `--group-size 8`
- `--max-tokens 512`
- `--max-train-examples 1024`
- `--max-val-examples 128`

To force regeneration of the shuffled GSM8K subset:

```bash
TINKER_BASE_URL=http://localhost:9000 TINKER_API_KEY=tml-dummy \
python grpo_client.py \
  --reprepare-data
```

For a faster single-step smoke:

```bash
TINKER_BASE_URL=http://localhost:9000 TINKER_API_KEY=tml-dummy \
python grpo_client.py \
  --max-train-steps 1 \
  --num-prompts 2 \
  --group-size 2 \
  --max-tokens 64 \
  --max-train-examples 32 \
  --max-val-examples 8 \
  --reprepare-data
```

## Multi-Node Note

Ray is installed in the image so cluster setup can be handled by normal Ray commands or a platform launcher.

Head node:

```bash
ray start --head --port=6379
```

Worker nodes:

```bash
ray start --address="$HEAD_NODE_IP:6379"
```
