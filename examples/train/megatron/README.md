# Megatron examples

## Tiny Moonlight MoE smoke

This is the smallest Moonlight-16B-A3B training smoke in this folder. It keeps the real SkyRL path: Megatron policy training, local vLLM rollout, GSM8K rule reward, colocated weight sync, MoE expert parallelism, and one optimizer step. It does not require DAPO, GSM8K download, Mini-SWE-Agent, E2B, or any external task service.

From a fresh checkout of the PR:

```bash
git fetch origin pull/<PR_NUMBER>/head:moonlight-smoke
git checkout moonlight-smoke
```

Install/use the Megatron image or environment described by the existing SkyRL Megatron examples, then make the Moonlight checkpoint available. Either let Hugging Face resolve it:

```bash
export MODEL_NAME=moonshotai/Moonlight-16B-A3B-Instruct
```

or point at a local checkpoint directory:

```bash
export MODEL_NAME=$HOME/moonlight16b
```

Run the smoke:

```bash
bash examples/train/megatron/run_megatron_moonlight_smoke.sh
```

The default smoke is sized for one 4-GPU node:

- `NUM_GPUS=4`
- Megatron `TP=4`, `PP=1`, `CP=1`, `EP=4`, `ETP=1`
- one vLLM engine with `INFERENCE_ENGINE_TP=4`
- `TRAIN_BATCH_SIZE=4`, `POLICY_MINI_BATCH_SIZE=2`, `N_SAMPLES_PER_PROMPT=1`
- `MAX_TRAINING_STEPS=1`
- `LOGGER=console`
- `REMOVE_MICROBATCH_PADDING=false`, which is the safer default for Moonlight MLA on A100-class GPUs
- `LORA_RANK=8`; set `LORA_RANK=0` to attempt a full-parameter smoke

Useful overrides:

```bash
MODEL_NAME=$HOME/moonlight16b \
GPU_MEMORY_UTILIZATION=0.45 \
MAX_GENERATE_LENGTH=32 \
bash examples/train/megatron/run_megatron_moonlight_smoke.sh
```

If the 4-GPU smoke succeeds, move to the larger existing Moonlight recipe:

```bash
bash examples/train/megatron/run_megatron_moonlight.sh
```
