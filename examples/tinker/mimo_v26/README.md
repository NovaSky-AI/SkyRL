# MiMo-V2.6-Flash-RL with Tinker and Megatron

This text-only attention-LoRA example uses MiMo-V2.6-Flash-RL (309B total /
15B active parameters) on two eight-GPU H200 nodes. One node runs the BF16
learner with TP2/EP8/ETP1; the other runs a TP8 sampler. Context parallelism
and packed sequences are disabled.

The example requires [Megatron-Bridge #6201](https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/6201),
the Bridge/Core dependency update in [SkyRL #2284](https://github.com/NovaSky-AI/SkyRL/pull/2284),
and [native quantized-export dtype preservation](https://github.com/NovaSky-AI/SkyRL/pull/2282).
The dependency update pins the official CUDA 13 vLLM nightly at
[`8644d2af`](https://github.com/vllm-project/vllm/commit/8644d2af2fb7c4588190a8d7f6c26a3cbef78a0f),
which includes fused-QKV sharding and BF16-router support. No serving source
patches are needed. This is a configuration example; full-model execution and
trainer/sampler parity are not yet validated.

Use an existing 16-H200 Ray cluster with the same environment on both nodes.
Set `NVTE_FLASH_ATTN=0` and `NVTE_FUSED_ATTN=1` before starting Ray. Mount
writable shared storage at the same path on both nodes and the API host.
Replace `/shared/skyrl/mimo-v26` below and in `backend_config.json` with that
mount. Delta synchronization keeps a local checkpoint copy and update payloads;
allow storage for the original 173 GB checkpoint, its working copy and updates.

Download the immutable model revision, keeping its safetensors index. Use the
same local snapshot for the API, learner and sampler:

```bash
MIMO_MODEL=$(uv run --isolated --extra megatron --extra tinker hf download \
  XiaomiMiMo/MiMo-V2.6-Flash-RL \
  --revision 5711b268169967567844e1e560e8a3966da959b1 \
  --cache-dir /shared/skyrl/huggingface)

RAY_ADDRESS=auto NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=1 \
uv run --isolated --extra tinker --extra megatron -m skyrl.tinker.api \
  --backend megatron \
  --base-model "$MIMO_MODEL" \
  --backend-config "$(cat examples/tinker/mimo_v26/backend_config.json)" \
  --checkpoints-base /shared/skyrl/mimo-v26/checkpoints
```

Create the Tinker training client with LoRA rank 32. Before publication, Bridge
merges the attention adapters and restores the source FP8/MXFP4 weights and
scales. The existing delta backend publishes that checkpoint and vLLM reloads
it; this does not require serving LoRA support. Requantization is lossy.

The sampler uses vLLM's native Omni architecture with `language_model_only`
to handle the checkpoint's auxiliary vision/audio tensors without enabling
multimodal requests. Explicit Marlin MoE keeps BF16 activations; verify that
backend in the serving startup log. The `mimo` reasoning and tool parsers are
native vLLM parsers.
