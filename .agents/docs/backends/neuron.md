# TorchNeuron (Trainium) Backend

## Overview

Native inference-engine backend that drives the policy model with transformers
`model.generate()` on the AWS Trainium2 (`neuron`) device, behind the existing
`InferenceEngineInterface`. vLLM cannot serve on Trainium (it requires CUDA torch and
has no RL weight-sync path), so this backend runs the rollout engine **in-process** — no
Ray actors, no HTTP server. It is the same `model.generate` path the TRL GRPO leg uses.

```
skyrl/backends/skyrl_train/inference_servers/
├── neuron_engine.py        # NeuronInferenceEngine (text + VLM generate/render)
├── engine_utils.py         # get_neuron_sampling_params (SamplingParams -> generate kwargs)
└── setup.py                # backend == "neuron" branch in build_new_inference_client
skyrl/backends/mm_codec.py  # vLLM-free safetensors codec for multimodal kwargs_data
```

## Install

Two steps. `torch` comes from `torch-neuronx`, which is published on the AWS Neuron index,
not PyPI, and must be installed first so the `neuron` extra resolves against it:

```bash
# 1. torch-neuronx (provides the patched torch + the neuron runtime)
pip install torch-neuronx --index-url https://pip.repos.neuron.amazonaws.com

# 2. the skyrl neuron extra (no vLLM, no CUDA torch)
uv run --isolated --extra neuron ...
```

The `neuron` extra mirrors `skyrl-train` minus the CUDA/vLLM-only pieces
(`vllm-router`, `flash-attn`, `flash-linear-attention`, `causal-conv1d`, `nixl`).

## Config

```yaml
generator.inference_engine.backend: neuron
```

No `num_engines`, `engine_init_kwargs`, or weight-sync transport config applies: the
engine is a single in-process instance on the driver.

## Status

- **Text rollouts**: implemented.
- **VLM rollouts**: implemented — `render_chat_completion` produces token ids + vision
  tensors via the processor and stashes them as one `mm_codec` blob; `generate` replays
  that blob into the model input. The processor call specifics are validated on hardware
  (a local CPU env has no VLM processor/model).
- **FSDP trainer device abstraction**: ported. `device.backend()`/`current_device()`/
  `collective_device()`/`ray_actor_options()` map CUDA↔neuron, and `create_device_mesh`
  pins the FSDP2 collective backend to `gloo` on neuron (the default auto-resolution
  errors on the "unknown accelerator"). `ray_actor_options` requests `neuron_cores`
  instead of `num_gpus`.
- **LoRA weight sync**: wired end-to-end for the native neuron engine. The engine holds a
  full model on the **driver** (not picklable through Ray), so the FSDP worker exports
  LoRA adapters to `lora_sync_path` (save-only) and `WorkerDispatch` then reloads them
  into the in-process engine via `update_named_weights(LoraLoadRequest)` on the driver.
  `init_weight_sync_state` is a no-op on neuron, and the worker's `broadcast_to_inference_engines`
  no longer tries to send the client over Ray.
- **Full-weight sync**: **not yet wired.** The trainer's full-weight path
  (`_sync_weights_to_inference_engines` → `engine.send_weights`) is the vLLM
  trainer-send transport and has no neuron equivalent. The engine's `update_named_weights`
  already accepts a full state-dict, but the trainer side must gather full FSDP weights and
  hand them to the in-process engine. See the RFC open question on full-weight sync.
- **Processed-logprob exactness**: `_sampled_token_logprobs` re-implements transformers'
  sampling filters; to be validated bit-for-bit against the trainer's `processed_logprobs`.

## Gotchas

- The base `[project].dependencies` floor pins `transformers<=5.16.1`. Some models on
  neuron (e.g. Qwen3.5) require a newer transformers release; if so, override it in the
  run's environment rather than bumping the shared floor.
- `do_sample` is derived from `temperature > 0` because transformers falls back to greedy
  at `temperature == 1.0` (a common RL default that must still sample).
