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

`torch` comes from `torch-neuronx`, which is published on the AWS Neuron index, not
PyPI. `uv run --isolated` builds a fresh environment per invocation, so a system-level
`pip install torch-neuronx` is **not** visible to it — the Neuron index must be passed to
`uv` itself so `torch-neuronx` resolves inside the isolated env alongside the extra:

```bash
uv run --isolated --extra neuron \
  --extra-index-url https://pip.repos.neuron.amazonaws.com \
  -m skyrl.train.entrypoints.main_base ...
```

(Keep the default PyPI index for everything else; the AWS Neuron index only supplies
`torch-neuronx` and its runtime deps.)

The `neuron` extra mirrors `skyrl-train` minus the CUDA/vLLM-only pieces
(`vllm-router`, `flash-attn`, `flash-linear-attention`, `causal-conv1d`, `nixl`).

## Config

```yaml
generator.inference_engine.backend: neuron
```

No `num_engines`, `engine_init_kwargs`, or weight-sync transport config applies: the
engine is a single in-process instance on the driver.

## Multi-core core pinning

On a multi-core run the FSDP workers own logical NeuronCores `0 .. world_size-1`: worker
`rank` pins `NEURON_RT_VISIBLE_CORES` to `rank` via `device.neuron_worker_core` (set
before its first neuron allocation, since the runtime reads the variable lazily at first
use). This mapping is **not** a stylistic choice — torch-neuronx's native `neuron`
collective backend sets the local device start index to the local rank, and its NCCL
topology layer aborts (`getNumOfVcoresPerDevice` assertion) when the actual core is
offset from it. The driver (the in-process inference engine) is a separate Ray process
and cannot share a core with any worker, so it pins the *last* logical core (63 on a
64-core trn2.48xlarge) via `device.neuron_driver_core()`.

The driver pin must be set in-process, not forwarded through the runtime env. The
engine loads inside the `cholec_entrypoint` actor, and Ray workers are spawned by the
raylet, so they only inherit env vars the runtime env forwards. `NeuronInferenceEngine.__init__`
pins the driver via `device.neuron_driver_core()` before its first neuron allocation.
`NEURON_RT_VISIBLE_CORES` must **not** ride the runtime env: Ray's neuron accelerator
manager reads it back as the worker's "original visible cores" and indexes worker core
assignments into it, so a runtime-env copy collapses every FSDP worker onto the driver's
single-core list (`IndexError`). `prepare_runtime_environment` forwards only
`NKI_ENABLE_TRACE_CACHE` and `USE_HUB_KERNELS`. Without the driver pin the engine's
`.to("neuron")` spreads across the default visible-core set and collides with the
workers' `0 .. world_size-1` cores.

The worker core must be the constant `rank`, **not** read back from
`NEURON_RT_VISIBLE_CORES` inside the worker. Ray's `NeuronAcceleratorManager` rewrites
that variable in each worker process — before the actor's `__init__` runs — to the
worker's *Ray-assigned* core (`0..world_size-1`, handed out in reverse rank order).
Deriving the worker core from it would collapse every worker onto the same core and every
worker then fails with `Failed to initialize Neuron Runtime: status code 1` (`cores busy`).
`neuron_worker_core` therefore ignores the env var and returns `rank`; see
`tests/backends/skyrl_train/utils/test_device.py`.

## Status

- **Text rollouts**: implemented.
- **VLM rollouts**: implemented — `render_chat_completion` produces token ids + vision
  tensors via the processor and stashes them as one `mm_codec` blob; `generate` replays
  that blob into the model input. The processor call specifics are validated on hardware
  (a local CPU env has no VLM processor/model).
- **FSDP trainer device abstraction**: ported. `device.backend()`/`current_device()`/
  `collective_device()`/`ray_actor_options()` map CUDA↔neuron. `ray_actor_options`
  requests `neuron_cores` instead of `num_gpus`.
- **FSDP2 collective backend**: `neuron`, not `gloo`. torch-neuronx registers a native
  `neuron` collective backend (NKI kernels) in `dist.Backend.default_device_backend_map`,
  and `init_device_mesh` auto-resolves to it. The earlier `gloo` override was wrong: gloo
  is CPU-only and cannot run the sharded-`neuron` DTensor all-gather/reduce-scatter that
  FSDP2 needs (`RuntimeError: No backend type associated with device type neuron`). The
  worker initialises `init_process_group(backend="cpu:gloo,neuron:neuron")` so scalar /
  metric CPU reductions still use gloo while sharded DTensor collectives use neuron.
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
- **Processed logprobs**: exact. `generate(output_scores=True)` returns the fully-processed
  per-step logits (temperature / top-k / top-p / min-p / penalties already applied), so the
  response logprob of the sampled token is `log_softmax(scores)[sampled]` — bit-for-bit the
  distribution the token was actually drawn from, with no filter re-implementation.
- **Qwen3.5-VL device-safety patches**: `patches/neuron/qwen3_5_vl.py` fixes three vision-tower
  paths that crash on neuron (vision RoPE `inv_freq` device-follow, and two dynamic
  `repeat_interleave` calls forced onto CPU). Applied from both the in-process engine and the
  FSDP worker's `HFModelWrapper` VLM branch; no-op off neuron.
- **Qwen3-VL device-safety patch**: `patches/neuron/qwen3_vl.py` fixes three vision-tower
  paths that crash on neuron, mirroring the Qwen3.5 patch but for Qwen3-VL's own classes:
  (1) `Qwen3VLVisionRotaryEmbedding` and (2) `Qwen3VLTextRotaryEmbedding` both register
  non-persistent `inv_freq` buffers that FSDP `cpu_offload` leaves on CPU — the forward does
  an unsynchronized device-crossing multiply/matmul; (3)
  `get_vision_interpolation_indices_and_weights` calls a dynamic `repeat_interleave` that
  neuronx-cc cannot lower (`stablehlo.concatenate` shape mismatch). Qwen3-VL binds that last
  helper into `modeling_qwen3_vl` via `from transformers.vision_utils import ...`, so the
  patch targets the *bound name* in `modeling_qwen3_vl` (not the `vision_utils` attribute the
  model never re-reads). Applied from both the in-process engine and the FSDP worker's
  `HFModelWrapper` VLM branch, gated on `model_type == "qwen3_vl"`; no-op off neuron. Verified
  end-to-end on a 4-NeuronCore geometry3k smoke (`run_geometry3k_neuron.sh`): the full VLM GRPO
  loop (generate + training step) completes with no `stablehlo` / `TorchMlirCompilerError`
  fallbacks, and the `vision_utils` wrong-target line is absent.

## Known issues

- **Multi-core worker core offset (root-caused, fixed).** The earlier multi-core
  SIGSEGV (`getNumOfVcoresPerDevice` assertion in NCCL's `path_neuron.cc`) was not OOM
  and not the `LOCAL_RANK`/`LOCAL_WORLD_SIZE` env — it was the worker core offset
  (`NEURON_RT_VISIBLE_CORES = 1 + rank`). torch-neuronx's `neuron` collective backend
  sets the local device start index to the local rank, and NCCL aborts when the visible
  core is offset from it. Workers must pin core `rank` (`0 .. world_size-1`); the driver
  moves to the last core. Verified standalone on 4 ranks (cores `0,1,2,3` pass; `1,2,3,4`
  abort) and on the full multi-core Ray smoke (`run_multiply_neuron.sh`).

- **Weight-sync SIGSEGV from NCCL P2P/SHM disable (root-caused, fixed).** A second
  multi-core crash — "No transport found between devices 0 and 1" → `neuronGetCommInfo`
  assertion in the FSDP2 LoRA weight-sync path — was caused by `prepare_runtime_environment`
  setting `NCCL_P2P_DISABLE=1` / `NCCL_SHM_DISABLE=1` on every worker: `peer_access_supported()`
  returns `False` for neuron, so the CUDA-only NCCL transport tuning leaked to the neuron
  backend and disabled the intra-NeuronCore transport libnccom/libnrt needs. The peer-access
  probe is now skipped for neuron (`if not device.is_neuron() and ...`). Verified on the
  multi-core Ray smoke: all `sync_weights` complete with no SIGSEGV.

## Gotchas

- The base `[project].dependencies` floor pins `transformers<=5.16.1`. Some models on
  neuron (e.g. Qwen3.5) require a newer transformers release; if so, override it in the
  run's environment rather than bumping the shared floor.
- `do_sample` is derived from `temperature > 0` because transformers falls back to greedy
  at `temperature == 1.0` (a common RL default that must still sample).
