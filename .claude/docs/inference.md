# Inference

## Two Codepaths

- **Legacy** (`inference_engines/`): `InferenceEngineClient` with Ray actors. Currently the default (`_SKYRL_USE_NEW_INFERENCE=0`).
- **New** (`inference_servers/`): `RemoteInferenceClient` with HTTP endpoints. Opt-in via `_SKYRL_USE_NEW_INFERENCE=1`.

The new codepath uses:
- **Data plane**: HTTP proxy URL for completions requests.
- **Control plane**: Fan-out to individual server URLs for weight sync, pause/resume.

## vLLM Router

- `VLLMRouter` in `skyrl/backends/skyrl_train/inference_servers/vllm_router.py` wraps a child process running the `vllm-router` binary.
- Configured via constructor args: `server_urls`, `host`, `port`, `policy` (default `consistent_hash`), etc.

## PD Disaggregation

Prefill-Decode separation:
- **Config**: `enable_pd=true` passed to `ServerGroup` constructor.
- **Server groups**: Separate prefill and decode `ServerGroup`s, one per engine.

## Key Config Knobs

All under `generator.inference_engine.*`:
- `enforce_eager` (bool, default true)
- `gpu_memory_utilization` (float, default 0.8)
- `max_num_batched_tokens` (int, default 8192)
- `max_num_seqs` (int, default 1024)
- `enable_prefix_caching` (bool, default true)
- `enable_chunked_prefill` (bool, default true)
- `distributed_executor_backend` ("ray" or "mp")
- `engine_init_kwargs` (dict, pass-through to vLLM EngineArgs)
