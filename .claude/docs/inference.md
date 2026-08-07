# Inference

For training-to-inference weight transfer (`NewInferenceWorkerWrap`, broadcast vs. CUDA IPC, lifecycle), see [`weight_sync.md`](weight_sync.md).

## Architecture

- Key abstractions: `RemoteInferenceClient` , `ServerGroup`, `VLLMServerActor`, `VLLMRouter`
- `RemoteInferenceClient` interacts with HTTP endpoints: 
    - **Data plane**: Interact with router for completions requests.
    - **Control plane**: Fan-out to individual server URLs for weight sync, pause/resume.
- Shared inference interfaces and types live in `inference_servers/base.py` (`InferenceEngineInterface`, `InferenceEngineInput`/`Output`, `ConversationType`); shared helpers (`build_engine_runtime_env`, `get_sampling_params_for_backend`) live in `inference_servers/engine_utils.py`.

## vLLM Router

- `VLLMRouter` in `skyrl/backends/skyrl_train/inference_servers/vllm_router.py` wraps a child process running `vllm-router`. 

## PD Disaggregation

Prefill-Decode disaggregation:
- **Config**: `enable_pd=true` and `num_prefill` passed to `ServerGroup` constructor. Requires a `kv_connector`
- **Server groups**: Separate prefill and decode `ServerGroup`s, one per engine.

## Engine Fault Tolerance

`generator.inference_engine.fault_tolerance.enabled=true` lets training continue when
an inference server dies mid-run. Off by default; when off, every path below is inert.

**Detection is reactive, not a monitor.** There is no timer and no background task. A
data-plane failure (transport error, or a router 502/503/504) triggers **one**
`GET /health` probe across `active_server_urls`; non-responders are marked dead and
`POST {router}/remove_worker?url=` de-routes them. Concurrent failures share one probe
(callers pass the `membership_generation` they failed under), so a batch of 512
trajectories failing at once costs one probe.

**Two URL views on `RemoteInferenceClient`, and the difference matters:**

| | meaning | used for |
|---|---|---|
| `server_urls` | PROVISIONED, never mutated, never compacted | identity + geometry (`replica_rank = index // dp`, `num_consumers`, `get_world_size`) |
| `active_server_urls` | the subset not yet found dead | anything talking to backends now (control plane, weight sync) |

Deriving geometry from the degraded list is the one dangerous mistake available here:
it would silently re-map every surviving consumer's weight slice.

**Recovery is per-trajectory.** A trajectory that dies on a transport error is re-run
from scratch (fresh env, salted session id) up to `max_trajectory_retries`; non-transport
errors still fail fast. `_gather_trajectories` cancels and drains the siblings when one
trajectory fails — that part is unconditional, FT or not.

Requires `weight_sync_backend="sharded_rdt"`, `colocate_all=false`,
`data_parallel_size=1`, `enable_pd=false`, `run_engines_locally=true` and
`num_engines>=2`; `_validate_inference_fault_tolerance_cfg` explains each refusal. See
[`weight_sync.md`](weight_sync.md) for the degraded weight sync.

Part 1 only **survives** a death — nothing is restarted, and the dead slot stays dead
for the rest of the run. Design doc: `~/default/rdt_fault_tolerance.md`.

### Router worker management

`vllm-router` exposes runtime membership routes, wrapped on both `VLLMRouter` (sync,
driver-side) and `RemoteInferenceClient` (async). Verified against 0.1.14.post1:

- `POST /add_worker?url=<backend>` — **blocks** until the backend answers `/health`, so
  it cannot pre-register a URL that is not serving yet. `400 already exists` for a known
  worker (normalized to success).
- `POST /remove_worker?url=<backend>` — idempotent; `200` even for a URL it does not have.
- `GET /list_workers` → `{"urls": [...]}`.

`url` is the only accepted query-parameter spelling; anything else is
`400 missing field \`url\``. add/remove answer with plain text, not JSON.

## Key Config Knobs

All under `generator.inference_engine.*`:
- `enforce_eager` (bool, default true): With `enforce_eager=false`, there can be more mismatch between inference logprobs and trainer logprobs. It is recommended to use off policy correction methods like Truncated Importance Sampling (see `docs/content/docs/algorithms/off_policy_correction.mdx` for details) to prevent logprobs drift. 
- `gpu_memory_utilization` (float, default 0.8)
- `max_num_batched_tokens` (int, default 8192)
- `max_num_seqs` (int, default 1024)
- `enable_prefix_caching` (bool, default true)
- `enable_chunked_prefill` (bool, default true)
- `distributed_executor_backend` ("ray" or "mp")
- `engine_init_kwargs` (dict, pass-through to vLLM EngineArgs)
- `fault_tolerance.*` (see [Engine Fault Tolerance](#engine-fault-tolerance)): `enabled`
  (default false), `health_probe_timeout_s` (5.0), `request_timeout_s` (900.0),
  `max_trajectory_retries` (2), `min_live_engines` (1), `stall_timeout_s` (300.0)

## Placement
- Colocated: vLLM and training workers (FSDP/Megatron) are placed on the same set of GPUs. We offload/backload each component as needed. During weight syncing, model weights from vLLM as well as model weights from the training workers remain on GPU
- Non-colocated: vLLM and training workers (FSDP/Megatron) are placed on a different set of GPUs. This reduces the number of available GPUs per component by half, but is in fact the preferred setup for agentic RL with SkyRL. This is because non-colocated setups allow for asynchronous training, where training and inference can progress together. Inference is typically dominated by a long tail of stragglers, and is also typically the time consuming component, and thus using half the number of GPUs doesn't affect inference time for a batch as much.
