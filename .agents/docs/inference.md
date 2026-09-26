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
- **Config**: `enable_pd=true` and `num_prefill` (decode engines = `num_engines - num_prefill`). Requires a
  `kv_transfer_config` with a supported P2P transfer connector (see below).
- **Server groups**: Separate prefill and decode `ServerGroup`s, one per engine.
- **P2P transfer connectors** (`get_pd_p2p_connector_name` in `inference_servers/utils.py`): two flavors are
  supported, either bare or wrapped in a vLLM `MultiConnector` (alongside store connectors such as
  `MooncakeStoreConnector` for KV offloading):
    - `NixlConnector` — pull-based default (NIXL side channel).
    - `MooncakeConnector` — push-based (bootstrap-server handshake). The router runs in `kv_connector=mooncake` mode and
      is given each prefill server's mooncake bootstrap server port. The `MultiKVConnectorPromMetrics.observe` monkeypatch
      (`patches/vllm/patch_multi_connector_stats.py`) is applied to avoid an assert-crash when a child connector
      reports stats without Prometheus metrics.
- **Role-specific engine kwargs**: `prefill_init_kwargs` and `decode_init_kwargs` are per-role pass-through vLLM engine
  kwargs (e.g. a different `all2all_backend`, or per-role `kv_role`) applied on top of the base args by
  `get_pd_cli_args`. They are **mutually exclusive** with `engine_init_kwargs` and require `enable_pd=true`; when used,
  both must be set and each must carry its own `kv_transfer_config` (enforced in `validate_inference_engine_cfg`).
- **Troubleshooting — configuring Mooncake GPU memory registration.** `WITH_NVIDIA_PEERMEM` selects how Mooncake pins
  KV-cache GPU memory. Decide with:

  ```
  lsmod | grep nvidia_peermem                          # is peer-memory available?
  cat /sys/class/infiniband/*/ports/1/link_layer       # InfiniBand | Ethernet (= RoCE)
  ```

  1. **`nvidia_peermem` loaded** (IB or RoCE alike) → leave the default. Mooncake registers via `ibv_reg_mr()`.
  2. **`nvidia_peermem` absent or unloadable** — module not shipped, or the container lacks `CAP_SYS_MODULE`
     (e.g. GKE/COS nodes, driver 580+) → set **`WITH_NVIDIA_PEERMEM=0`** to use the dmabuf path
     (`cuMemGetHandleForAddressRange` + `ibv_reg_dmabuf_mr`). Requires a GPU/driver with dma-buf support.
  3. **Got it wrong?** Symptom is a *silent hang, not an error*: hundreds of
     `Failed to register memory ...: Bad address [14]` at engine startup, servers still report healthy, then the first
     request stalls forever with `Memory region not registered by any active device(s)`.


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

## Startup: overlapping engine boot

`_setup_trainer` builds the inference client with `wait_for_ready=False`: `create_inference_servers`
launches the engines, reads their URLs from the actors' constructor state
(`ServerGroup.get_server_infos_nowait`), and constructs but does not start the router (the router
waits for its backends to pass health checks, so it can only start once they are up; its port is
reserved in `__init__`, so the proxy URL is known before then). The training workers are then built
while the engines load and compile, and `InferenceServerSetup.wait_until_ready()` resolves the start
refs and starts the router before `train()`. Entrypoints that call `get_inference_client()` directly
(`serve`, `main_generate`) still get fully-ready engines.

`RayPPOTrainer.build_models` is split into `create_actor_groups` (spawns the policy/ref/critic groups
concurrently in threads, each blocking only on its own actors) and `init_models`. Actor creation is
where each worker process starts, re-execs through the uv runtime env, imports the backend (tens of
seconds for Megatron + TransformerEngine) and joins its process group; it costs no GPU memory beyond
a CUDA context. The entrypoint runs the spawn while the engines start in both placements. Colocated
runs then wait for the engines, sleep them, and only then load the models, because the engines must
be healthy to be slept; non-colocated runs load immediately on their own GPUs.
`placement.overlap_worker_spawn=false` restores the fully sequential order — worth doing if
`gpu_memory_utilization` is pushed to the limit, since the trainer processes otherwise hold a CUDA
context on the shared GPUs during vLLM's memory profiling and shrink the KV cache by that much.

The info RPCs must be **resolved** before `start` is submitted, not merely submitted first: the
engine build runs synchronously inside the async `start` and holds the actor's event loop until the
engine is healthy, so an info RPC that lands behind it waits for the whole engine startup and the
overlap silently degrades to the sequential order.

## Placement
- Colocated: vLLM and training workers (FSDP/Megatron) are placed on the same set of GPUs. We offload/backload each component as needed. During weight syncing, model weights from vLLM as well as model weights from the training workers remain on GPU
- Non-colocated: vLLM and training workers (FSDP/Megatron) are placed on a different set of GPUs. This reduces the number of available GPUs per component by half, but is in fact the preferred setup for agentic RL with SkyRL. This is because non-colocated setups allow for asynchronous training, where training and inference can progress together. Inference is typically dominated by a long tail of stragglers, and is also typically the time consuming component, and thus using half the number of GPUs doesn't affect inference time for a batch as much.
