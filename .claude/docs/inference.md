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

## Fireworks (external generation, eval-only)

`generator.inference_engine.backend=fireworks` sends generation to the external Fireworks endpoint
via `FireworksInferenceClient` (`inference_servers/fireworks_client.py`, built on the `fireworks-ai`
SDK — install the `fireworks` extra). Token-in/token-out is preserved: prompts are sent as raw token
ids and Fireworks returns the generated `token_ids` (`return_token_ids`), so the stock
`SkyRLGymGenerator` works unchanged.

- **Eval-only**: accepted only by `skyrl.train.entrypoints.main_generate` (`EvalOnlyEntrypoint`
  overrides `get_inference_client`); `BasePPOExp.get_inference_client` raises for any non-vllm
  backend, and the client raises on weight-sync methods.
- **Config** (all under `generator.inference_engine.*`): `run_engines_locally=false`,
  `served_model_name` (the Fireworks model id, e.g. `accounts/fireworks/models/gpt-oss-20b`),
  `api_key`, and `hf_tokenizer_name` (the served model's HF tokenizer, e.g. `openai/gpt-oss-20b`;
  only settable with this backend — `EvalOnlyEntrypoint.get_tokenizer` loads it instead of
  `trainer.policy.model.path`, which this backend does not use) are required; `external_proxy_url`
  is optional and is the server root **without** `/v1` (the SDK appends `/v1/completions`;
  defaults to `https://api.fireworks.ai/inference`).
- **Tokenizer pairing**: `hf_tokenizer_name` must be the served model's tokenizer — token ids are
  consumed raw by the server, so a mismatch degrades generations silently instead of erroring.
- **Sampling params**: converted by `get_fireworks_sampling_params` — vLLM-only keys
  (`min_tokens`, `skip_special_tokens`, `include_stop_str_in_output`) are dropped with a warning.
  Verified against the live endpoint (gpt-oss-20b): a matched `stop` string is excluded from
  Fireworks' `text` field but its tokens are **included in `token_ids`**, and a natural stop ends
  with the EOS token id. Since the client builds `response_ids` from `token_ids` and decodes
  `responses` locally, the effective behavior matches the vLLM path
  (`include_stop_str_in_output=True`, `skip_special_tokens=True`), including
  `append_eos_token_after_stop_str_in_multi_turn` detection. `min_tokens` is the only true gap
  (no Fireworks equivalent; zero-length completions are possible).
- Example: `examples/eval/run_eval_fireworks.sh`. Tests:
  `tests/backends/skyrl_train/inference_servers/test_fireworks_client.py` (offline, mocked HTTP
  transport through the real SDK).

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

## Placement
- Colocated: vLLM and training workers (FSDP/Megatron) are placed on the same set of GPUs. We offload/backload each component as needed. During weight syncing, model weights from vLLM as well as model weights from the training workers remain on GPU
- Non-colocated: vLLM and training workers (FSDP/Megatron) are placed on a different set of GPUs. This reduces the number of available GPUs per component by half, but is in fact the preferred setup for agentic RL with SkyRL. This is because non-colocated setups allow for asynchronous training, where training and inference can progress together. Inference is typically dominated by a long tail of stragglers, and is also typically the time consuming component, and thus using half the number of GPUs doesn't affect inference time for a batch as much.
