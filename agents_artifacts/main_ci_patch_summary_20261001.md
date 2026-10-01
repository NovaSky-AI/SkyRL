# Main CI patch audit (2026-10-01)

Branch: `fix/outstanding-main-ci-20260930` | PR: [#2363](https://github.com/NovaSky-AI/SkyRL/pull/2363) | E2E training validation SHA: `2fc91fbb` | Base: `c6cce532`.

This separates observed failures from inferred causes and operational mitigations. A green test on an earlier branch SHA is not claimed as exact-SHA validation of the final patch.

## Root-cause fixes

### vLLM worker-local model runner registry

- **Files:** `skyrl/backends/skyrl_train/patches/vllm/patch_model_runner_registry.py`, `skyrl/backends/skyrl_train/inference_servers/new_inference_worker_wrap.py`.
- **Observed:** On main at `5e580e8`, [SkyRL-GPU](https://github.com/NovaSky-AI/SkyRL/actions/runs/36755420630) and [H100-GPU-CI](https://github.com/NovaSky-AI/SkyRL/actions/runs/36755420577) reached `/finish_weight_update` and failed to import `skyrl.backends.skyrl_train.patches.vllm.patch_model_runner_registry` in vLLM workers. The receive-side FP8 scale hook already imported `current_model_runner()` from that missing module.
- **Cause:** The hook requires the process-local `GPUModelRunner`, but vLLM's weight-transfer engine receives the model object, not its runner. The helper file was absent on main, so the import failed before the hook could run. This is a concrete missing-module error, not a timeout.
- **Patch:** Add the registry helper and install its `GPUModelRunner.load_model` hook inside every vLLM worker via `new_inference_worker_wrap.py`. The registry holds a weak reference and is idempotently installed.
- **Validation:** Minimal L4 weight-sync job `prodjob_s5171d4ap1svn8m7xxbyrn25vw` succeeded; [SkyRL-GPU](https://github.com/NovaSky-AI/SkyRL/actions/runs/36835088606) and [H100-GPU-CI](https://github.com/NovaSky-AI/SkyRL/actions/runs/36835088620) succeeded on branch SHA `1e8b992` (same source fix). Confidence: high.

### Tinker protobuf retrieval of completed futures

- **Files:** `skyrl/tinker/proto_serialization.py`, `tests/tinker/test_proto_serialization.py`.
- **Observed:** The sync Tinker E2E rerun at `7719663f` repeatedly returned HTTP 500 from `retrieve_future`. The API traceback ended at `float(None)` in `_serialize_forward_backward_output`. Read-only inspection of five completed `FORWARD_BACKWARD` futures (request IDs 3371-3375) found `importance_ratio:mean: null`; their total and policy losses were finite.
- **Cause:** The worker emitted a non-finite scalar diagnostic. `ForwardBackwardOutput.model_dump_json()` serialized that value as JSON `null` in the future database. The protobuf encoder then assumed every stored metric was float-convertible. The exact reason the upstream importance-ratio diagnostic became non-finite is not proven. One possible code path is division by zero when a microbatch has zero total `loss_mask`; this needs separate evidence before changing the loss calculation.
- **Patch:** Omit only `None` metrics when encoding a stored future. Finite metrics, loss outputs, gradients, and optimizer steps are unchanged. This repairs the JSON-to-protobuf boundary; it does **not** claim to cure the non-finite diagnostic at its source.
- **Validation:** New round-trip test passes through `model_dump_json()`, JSON parsing, protobuf encoding, and SDK decoding. The full proto test file passed (`18 passed`); the latest PR Tinker CPU check is green. Exact-SHA sync GPU E2E [run 36874991711](https://github.com/NovaSky-AI/SkyRL/actions/runs/36874991711) is still in progress at the time of this note. Confidence: high for the 500, pending for end-to-end convergence.

## Configuration and observability

### Plain GPU E2E wait budget: 5000 to 7200 seconds

- **File:** `.github/workflows/gpu_e2e_ci.yaml`.
- **Observed:** [Main run 36732329780](https://github.com/NovaSky-AI/SkyRL/actions/runs/36732329780) entered `RUNNING` at 15:02:59 UTC and the CI helper returned failure at 16:26:45 UTC while Anyscale still reported `RUNNING`; the driver had recently logged healthy training/checkpoint progress. The helper uses the same 5000-second value for Anyscale runtime and its terminal-state wait. [Branch run 36835088645](https://github.com/NovaSky-AI/SkyRL/actions/runs/36835088645) entered `RUNNING` at 08:17:12 and succeeded at 09:40:36, about **5004 seconds** later: four seconds beyond the old limit.
- **Cause supported by evidence:** The old cap had essentially no headroom for an ordinary successful run. The prior failure was reported by the wait helper while the job was still active, not by a failed accuracy or loss assertion. This explains the CI false negative; it does **not** explain run-to-run throughput variation.
- **Patch:** Raise only this workflow's budget to 7200 seconds. This is an operational limit correction, not a training-code fix. The successful branch run demonstrates the test completes, but a future genuine training hang would now take longer to surface. Confidence: high for the budget mismatch, unknown for throughput variance.

### Fully async Tinker L4 memory budget: 0.8 to 0.7

- **File:** `tests/train/gpu_e2e_test/gsm8k_tinker_fully_async.sh`.
- **Observed:** [Main run 36733926676](https://github.com/NovaSky-AI/SkyRL/actions/runs/36733926676), Anyscale `prodjob_reiwen3usdq5gtwzka4z5my678`, logged CUDA allocation OOM on devices 2 and 3 during startup: 228,589,568 bytes requested with 28,442,624 free on device 2; 216,006,656 requested with 217,186,304 free on device 3. The script set vLLM's GPU-memory-utilization target to 0.8.
- **Cause supported by evidence:** vLLM startup peak at the 0.8 memory target left insufficient free memory on the 22-GiB L4s. The exact source of every allocation is not established; the OOM itself is direct worker-log evidence.
- **Patch:** Set only the fully async CI script's `gpu_memory_utilization` to 0.7, matching the colocated Tinker CI headroom. This reduces KV capacity for this test; it does not alter library defaults.
- **Validation:** [Fully async E2E branch run 36835088671](https://github.com/NovaSky-AI/SkyRL/actions/runs/36835088671) completed successfully on `1e8b992`. Confidence: high that memory pressure caused the startup failure; moderate that 0.7 is the best long-term value.

### Tinker metrics-check dependency isolation

- **Files:** `tests/train/gpu_e2e_test/gsm8k_tinker.sh`, `tests/train/gpu_e2e_test/gsm8k_tinker_fully_async.sh`.
- **Observed:** Sync Tinker job `prodjob_zdpufh6sxhkl2zasssqbtznvld` on `2fc91fbb` completed all 14 training batches (`progress/done_frac=1.0`) with final reward `0.547510` and KL `0.000685`, both within the test thresholds. Only afterward, the `get_summary.py` invocation failed because `uv run --isolated --extra fsdp` tried to fetch the unrelated `transformer-engine-torch` wheel from GitHub and received HTTP 500. Its automatic retry was stopped before another GPU entrypoint ran.
- **Cause:** `get_summary.py` imports only `wandb`, but its invocation resolved the entire FSDP project environment. A transient external wheel failure could therefore mark an otherwise successful training run red. This is a post-training dependency-resolution issue, distinct from the earlier API 500.
- **Patch:** Run the summary script in an isolated no-project environment with `wandb==0.30.0` (the version in `uv.lock`) in both Tinker E2E scripts. No training or assertion threshold changes.
- **Validation:** The minimal command imported `wandb 0.30.0` locally without pulling backend extras; running `get_summary.py --help` in that environment, shell syntax, and diff checks passed. The next exact-SHA workflow rerun is pending. Confidence: high in the cause; final workflow result pending.

### Tinker server log survival

- **File:** `tests/train/gpu_e2e_test/gsm8k_tinker.sh`.
- **Observed:** The sync Tinker client could not retrieve futures, but `server.log` was missing from its run directory. The cookbook's `behavior_if_log_dir_exists=delete` removes that directory after the server has opened the log file. Reading the still-open server stdout descriptor exposed the HTTP 500 traceback.
- **Patch:** Keep the server log next to, rather than inside, the cookbook run directory and tail that same path on failure. This is observability only; it does not make training pass. Exact-SHA E2E validation is pending.

### Uvicorn error logger

- **Files:** `skyrl/utils/log.py`, `tests/tinker/test_uvicorn_log_config.py`.
- **Observed:** In the first sync Tinker rerun, a live `py-spy` sample placed the API server's event-loop thread inside `RichHandler.emit`/syntax rendering while reporting an ASGI exception. The underlying exception was later identified as the protobuf `float(None)` error.
- **Patch:** Route `uvicorn.error` to a plain stream handler; a focused logging-config test passes. This avoids expensive Rich rendering on the API error path and preserves plain traceback output.
- **Scope caveat:** This is defense in depth against error-reporting stalls, **not** the fix for the 500. It affects all Uvicorn users of this config. The PR can drop it if the desired scope is strictly the known protobuf fault; no claim that the E2E needs it after the serializer fix. Confidence: high in the observed stall, lower in its necessity after the primary fix.

### Nested Alembic test environment

- **File:** `tests/tinker/test_db.py`.
- **Observed:** `tests/tinker/test_db.py` launches nested `uv run` processes. Its prior commands omitted `--isolated` and one omitted the Tinker/dev extras, so nested commands could resolve a different environment from the test's isolated parent. The prior main CPU failure was a Tinker API startup timeout, **not** an Alembic assertion.
- **Patch:** Give nested Alembic commands `--isolated --extra tinker --extra dev`. Local `test_db.py` passed (2 tests); a full Linux Tinker CPU Anyscale job and the latest PR Tinker CPU check passed.
- **Scope caveat:** This is reproducibility/test-harness cleanup, not a demonstrated fix for the reported nightly failure. Consider moving it to a separate PR if keeping this repair narrowly scoped.

## Excluded changes and remaining gates

- No Megatron-specific change is in this branch. [Megatron run 36755420482](https://github.com/NovaSky-AI/SkyRL/actions/runs/36755420482) on `5e580e8` passed after the earlier `max_generate_length` fix on main.
- A speculative continuous-sampler timeout and `test_unload_model` teardown hardening were removed because their proposed causes were not established by the before/after runs. Neither is in the current diff.
- The `2fc91fbb` sync Tinker run completed training and met reward/KL thresholds but failed during the external wheel fetch in its final metrics-check command. The summary-command patch above still needs exact-SHA workflow validation. CPU, Tinker CPU, JAX CPU/GPU, gym, skycap, and code-quality checks passed on `2fc91fbb`. The branch should remain draft until the final summary command is validated.
- The non-finite `importance_ratio` producer deserves a separate targeted investigation. Do not infer numerical health solely from a green API retrieval: inspect loss and reward metrics in the E2E run.
