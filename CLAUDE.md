# SkyRL + Harbor Log Separation — Session Context

## What was done

Added the ability to separate vLLM server-side logs from SkyRL client-side logs into
dedicated files, controlled by environment variables. Tested on a live 14-minute training
run with 4xH100 GPUs on Anyscale.

## Repo layout

- **Working (dirty) repo**: `/home/ray/default/SkyRLHarbor3` — has the changes plus
  other unrelated in-flight modifications (debug prints, config tweaks).
- **Clean repo with commit**: `/home/ray/default/SkyRL-clean` — fresh clone from
  `https://github.com/NovaSky-AI/SkyRL`, commit `68f77a8` contains only the log
  separation changes.
- **Harbor** (dependency): `/home/ray/default/harbor` — used via `Trial()` / `Trial.run()`
  for rollouts. Not modified.

## Files changed (3 files)

### 1. `skyrl-train/skyrl_train/utils/utils.py`
- `_is_vllm_log(record)` / `_is_skyrl_log(record)` — Loguru filter functions. Check
  `record["extra"]["stdlib_logger"]` (for intercepted stdlib logs) and `record["name"]`
  (for direct Loguru calls) to classify logs as vLLM or SkyRL.
- `configure_ray_worker_logging()` — added optional file sinks (when env vars are set)
  and modified `_InterceptHandler.emit()` to use `logger.bind(stdlib_logger=record.name)`
  to preserve the origin logger name.
- `prepare_runtime_environment()` — propagates `VLLM_LOG_FILE`, `SKYRL_LOG_FILE`,
  `LOG_LEVEL` to Ray worker env vars.

### 2. `skyrl-train/skyrl_train/inference_engines/vllm/vllm_engine.py`
- `BaseVLLMInferenceEngine.__init__()` — calls `configure_ray_worker_logging()` at top
  (vLLM engine actors otherwise have no Loguru setup). After `self._create_engine()`,
  iterates over all `vllm.*` stdlib loggers and sets `handlers.clear()` +
  `propagate=True`. This is necessary because vLLM sets `propagate=False` and adds its
  own handlers during engine init, which bypass our root-level Loguru intercept.

### 3. `skyrl-train/examples/terminal_bench/run_codecontest.sh`
- Example env var setup: `VLLM_LOG_FILE` and `SKYRL_LOG_FILE`.

## Usage

```bash
export VLLM_LOG_FILE="/home/ray/logs/vllm.log"
export SKYRL_LOG_FILE="/home/ray/logs/skyrl.log"
bash run_codecontest.sh 2>&1 | tee /home/ray/logs/combined.log
```

If the env vars are not set, behavior is identical to upstream (all to stderr).

## Architecture / how it works

### Logging flow in SkyRL

```
Main process (head node)
  └─ initialize_ray(cfg)        → sets runtime env vars for workers
       └─ ray.init()

Worker process (skyrl_entrypoint)
  └─ _setup_trainer()
       ├─ get_inference_client() → creates 4x AsyncVLLMInferenceEngine Ray actors
       │    └─ Each actor's __init__:
       │         1. configure_ray_worker_logging()   ← sets up Loguru + file sinks
       │         2. self._create_engine()            ← vLLM sets up its own loggers
       │         3. Reroute vllm.* loggers           ← undo vLLM's propagate=False
       │
       ├─ get_generator()        → TerminalBenchGenerator (uses Harbor Trial.run())
       └─ get_trainer()          → RayPPOTrainer.__init__
            └─ configure_ray_worker_logging()   ← also here for the entrypoint worker
```

### What each log file captures

- **vllm.log**: `vllm.entrypoints.logger` (request received/added), `vllm.v1.engine.async_llm`
  (request lifecycle), `vllm.entrypoints.chat_utils` (chat template detection),
  `vllm.entrypoints.openai.serving_chat` (errors like context length exceeded).
- **skyrl.log**: `skyrl_train.trainer` (training steps, weight sync, generate timing),
  `skyrl_train.workers.worker` (actor group init, mesh ranks),
  `examples.terminal_bench.generator.terminal_bench_generator` (trial results, retries,
  rewards), `harbor.*` (trial execution), `skyrl_train.utils.rate_limiter` (rate limit
  config).

## Caveats / known limitations

### 1. EngineCore subprocess logs are NOT captured in vllm.log
vLLM v1's `EngineCore` runs as a **separate subprocess** (forked via multiprocessing).
Its logs (model loading, KV cache allocation, checkpoint shard progress, flashinfer
autotuning) go directly to the subprocess's stdout/stderr, which Ray captures via actor
log forwarding. These appear in `combined.log` with the prefix:
```
(AsyncVLLMInferenceEngine pid=XXXX) (EngineCore_DP0 pid=YYYY) INFO ...
```
They bypass Python's `logging` module entirely, so our Loguru intercept cannot catch them.
Neither verl (`/home/ray/default/verl`) nor slime (`/home/ray/default/slime`) solve this.
verl suppresses them via `VLLM_LOGGING_LEVEL: "WARN"`. A future fix could redirect the
subprocess's stderr via `VLLM_LOGGING_CONFIG_PATH` or by patching the subprocess launch.

### 2. Multiple processes write to the same log file
With 4 vLLM engine actors, all 4 write to the same `vllm.log`. Loguru's `enqueue=True`
provides thread safety within a process, but not cross-process. On Linux, small writes
(<4KB) to the same file are atomic, so lines won't interleave in practice. For very high
throughput, consider per-rank log files (e.g. `vllm_rank_{rank}.log`).

### 3. `configure_ray_worker_logging()` is called multiple times
It's called in: `RayPPOTrainer.__init__`, `Worker.__init__` (policy/critic/ref actors),
and `BaseVLLMInferenceEngine.__init__` (vLLM actors). Each call does `logger.remove()`
+ re-adds sinks, which is idempotent but slightly wasteful. Could be guarded with a
module-level flag if it becomes a problem.

### 4. Startup sequence matters
vLLM engine actors are created BEFORE `RayPPOTrainer.__init__()` (see `_setup_trainer()`
in `main_base.py`). That's why we had to add `configure_ray_worker_logging()` to
`BaseVLLMInferenceEngine.__init__` — the trainer's call comes too late for vLLM actors.

### 5. vLLM logger rerouting is version-sensitive
The `lgr.handlers.clear(); lgr.propagate = True` loop targets all `vllm.*` loggers.
If a future vLLM version changes its logging setup (e.g. adds handlers lazily), this
may need adjustment. Tested with vLLM 0.13.0.

## Test results (14-min run)

| File | Lines | Size | Content |
|------|-------|------|---------|
| combined.log | 1304 | 876 KB | Everything (same as before) |
| vllm.log | 91 | 14 KB | Only vLLM actor-process logs |
| skyrl.log | 672 | 436 KB | Only SkyRL client-side logs |

Zero cross-contamination between vllm.log and skyrl.log.

## Other frameworks' approaches (for reference)

- **verl** (`/home/ray/default/verl`): Suppresses vLLM logs via `VLLM_LOGGING_LEVEL: "WARN"`
  in Ray runtime env. No file-based separation. Uses basic Python `logging.basicConfig()`.
  Relevant file: `verl/trainer/constants_ppo.py`.
- **slime** (`/home/ray/default/slime`): Uses SGLang (not vLLM). Server subprocess launched
  via `multiprocessing.Process`. No log separation. Router log level set to `"warn"`.
  Relevant file: `slime/backends/sglang_utils/sglang_engine.py`.

## Environment

- 8xH100-80G on Anyscale (used 4 GPUs for this test)
- Python 3.12, Ray, vLLM 0.13.0, Loguru
- SkyRL upstream: `https://github.com/NovaSky-AI/SkyRL`
- Model: Qwen/Qwen3-8B
