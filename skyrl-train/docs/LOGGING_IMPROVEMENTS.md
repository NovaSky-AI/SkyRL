# Logging Improvements for SkyRL

## Goal
Separate infrastructure logs from training progress logs:
- **Training progress** (config, steps, rewards, metrics, timings) → stdout
- **Infrastructure logs** (vLLM engine, model loading, KV cache) → log file only
- **SKYRL_LOG_LEVEL=DEBUG** → show all logs on stdout (for debugging)

## Implementation

### Approach: OS-level stdout/stderr redirection with `os.dup2`

The solution redirects stdout/stderr at the OS level for specific Ray actors (vLLM engines, workers), while leaving the training entrypoint unredirected so training progress reaches stdout.

### Files Modified

1. **`skyrl_train/env_vars.py`** - Added environment variables:
   - `SKYRL_LOG_DIR` - Base directory for logs (default: `/tmp/skyrl-logs`)
   - `SKYRL_LOG_LEVEL` - Log level; DEBUG shows all logs on stdout

2. **`skyrl_train/utils/ray_logging.py`** (new file) - Helper module:
   ```python
   def redirect_actor_output_to_file():
       """Redirect stdout/stderr to log file."""
       log_file = os.getenv("SKYRL_LOG_FILE")
       if log_file:
           log_fd = open(log_file, "a", buffering=1)
           os.dup2(log_fd.fileno(), sys.stdout.fileno())
           os.dup2(log_fd.fileno(), sys.stderr.fileno())
   ```

3. **`skyrl_train/utils/utils.py`**:
   - `initialize_ray()` - Sets up log file path and passes to workers via runtime_env
   - `configure_ray_worker_logging()` - Only configures loguru (no redirect here)

4. **`skyrl_train/inference_engines/vllm/vllm_engine.py`**:
   - `VLLMInferenceEngine.__init__()` - Calls `redirect_actor_output_to_file()`
   - `AsyncVLLMInferenceEngine.__init__()` - Calls `redirect_actor_output_to_file()`

5. **`skyrl_train/workers/worker.py`**:
   - `BaseWorker.__init__()` - Calls `redirect_actor_output_to_file()`

### How It Works

1. `initialize_ray()` creates log directory and sets `SKYRL_LOG_FILE` env var
2. `log_to_driver=True` allows Ray to forward actor output to driver stdout
3. vLLM engines call `redirect_actor_output_to_file()` in `__init__` → logs go to file instead of being forwarded
4. Workers call `redirect_actor_output_to_file()` in `__init__` → logs go to file
5. `skyrl_entrypoint` does NOT redirect → training progress reaches stdout

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SKYRL_LOG_DIR` | `/tmp/skyrl-logs` | Base directory for log files |
| `SKYRL_LOG_FILE` | (auto-set) | Full path to infra.log (set by initialize_ray) |
| `SKYRL_LOG_LEVEL` | `INFO` | Log level; `DEBUG` shows all logs on stdout |

## Results

### Before
- 300+ lines on stdout including vLLM model loading, KV cache allocation, tensor parallel setup, etc.

### After
- **Stdout**: Config dump, dataset loading, training progress (steps, metrics, rewards)
- **Log file** (`/tmp/skyrl-logs/{run_name}/infra.log`): vLLM engine logs, worker initialization, model loading

### Known Limitations

1. **Ray `(raylet)` logs still appear on stdout** - These are Ray system logs that occur before our actors start. They're relatively few lines and not the noisy vLLM output.

2. **Ray dedup messages** - Ray may show "repeated Nx across cluster" messages which are informational.

## Usage

```bash
# Normal run - training progress on stdout, infra logs to file
bash examples/gsm8k/run_gsm8k.sh

# Debug mode - all logs on stdout
SKYRL_LOG_LEVEL=DEBUG bash examples/gsm8k/run_gsm8k.sh

# Custom log directory
SKYRL_LOG_DIR=/path/to/logs bash examples/gsm8k/run_gsm8k.sh

# View infrastructure logs
tail -f /tmp/skyrl-logs/{run_name}/infra.log
```
