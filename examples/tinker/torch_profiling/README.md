# Tinker torch.profiler Example

This example shows how to profile a running SkyRL Tinker server: the client trains normally and brackets a few optim steps with `/start_profiling` and `/stop_profiling`.

Profiling is driven over plain HTTP rather than through the Tinker SDK, so a long-lived server can be profiled at any point without a restart.

Setup:
- base model: `Qwen/Qwen3-0.6B`
- backend: `--backend fsdp` with LoRA
- traces written to `$EXPORT_DIR` (default `/tmp/skyrl_traces`), one directory per profiling session

## 1. Start the Tinker API server

```bash
bash examples/tinker/torch_profiling/run_tinker_server.sh
```

`--torch-profiler` is what enables the endpoints; without it they return `404`. To write traces elsewhere, including cloud storage:

```bash
EXPORT_DIR=s3://my-bucket/skyrl-traces bash examples/tinker/torch_profiling/run_tinker_server.sh
```

For a cloud `export_dir` each closed window is uploaded as it is produced and the local copy dropped, so all ranks' traces collect in one place.

## 2. Run the profiling demo client

```bash
TINKER_API_KEY=tml-dummy uv run --extra tinker --with torch --with transformers \
    python examples/tinker/torch_profiling/profile_training_demo.py
```

The client runs a couple of unprofiled warmup steps, starts profiling, runs more steps, then stops. Traces land in `{export_dir}/{global_step}_{tag}` and can be opened in `chrome://tracing`, Perfetto, or [HolisticTraceAnalysis](https://github.com/facebookresearch/HolisticTraceAnalysis).

## Endpoints

| Endpoint | Body | Purpose |
|----------|------|---------|
| `POST /start_profiling` | `model_id`, `global_step`, `export_path_extra`, `schedule_options`, `profile_options`, `overwrite` | Claim the profiling slot and start a session |
| `POST /stop_profiling` | `model_id` | Stop, flush the open window, and upload |
| `GET /profiling_status` | — | `active`, `model_id`, `export_path`, `step`, `error` |

## Notes

- **The profiler advances one step per `optim_step`**, for the requesting `model_id` only. `schedule_options` is therefore expressed in optim steps and is passed to [`torch.profiler.schedule`](https://docs.pytorch.org/docs/stable/profiler.html#torch.profiler.schedule); `profile_options` is passed to [`torch.profiler.profile`](https://docs.pytorch.org/docs/2.14/profiler.html#torch.profiler.profile).
- **A trace is written when a window closes**, on the transition out of the last active step — not during it. Run at least one step past `active`, or call `/stop_profiling`, which flushes an open window.
- **One session at a time, server-wide.** A second `/start_profiling` gets a `409` while one is running. Only the owning `model_id` may stop it.
- **Profiling sessions expire.** `max_session_duration_sec` (default 7200) releases the profiling slot if a client never calls `/stop_profiling`.
- **Re-running with the same `global_step`** is rejected with a `409` so an earlier capture is not overwritten. Pass a different `export_path_extra`, or `overwrite: true`.
- **Only policy workers are profiled**, on the ranks given by `ranks` at startup. Inference engines are not.
- **`with_stack` is off by default here.** Recording a Python stack for every operator adds substantial per-step overhead — enough to distort the step timings you are trying to measure — and inflates traces several-fold, which slows the upload and makes them harder to open. Turn it on when you specifically need to attribute kernels to Python call sites.
