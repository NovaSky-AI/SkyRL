# GLM training checks and profiling

`run_client.run()` shows the benchmark: create → initial publish/sample →
repeat reference scoring, GSPO forward/backward, optimizer, publish/sample →
checkpoint/unload. The first update is warmup; subsequent updates are measured.
Each update uses two fixed full-context sequences with opposite synthetic advantages.
This checks mechanics and latency, **not learning or LoRA numerical agreement**.

## Run

Use an owned Ray cluster and the same frozen checkout/environment on every node.
Download the model first. State/adapter paths must be shared; database and traces use local scratch.
Start uv-managed Ray with `--block`. For GLM, set
`SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S=1200` and
`SKYRL_WORKER_NCCL_TIMEOUT_IN_S=1800` before starting Ray on every node.

```bash
export RAY_ADDRESS=auto
uv run --isolated --extra tinker --extra megatron python examples/tinker/glm53/run_server.py glm53-32k-2n \
  --model-path /shared/models/glm53 --state-dir /shared/glm-check \
  --database-path /local/glm.db --profile-dir /local/glm-traces

timeout --signal=TERM --kill-after=30s 2h \
  uv run --isolated --extra tinker examples/tinker/glm53/run_client.py \
  --model-path /shared/models/glm53 --context 32768 --batch-size 2 --steps 3 \
  --output-dir /local/glm-results
```

Use fresh state/output paths and create parent directories. Keep the explicit `python` in the server command.
`run_server.py --help` lists profiles; `--print-config` renders without GPUs.
The Qwen3-0.6B smoke profile needs two GPUs. Other GLM/context profiles require their own qualification.

## Results

The client saves exact inputs, replay batches, model metadata and per-phase JSONL.
Trainer traces include CUDA/memory events on every rank. Optimizer metrics separate
dispatch from trace processing; the full optimizer API duration still includes both.
For vLLM traces, enable its profiler and pass `--inference-profile-url` or
`--inference-profile-url-file` (read after initial publication). Capture-control/export
time is outside publication/sample timings. Verify trace files, not just HTTP success.

Short samples do not prove full-context inference. OOM trace export is best-effort;
cold loading, SIGKILL and failed exports are not covered. Unload does not release the
deployment: its owner must enforce deadlines and tear it down.

Completed GLM-5.3 32K receipt: source `97e14ca42d85539958a0c40f72f0f43b4dce42dc`,
model `zai-org/GLM-5.3-BF16` revision `304b8051cfb2b260b61ce0cbe330e02a98e73639`,
image `novaskyai/skyrl-train-ray-2.57.0-py3.12-cu13.0-megatron@sha256:d3efc4bc84b9013c61f320a470c04d7cca39ab09176f96c571a40c40b0cf4edd`.
New instrumentation/source needs its own validation; that receipt is not credited to later revisions.
