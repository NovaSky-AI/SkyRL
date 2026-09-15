# GLM training checks and profiling

The benchmark uses two fixed full-context sequences and one reference forward,
one GSPO forward/backward, one optimizer step, and one publication/sample per update.
It checks mechanics and timing, not learning or LoRA numerical agreement.

Cold work is recorded separately: model creation, input preparation, initial
publication and initial sample. The first update is warmup; later updates are
measured. Initial publication includes lazy vLLM startup. Server timing logs
separate `sampler_inference_init` (with a cold flag) from `sampler_weight_sync`;
the client publication envelope additionally includes RPC and scheduling overhead.
Service boot before the client connects is outside these measurements.

## Run

Use an owned Ray cluster and the same frozen checkout/environment on every node.
Download the model first. State/adapter paths must be shared; database and traces
use scratch. Start uv-managed Ray with `--block`. For GLM, set
`SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S=1200` and
`SKYRL_WORKER_NCCL_TIMEOUT_IN_S=1800` before starting Ray on every node.

```bash
export RAY_ADDRESS=auto
uv run --isolated --extra tinker --extra megatron python examples/tinker/glm53/run_server.py glm53-32k-2n \
  --model-path /shared/models/glm53 --state-dir /shared/glm-check \
  --database-path /local/glm.db --profile-dir /local/glm-traces --profile-ranks 0

uv run --isolated --extra tinker python examples/tinker/glm53/run_client.py \
  --model-path /shared/models/glm53 --context 32768 --batch-size 2 --steps 4 \
  --profile-mode none --receipt-metadata /local/receipt-metadata.json \
  --output-dir /local/glm-results
```

`receipt-metadata.json` is validated before the first API call and supplies the
run ID, model name/revision/config SHA-256, exact SkyRL commit and dirty state,
immutable image URI/digest, arm, transport implementation/revision, and profiler
mode. Generate it from the versioned schema published by this branch; do not
reuse it after any provenance or capture-mode change. A baseline file has this
shape:

```json
{
  "run_id": "unique-run-id",
  "model": {"name": "model-name", "revision": "immutable-revision", "config_sha256": "<64 hex>"},
  "provenance": {
    "repository_url": "https://github.com/NovaSky-AI/SkyRL.git",
    "commit": "<40 hex>",
    "dirty": false,
    "image_uri": "registry.example.com/skyrl:immutable-tag",
    "image_digest": "sha256:<64 hex>"
  },
  "arm": "baseline_a",
  "transport": {"implementation": "safetensors", "revision": "baseline-v1"},
  "profiler_mode": "none"
}
```

`--steps 4` means one warmup plus three measured updates. Use new output paths.
The Qwen3-0.6B smoke profile needs two GPUs and exercises the same client flow.
The 256K profiles are candidates, not demonstrated memory/capacity qualifications.
`run_server.py --print-config` renders the backend configuration without GPUs.

## Capture modes

All modes use the same inputs and number/order of training and publication calls:

- `none` (default): no trainer or receiver capture. Use this for latency comparisons.
- `trainer`: bracket each warmup/measured update with the existing Tinker
  `/start_profiling` and `/stop_profiling` endpoints. Cold work stays outside
  capture. The server owns trace location/ranks; the client owns schedule/options.
  Stop flushes the publication work after the optimizer boundary, with no extra
  optimizer update. Each update uses its own export path.
- `receiver`: capture each warmup/measured publication and sample on vLLM.
  Enable vLLM profiling and supply `--inference-profile-url` or
  `--inference-profile-url-file`. The endpoint is resolved after cold publication.

Trainer and receiver captures are separate modes. The server enables trainer
endpoints with `--torch-profiler`; backend static profiler overrides are rejected.
The client waits for worker acknowledgements, stops an owned session before
unload, and records cleanup errors without replacing the original training error.

## Evidence and limits

The client writes exact inputs, replay batches, model metadata, capture mode,
cold/measured phase classifications and per-phase JSONL. Optimizer metrics separate
dispatch from engine-owned trace processing. Start/stop/export timings are separate
phase records. Profiling still perturbs the work it encloses; do not compare a
profiled latency against an unprofiled one as a speedup.

Trainer files use `rank<N>_w<M>.pt.trace.json[.gz]` under the returned export path.
Verify complete trace files and CUDA/memory events, not only HTTP success.
Megatron schedule timers are enabled by the live trainer profiler, not static
backend configuration. Kernel-summary aggregation can be disabled without
disabling window advancement or cloud uploads.

Samples generate eight tokens and do not qualify full-context inference. This
example does not restart profilers after OOM or guarantee traces after SIGKILL.
Unload removes the model; the deployment owner must enforce deadlines and release
the underlying service. Historical measurements remain associated with their
original source/configuration and must not be relabeled as this client revision.
