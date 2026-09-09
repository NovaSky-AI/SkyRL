# GLM runtime checks and profiling

Shared runtime profiles; each command has a separate, explicit test purpose.
`run_client.py` measures training mechanics, not learning or sampler parity.
Start with Qwen3-0.6B (one trainer GPU + one inference GPU); it is **not** offered
by hosted Tinker. A hosted comparison needs a common model and matched loss/gradients.

Use an owned Ray cluster, the same pinned checkout/environment on each node, and a
downloaded model. Adapter/checkpoint paths must be shared; database/traces use local scratch.

The completed GLM-5.3 32K profiling receipt used source `97e14ca42d85539958a0c40f72f0f43b4dce42dc`,
`zai-org/GLM-5.3-BF16` revision `304b8051cfb2b260b61ce0cbe330e02a98e73639`, and image
`novaskyai/skyrl-train-ray-2.57.0-py3.12-cu13.0-megatron@sha256:d3efc4bc84b9013c61f320a470c04d7cca39ab09176f96c571a40c40b0cf4edd`.
Use the checkout's frozen lockfile; newer source or model revisions need revalidation.

```bash
export RAY_ADDRESS=auto
uv run --isolated --extra tinker --extra megatron python examples/tinker/glm53/run_server.py qwen3-0.6b \
  --model-path /shared/models/qwen3-0.6b --state-dir /shared/qwen-control \
  --database-path /local/qwen.db --profile-dir /local/qwen-traces

timeout --signal=TERM --kill-after=30s 2h \
  uv run --isolated --extra tinker examples/tinker/glm53/run_client.py \
  --model-path /shared/models/qwen3-0.6b --context 32768 --batch-size 2 --steps 3 \
  --output-dir /local/qwen-results
```

Create the parent directories; use fresh output/state paths. Keep the explicit
`python` in the server command for the API's uv-environment discovery.
Start uv-managed Ray with `--block` so its temporary environment stays alive.
For GLM, set both deadlines before starting Ray on every node:

```bash
export SKYRL_WAIT_UNTIL_INFERENCE_SERVER_HEALTHY_TIMEOUT_S=1200
export SKYRL_WORKER_NCCL_TIMEOUT_IN_S=1800
```

The worker collective deadline also covers ranks waiting while rank 0 exports
and loads the adapter. Keep an outer job deadline; these are not speed optimizations.

`run_client.run()` shows the protocol: create → initial publish/sample →
warmup + two measured updates → checkpoint/unload. Each update is one batched
reference forward, one batched GSPO forward/backward, optimizer, publication and short sample.
Two repeated-text fixtures each score exactly 32,768 positions. Raw advantages are
+1/-1; the API sums them. This is not yet the hosted sequence-mean GSPO workload.

Profiles are in [configs/](configs/); `run_server.py --help` lists GLM variants.
`--print-config` renders the effective server config without starting GPUs.
GLM/256K profiles still need their own qualification.

The client saves exact datums, replay batches and phase JSONL. Every trainer rank
profiles warmup and updates; verify CUDA traces. Optimizer request time includes
trace export; this example disables eager kernel-summary aggregation. OOM
export/restart is best-effort; it does not recover model state.
Cold model loading, vLLM, SIGKILL and failed exports are outside profiler coverage.
Short samples do not qualify full-context inference. Unload does not release the
deployment; the owner must enforce deadlines and tear down its resources.

## LoRA scores

On an owned Ray cluster, `run_lora_logprobs.py` checks zero-init agreement,
a seeded adapter change, withheld publication, weight sync, and updated scores:

```bash
uv run --isolated --extra tinker --extra megatron python -m examples.tinker.glm53.run_lora_logprobs \
  --backend-config /local/backend-config.json --output-dir /local/lora-check \
  --mean-atol "$MEAN_ATOL"
```

Use the rendered backend config and a reviewed absolute error budget. The actual withheld
adapter must fail that budget before the published adapter passes it; direct update deltas
remain diagnostic. These short synthetic inputs do not prove
full-context capacity, real optimizer behavior, or learning.
