# Full-context profiling

Native SkyRL benchmark, not a learning or sampler-parity test.
Start with Qwen3-0.6B (one trainer GPU + one inference GPU); it is **not** offered
by hosted Tinker. A hosted comparison needs a common model and matched loss/gradients.

Use an owned Ray cluster, the same pinned checkout/environment on each node, and a
downloaded model. Adapter/checkpoint paths must be shared; database/traces use local scratch.

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

`run_client.run()` shows the protocol: create → initial publish/sample →
warmup + two measured updates → checkpoint/unload. Each update is one batched
reference forward, one batched GSPO backward, optimizer, publication and short sample.
Two repeated-text fixtures each score exactly 32,768 positions. Raw advantages are
+1/-1; the API sums them. This is not yet the hosted sequence-mean GSPO workload.

Profiles are in [configs/](configs/); `run_server.py --help` lists GLM variants.
`--print-config` renders the effective server config without starting GPUs.
GLM/256K profiles still need their own qualification.

The client saves exact datums, replay batches and phase JSONL. Every trainer rank
profiles warmup and updates; verify CUDA traces. Optimizer request time includes
trace export. OOM export/restart is best-effort; it does not recover model state.
Cold model loading, vLLM, SIGKILL and failed exports are outside profiler coverage.
Short samples do not qualify full-context inference. Unload does not release the
deployment; the owner must enforce deadlines and tear down its resources.
