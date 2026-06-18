# `sample_full_batch` — verification findings (2026-06-18)

Verification notes for the fully-async `sample_full_batch` work (PR #1802, branch
`async-sample-full-batch`). All runs below: GSM8K, `Qwen/Qwen2.5-1.5B-Instruct`, fully async, FSDP,
2 inference GPUs + 2 policy GPUs on a B200 box, `--extra fsdp`.

## 1. vLLM `/health` 500 → "Server failed to become healthy within 600s" (FIXED)

### Symptom
Runs intermittently failed during inference-server startup with the entrypoint timing out
(`TimeoutError: Server failed to become healthy within 600s`) and Ray retrying. The vLLM server
logs showed every `/health` request returning **500**:

```
GET /health HTTP/1.1" 500 Internal Server Error
AttributeError: '_IncludedRouter' object has no attribute 'path'
  prometheus_fastapi_instrumentator/routing.py:55 in _get_route_name -> route.path
```

### Root cause
**FastAPI 0.137.0** refactored `include_router()` to store `_IncludedRouter` wrapper objects in
`app.routes` instead of copies of the individual routes. `prometheus_fastapi_instrumentator` (8.0.0,
pulled in transitively by vLLM) iterates `app.routes` in its metrics middleware and unconditionally
reads `route.path`, which raises on the new `_IncludedRouter` objects — so **every request through the
instrumented app 500s**, including the `/health` probe the trainer waits on.

This is an ecosystem incompatibility, not SkyRL code. It bit us because a re-lock on 2026-06-18
05:40 bumped `fastapi` to `0.137.1`. (Tracked upstream:
<https://github.com/trallnag/prometheus-fastapi-instrumentator/issues/370>,
<https://github.com/vllm-project/vllm/issues/45596>.)

### Fix
Cap `fastapi` below the breaking release in `[tool.uv].constraint-dependencies` and re-lock:

```toml
constraint-dependencies = [
    ...
    "fastapi<0.137",
]
```

`uv lock` → `fastapi 0.137.1 → 0.136.3`. (Drop the cap once
`prometheus-fastapi-instrumentator` ships a fix and vLLM picks it up.)

### Verified
A `sample_full_batch` run started healthy and trained 3 steps to completion — no `/health` 500, no
startup timeout.

## 2. Metrics comparability with vs. without `sample_full_batch` (VERIFIED)

### Change under test
Under `sample_full_batch`, zero-variance groups are dropped before training, so reward/generation
metrics would otherwise be computed over only the kept (non-zero-variance) subset — biasing them.
The fix includes the dropped groups in the reward + generation metrics (but not in training), via an
optional `metrics_generator_output`/`metrics_uids` superset arg on
`postprocess_generator_output`.

### Result — exact arithmetic proof
With `mini_batch_size=8`, the active run reported `reward/avg_pass_at_5` exactly equal to
`kept / (kept + dropped)` each step (kept groups have reward variance → pass@5 ≥ 1 contribution;
the dropped zero-variance groups here were all-fail → 0 contribution):

| step | kept | dropped | reported `avg_pass_at_5` | `kept/(kept+dropped)` |
|------|------|---------|--------------------------|-----------------------|
| 1    | 8    | 2       | 0.800                    | 8/10 = 0.800          |
| 2    | 8    | 13      | 0.381                    | 8/21 = 0.381          |
| 3    | 8    | 11      | 0.421                    | 8/19 = 0.421          |

So the dropped groups **are** counted. Without the fix this would report `8/8 = 1.0` every step
(biased high). With the fix, the active values land in the same range as a **no-filter baseline**
(same seed/config, `zero_variance_filter=false`, `sample_full_batch=false`):

| metric             | active (`sample_full_batch`) | baseline (no filter) |
|--------------------|------------------------------|----------------------|
| `avg_pass_at_5`    | 0.38 – 0.80                  | 0.375 – 0.50         |
| `avg_raw_reward`   | 0.086 – 0.26                 | 0.075 – 0.15         |

(Per-step values are noisy — small 8-prompt batches + stochastic generation + the two runs consume
different prompt sets — but they are in the same range, and the kept-only bias is removed.)

### Notes
- The **passive** path (`zero_variance_filter=true`, `sample_full_batch=false`) already counted
  filtered groups, since it loss-masks rather than drops and computes metrics over the full batch
  before masking. This change brings the **active** drop path in line.
- **Loss** metrics remain over the trained tokens only (dropped groups are not trained); this is
  consistent between the passive and active paths (same non-zero-variance tokens). Making the
  loss-block stats (`loss/avg_final_rewards`, `response_length`) span the union too would require
  threading the dropped groups into the training tensors — deferred; flag if wanted.
- Unit test: `tests/train/test_generator_postprocess.py::test_postprocess_metrics_over_superset`.

## 3. Epoch boundary under `sample_full_batch` (VERIFIED)

### Test
Tiny 48-row train set (`num_steps_per_epoch = 48/8 = 6`), `epochs=2`, `sample_full_batch=true`. The
high early-training drop rate exhausts an epoch's prompts before 6 trained steps, exercising the
end-of-epoch path.

### Result
```
epoch 0: step (dropped 3), step (dropped 4)
  WARNING sample_full_batch: epoch 0 exhausted with a partial mini-batch of 7 group(s);
          discarding and ending the epoch.
epoch 1: step (dropped 5), step (dropped 12), step (dropped 6)   [trainer/epoch: 1]
Training done!
```

- Epoch 0 exhausted mid-mini-batch → the partial batch (7 groups) was **discarded and marked
  consumed/filtered** (not trained), and the epoch ended early.
- Training **continued into epoch 1** with a reset dataloader.
- No `AssertionError`, no `Unexpected number of trained...`, no `Traceback`. `validate_state_at_epoch_end`
  and the per-step consumed-UID accounting passed across the boundary, and the run finished cleanly.

This confirms the dynamic-epoch design: per-epoch step count is an upper bound; the partial last
mini-batch is skipped and the next epoch proceeds.

## Open items / follow-ups
- **Per-step resample cap** (analogous to DAPO `max_sample_batches`): `sample_full_batch` can churn a
  lot of generation when the drop rate is high (e.g. step 2 above generated 8 kept + 13 dropped). A
  cap that warns and falls back to loss-masking would bound generation cost and the per-step memory
  held by `cur_dropped_groups` (kept for metrics). Not yet implemented.
- Optionally route the loss-block reward/length stats over the kept+dropped union (see §2 notes).
- Drop the `fastapi<0.137` cap once the upstream `prometheus-fastapi-instrumentator` fix lands.
