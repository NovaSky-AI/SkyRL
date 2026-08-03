# Tinker API Server

SkyRL's implementation of the [Tinker API](https://tinker-docs.thinkingmachines.ai/) for local post-training. Full user-facing docs live at `docs/content/docs/tinker/` -- refer there for quickstart, cookbook recipes, and architecture diagrams.

## Code Layout

- **`skyrl/tinker/api.py`** -- FastAPI HTTP server. Receives Tinker SDK requests, writes them to SQLite/Postgres, returns future IDs.
- **`skyrl/tinker/engine.py`** -- Background subprocess (`TinkerEngine`). Polls DB, batches compatible requests, dispatches to backend.
- **`skyrl/tinker/types.py`** -- Internal Pydantic models (distinct from API request/response models in `api.py`). `LOSS_TYPES` dict defines valid loss functions.
- **`skyrl/tinker/config.py`** -- `EngineConfig` Pydantic model. `add_model()` auto-generates argparse flags from Pydantic fields.
- **`skyrl/tinker/db_models.py`** -- SQLModel tables: `FutureDB`, `ModelDB`, `CheckpointDB`, `SessionDB`, `SamplingSessionDB`. Also owns SQLite pragmas (`enable_sqlite_wal`), the JSON codec for payload columns (`json_engine_kwargs`), and `create_missing_indexes`.
- **`skyrl/tinker/futures.py`** -- `FutureWaiter` (one shared batched poller backing every `retrieve_future` call) and `complete_future` (terminal result write-back).
- **`skyrl/tinker/loss_fns.py`** -- JAX loss function implementations (cross_entropy, importance_sampling, ppo, cispo). Only used by the JAX backend.
- **`skyrl/tinker/extra/`** -- `ExternalInferenceClient` for offloading sampling to external vLLM.
- **`skyrl-agent/skyrl_agent/integrations/tinker/`** -- Agent-side Tinker integration (separate package).

## Starting the Server

```bash
uv run --extra tinker --extra fsdp -m skyrl.tinker.api \
    --base-model "Qwen/Qwen3-0.6B" --backend fsdp --port 8000
```

The API process spawns the engine as a child subprocess (via `uv run -m skyrl.tinker.engine`). If the engine crashes, the API server auto-terminates.

## Key API Endpoints

All endpoints are under `/api/v1/`. Requests are async -- submit via POST, get a `request_id`, poll `retrieve_future`.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/create_session` | POST | Initialize a session (required before model creation) |
| `/create_model` | POST | Create a LoRA (or full-param) training model |
| `/forward_backward` | POST | Forward + backward pass, accumulates gradients |
| `/forward` | POST | Forward-only pass (logprobs, no gradients) |
| `/optim_step` | POST | Apply accumulated gradients |
| `/asample` | POST | Generate samples from current or base model |
| `/save_weights` | POST | Save full training checkpoint (weights + optimizer) |
| `/save_weights_for_sampler` | POST | Sync weights to inference engines |
| `/load_weights` | POST | Load a previously saved checkpoint |
| `/training_runs/{unique_id}/checkpoints/weights/{checkpoint_id}` | DELETE | Delete a saved training checkpoint archive from disk |
| `/training_runs/{unique_id}/checkpoints/sampler_weights/{checkpoint_id}` | DELETE | Delete a saved sampler checkpoint archive from disk |
| `/retrieve_future` | POST | Long-poll for async result (300s timeout) |
| `/healthz` | GET | Liveness check |

## Concurrency and Batching

- `forward_backward` and `forward` requests are batched using look-ahead scheduling -- the engine groups all pending ops before the next barrier (`optim_step` or `load_weights`).
- `sample` requests are batched ensuring one checkpoint_id per model_id per batch.
- `optim_step`, `create_model`, `save_weights`, `load_weights` are processed individually and act as barriers.
- DB uses SQLite WAL mode with `synchronous=NORMAL`, a 64 MB page cache, and 30s busy timeout by default. Default DB path is `/tmp/skyrl_tinker/tinker.db` (node-local). `prepare_sqlite_path` in `db_models.py` creates missing parent dirs and warns if the SQLite file is on a network filesystem, where locking/WAL are unreliable.

### Keeping the DB off the critical path

Request payloads are large (a `forward_backward` with 4x512 tokens is ~76 KiB of JSON), so this layer is easy to make accidentally quadratic. Three rules hold it in place:

1. **Scheduling never loads payloads.** `TinkerEngine.scan_pending_requests` reads only `(request_id, model_id, request_type)` through the `ix_futures_pending_scan` covering index; `find_batchable_*` / `find_single_requests` decide what runs from that metadata, then fetch `request_data` for the chosen requests only. The main loop scans once per iteration and passes the result to all four finders. Loading full rows instead means deserializing the whole backlog every poll -- including requests parked behind a barrier that cannot run yet, which is the multi-LoRA steady state.
2. **One shared poller for results.** `retrieve_future` awaits `FutureWaiter.wait`, which registers an asyncio future; a single background task resolves all of them with one `WHERE request_id IN (...)` query per tick. Polling per caller instead costs one query per in-flight request per tick. Results written by the API process itself (forwarded samples) call `FutureWaiter.notify` and skip the round trip entirely.
3. **Payload columns use the fast JSON codec.** `json_engine_kwargs()` installs orjson when available (~7x faster encode, ~3x faster decode on numeric arrays) and falls back to the stdlib. Pass it to every `create_engine`/`create_async_engine` that touches these tables.

### An EXTERNAL request has exactly one chance to complete

When sampling is forwarded (`external_inference_url` set, or a non-colocated backend), `asample` writes an `EXTERNAL` future and hands off to a fire-and-forget `asyncio.create_task`. That task's result write is the **only** thing that can ever complete the request: the engine excludes `EXTERNAL` from scheduling (`_BATCHED_REQUEST_TYPES`) and nothing reaps pending rows. So a failure there does not fail one request -- it leaves it pending forever while the client polls until its own deadline, which surfaces as a hang rather than an error.

Hence `complete_future` retries `SATimeoutError` and `OperationalError` (pool-checkout timeouts and `database is locked` are exactly what a burst of simultaneous completions produces), and both forwarding clients catch and log anything that still escapes. If you add another path that completes futures, preserve both properties.

The pool is the thing to watch: the async engine takes SQLAlchemy's defaults, so it is **15 connections (5 + 10 overflow) with a 30s checkout timeout**. Any per-request DB polling scales checkouts with in-flight requests and starves unrelated endpoints -- a symptom that shows up first as `session_heartbeat` timing out client-side, long before sampling visibly breaks.

Measure changes here with `skyrl/benchmarks/bench_tinker_db.py` (no GPU needed). It reports SQL statement and commit counts alongside timings, and tolerates an engine without `scan_pending_requests` so it can be run against an older checkout for comparison.

Rejected deliberately: group-committing submissions. It measured only ~1.18x throughput once orjson removed the JSON encoding cost, and batching some endpoints while `save_weights`-style endpoints commit directly would reorder `request_id` allocation, which barrier scheduling depends on.

## Weight Sync Modes

- **Persistent**: `save_weights_for_sampler(name="...")` -- syncs to inference engines AND writes HF checkpoint to disk. Expensive.
- **Ephemeral**: `save_weights_and_get_sampling_client(name="...")` -- syncs to inference engines only, skips disk write. Triggered when `sampling_session_seq_id` is present in the request.
- In RL loops, always prefer ephemeral mode; reserve persistent saves for periodic checkpoints.
- Delete persistent checkpoints after they are no longer needed. The delete endpoint requires an explicit checkpoint type, since a training and a sampler checkpoint can share an id:
  - `DELETE /training_runs/{unique_id}/checkpoints/weights/{checkpoint_id}` (training checkpoint)
  - `DELETE /training_runs/{unique_id}/checkpoints/sampler_weights/{checkpoint_id}` (sampler checkpoint)
  - or `DELETE /training_runs/{unique_id}/checkpoints/{checkpoint_id}?checkpoint_type=training|sampler`

  A bare `.../checkpoints/{checkpoint_id}` with no type prefix and no `checkpoint_type` query param is rejected with a `400`. This removes the saved archive from `checkpoints_base`; it does not unload the live model.

## Testing

```bash
# Unit tests (CPU, no GPU needed, requires jax extra)
uv run --extra dev --extra jax pytest tests/tinker/ -v

# Integration tests (test_api.py) spin up a real server subprocess -- slow, need port 8000/8001 free
uv run --extra dev --extra jax pytest tests/tinker/test_api.py -v
```

- `tests/tinker/conftest.py` -- `wait_for_condition` helper for polling.
- `tests/tinker/test_api.py` -- Integration tests using the real `tinker` SDK client. `start_api_server` context manager launches a subprocess.
- `tests/tinker/test_engine.py` -- Unit tests for `TinkerEngine` (model creation, unload, stale session cleanup, batch preparation).
- `tests/tinker/test_api_validation.py` -- Pydantic validation edge cases (loss_fn_config, chunk discriminator, base64 image round-trips).
- `tests/tinker/test_db.py` -- Alembic migration smoke tests.
- `tests/tinker/test_loss_fns.py` -- JAX loss function correctness (cispo clipping, gradient stop).

## Gotchas 

- **Token shifting**: Tinker pre-shifts inputs/targets; SkyRL-Train shifts internally. The backend appends the last target token to reconstruct full sequences during batch conversion -- be careful if modifying `prepare_model_pass_batch`.
- **Left-padding**: SkyRL-Train expects left-padded tensors. The backend handles this during batch prep.
- **API models vs internal types**: `api.py` defines its own Pydantic models (e.g., `api.ForwardBackwardInput`) that mirror but differ from `types.ForwardBackwardInput`. Each API model has a `.to_types()` method for conversion. Do not confuse the two.
