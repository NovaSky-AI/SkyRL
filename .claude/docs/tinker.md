# Tinker API Server

SkyRL's implementation of the [Tinker API](https://tinker-docs.thinkingmachines.ai/) for local post-training. Full user-facing docs live at `docs/content/docs/tinker/` -- refer there for quickstart, cookbook recipes, and architecture diagrams.

## Code Layout

- **`skyrl/tinker/api.py`** -- FastAPI HTTP server. Receives Tinker SDK requests, writes them to SQLite/Postgres, returns future IDs.
- **`skyrl/tinker/engine.py`** -- Background subprocess (`TinkerEngine`). Polls DB, batches compatible requests, dispatches to backend.
- **`skyrl/tinker/types.py`** -- Internal Pydantic models (distinct from API request/response models in `api.py`). `LOSS_TYPES` dict defines valid loss functions.
- **`skyrl/tinker/config.py`** -- `EngineConfig` Pydantic model. `add_model()` auto-generates argparse flags from Pydantic fields.
- **`skyrl/tinker/db_models.py`** -- SQLModel tables: `FutureDB`, `ModelDB`, `CheckpointDB`, `SessionDB`, `SamplingSessionDB`.
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
- DB uses SQLite WAL mode with 30s busy timeout by default.

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
