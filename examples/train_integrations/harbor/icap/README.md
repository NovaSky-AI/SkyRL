# Harbor through inference-capture

Harbor runs **unmodified, in text space**. The proxy renders the prompt, calls
the engine with token IDs, and keeps a message graph, so training gets the exact
tokens without Harbor knowing anything about tokens.

The difference from the sibling `harbor/` integration is what Harbor is asked to
do. There it runs with `collect_rollout_details=True` and emits per-turn token
IDs itself — which is why that integration has to ban summarization:

```python
if agent_kwargs.get("enable_summarize", False):
    raise ValueError("step_wise_trajectories=true is incompatible with enable_summarize=true")
```

Compaction breaks the harness's own token accounting. A proxy-side graph does
not have the problem: a rewritten history stops matching at the last unchanged
message and branches there. **Summarization is allowed here**, and each branch
becomes its own training row.

## Two hooks

**Run it** — `entrypoints/main_harbor_icap.py` is complete, and is the sibling
generate entrypoint with the generator swapped:

```bash
python -m examples.train_integrations.harbor.icap.entrypoints.main_harbor_icap \
    trainer.policy.model.path=Qwen/Qwen3-4B-Instruct-2507 \
    data.train_data="['/path/to/harbor/tasks']"
```

**Inference setup**, once per run, beside the engine — `start_capture()` in that
file, which is the whole of it:

```python
service = CaptureService(data_dir="./icap-data", port=8080, num_workers=4)
service.start()                      # returns once /healthz answers
asyncio.run(service.ensure_target(
    name="policy", type="skyrl", url=router_url,
    model=model_name, tokenizer=tokenizer_name,
    config={"max_model_len": max_seq_len},
))
```

`type="skyrl"` is capture's upstream kind for this router, pointed at its
**root**. It knows the `/skyrl/v1/generate` path, the singular request shape,
the `X-Session-ID` affinity header and vLLM's sampling-parameter rules —
including that `logprobs` is a *count* here and a boolean in the OpenAI shape
capture speaks. A `tokens` target gets all five wrong against this router.

Nothing has to exist first. With no `DATABASE_URL`, capture starts its own
PostgreSQL and keeps the database, the queue and the payloads under
`./icap-data`. `ensure_target` is idempotent, so a restarted job or a second
node does not fail on its second call — it is a `PUT`, so the second call
updates rather than failing on a taken name.

**The agent loop**, once per trial — see `harbor_generator.py`:

```python
trajectory = create_trajectory(project=..., target="policy",
                               trajectory_id=session_id, upstream=upstream)
try:
    config["agent"]["kwargs"]["api_base"] = trajectory.base_url
    # Terminus-2 takes `api_base` as its own parameter but has no `api_key`
    # one: it forwards `llm_kwargs` to the LiteLLM constructor and swallows
    # anything else. A key set beside `api_base` is accepted, ignored, and the
    # route answers 401 with nothing in the config to explain it.
    config["agent"]["kwargs"].setdefault("llm_kwargs", {})["api_key"] = trajectory.api_key
    await harbor.run(config)
finally:
    trajectory.finish(annotations={"reward": reward})

rows = trajectory.export("token-samples")
```

Naming the trajectory also names the engine's session key, so capture's session
and SkyRL's are the same one. `upstream={"body": {"cache_salt": ...}}` carries
the policy version, which has to be per trajectory because the weights move
every step while the target stays put.

## What `compose` does

`export("token-samples")` returns one row per root-to-leaf branch. `compose`
splits each into `prompt_token_ids` / `response_ids` at its **first trainable
token** and builds a `GeneratorOutput`.

Capture has already applied the rule that a sampled node reachable from several
branches is trainable in exactly one, so summing masks never double-counts.
Two decisions stay on this side because only the harness knows them:

- a trial that timed out or errored is masked;
- with `step_wise` off, a trajectory that branched has no single row, so it is
  masked rather than guessed at. **A summarizing agent branches by design, so
  summarization wants step-wise on.**

Masked trajectories are masked, not dropped: the batch keeps one row per
trajectory so rewards and rollouts stay aligned.

## Running

For development, resolve capture from a checkout:

```bash
uv run --with-editable /path/to/anyscale-capture \
  -m examples.train_integrations.harbor.icap.entrypoints.main_harbor_icap
```

Or from a release, once there is one:

```bash
uv run --with inference-capture -m examples.train_integrations.harbor.icap.entrypoints.main_harbor_icap
```

The adapter's tests need neither capture nor a cluster:

```bash
uv run --with pytest python -m pytest examples/train_integrations/harbor/icap/test_compose.py -q
```

## Status

`compose.py` is complete and tested. `harbor_generator.py` has one hole:
`_run_harbor` must be wired to the same Harbor runner the sibling integration
uses — the only difference is that its config already points at the capture
route. The entrypoint is a sketch showing where `start_capture()` and
`build_generator()` slot into the existing `main_harbor.py`.
