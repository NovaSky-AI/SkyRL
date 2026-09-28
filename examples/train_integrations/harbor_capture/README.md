# Harbor through skyrl-capture

Harbor runs **unmodified, in text space**. A per-trajectory capture route
renders its prompt, calls the engine with token IDs, and keeps a message graph
-- so training gets the exact tokens the model was conditioned on without
Harbor knowing anything about tokens.

The difference from the sibling `harbor/` integration is what Harbor is asked
to do. There it runs with `collect_rollout_details=True` and records the token
IDs vLLM reports, which is why that integration has to ban summarization:

```python
assert len(traj.rollout_details) == 1, "Expected exactly one rollout segment"
# TODO(Charlie): Support summarization.
```

Compaction breaks its token accounting. A proxy-side graph does not have the
problem: a rewritten history stops matching at the last unchanged message and
branches there. **Summarization is allowed here**, and each branch becomes its
own training row.

## What this guarantees

> Every token the trainer trains on is a token the engine sampled from.

The harness talks messages, so capture re-renders each turn. If re-rendering
perturbed anything the model had already seen, the training sequence would
contain tokens that were never in any prompt the engine sampled from. It does
not, because capture bridges forward in token space -- the previous prompt and
its completion are carried verbatim and only the new messages are rendered.

This is not a claim that capture reproduces the sibling's tokens bit for bit.
From turn two the two build different prompts -- capture carries the model's
own tokens, the sibling re-renders from the text it was handed -- and each is
faithful to its own run. `validation/` measures the guarantee above directly,
without reference to the other implementation.

## Two hooks

**Inference setup**, once per run, beside the engine -- `start_capture()` in
`entrypoints/main_harbor_capture.py`:

```python
config = Config(
    record_dir=record_dir,
    upstream=TitoUpstream(type="skyrl", url=engine_url, tokenizer=..., model=...),
    upstream_modules=("examples.train_integrations.harbor_capture.upstream",),
)
service = CaptureService(config=config, port=port)
service.start(blocking=False)
```

One capture process owns one configured upstream. There is nothing to register
afterwards and no target to name; several policy endpoints mean several
capture processes. It writes a record directory and needs no database.

`type="skyrl"` is capture's upstream kind for this router, pointed at its
**root**. `upstream.py` knows the `/skyrl/v1/generate` path, the singular
request shape, the `X-Session-ID` affinity header, vLLM's sampling-parameter
rules, `cache_salt`, and the packed routed-expert payload. A `tokens` upstream
gets all of them wrong against this router.

**The agent loop**, once per trial -- see `harbor_generator.py`:

```python
trajectory = create_trajectory(project=..., run_id=..., task_id=..., step=...,
                               trajectory_id=session, client=self.capture_client)
try:
    config["agent"]["kwargs"]["api_base"] = trajectory.base_url
    # Terminus-2 takes `api_base` as its own parameter but has no `api_key`
    # one: it forwards `llm_kwargs` to the LiteLLM constructor and swallows
    # anything else. A key set beside `api_base` is accepted, ignored, and the
    # route answers 401 with nothing in the config to explain it.
    config["agent"]["kwargs"]["llm_kwargs"]["api_key"] = PLACEHOLDER_API_KEY
    await harbor.run(config)
finally:
    envelope = trajectory.finish(annotations={"reward": reward}, format="token_samples")
```

Naming the trajectory also names the engine's session key, so capture's
session and SkyRL's are the same one. `cache_salt` rides in
`llm_kwargs.extra_body` when `generator.use_cache_salt` is on; it carries the
policy version, which has to be per trajectory because the weights move every
step while the engine stays put.

## What `compose` does

`finish(format="token_samples")` returns one row per root-to-leaf branch.
`compose` splits each at its **first trainable token** and emits every
trainable path as one complete multi-turn sample.

One Harbor execution is one physical rollout with one reward, and compaction
may make it several paths. Those are sample shards of the same rewarded
execution, not independent reward observations, so graph shape must not change
what a rollout is worth. SkyRL already expresses "several rows, one rollout,
one advantage" -- step-wise training -- so this reuses it rather than
generalising a second contract. A rollout's paths are emitted contiguously
under one `TrajectoryID`, every path carries the rollout's reward, and
`is_last_step` marks the last of them.

Two settings go with that shape, and `_require_grouped_output` refuses to
start without them:

* `generator.step_wise_trajectories=true`
* `generator.merge_stepwise_output=false`

Capture has already applied the rule that a sampled node reachable from
several branches is trainable in exactly one, so summing masks never
double-counts. A trial that timed out or errored is masked, not dropped: the
batch keeps one row per rollout so rewards and rollouts stay aligned.

## Running

```bash
uv run --extra harbor-capture \
  -m examples.train_integrations.harbor_capture.entrypoints.main_harbor_capture \
  trainer.policy.model.path=Qwen/Qwen3-4B-Instruct-2507 \
  data.train_data="['/path/to/harbor/tasks']"
```

`skyrl-capture` lives in this repo, under `skyrl-capture/`, and the extra
installs it from there.

Capture runs in this process by default. Setting `CAPTURE_ENDPOINT` points at
a separate `skyrl-capture serve` instead -- naming an endpoint is the whole
intent, so there is no second switch. That process needs the same upstream
module on its command line, and `PYTHONPATH` set to this checkout, since
`examples.…` is a path rather than an installed package.

`CAPTURE_ENDPOINT` is the only environment variable this example reads, and it
is the capture SDK's own. Everything else -- the record directory, the port,
the tokenizer -- is an argument or comes from the run config, so there is one
place to look for each.

The plugin imports SkyRL's `generate_wire` for the routed-expert decoder, so a
standalone capture process has to run from SkyRL's environment.

## Tests

```bash
uv run --extra dev pytest tests/integrations/harbor_capture/ -q
```

They need neither capture nor a cluster: `conftest.py` stubs capture's
protocol base class, because capture is an optional dependency of one example
and the CPU pipeline does not install it. They run in `cpu_skyrl.yaml`, which
collects `tests/` wholesale.

## Validation

`validation/` holds the end-to-end checks, which do need a live engine, a
capture service and Harbor sandboxes. See `validation/README.md`.
