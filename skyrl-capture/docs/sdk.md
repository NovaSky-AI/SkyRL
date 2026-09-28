# Python SDK

```python
from skyrl_capture.sdk import CaptureClient, capture, create_trajectory
```

The design goal is that the workload does not change. You create a trajectory,
apply its environment, run the existing client, and finish.

The SDK is for **lifecycle and export integration** — creating a trial,
scoping a workload to it, scoring it, and pulling training rows out. It is
deliberately not a browser for captured data: reading is the
[viewer](viewer.md) and [`/v1`](api-reference.md), and a thin wrapper around
every route would be a second surface to keep in step with the first.

## The lifecycle

```python
from skyrl_capture.sdk import capture

with capture(project="terminal-bench", labels=["task-17"]) as trajectory:
    run_client()                           # existing application, untouched
    trajectory.annotate(rlvr_reward=1.0)
```

`capture()` does four things, and all four matter: it creates the trajectory,
applies its environment, finishes it — with an `error` outcome if the block
raised, so an abandoned trial is a record rather than a journal nobody came
back for — and releases the HTTP client it made.

By hand, the `finally` has to do the last two:

```python
trajectory = create_trajectory(project="terminal-bench")

try:
    with trajectory.environment():
        run_client()                       # existing application, untouched
finally:
    trajectory.finish(labels=["task-17"],
                      annotations={"rlvr_reward": 1.0})
    trajectory.close()                     # see "Whose HTTP client it is"
```

`environment()` sets the provider environment variables the upstream's client
protocol reads, and restores the previous values on exit:

| Target type | Variables set |
| --- | --- |
| `openai`, `tokens` | `OPENAI_BASE_URL`, `OPENAI_API_BASE`, `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN` |

Both the modern and legacy OpenAI names are set, because SDK versions in the
wild read different ones. `SKYRL_CAPTURE_TRAJECTORY_ID` is also set so a
wrapped process can report which trial it is.

No client patching and no per-request headers are involved: the route identity
is the correlation key.

## The context manager

For the common case, `capture()` does all three steps:

```python
from skyrl_capture.sdk import capture

with capture(project="terminal-bench", labels=["task-17"]) as trajectory:
    run_client()
    print(trajectory.id)
```

On an exception the trajectory is finished with an `error` outcome and the
exception propagates. Nothing expires a trajectory, so a trial that is never
finished stays in the record as an unfinished journal until somebody finishes
it — which is recoverable, and is why this does it for you.

## Configuration

```python
create_trajectory(
    project="terminal-bench",
    run_id="run-2026-09-17a",          # the run; created on first use
    task_id="task-17",                 # which task this attempt attempted
    step=7,                            # which training step produced it
    endpoint="http://capture.internal:8080",   # or CAPTURE_ENDPOINT
    labels=["nightly"],
    bodies="full",                     # or "sampled"
    source_metadata={"git_sha": "abc123"},
    trajectory_id="rollout-step7-042",         # name it instead of generating
)
```

There is no upstream to choose. The capture process at `endpoint` was started
against one inference server, and the `Trajectory` you get back carries what a
launcher needs to know about it: `trajectory.protocol`, the wire your client
should speak, and `trajectory.mode`, whether tokens were captured. Token
capture reports `openai`, because that is what a client sends it; which engine
wire capture uses behind the route is capture's business and is on the upstream
snapshot.

The full upstream snapshot — url, model, tokenizer — is on the HTTP
representation at `GET /v1/trajectories/{id}`, not on this object.

`run_id`, `task_id` and `step` place the attempt in the hierarchy — project /
run / trajectory. All three are optional: a wrapped agent command is one trial
and belongs to no run. They are indexed columns rather than annotations,
because every question an RL run is read with groups by them.

`trajectory_id` names the trajectory. **The SDK generates one before it sends
the request** when you do not, which is what makes a create whose response was
lost safe to repeat: the same id with the same body is the same trajectory, and
the server says so from a hash it persisted. The same id with a *different*
body is a conflict rather than a silent reuse of another trial's route.

It becomes the first path segment of the route and the name of this
trajectory's files, so it must be URL-safe. It is also the session key sent to
the inference engine, so naming it aligns capture's session with your own.

A trajectory does not reconfigure the upstream. There is nothing here that
changes the call the proxy makes on your behalf, and that is the rule:

* deployment and upstream settings come from the capture process's
  configuration;
* authentication comes from the capture process's configuration;
* per-inference sampling and behaviour come from the inference request --
  a `cache_salt` keyed on a weight version rides in the request that wants it,
  where your own client already puts every other sampling field;
* what is on the trajectory is capture metadata.

`source_metadata` is stored on the trajectory and on every exchange, which is
the place to put launcher, host, and commit information.

### Whose HTTP client it is

`create_trajectory` makes a `CaptureClient` when you do not pass one, and that
client owns a connection pool. One trial per rollout then means one pool per
rollout, for as long as the training loop runs, so release it:

```python
trajectory = create_trajectory(project="rl")
try:
    ...
finally:
    trajectory.finish()
    trajectory.close()          # closes the client it made
```

`capture()` does this for you. The other way is to own the client yourself,
which is what a loop making many trials should do — then the pool is shared,
`trajectory.close()` is a no-op, and closing it is yours:

```python
with CaptureClient() as client:
    for task in tasks:
        trajectory = create_trajectory(project="rl", client=client)
        ...
```

There is no idempotency key. Creation is idempotent by the trajectory id and a
hash of the body, both on disk before the route comes back — which is stronger,
because it survives the process that answered the first time.

## Reading captured data

Not here. `skyrl-capture view` reads a live run and a finished record alike,
and everything it shows comes from [`/v1`](api-reference.md), so a script that
needs the timeline or the graph fetches the route. What the SDK gives you back
is training rows — see below.

## Getting the rows

**Finishing returns them.** A trajectory's own training rows need no export
job: `finish` compiles the record and renders it in whichever format you ask
for, from what it just committed.

```python
result = trajectory.finish(
    annotations={"rlvr_reward": 1.0},
    format="token-samples",
)
for row in result["records"]:
    train_on(row["input_ids"], row["loss_mask"])
```

One row per root-to-leaf branch of the message graph. A linear run is one row;
a compaction branches, so it is two; a sub-agent fan-out is more. A sampled
node reachable from several branches is trainable in exactly one of them, so
summing `loss_mask` across a trajectory's rows never double-counts.

`format` is `graph` (the default — the whole trajectory), `replay`,
`text-samples` or `token-samples`, and `options` takes the same flags the
exporters do.

For a trajectory finished earlier, `trajectory.export("token-samples")` runs
the same formatters through `/v1/exports`. For a dataset over a whole run or
project, use the CLI or `/v1/exports` directly.

## Annotating

```python
trajectory.annotate(rlvr_reward=0.75, critique="omitted units")
trajectory.tag("success", remove=["draft"])
# {"labels": ["success"], "annotations": {"rlvr_reward": 0.75, ...}}
```

Both return the merged metadata, which is also what reading the trajectory
gives: `labels` and `annotations` are fields on it, not a document of their
own.

Metadata belongs to the whole trajectory, so a branched run needs no choice of
which node to attach to. Annotations merge rather than replace, nothing is
sealed, and no history is kept.

## Finishing

```python
result = trajectory.finish(
    labels=["task-17", "success"],
    annotations={"grader": "rubric-v2", "rlvr_reward": 1.0},
    command_result="success",
    format="graph",
)
# {"id": "...", "status": "finished", "format": "graph", "records": [...]}
```

**Synchronous.** By the time this returns, the trajectory's record is on disk
and `records` was rendered from it.

**Idempotent, by a hash of the outcome**, which is persisted with the record —
so a retry gets the same answer after a restart as well as before one. The
format is not part of that hash: asking again in a different format is a retry.
A different outcome is a `409`.

**Retried, exactly three times.** This is the one call whose failure loses a
trial's whole record, so the SDK retries a connection failure, a timeout, or a
`502`/`503`/`504` with short exponential backoff and jitter, sending the
identical body each time. A `409` is a decision and is never retried.

A `503` means the record was not written -- the disk refused it, or a turn was
still in flight when the barrier ran out. The trajectory is untouched either
way, and a finish already recorded on it is not recorded again, so retrying
merges the same metadata once rather than twice. If all three attempts fail the
trajectory stays open in the record and can be finished later, by id, from
anywhere.

Metadata is not sealed by finishing — labels and annotations given here merge
like any other write and stay editable afterwards, including on the committed
record.

There is no reward parameter: record a reward as an annotation under whatever
key you use. It belongs to the whole trajectory, so a branched run cannot
reject it.

## The control-plane client

`CaptureClient` carries the operations the lifecycle needs by name. There is no
public `get(path)`: a generic verb makes every route look supported and turns
removing one into a breaking change, so the surface is the list below.

```python
from skyrl_capture.sdk import CaptureClient

with CaptureClient() as client:
    # A run is a flat listing; task and step are filters on it.
    page = client.list_trajectories(run_id="run-2026-09-17a", step=7)

    # Rewards usually arrive after the trial, when an id is all you have.
    client.annotate_trajectory("tr_123", annotations={"rlvr_reward": 0.75},
                               labels=["success"], remove_labels=["draft"])

    job = client.create_export(project="terminal-bench", format="token_samples")
    job = client.wait_for_export(job["id"])
    open("dataset.jsonl.zst", "wb").write(client.download_export(job["id"]))

    client.health()
```

| | |
| --- | --- |
| `health()` | The `/healthz` document |
| `list_trajectories(**filters)` | One page, filtered by `project`, `run_id`, `task_id`, `step`, `status` |
| `annotate_trajectory(id, ...)` | Merge labels and annotations, by id |
| `create_export` / `get_export` / `wait_for_export` / `download_export` | The export job, end to end |

Anything else is a `curl` at [`/v1`](api-reference.md). Run listing and run
metadata are routes with no wrapper here: they exist and are documented, they
just do not have a convenience method yet.

Errors raise `CaptureError` carrying the status code and the server's detail.
So do transport failures — an unreachable endpoint, a timeout — everywhere,
including `health()` and the export download, which may fetch a signed URL at
the object store rather than a control-plane path. No `httpx` exception escapes
this client.

## Running the service from Python

`skyrl-capture serve` is one caller of `CaptureService`; a training script that brings
capture up beside its inference server is another.

```python
from skyrl_capture.service import CaptureService

from skyrl_capture.config import UpstreamConfig, load_config

service = CaptureService(
    config=load_config().with_overrides(
        upstream=UpstreamConfig(
            type="tokens",
            url=f"{engine_url}/generate",
            model=model_name,
            tokenizer=tokenizer_name,
            max_model_len=32768,
        )
    ),
    data_dir="./capture-data",
    port=8080,
)
service.start(blocking=False)         # returns once it is serving

try:
    run_training(service.base_url)
finally:
    service.stop()
```

This is the same call `skyrl-capture serve` makes; the CLI passes
`blocking=True` and serves on its own thread. There is one serving path, so an
embedded caller and the command line cannot drift.

Nothing has to exist first: there is no database to install or start. State is
held in memory and the queue and payloads go under one directory.
`start(blocking=False)` returns only once the socket is bound and startup has
finished, so trajectory URLs handed out on the next line are usable rather than
a race. A service is single-use — `stop()` is safe to repeat, and restarting
means building another one. `with CaptureService(...) as service:` does both
ends for you.

Pass `record_dir` if the run's results should outlive the process -- without
it, everything this service captured goes when it stops.

The upstream is part of that config, so there is no second call to make after
`start()`. That is what makes a bootstrap safe to run twice: there is no
mutable definition two nodes could race to declare.

## Running many trials

Each trial needs its own trajectory: a route serves one trial, and finishing
revokes it.

```python
from concurrent.futures import ThreadPoolExecutor
from skyrl_capture.sdk import CaptureClient, capture

client = CaptureClient()

def run_trial(task):
    # environment() mutates os.environ, which is process-wide -- so with
    # threads, read trajectory.env() and pass it to the subprocess explicitly
    # rather than relying on the context manager.
    with capture(project="terminal-bench", labels=[task], client=client) as trajectory:
        subprocess.run(["python", "agent.py", task],
                       env={**os.environ, **trajectory.env()}, check=True)

with ThreadPoolExecutor(max_workers=8) as pool:
    pool.map(run_trial, tasks)
```

The comment matters: `environment()` is a process-wide mutation, so it is only
safe when one trial runs at a time in a process. For concurrent trials, use
`trajectory.env()` and pass it into each subprocess.
