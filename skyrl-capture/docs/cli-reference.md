# CLI reference

```
skyrl-capture [COMMAND] --help
skyrl-capture --version
```

Seven commands, and they are the product loop:

```
serve  →  run  →  view / annotate  →  export
```

`list` is the scripting escape hatch that finds IDs, and `health` is the
operator's diagnosis. Everything else a terminal used to do — reading one
trajectory, its calls, its graph — the [viewer](viewer.md) does better, for a
live run and a finished record alike.

Every command that talks to the control plane accepts `--endpoint`, or reads
`CAPTURE_ENDPOINT`. There is no control key: see
[api-reference.md](api-reference.md#authentication).

```bash
export CAPTURE_ENDPOINT=http://127.0.0.1:8080
```

## Running the service

```bash
skyrl-capture serve --record-dir DIR
           [--upstream-type T] [--upstream-url U] [--model M] [--tokenizer T]
           [--max-model-len N] [--host H] [--port P]
           [--viewer | --disable-viewer] [--upstream-module MODULE]...
```

Runs the proxy and the control plane against **one** inference server, and
writes what it captures.

```bash
UPSTREAM_API_KEY=$OPENAI_API_KEY skyrl-capture serve --record-dir ./traces \
  --upstream-type openai --upstream-url https://api.openai.com/v1

skyrl-capture serve --record-dir ./traces --mode tokens --upstream-type tokens \
  --upstream-url http://skyrl-router/generate \
  --model glm-5.2 --tokenizer zai-org/GLM-5.2 --max-model-len 32768
```

The upstream is fixed for the life of the process: there is nothing to register
afterwards, and changing it means relaunching. Every flag has an environment
equivalent (`CAPTURE_MODE`, `UPSTREAM_TYPE`, `UPSTREAM_URL`, `UPSTREAM_MODEL`,
`UPSTREAM_TOKENIZER`, `UPSTREAM_MAX_MODEL_LEN`), and the credential is
**only** an environment variable — `UPSTREAM_API_KEY` — so it never reaches a
shell history or a process listing.

`--mode` states which of the two the process is, rather than inferring it from
the type name: it decides which registry `--upstream-type` is resolved in, so a
contributed kind that is loaded by `--upstream-module` cannot be classified
before it exists.

An unknown `--upstream-type`, or a `tokens` upstream with no tokenizer, fails
here rather than on the first request.

`--record-dir DIR` is **required**, and is where the [record](record.md) goes:
one journal per trajectory in flight, one compiled record per finished one,
readable afterwards with no database and no server. Persistence is not a mode,
so a process with nowhere to write refuses to start rather than discovering
that at the first request. `CAPTURE_RECORD_DIR` is the environment equivalent.

Started against a directory that already holds trajectories, the process
**does not read it**. It picks up one journal when a request for that
trajectory arrives — so a replacement serving four trajectories does not scan a
hundred thousand files to find them.

`--disable-viewer` drops the read API and bulk exports from this replica,
leaving create, finish, annotate, health and capture itself. Exactly one
process per record directory should serve the viewer, because there is one
indexer per directory; a fleet runs one with it and the rest without.

To capture two providers at once, run two capture processes. They may share a
record directory — a trajectory's files are named by its id — as long as
routing gives each trajectory exactly one writer.

Writing happens in this process, always. There is no flag for it.

`--upstream-module` imports a module before serving, so an upstream kind that
module registers becomes a usable `--upstream-type`. This process only knows
the types it has imported, and a contributed wire lives in its own package:

```bash
skyrl-capture serve --port 8080 --upstream-module my_project.capture_upstream
```

Repeatable, `-u` for short. A module that will not import fails here rather than
later. See [tokens.md](tokens.md#which-wire-the-engine-speaks).

## Running a trial

```bash
skyrl-capture run \
  --project terminal-bench \
  --run-id run-2026-09-17a --task-id task-17 --step 7 \
  --tag task-17 \
  --annotate attempt=1 \
  --annotate rlvr_reward=1 \
  -- uv run harbor run --dataset terminal-bench@2.0 --agent terminus-2
```

Everything after `--` is your command, run unchanged with inherited
stdin/stdout/stderr.

`skyrl-capture run`:

1. creates a trajectory,
2. injects its base URL through the environment variables the upstream's client
   protocol uses (`OPENAI_BASE_URL`, or `ANTHROPIC_BASE_URL`) -- with a
   placeholder in the key variable, because most SDKs refuse to start without
   one and capture authenticates nothing inbound -- plus
   `SKYRL_CAPTURE_TRAJECTORY_ID`,
3. runs the child command,
4. finishes the trajectory,
5. exits with the child's exit code.

`--run-id`, `--task-id` and `--step` place the attempt in the hierarchy:
project / run / trajectory, with the task it attempted and the step that
produced it. All three are optional — a wrapped agent command is one trial and
belongs to no run — and the run is created on first use. The flag is
`--run-id` rather than `--run` because `run` is this command.

`SIGINT`, `SIGTERM`, and `SIGHUP` are trapped and forwarded to the child, and
the trajectory finishes with an `aborted` outcome. A launcher killed outright
leaves an unfinished journal, which nothing expires: finish it again by id when
you know the outcome, or leave it. A child killed by a signal exits `128 + N`
— `130` for `SIGINT` — which is what a shell reports and what `set -e` and a CI
runner are written against.

Useful flags:

| Flag | Effect |
| --- | --- |
| `--bodies sampled` | Capture metadata for every call but bodies for only a sample |
| `--tag NAME` | Repeatable. A bare string tag, for filtering |
| `--annotate k=v` | Repeatable. Values parse as JSON when possible, else as strings |

One `skyrl-capture run` is one trial. If a wrapped command internally runs several
trials, their boundaries cannot be recovered from inference requests alone —
split the command.

The trajectory finishes when the command does. There is no separate finish step:
an SDK caller has `Trajectory.finish()`, and a `run` caller has the child exiting.

## The viewer

```bash
skyrl-capture view [--api http://127.0.0.1:8080] [--port 8750]
skyrl-capture view --record ./traces
```

Opens the [viewer](viewer.md): projects, runs, the trajectories inside them, and
for one trajectory its **Tree** and **Path** views, its **Calls** tab and its
raw **JSON**. This is the inspection surface — there are no per-view terminal
commands, because one UI that reads both sources beats six commands that read
one each.

It is a Node program under `viewer/` and only ever an HTTP client of `/v1`,
which is why `--api` and `--record` are interchangeable: a capture process
serving the viewer and `view --record` mount the same app over the same reader.
With `--record` it starts that read API itself, in this process, so looking at
a run is one command rather than two — and because it reads files rather than a
proxy's memory, it works equally on a run still in progress.

A large directory is indexed in the background: the first page arrives at once,
the listing says `indexing` while the scan continues, and the pager withholds a
total until there is a stable one to give.

Needs Node 18 or newer on `PATH`; when there is none it says so and points at
the API, because every number the viewer shows comes from `/v1`.

TITO records carry the decoded text and exact per-token character offsets
captured with their token IDs, so viewing a record never loads a tokenizer.

## Listing

```bash
skyrl-capture list --project terminal-bench [--status finished] [--limit 50]
skyrl-capture list --run-id run-2026-09-17a --task-id task-17 --step 7
skyrl-capture list --project terminal-bench --ids        # IDs only, for scripting
skyrl-capture list --json
skyrl-capture list --record ./traces
```

Finds IDs and feeds scripts. `--run-id`, `--task-id` and `--step` filter at
every level of the hierarchy; `--step 0` means step zero, not "no step given".
There is no grid command — `task_id` and `step` are filters on one flat
listing, and nothing guarantees a task is attempted at every step.

The trajectory ID is never truncated, because it is the field you copy. A
trajectory whose capture dropped events is marked.

`--limit` pages, and `--cursor` continues. The cursor is opaque — a keyset from
the live service, an offset from a record — so it is handed back rather than
read. For detail on one trajectory, use `--json`, the [API](api-reference.md),
or the viewer.

`--record DIR` makes `list` and `export` read a [record directory](record.md)
instead of a running service — no database and no server. It reads the whole
directory before printing, because a command that is about to print a page
wants the whole answer and there is nobody waiting on a first paint.

## Annotating

```bash
skyrl-capture annotate tr_123 --annotate rlvr_reward=0.75
skyrl-capture annotate tr_123 --annotate critique='"incorrect tool"'
skyrl-capture annotate tr_123 --tag success --untag draft
skyrl-capture annotate tr_123 --unset draft_note
```

Rewards and labels usually arrive after the trial, which is why this is its own
command. Labels and annotations belong to the whole trajectory, not to a node,
and stay editable for its life: finishing does not seal them and no history is
kept. Annotation values parse as JSON when possible, so quote strings: `'"text"'`.

## Exporting

```bash
skyrl-capture export --project terminal-bench --format text-samples \
  --mask-abandoned --output terminal-bench-traces.jsonl

skyrl-capture export --trajectory tr_123 --format replay --output tr_123.jsonl

skyrl-capture export --record ./traces --run-id run-a \
  --format token-samples --output rl.jsonl
```

Creates the job, waits for it, and downloads the result. Formats: `graph`,
`replay`, `text-samples`, `token-samples` — one per use case.

For **one** trajectory there is usually no job to create: `finish` returns the
same rows, rendered from the record it just committed, in whichever format you
ask for. This command is for a run or a project, and for re-exporting later.

| Flag | Effect |
| --- | --- |
| `--run-id RUN` | Export one run's committed trajectories |
| `--allow-repeated-targets` | `text-samples`, `token-samples`: let one sampled message be a target in several rows. Off by default |
| `--mask-abandoned` | `text-samples`, `token-samples`: rows whose branch lost a race train on nothing |
| `--overlong-filtering` | `token-samples`: mask rollouts that stopped at the context limit |
| `--compression none\|gzip\|zst` | How `--output` is written. Default `none` |

Sample rows are never dropped; these flags only decide what `trainable` says.
See [exports.md](exports.md).

`--compression` decides what lands on disk, not how the artifact is stored:
`none` (default) writes `.jsonl`, `gzip` writes `.jsonl.gz`, `zst` writes
`.jsonl.zst`. Scope makes no difference — a project export is a concatenation
of its trajectories, in the same shape as a single one.

The default is uncompressed because the first thing you usually do with an
export is look at it, and `head` and `jq` cannot read zstd. The copy kept
under `exports/artifacts/` stays compressed regardless — on a 60-call
trajectory that is 9.7× for `graph` and 78× for `replay`, which is worth
keeping for storage and worth giving up for a file you are about to read.

An `--output` name whose extension disagrees with `--compression` is refused
rather than written, so a file never lies about its contents.

Omitting `--output` prints the job record — the download URL and the checksum —
and fetches nothing. `export` waits for the job itself; if you lose the terminal
before it lands, the job is still at `GET /v1/exports/{id}`.

With `--record` the export runs here, against a record directory, with no
service and no database. The four exporters are the same code either way, so
the artifact is identical.

## Health

```bash
skyrl-capture health
```

Service and capture health, as JSON: the same document `/healthz` serves. See
[operations.md](operations.md).

## Developer tooling

Verification and benchmarking are not part of the product loop, and live under
[`tools/`](../tools/README.md):

```bash
uv run python -m tools.verify.cli --record ./traces --run-id run-a \
  --engine-url http://127.0.0.1:8000/generate --model policy

uv run python -m tools.bench.cli --requests 4000 --concurrency 8 --processes 3
```

`verify.py` re-feeds a record's prompts to the engine and compares the
completions — the second [verification layer](verification.md). `bench.py`
compares capture-on against capture-off through the same proxy; see
[benchmarks.md](benchmarks.md).
