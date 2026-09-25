# The viewer

```bash
# against a running capture process
skyrl-capture view --api http://127.0.0.1:8080

# against a record directory, with no database and no capture process
skyrl-capture view --record ./traces
```

Then open <http://127.0.0.1:8750>.

The viewer is a separate program under [`viewer/`](../viewer), written in
Node, and it is **only ever an HTTP client of `/v1`**. That one fact is what
makes the two commands above interchangeable: they mount the same app over the
same reader, and **both read files** — a capture process serving the viewer
does not show you what a proxy is holding, it shows you what survived. There is
no offline mode to fall out of date, and nothing on screen that a crash would
erase. It is also R9 read from the other side: the viewer has no private API,
so anything on screen can be fetched with `curl`.

One process per record directory serves it. A capture replica started with
`--disable-viewer` has no read routes at all; run the viewer on one replica, or
standalone against the shared directory.

It has **no dependencies and no build**: Node's own HTTP server, ES modules
served as they sit. `node_modules` and a lockfile are both CI failures. The
only thing it needs that the Python package does not is Node 18 or newer; when
that is missing, `skyrl-capture view` says so and points at the API.

## The shape

Projects and runs down the left, the run in the middle, one trajectory in a
drawer on the right. The drawer rather than a page because the question is
nearly always comparative — *this* attempt against the others in its run — and
navigating away to answer it loses the place you were looking from.

The route is in the hash, so `#/run/<run>/<trajectory>` is a link to an
attempt that anyone can paste.

## What "complete" means

Written down before it was built, because this is the largest surface in the
system and the one most likely to grow without a boundary. Five things, and
when they work the viewer is done.

1. **The run's attempts, filtered.** A flat list of trajectories, narrowed by
   step, task, status or any annotation, and clicking one opens it. Not a
   matrix: nothing guarantees the same task is attempted at every step, so a
   `task × step` grid is a dense view of sparse data that degrades into empty
   cells. `step` is a filter; `task_id` is one dimension among whatever else a
   harness annotates. See [design/run-dimensions.md](design/run-dimensions.md).

2. **The trajectory tree.** One node per message, one parent per node, forks
   marked, with the run, task and step this attempt belongs to.

3. **Token inspection.** A trajectory captured in token space rendered as
   decoded text with the mask laid over it: four kinds (`sampled`, `replayed`,
   `scaffold`, `given`), cut on the mask *and* the turn boundary, every block
   carrying its token range, the mask drawn as a proportional strip above the
   text, and special tokens shown rather than stripped.

4. **Forks read where they happen.** Opening a node in the tree shows it and
   what follows it; a branch point lays its branches side by side. Described
   below.

5. **Both sources, and a run in progress.** Everything above works against a
   capture process and against a [record directory](record.md), through the
   same `/v1` paths — and against a run that has not finished, because a
   trajectory in flight is a journal the reader replays.

### Not in it

Stated so the boundary holds:

- **Training metrics.** Reward curves, loss, throughput belong to the trainer.
  The viewer's job stops at *is this record right*. Reward appears in the list
  because it is how you choose which attempt to open, not as a measurement.
- **Editing captured data.** Labels and annotations are writable, as they
  already were. Tokens are not editable by anyone, here least of all.
- **Re-tokenizing in the browser.** The capture process stored text and offsets
  with the IDs. The browser owns no tokenizer, and a second tokenizer would be
  a second answer.

## Record health: what is worth looking at

The band above the list is the part that earns its place during a training run.
Nobody scrolling a run is reading trajectories; they are looking for the one
that is not like the others, and the useful signal is rarely a number. Each
flag is derived from what `/v1` already returns, and clicking one filters the
table to it.

| Flag | Level | What it means |
| --- | --- | --- |
| `calls-missing` | error | the harness made calls this trajectory never received |
| `incomplete` | error | capture did not see the whole trajectory |
| `after-close` | error | calls arrived after the trajectory was finished |
| `no-logprobs` | error | sampled tokens without logprobs cannot be importance-weighted |
| `replayed` | warn | assistant text the model did not produce is in the prompt |
| `no-train` | warn | nothing in this path is trainable: the export row would be empty |
| `truncated` | warn | generation stopped on the token budget, not on the model |
| `degraded` | warn | blocks were cut on the mask alone, so the scaffold is not separable |
| `abandoned` | info | a branch the harness walked away from |
| `masked` | info | excluded from training, and the record says why |
| `forked` | info | more than one root-to-leaf path: the history diverged |

None of these is a training metric. Every one of them is a way the *record* can
be wrong, or a way the run can be going wrong that the record can see.

## The run's attempts

`GET /v1/trajectories?run_id=<id>` is the list, and every other question is a
filter on it:

```
?run_id=<id>&step=0            attempts at one step
?run_id=<id>&task_id=t1        one task across the run
?run_id=<id>&status=error      what went wrong
```

A run's entry in `GET /v1/runs` carries `steps` — the values the step filter
can offer — along with `trajectory_count` and `task_count`. They are counts and
a set of values, not the axes of anything.

- Reward comes from `annotations.reward` when that value is numeric. There is
  no reward column; a reward is an annotation like any other.
- Two attempts at the same task and step are **two rows**. A grid had to decide
  what a resample meant and showed their mean; a list does not have to decide.
- A task never attempted at some step is simply absent from that step's list,
  which is the same thing as "not run" without needing a rule to tell it apart
  from a zero.

The same view is in the terminal: `skyrl-capture list --run-id <run> --step 0`.

## Token inspection

`GET /v1/trajectories/{id}/paths` returns one entry per root-to-leaf path,
because **a path is what an export row is** — so what is read on screen and
what reaches the trainer are the same object.

Each block carries `kind`, `role`, `start`, `end`, `token_count` and `text`.
The kinds are structural rather than textual:

| Kind | What it is | Why it is its own kind |
| --- | --- | --- |
| `sampled` | the model produced it and it is in the loss | the thing being trained on |
| `replayed` | assistant text the model did **not** produce | looks like output, is not — the case tokens mode exists for |
| `scaffold` | the template's generation prefix inside an assistant turn | the `sampled_start` boundary, made visible |
| `given` | user messages, tool results, system prompt | context |

`scaffold` is *an assistant block whose next block is trainable* — no string
matching, which matters because SkyRL's template puts `<think>\n` in the
scaffold and the built-in one does not.

Each path says how its blocks were cut. `blocking: "mask+turns"` distinguishes
all four kinds; `"mask"` means the tokenizer has no `<|im_start|>` turn marker,
so roles cannot be read and `replayed` and `scaffold` are **unreachable** —
everything untrainable comes back as `given`. What is trainable is still
correct either way.

`start` and `end` index `input_ids`, `loss_mask` and `rollout_logprobs`
identically, so a range read off the screen can be pasted into whatever is
being debugged.

### Forks, compared (R7)

A fork is the one thing a tree draws badly: the tree says *that* there was a
branch, and what anyone needs is *where* the histories stopped matching — which
is the harness's behaviour, and regularly a surprise to whoever wrote the
harness. The **Forks** tab picks two paths, names the last node they agree on,
dims everything before it and shows the rest side by side.

It also checks the **train-once rule**: a sampled node reachable from several
branches must be in the loss exactly once, or it is trained on twice. That is a
statement about paths in the plural, so it is checkable here and nowhere else.
Each block carries the `node_id` it came from, which is what makes the check
possible; where something bridged between the nodes and the export row the
server returns `node_id: null` rather than an attribution that is quietly one
node out.

### Logprobs on the text's axis (R8)

Off by default. Each sampled block gets a ribbon above it, one segment per
token, shaded by `exp(logprob)` — paler is less confident. `rollout_logprobs`
shares an index with `input_ids`, so a block's `[start:end]` slices it directly
and the shading costs no alignment work. It is how a misalignment announces
itself: confident text over implausible values.

### Token text

A TITO node carries its exact token IDs together with the decoded text and
verified character offsets produced by the same renderer during capture.
`skyrl-capture view` therefore needs neither the tokenizer nor the tokenization
extra. Token `i` is always `text[offsets[i]:offsets[i+1]]`, which links the
readable span, its ID, mask, and logprob on hover.

Capture refuses a TITO turn if the renderer cannot produce an exact mapping.
Neither the live `/paths` route nor the record viewer decodes or retokenizes.

## Watching a run that is still going

A record directory changes while it is being read: the process that owns it is
appending to journals, and trajectories are moving from `active/` to
`committed/` as they finish. The viewer is built for that.

- The index is **progressive**. A directory with a hundred thousand
  trajectories in it does not delay the first page: listings answer at once,
  carry `indexing` while the scan continues, and withhold `total` until there
  is a stable one — a pager prints whatever count it is given as fact, so it is
  not given a provisional one.
- A refresh reaches **both directories**, so a trajectory that has just
  finished is not shown twice or lost for a moment.
- A committed record is written once *and rewritten* -- a reward that arrives
  after a trajectory finished replaces it with a new revision -- so the index
  compares each file's size and modification time rather than only its id. A
  viewer sharing a process with the writer is told what changed; a standalone
  one over a shared volume has only the filesystem, and this is what it reads.
- A refresh also reaches the **open trajectory**, which is the one somebody
  watching a run is looking at.
- There is no WebSocket, no server push and no change feed. It polls, which is
  what a reader of files can do without the writer knowing it exists.

## What the viewer cannot answer

`skyrl-capture view` is a reader. Nothing in it creates, annotates or finishes,
because those belong to the process that owns the trajectory. Two reads are
also absent, by construction rather than by omission:

- **Raw HTTP payloads.** The record holds them, and the viewer shows the graph
  and the tokens; there is no route that serves a request body.
- **Node payloads by node id alone.** They are embedded in the trajectory, so
  they are read through `/v1/trajectories/{id}/paths`.

Exporting from a record is a shell command rather than a button:

```bash
skyrl-capture export --record ./traces --run-id run-a --format token-samples \
  --output rl.jsonl
```
