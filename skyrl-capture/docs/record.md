# The record

A capture run as a directory: one append-only journal per trajectory in
flight, and one compiled record per finished one. No database, no server, and
`rm -rf` is the retention policy.

```bash
skyrl-capture serve --upstream-url ... --record-dir ./traces
```

```
traces/
  manifest.json                        versions, and the upstream, credential-free
  active/
    3f/tr_01J....capture               a journal per trajectory in flight
  committed/
    a1/tr_01H....json.zst              a compiled record per finished trajectory
    a1/tr_01H....head.json             its document, which every listing reads
  exports/
    jobs/exp_....json                  what was asked for
    artifacts/exp_.../run-graph.jsonl.zst   what it produced
```

`--record-dir` is required. Persistence is not a mode: every trajectory is
written as it runs, and a process with nowhere to write refuses to start.

A trajectory is in the record from the moment it is created -- creation is
durable before the route is handed back -- so a record holds work in progress
as well as finished work. Reading needs nothing but the directory:

```bash
skyrl-capture view  --record ./traces
skyrl-capture list  --run-id run-2026-09-17a --step 0 --record ./traces
skyrl-capture list  --record ./traces --task-id task-17
skyrl-capture export --record ./traces --run-id run-2026-09-17a \
  --format token-samples --output rl.jsonl
```

## Two forms, and why

**An active journal is a history.** Records are appended as they happen:
creation, each committed exchange with its graph delta and its raw bodies,
delivery confirmations, capture gaps, metadata edits, the finish request. It is
what a replacement process picks up, and it is append-only because a crash
must leave something readable rather than something half-rewritten.

**A committed record is an artifact.** `finish` reads the journal back,
compiles it once, and writes the result: the public document, the exchanges
with their payloads, and the graph nodes in order, including the exact token
arrays. Opening one runs no reducer and invokes no tokenizer, which is what
makes the viewer's detail view a file read rather than a replay.

The path of each is a deterministic function of the trajectory id -- two hex
characters of a stable hash, then the id -- so nothing has to allocate a
location and two processes never have to agree on one.

## Text comes with the tokens

A `tokens`-mode node carries its own decoded text, cut where a path is cut: at
the turn marker and at the sampled boundary. So a block's text is the
concatenation of the whole segments it spans, and **reading a record needs no
tokenizer** -- not `skyrl-capture[tokens]`, not `transformers`, and not the
model. That is what makes "a directory you can copy to a laptop" true rather
than aspirational.

The process that writes it rendered those ids and still has the renderer
loaded, so the decode is free there and a dependency everywhere else. It costs
about 2% of the compressed record, because the text largely repeats what
`payload.message` already holds.

Token IDs stay authoritative for training. The captured text and verified
offsets are authoritative for presentation; a TITO commit is refused if the
renderer cannot produce that exact association. Reading never loads a
tokenizer.

## Why a read off it is identical

There is one reader. `RecordReader` prefers a committed record where one
exists and replays the journal where one does not, and both come back as the
same view -- the one the exporters, the `paths` route and the viewer all take.
So there is no second implementation of a read path to drift:
`tests/test_record.py` renders an export from the live aggregate and from the
committed record and compares them, and `tests/test_reader_contract.py` fetches
every read route from a running process and from its directory and diffs the
documents.

This is why the raw bodies live in the record rather than in an object store. A
record is meant to be a directory you can copy; a URI pointing at a bucket
somebody else owns is not.

## Recovery

A process started against an existing record directory does **not** read it.
There is no startup replay: a replacement that serves four trajectories does
not read a hundred thousand journals to find them. It recovers one when a
request for it arrives, truncating a torn tail first so the next append lands
after the last whole record.

What a recovery cannot know, it says:

- **Text capture** commits behind the response, so a previous process may have
  served an exchange inside its commit window and died before writing it.
  Nothing identifies which, so the doubt is trajectory-wide:
  `recovery_uncertain`. It is written into the journal, not merely held in
  memory, so a viewer sees what the writer knows.
- **Token capture** commits before the response closes, so every exchange it
  captured is exact. What it cannot vouch for is whether the client received
  one, which is per exchange: `delivery_uncertain`. It is resolved only by
  evidence -- a later request whose own message history contains that assistant
  output -- and never by assuming the absence of one.

A finished trajectory is evicted from memory and its journal is deleted once
the committed record is visible. An unfinished one stays until somebody
finishes it; nothing expires it on a clock.

## Versioning

`manifest.json` carries `record_version` and `journal_format_version`, and
nothing that changes while a run is in progress -- so several capture processes
can share a directory without racing to rewrite it. Three things read this
format — the exporters, the viewer, and eventually an upload path into whatever
a deployment keeps long-term — so it is a public format, not a convenience
dump. A record from a version this build does not read is refused with a
message rather than misread, and a global event log from before per-trajectory
persistence is rejected outright: there is no migration.
[design/record-format.md](design/record-format.md) is the specification.

## The edges

- **A torn tail stops a reader**, which says so rather than guessing past it. A
  writer that adopts such a journal truncates to the last whole record first,
  because a record written behind a tear would be durable and invisible.
- **One writer per trajectory, and routing is what guarantees it.** Two
  processes appending to one journal interleave records that are each
  individually valid, so nothing in the format can detect it. Several replicas
  may share one directory; they must not share a trajectory. See
  [operations.md](operations.md#routing-pinning-a-trajectory-to-a-replica).
- **One indexer per directory.** The viewer scans it; running two is running
  the same scan twice. Other replicas use `--disable-viewer`.
- **No cross-run deduplication.** A shared prefix is stored once per
  trajectory, not once per fleet. Storage efficiency traded for a run you can
  delete with one `rm -rf`.
- **A failed append is a gap, not an exception**, so a full disk cannot take
  text inference down. It shows up as `commits.failures` and
  `commits.unwritten_gaps` on `/healthz`, and it is written into the journal as
  soon as the disk takes anything again. Alert on it. Token capture is the
  exception: there the response fails instead.
- **A lost exchange never costs the graph.** Its row is gone, but the nodes it
  created ride on the next record that can carry them -- otherwise every later
  record would reference ancestors the journal does not contain, and a recovery
  or an export would produce a path that starts in the middle.
- **A finished journal is closed for good.** An append after the record is
  committed fails rather than re-creating the file or being dropped, so a turn
  that outlived `FINISH_GRACE_SECONDS` is a counted gap in text mode and a
  failed connection in token mode. A journal left beside a committed record --
  by a process that died between the two -- is removed rather than adopted;
  the committed record wins for writers as well as readers.
- **Reading is a directory scan**, which is progressive and is not a query
  engine. What you do not get is cross-run queries or anything resembling a
  join -- the shape to reach for is an export into whatever does.
