# The record format

A record is a directory. It holds one append-only journal per trajectory in
flight and one compiled record per finished one, and between them they are the
whole of what a capture run produced. There is no global log, no global
sequence, and no other encoding of the graph.

This document is the normative specification. The framing is fixed-width,
little-endian and dependency-free, so a reader or a writer in another language
needs nothing but this file and a zstd decoder.

```
<record>/
  manifest.json                         format versions and upstream provenance
  active/
    <xx>/tr_....capture                 a journal per trajectory in flight
  committed/
    <xx>/tr_....json.zst                a compiled record per finished trajectory
    <xx>/tr_....head.json               its document, which every listing reads
  exports/
    jobs/<export id>.json               export job records
    artifacts/<export id>/<name>.zst    export artifacts
```

`<xx>` is the first two hexadecimal characters of a BLAKE2b-1 hash of the
trajectory id -- a stable function of the id alone, so every process computes
the same path and none has to allocate one. A shard directory is removed as
soon as its last journal is, so a record whose work is finished does not read
as 256 empty directories.

`tr_....head.json` is the trajectory document -- what `/v1/trajectories`
returns -- written beside its record as plain, indented JSON. Every listing
reads these and no records: the dozen fields a listing needs sit at the front
of a trajectory, and decompressing every exchange, every node and every token
array to reach them is the difference between a viewer that opens over a large
directory and one that does not. `jq` reads them for the same reason, with no
capture process, no zstd and no tokenizer.

The layout stays flat and addressed by id. Grouping by project and run is not
a directory structure but a query: `/v1/runs` computes it from these documents
on read, because a run is the grouping its members imply. Putting that
hierarchy on disk would turn arbitrary user strings into path segments and
give a trajectory a location its own id could no longer find -- and a reward
that arrives after a trajectory finished arrives with an id and nothing else.

The headers are authoritative: a listing is what they say, and nothing on the
read path falls back to reading records. Two rules make that safe. A header is
written before the journal that produced it is deleted, so a crash between
them leaves the journal and replaying it writes both again. And a header is
addressed by id like its record, so a later revision -- a reward arriving late
-- replaces it where it already is, atomically. `skyrl-capture reindex`
rebuilds the headers from the records for a directory those rules never
covered: one written by an older build, or assembled by copying records in.

## Authority

- **A trajectory's own files are the record of it.** Nothing else is.
- Where both forms exist -- the window between the committed rename and the
  journal's deletion, or for good if a process died in it -- **the committed
  record wins, for writers as well as readers**. A journal beside a committed
  record is redundant, not a trajectory to reopen: whoever notices removes it,
  and a request for that trajectory gets `410`.
- `manifest.json` identifies the versions and names the upstream the run
  captured, credential-free. It holds nothing that changes while a run is in
  progress, so several capture processes sharing a directory never race to
  rewrite it. A reader may use it to refuse a record from a format it does not
  understand; it needs nothing else from it.
- `exports/` holds derived artifacts. Deleting it loses no capture.

## Journal framing

```
journal := header record*
header  := magic[8] version[u16] flags[u16] reserved[u32]
record  := length[u32] crc32[u32] kind[u16] flags[u16] body[length]
body    := doc_len[u32] doc[doc_len] blob_count[u16] (blob_len[u32] blob)*
```

- `magic` is `ICAPTRJ1`; `version` is `JOURNAL_FORMAT_VERSION`.
- `length` counts `body` only. `crc32` is over `body`.
- **There is no sequence in the frame.** A journal is one trajectory's, and its
  records are in the order they were appended. The per-trajectory exchange
  sequence lives inside `ExchangeCommitted`, where it orders exchanges rather
  than files.
- `flags & 1` (`FLAG_ZSTD`) means `body` is zstd-compressed as a whole. Records
  above a few kilobytes are compressed; below that the frame costs more than it
  saves.
- `doc` is JSON and carries every field of the record except raw byte strings.
  Those travel as `blob`s, in a fixed order per kind, so a captured request
  body is stored as the bytes that were sent rather than as base64 of them --
  which would cost a third of the record's size and an encode on every commit.

## Record kinds

Kinds are additive and are never renumbered. A reader that meets a kind it does
not know **skips it by its length** and carries on, so a journal written by a
newer build stays readable.

| Kind | Record | Says |
| ---: | --- | --- |
| 1 | `TrajectoryCreated` | The header: project, run, task, step, mode, upstream, the creation-body hash |
| 2 | `ExchangeCommitted` | One complete exchange, its graph delta, its bytes, its exact tokens |
| 3 | `ExchangeDeliveryConfirmed` | The complete response reached the downstream transport |
| 4 | `CaptureGap` | Exchanges observed and not captured -- or, with a count of zero, uncertainty |
| 5 | `MetadataUpdated` | Labels and annotations merged |
| 6 | `FinishRequested` | The route closed, with the final metadata and the finish hash |
| 7 | `TrajectoryPoisoned` | A turn could not be attributed; the graph may not be extended |

A record says what happened, not how to store it. `ExchangeCommitted` carries
the exchange as capture derived it, the `GraphDelta` capture decided, and the
raw bodies -- so applying it reproduces exactly what the writing process held,
and nothing has to be re-derived at read time. There are **no partial-stream
records**: an exchange is written whole or not at all.

**A delta carries whatever nodes the journal does not yet have, not only the
ones its own exchange introduced.** Normally those are the same set. They
differ after a refused or failed append: that exchange's row is gone for good,
but every later record still references the nodes it created, so a journal that
had only its own would be full of dangling parents -- which a recovery or a
compile turns into an orphaned subtree and a truncated path in every export off
it. The exchange row is expendable and the graph is not, so the nodes ride on
the next record that can carry them, ahead of its own and in graph order. The
loss is reported as `calls_missing`, and `capture_gaps_unwritten` and
`undelivered_nodes` on `/healthz` say what is still waiting for a record.

It also carries a `request_fingerprint`, a hash of the standard request body.
It is **diagnostic only and must never deduplicate**: two turns of one
trajectory may send byte-identical requests on purpose, and that is resampling.

`CaptureGap` with `count == 0` is uncertainty rather than loss -- what a
replacement process writes when it adopts a text trajectory it cannot vouch
for. It does not make the trajectory incomplete; it makes it uncertain.

Corrections are later records. A reward computed after a trajectory finished is
applied to the committed record instead, which is rewritten with a new
`revision`; nothing in a journal is ever rewritten in place.

## The committed record

`<record>/committed/<xx>/<id>.json.zst` is a zstd frame around one JSON
document:

```json
{
  "record_version": 5, "schema_version": 4, "derivation_version": 2,
  "revision": 0, "finish_request_hash": "…",
  "trajectory": { "id": "tr_…", "status": "finished", "capture": { … }, … },
  "exchanges": [
    { "id": "ex_…", "sequence": 0, "row": { … },
      "delivery_confirmed": true, "delivery_uncertain": false,
      "request":  {"encoding": "utf-8", "data": "{…}"},
      "response": {"encoding": "utf-8", "data": "{…}"},
      "chunks": { … }, "tokens": { … } }
  ],
  "nodes": [ { "id": "nd_…", "parent_id": null, … } ],
  "node_order": ["nd_…"]
}
```

Bodies are stored as text when they decode as UTF-8 -- which they are, being
JSON request and response payloads -- and base64 otherwise, named by
`encoding`. That keeps a committed record legible to anything that can read
JSON, and compresses far better than base64 of the same bytes.

It is written by serializing and compressing off the event loop, writing a
temporary file **in the same directory**, fsyncing it, renaming it onto the
deterministic path, and fsyncing the directory. Rename is atomic only within a
filesystem, and a record directory may be a mount of its own, which is why the
temporary file is not in `/tmp`.

Compilation is deterministic: given the same journal and the same finish
request, two processes produce the same bytes. That is what makes a retry after
a crash safe -- whichever attempt reaches the rename first, the record is the
same record.

## Reading a record

For one trajectory: read `committed/`, and if there is nothing there, replay
`active/`. Replaying is applying the journal's records in order to an empty
aggregate; applying an exchange is idempotent by its id, so a re-read
converges rather than duplicating.

For a listing: scan both directories, prefer the committed entry where a
trajectory appears in both, and sort by id -- ids are time-sortable, so
descending is newest-first and is also what a cursor pages through.

An active journal can be read **incrementally**: a scan reports how many bytes
of whole records it consumed, and the next scan starts there. That is what
makes a viewer's refresh proportional to what was appended rather than to the
file.

## A torn tail

A writer killed mid-append leaves a partial record, and a partial record is
detected two ways: a body shorter than its `length`, or a `crc32` that does not
match. Either ends the scan. **A reader stops there and reports the
truncation.** It does not skip forward looking for the next plausible header.

A writer that adopts a journal ending in a tear **truncates to the last whole
record before appending**. Appending after the tear would produce records that
are durable and invisible, which is worse than not writing them.

A journal whose first whole record is not `TrajectoryCreated` was torn before
creation became durable. There is nothing to attach the rest of it to, and it
is ignored rather than guessed at.

## A journal that has been finished

`finish` deletes the journal once the committed record is visible, and an
append after that **fails**. It does not re-create the file -- that would leave
an orphan journal for a finished trajectory, invisible because readers prefer
the committed record, and never collected -- and it does not drop the record
quietly, which would let a TITO response close cleanly with its exchange
nowhere.

A turn that outlives `FINISH_GRACE_SECONDS` does not reach that state,
because `finish` does not finalize while one is open: it returns a retryable
`503` and leaves the trajectory alone. What the closed journal is for is the
narrower race -- a turn that was inside the barrier when it passed and lands a
moment after the record is written. Text counts a gap; TITO fails the
connection.

## Durability

`RECORD_FSYNC` is `always` (the default), `interval` or `never`.

**`always` is what the product contract rests on.** Token capture promises that
a cleanly closed response has its exact exchange on disk, and that promise is
an fsync. Creation makes the same promise: a route handed out for a trajectory
that is not on disk is a route no replacement process could serve.

The other two exist to measure what that costs, and they weaken it. Under
`interval` a crash loses whatever the last window held; under `never`, whatever
the operating system had not flushed.

Text capture is unaffected either way at the request path: its appends run
behind the response, so the fsync costs writer throughput rather than inference
latency.

## Versioning

| Constant | Bumped when |
| --- | --- |
| `JOURNAL_FORMAT_VERSION` | The journal framing or its record kinds change |
| `RECORD_FORMAT_VERSION` | The directory layout or the committed record's shape changes |
| `SCHEMA_VERSION` | The projected trajectory or exchange shape changes |
| `DERIVATION_VERSION` | Prefix matching or graph derivation changes |

A record whose `record_version` this build does not read is refused with a
message naming both versions. A global event log written before per-trajectory
persistence is rejected by its magic: there is no migration, and it must be
exported with the build that wrote it or re-captured. Three things read this
format -- the exporters, the viewer, and anything reading it later -- so a
silent misread is worse than a stop.
