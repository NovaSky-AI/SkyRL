# skycap record format, version 1

This is the contract between skycap, which writes trajectories, and any reader
of them, such as the skycap viewer. A reader that follows this document needs
no Python and no tokenizer.

## Files

A record directory holds, per trajectory `{id}`:

| File | Present | Holds |
| --- | --- | --- |
| `{id}.json.zst` | always | the document (graph) |
| `{id}.tokens.zst` | optional: token mode only, when any node has tokens; never in text mode | token ids, logprobs, the text the tokens decode to, and each token's byte offset in it |
| `{id}.experts.zst` | optional: when routed experts were captured (R3) | routed experts |
| `{id}.sampling_mask.zst` | optional: when a captured sampling mask has at least one row | per sampled token, the ids it could have been drawn from |

Every sidecar is optional. The document's `sidecars` manifest lists the ones
the trajectory captured, and a kind missing from it is one the trajectory
doesn't have.

Every file is exactly one zstd frame: compress the whole payload in one call
and write it once. Never add to a file that already exists, whether by
appending a second compressed chunk or by flushing a frame partway through a
stream. A file with more than one frame is valid zstd, but many decoders,
including Node's and the `zstandard` Python reader skycap itself uses, stop
after the first frame without raising an error. The reader gets only the first
chunk of the data and has no sign that anything is missing. To change a file,
rewrite the whole file as a single frame.
A trajectory is not partially written when it is live, it is only written when it ends.
Each document has a `sidecars` field naming the sidecar files that belong to
it. The writer writes those sidecars before the document, and writes every file
under a temporary name and then renames it, so no file is ever seen
half-written. In a record directory, if a document exists, every sidecar named
in its `sidecars` field exists too. A crash can leave sidecars with no
document, but never a document with a missing sidecar.

A copy of a record directory, such as a mirror that leaves sidecar kinds out,
may lack sidecar files its manifest lists. A reader treats a listed sidecar
whose file is missing as absent, as if the trajectory hadn't captured it, and
doesn't tell it apart from one that went missing: without `tokens` it has
message text only and should say so; without `experts` or `sampling_mask` it
loses nothing a reader shows, since only training uses them, and training
takes its samples from `finish`.
A reader lists trajectories by listing `*.json.zst`.

### A mirror

A server started with a record mirror (`--record-mirror URL`, an fsspec URL
such as `s3://bucket/run-7`) also copies each trajectory's files, under the
same names, to `{URL}/{name}`, after writing them to the record directory.
The copy is made in the background, sidecars before the document, so a
mirrored document also has its sidecars beside it. The copy fails open: a
record the store never got is missing from the mirror (the server's
`/healthz` counts them under `record_mirror`), and the record directory is
always complete. A reader reads a mirror exactly as it reads a record
directory.

A mirror can leave sidecar kinds out (`--record-mirror-config '{"exclude":
["experts"]}'`), e.g. when the remote copy is for reading rather than
retraining. It still copies the document unchanged, so the document lists
sidecars the mirror doesn't hold, which a reader reads as absent (above). The
record directory's copy is untouched. Leaving out `tokens` is allowed, but a
reader of the mirror then has message text only.

### Where a trajectory's record is

`finish` answers with the document's location as `record`, or `null` when the
server has no record directory or couldn't write the record:

```json
{"record": {"host": "10.0.0.5",
            "path": "/data/record/tr_ab12.json.zst",
            "mirror": "s3://bucket/run-7/tr_ab12.json.zst",
            "files": ["tr_ab12.tokens.zst", "tr_ab12.json.zst"]}}
```

`path` is on the server's own disk, and `host` is that machine: its IP address
(or hostname), so another node can reach a record that exists only there, e.g.
`scp 10.0.0.5:/data/record/tr_ab12.json.zst .` (an IPv6 host is bracketed
there, `[fe80::1]:/data/...`). The server reports the address it is reached
at, or its primary IP when it has none to report; `--record-host` sets it. `mirror` is null without a mirror, and
otherwise where the copy is going: it may not be there yet, or at all. The
sidecars are beside the document in both places.

`files` names the record's files, sidecars first and the document last. With a
mirror they are the files the mirror holds (or will hold), so the kinds its
`exclude` leaves out are not listed; without one they are the files in the
record directory. A sidecar the trajectory didn't capture is never listed.
Each file is beside the document: `{dirname(mirror)}/{name}`, or
`{dirname(path)}/{name}` without a mirror.

## The document

The decompressed document is a UTF-8 JSON object:

| Field | Type | Meaning |
| --- | --- | --- |
| `format_version` | int | `1`. Covers both this document's shape and the files' encoding. A reader refuses a version it doesn't know |
| `id` | string | the trajectory id |
| `status` | string | `finished`, `failed`, `abandoned` (idle past the TTL) or `open` (written at shutdown) |
| `ended` | bool | whether the trajectory was ended (by `finish` or the TTL). False for one written at shutdown, which may be sealed as `failed` and still waiting for its finish |
| `meta` | object | what the creator passed at create |
| `capture` | object | how it was captured: `mode` (`text` or `tokens`), and for tokens the `engine`, `tokenizer`, `logprobs_mode` (`processed_logprobs` means logprobs are over the truncated, renormalized distribution), any `sampling_overrides`, and `use_raw_content` (whether replies carried the completion's own text as `content`, with no reasoning or tool-call parsing) |
| `annotations` | object | what the creator passed at finish, e.g. `{"reward": 1.0}` |
| `created_at`, `finished_at` | float or null | Unix seconds |
| `tools` | object | tool-set hash → the tool list, as sent |
| `failures` | array | calls that produced no node: `{t, status, error, input_leaf}` |
| `retries` | object | SDK retries answered from the original call: `{replayed, coalesced}` counts |
| `samples` | object or null | what `finish` returned as training samples: `{paths, rows}`. `paths` names the path rule that picked them: `all` (a row per root-to-leaf path, each model node a target in exactly one), `final` (one row, the path to the last model call's reply, every model node on it a target), or a custom rule's name. Each row is `{leaf, targets}`: the node its path ends at (the path is that node and its ancestors) and the model node ids it trains. No node is a target in two rows. Null for a trajectory not ended by `finish`, or one whose `finish` rule raised (a later `finish` that succeeds records it) |
| `nodes` | array | the graph, in creation order (below) |
| `sidecars` | object | kind → sidecar manifest (below). Empty in text mode, and in token mode when no node has tokens |

Fields a reader doesn't know are ignored. Adding a field does not change
`format_version`. Removing or redefining one does.

### Nodes

Node `i` is `nodes[i]`, and `nodes[i].id == i`. A node is one message:

| Field | Meaning |
| --- | --- |
| `parent` | parent node id, or null for a root |
| `depth` | distance from its root |
| `role` | the message's role (may be null for a non-message item) |
| `author` | `client` (sent by the harness) or `model` (sampled) |
| `message` | the message, exactly as the harness sent or received it |
| `match_hash`, `delta_hash` | identity hashes (see the graph module) |
| `created_at` | Unix seconds |
| `calls` | model-authored nodes: every call that produced this output, `{t_start, t_end, model, sampling, usage, finish_reason, tools, bridged}`, where `tools` is the key of the call's tool set in the document's `tools`, or null, and `bridged` (token mode) is whether the call's prompt extended an earlier call's prompt and completion token for token: `false` when it was rendered from the messages instead, null for a trajectory's first call or text mode |
| `shadowed_by` | null, or the sibling that a later request's matching message is matched to instead of this node: the latest model-authored sibling with the same match hash, or the latest client-authored one if there is none. In text mode, later history always continues from that sibling. In token mode, it continues from that sibling only if the turn's tokens agree with it: the turn extends a model call on that path, or the full render reproduces the sibling's tokens. Otherwise the message is stored as a client node under the same parent. That reuses a client sibling with exactly the rendered tokens, which can be this node, or else creates a new client sibling. A shadowed model-authored node never receives later history |
| `tokens` | null in text mode, else this node's slices of the sidecars (below) |

Every root-to-leaf path is one conversation as a model call saw it. Paths are
ordered by their leaf's id. A model node is a training target in the first
path that contains it, so a node shared by several paths trains once.

### A node's `tokens`

| Field | Meaning |
| --- | --- |
| `offset`, `length` | the node's tokens are positions `[offset, offset + length)` of the `tokens` sidecar's per-token arrays |
| `sampled_start` | null for a client node. For a model node, tokens before it are template scaffold and the rest were sampled |
| `has_logprobs` | whether `logprobs` holds real values for this node |
| `text_offset`, `text_bytes` | the node's text is bytes `[text_offset, text_offset + text_bytes)` of `text`; `text_offset` is null when no text was recorded |
| `experts_offset`, `experts_rows` | the node's rows of `routed_experts`; offset null when absent |
| `mask_offset`, `mask_rows` | the node's rows of the sampling mask (one per sampled token); offset null when absent. A node with `mask_rows` 0 has no rows even when its offset is set, and the sidecar may not exist |
| `media` | the multimodal items (images) whose placeholder tokens are in the node, in order: `{modality, offset, length, hash}`, where `offset` is relative to the node's first token and `hash` is the item's content hash. The item itself is in the node's `message`; its processed arrays are not recorded, and a repeated `finish` of a written trajectory rebuilds them from the message. Empty for a model node, and for a node with no images |

## Sidecars

A sidecar decompresses to raw arrays, little-endian and C-ordered, each
starting at a byte offset that is a multiple of 8. The document's `sidecars`
entry for a kind is:

```json
{"file": "tr_ab12.tokens.zst",
 "arrays": {"token_ids":    {"dtype": "int32",   "shape": [N], "offset": 0},
            "logprobs":     {"dtype": "float64", "shape": [N], "offset": 4N rounded up to 8},
            "text_offsets": {"dtype": "int32",   "shape": [N], "offset": ...},
            "text":         {"dtype": "uint8",   "shape": [B], "offset": ...}}}
```

`dtype` is one of `uint8`, `uint16`, `int16`, `int32`, `int64`, `float64`. To
read an array, decompress the file and view `prod(shape)` elements of `dtype`
starting at `offset`. Offsets are relative to the start of the decompressed
bytes. In Node, a decompressed `Buffer` can be a slice of a larger
`ArrayBuffer`, starting at its `byteOffset`, which need not be aligned for the
dtype. Copy it into an `ArrayBuffer` of its own before taking views:

```js
const bytes = new Uint8Array(decompressed); // a copy, at byte offset 0
const tokenIds = new Int32Array(bytes.buffer, offset, n);
```

### `tokens`

| Array | Shape | Meaning |
| --- | --- | --- |
| `token_ids` | `[N]` int32 | every token node's tokens, concatenated in node order |
| `logprobs` | `[N]` float64 | the rollout logprob of each token. In a node with `has_logprobs`, 0 for scaffold. NaN for every token of a node without, scaffold included |
| `text` | `[B]` uint8 | every node's text, UTF-8, concatenated in node order |
| `text_offsets` | `[N]` int32 | for each token, the byte offset in its node's text where the token starts |

A node's text is what its tokens decode to, special tokens included. Token
`k` of a node covers bytes `[text_offsets[k], text_offsets[k + 1])` of the
node's text, and its last token ends at `text_bytes`. Offsets never decrease.
A character that spans several tokens belongs to the token that completes it,
and the tokens before it have empty spans, so every span is whole UTF-8
characters and decodes on its own.

### `experts`

| Array | Shape | Meaning |
| --- | --- | --- |
| `routed_experts` | `[R, layers, k]` uint8, int16 or int32 | per token position of each node that has it, the experts each layer routed to |

A node's rows are `[experts_offset, experts_offset + experts_rows)`, one per
token of the node. The last position of a sequence is never forwarded by the
engine, so its row is a copy of the previous one.

### `sampling_mask`

| Array | Shape | Meaning |
| --- | --- | --- |
| `ids` | `[M]` int32 | support ids, all rows concatenated |
| `offsets` | `[rows + 1]` int64 | row `r` is `ids[offsets[r]:offsets[r + 1]]` |

A node's rows are `[mask_offset, mask_offset + mask_rows)`, one per sampled
token, in order.

## Sizes

Before compression, per token of the trajectory (`N` tokens, `S` of them sampled):

| Sidecar | Per unit | Scales as |
| --- | --- | --- |
| `tokens` | per token: 4 B id + 8 B logprob + 4 B text offset + its UTF-8 text (~4 B for English BPE) ≈ 20 B | `O(N)` |
| `experts` | per token: `layers × k × itemsize`, where `k` is the experts each token is routed to (top-k) and `itemsize` is 1 B while expert ids fit in uint8 (≤ 256 experts), 2 B above | `O(N × layers × k)` |
| `sampling_mask` | per sampled token: `4 B × support + 8 B`, where support is at most the sampling `top_k` | `O(S × top_k)` |

The total number of experts only sets `itemsize`. The document's size follows the messages' text.

For example, a 32k-token trajectory with 8k sampled tokens, on a Qwen3-30B-A3B-style model (48 layers, top-8 of 128 experts) sampled with `top_k = 50`:

| Sidecar | Per unit | Total |
| --- | --- | --- |
| `tokens` | ~20 B × 32k | ~0.6 MB |
| `experts` | 48 × 8 × 1 B = 384 B × 32k | ~12 MB |
| `sampling_mask` | up to 208 B × 8k | up to ~1.7 MB |

Routed experts dominate, which is why they have their own file that only training reads.
