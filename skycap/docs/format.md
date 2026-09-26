# skycap record format, version 1

This is the contract between skycap, which writes trajectories, and any reader
of them, such as the skycap viewer. A reader that follows this document needs
no Python and no tokenizer.

## Files

A record directory holds, per trajectory `{id}`:

| File | Always Outputted | Holds |
| --- | --- | --- |
| `{id}.json.zst` | yes | the document (graph) |
| `{id}.tokens.zst` | token mode | token ids, logprobs, the text the tokens decode to, and each token's byte offset in it |
| `{id}.experts.zst` | when routed experts were captured | routed experts (R3) |
| `{id}.sampling_mask.zst` | when sampling masks were captured | per sampled token, the ids it could have been drawn from |

Every file is one zstd frame. A trajectory is not partially written when it is live, it is only written when it ends.
Sidecars are written before the document, and every file is written to a
temporary name and renamed, so a document that exists always has its sidecars.
A reader lists trajectories by listing `*.json.zst`.

## The document

The decompressed document is a UTF-8 JSON object:

| Field | Type | Meaning |
| --- | --- | --- |
| `format_version` | int | `1`. Covers both this document's shape and the files' encoding. A reader refuses a version it doesn't know |
| `id` | string | the trajectory id |
| `status` | string | `finished`, `failed`, `abandoned` (idle past the TTL) or `open` (written at shutdown) |
| `ended` | bool | whether the trajectory was ended (by `finish` or the TTL). False for one written at shutdown, which may be sealed as `failed` and still waiting for its finish |
| `meta` | object | what the creator passed at create |
| `capture` | object | how it was captured: `mode` (`text` or `tokens`), and for tokens the `engine`, `tokenizer`, `logprobs_mode` (`processed_logprobs` means logprobs are over the truncated, renormalized distribution) and any `sampling_overrides` |
| `annotations` | object | what the creator passed at finish, e.g. `{"reward": 1.0}` |
| `created_at`, `finished_at` | float or null | Unix seconds |
| `tools` | object | tool-set hash → the tool list, as sent |
| `failures` | array | calls that produced no node: `{t, status, error, input_leaf}` |
| `retries` | object | SDK retries answered from the original call: `{replayed, coalesced}` counts |
| `nodes` | array | the graph, in creation order (below) |
| `sidecars` | object | kind → sidecar manifest (below). Empty in text mode |

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
| `calls` | model-authored nodes: every call that produced this output, `{t_start, t_end, model, sampling, usage, finish_reason, tools}`, where `tools` is the key of the call's tool set in the document's `tools`, or null |
| `shadowed_by` | null, or the sibling that later history with the same message continues from |
| `tokens` | null in text mode, else this node's slices of the sidecars (below) |

Every root-to-leaf path is one conversation as a model call saw it.

### A node's `tokens`

| Field | Meaning |
| --- | --- |
| `offset`, `length` | the node's tokens are positions `[offset, offset + length)` of the `tokens` sidecar's per-token arrays |
| `sampled_start` | null for a client node. For a model node, tokens before it are template scaffold and the rest were sampled |
| `has_logprobs` | whether `logprobs` holds real values for this node |
| `text_offset`, `text_bytes` | the node's text is bytes `[text_offset, text_offset + text_bytes)` of `text`; `text_offset` is null when no text was recorded |
| `experts_offset`, `experts_rows` | the node's rows of `routed_experts`; offset null when absent |
| `mask_offset`, `mask_rows` | the node's rows of the sampling mask (one per sampled token); offset null when absent |

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
starting at `offset`. In JavaScript that's `new Int32Array(buffer, offset, n)`.

### `tokens`

| Array | Shape | Meaning |
| --- | --- | --- |
| `token_ids` | `[N]` int32 | every token node's tokens, concatenated in node order |
| `logprobs` | `[N]` float64 | the rollout logprob of each token. NaN where unknown, 0 for scaffold |
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
