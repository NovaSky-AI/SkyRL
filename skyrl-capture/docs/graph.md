# The context graph

This is the primitive that skyrl-capture stores an agent's trace in. A
trace can be complex: many model calls, possibly nested agents with their own
execution graphs, and context inherited between calls in ways that are not
obvious from the outside. A lossless record of what happened, built from what
the server can actually see, is the thing a continuous learning loop needs —
whether the loop is improving inference performance or improving the model. So
the abstraction has to hold the timing of each call, the causality between
calls to the extent that causality can honestly be derived server-side, and the
token-level detail when there is any.

Everything here is either an observed fact or a derivation whose evidence is
stored alongside it. Let's unpack what a context graph is.

## One node per message

Each node holds exactly one message introduced into a trajectory, and has
exactly one parent. It does not matter whether the client provided that message
or the model produced it. Consequences:

- Every root-to-leaf path reconstructs one complete conversation, as one model
  call saw it.
- A shared prefix can be stored once, however many calls share it — where
  "shared" means the same messages under the same tools and the same model.
- Two distinct messages under one parent are a fork of that conversation: the
  harness spawning a sub-agent, say, or retrying one that failed.
- Joins are never inferred. A node never has two parents. If the harness
  combines the results of two parallel calls into a third call, that causality
  is not recorded, because it exists only inside the application.

### What a node holds

| Field | What it is |
| --- | --- |
| `node_id` | Identifier for this node |
| `parent_node_id` | The node this message follows. `null` at a root |
| `depth` | Distance from the root, so depth 0 is the first message |
| `role` | The message's own role: `system`, `user`, `assistant`, `tool` |
| `author` | Who produced it: `client` or `model` |
| `message_hash` | Hash of this message alone |
| `delta_hash` | What identifies the node — see [Node identity](#node-identity) |
| `context_hash` | Hash of the whole conversation ending here |
| `exchange_id` | The model call that introduced this message |
| `char_count`, `content_block_count` | Size of the message, without reading it |
| `token_count`, `sampled_start`, `sampled_token_count` | Token detail. `null` outside tokens mode |
| `has_logprobs`, `has_routed_experts`, `tokenizer` | What else the token payload carries |
| `inserted_at`, `output_started_at`, `output_ended_at` | When it was recorded, and when the model produced it |
| `derivation` | How this node was placed: how many messages matched, and where in the tail it sat |

`GET /v1/trajectories/{id}/graph` returns all of these except `context_hash`,
plus `has_payload`; the `graph` export returns `context_hash` and the message.

The message itself is not in this row — it lives on the node's payload, so
walking a large graph does not mean pulling every message with it. To read the
messages, take the whole tree with them inlined from the
[`graph` export](exports.md#graph), which is one artifact rather than one call
per node.

`has_payload` is a capture diagnostic rather than a pointer: it is false when
the body was never stored, which is what `--bodies sampled` decides. A node
without one is a node the `graph` and `replay` exports have no message for.

```json
{
  "node_id": "nd_01M275ES48ZMGZV3FM1FY5NWDD",
  "parent_node_id": "nd_01M275ES41BTVSDNBJ9A4EETW3",
  "depth": 1,
  "role": "assistant",
  "author": "client",
  "message_hash": "bd73fe82b5ce04c6...",
  "delta_hash": "4a1e950f0b11dac2...",
  "char_count": 23,
  "derivation": {...}
}
```

### Who wrote the message

`author` distinguishes a message the client sent (`client`) from one the model
sampled (`model`). That draws a distinction nothing else can: an assistant
message with `author: client` was written by the harness — a repair, or a
replay of an earlier reply — and was never emitted by any model.

It matters because the two look identical in a transcript and mean opposite
things to a trainer. Training on a repaired message is training on the harness.

[`examples/agents/repair.py`](../examples/agents/repair.py) is the case in
three calls: the model answers, the harness corrects the answer, and the next
turn sends the correction back as history.

```
5 nodes, 2 leaves, 1 branch points
- user          38ch  9A4EETW3 <branch>
  * assistant     22ch  Q9GV4ZJD        "It retries four times."   (model)
  - assistant     23ch  1FY5NWDD        "It retries three times."  (client)
    - user          16ch  EH3N8SG4
      * assistant     20ch  KXD4DPRY
```

Both children of the user node are assistant messages. `*` is the one the model
sampled; `-` is the one the harness wrote. The conversation continues from the
repair, and the model's original is preserved rather than overwritten.

## The matching rule

Every time a call arrives, its messages have to be placed against the tree the
trajectory has built so far. The rule is: for each exchange with a message
history,

1. Hash every parsed request message (canonical JSON, sorted keys), together
   with the call's tool set and model.
2. Walk the existing graph for the **longest exact hash prefix**.
3. Commit one node per unmatched request message, in order.
4. Commit one `model`-authored node for the observed assistant response.

That is the whole algorithm. Nothing below is special-cased; each case is what
those four steps do when the input has a particular shape.

| Situation | What the graph does |
| --- | --- |
| [Ordinary next turn](#ordinary-next-turn) | Matches the whole previous path, commits the new message and the new reply |
| [Identical retry](#identical-retry) | Matches everything including the reply, commits nothing |
| [Re-sampled reply](#re-sampled-reply) | Same parent, two different replies: a fork |
| [Repaired reply](#who-wrote-the-message) | Forks at the edited message, which is marked `client` |
| [Compacted history](#compacted-history) | Stops matching at the last unchanged message and forks there |
| [Nothing in common](#nothing-in-common) | Starts a second root |
| [Changed tool set](#changed-tool-set) | Starts a second root, even with identical messages |
| [Changed model](#changed-model) | Starts a second root, even with identical replies |

Each is a runnable script in [`examples/agents/`](../examples/agents/). They
need the service and the mock provider from
[quickstart.md](quickstart.md#1-start-the-service), and each one is a
trajectory of its own:

```bash
skyrl-capture run --project demo -- python examples/agents/repair.py
skyrl-capture view      # the Tree view, on the trajectory that just finished
```

In the trees below, `-` is a message the client sent, `*` one the model
produced, and the eight characters at the end of each line are the tail of the
node's ID.

### Ordinary next turn

[`next_turn.py`](../examples/agents/next_turn.py) — the agent asks, appends the
reply to its history, and asks again.

```
4 nodes, 1 leaves, 0 branch points
- user          33ch  37AH9R2F
  * assistant     31ch  TJQBRBHM
    - user          27ch  XTMRW4EF
      * assistant     25ch  FEPCG741
```

The history the second call sent *is* the first call's path plus a tail, so the
match runs all the way down and only the two new messages are committed. This
is the common case, and it is what makes a shared prefix worth having: four
nodes, not six.

### Identical retry

[`retry.py`](../examples/agents/retry.py) — a transport hiccup makes the agent
resend a request it already sent, and the same reply comes back.

```
2 nodes, 1 leaves, 0 branch points
- user          31ch  9YAR0135
  * assistant     24ch  Y8F934Y4
```

Two calls, two nodes. Every request message matched, and the reply was one the
graph already had, so there was nothing to commit — a node is a message, and
the retry introduced no message.

The retry is not lost, though. It is recorded as a second *exchange* carrying
`is_duplicate_retry: true`, and both exchanges resolve to the same
`input_leaf_node_id` and the same `output_node_id`. That is what makes the
detection exact rather than a guess about two calls that looked alike.

### Re-sampled reply

[`resample.py`](../examples/agents/resample.py) — the same question asked twice
from the same history, to vote on the answers or because the first was no good.

```
3 nodes, 2 leaves, 1 branch points
- user          38ch  MDFW6R9Z <branch>
  * assistant     35ch  9ZFFSD31
  * assistant     48ch  YMGDY9W7
```

The request messages matched both times; the replies differed. So the shared
user node has two `model`-authored children, and that is a fork. Both are real
generations from the same context, and an export gives two training samples
here — which is the right count, because the model produced two.

Compare this with the retry above. The difference between "the same call twice"
and "two samples from one context" is entirely whether the reply differed, and
the graph decides it by looking rather than by counting calls.

### Compacted history

[`compaction.py`](../examples/agents/compaction.py) — the agent runs out of
context, asks for a summary, throws the transcript away, and carries on from
the system prompt plus that summary.

```
10 nodes, 2 leaves, 1 branch points
- system        30ch  X37PQ8DQ <branch>
  - user          35ch  1FV5G4EF
    * assistant     34ch  0R9HFEPQ
      - user          14ch  2BCHJT0D
        * assistant     27ch  M9VWEAHX
          - user          40ch  9JJZ2NAG
            * assistant     41ch  EXZ7D87S     <- the summary, sampled
  - user          55ch  2P888YTC               <- the summary, replayed
    - user          13ch  HGBKSSS4
      * assistant     26ch  FMCM84QF
```

The rebuilt history still opens with the same system message, so the match gets
exactly one node deep and forks there. Everything before the rewrite is
genuinely shared and is kept once; everything after is a different conversation
and is stored as one.

Note where the summary ends up: once as a `model`-authored assistant node
ending the old branch, and once as a `client`-authored user node opening the
new one. Same text, different authors, and only one of them is a generation.

A long-running agent that compacts repeatedly therefore produces a fan of
shallow branches off its system message rather than one deep spine.
[`scripts/compaction.py`](../scripts/compaction.py) is the same idea at full
length, with the export and replay consequences worked through.

### Nothing in common

[`new_system_prompt.py`](../examples/agents/new_system_prompt.py) — one
harness run driving two differently-configured agents, so the system prompts
differ.

```
6 nodes, 2 leaves, 0 branch points
- system        34ch  NFYFT0XG
  - user          27ch  W6QMDYX5
    * assistant     25ch  GZWBAGN5
- system        38ch  WG96TKR7
  - user          27ch  V6BSD01E
    * assistant     46ch  V87Q99S9
```

The match failed at the very first message, so there was nothing to attach to
and the second call started a root of its own. A trajectory is free to hold
several roots — the structure is a forest rather than a tree, and a root is
just a node with no parent. Note that this is *not* a branch point: nothing
forked, because nothing was shared.

### Changed tool set

[`tool_change.py`](../examples/agents/tool_change.py) — the agent asks with
read-only tools, the model answers without calling anything, so the agent
widens the tool set and asks the same question again. The messages are
byte-for-byte identical and the replies are pinned identical, so the tool set
is the only thing that differs.

```
4 nodes, 2 leaves, 0 branch points
- user          32ch  FNV3HPYH
  * assistant     30ch  QJZY7BZV
- user          32ch  Q8EZX8Q2
  * assistant     30ch  HHT1F62Q
```

Two roots, from two calls whose message lists were the same. That is not a
quirk: a chat template renders tool schemas into the prompt, so the second call
never saw the first call's context — it saw a different one, differing from its
very first token. Merging them would report one generation where there were
two, under conditions that were not the same.

Changing tools here, on the first message, has nothing to share anyway. The
case worth understanding is a change **partway through** a conversation, in
[`tool_change_midway.py`](../examples/agents/tool_change_midway.py) — one turn
with `search`, then a second turn continuing that history with `search` and
`edit`:

```
6 nodes, 2 leaves, 0 branch points
- user          20ch  G2KV2ZN9
  * assistant     23ch  YNPD6YY6
- user          20ch  8CTDCR1Y        <- the same two messages, committed again
  - assistant     23ch  KCP780GS
    - user          11ch  MDGHT16E
      * assistant     23ch  MDNF1WK1
```

Still a second root, and the entire prefix is duplicated. The tool set is
folded into *every* message of a call, not only the new ones, so the match
fails at the first message rather than at the turn where the tools changed.

Two consequences to know about. The prefix is stored twice, so an agent that
changes its tools every turn stores a triangle rather than a path. And the
replayed assistant message on the new root is `-`, `author: client` — the model
sampled it on the *other* root, and this copy was sent by the harness. Only
`model`-authored messages are trainable, so that generation still trains once,
from the branch it was actually produced on.

### Changed model

[`model_change.py`](../examples/agents/model_change.py) — the same question put
to a large model and a small one, which is the shape a distillation set is
gathered in. On an easy turn both say the same thing.

```
4 nodes, 2 leaves, 0 branch points
- user          28ch  EPFCMCHN
  * assistant     35ch  86M505ZY
- user          28ch  3AX61JNR
  * assistant     35ch  S44N8ZX5
```

Identical messages, identical replies, two roots. Nothing on the first path was
produced by the second model, and a dataset meant to train one of them, or to
distil one into the other, has to be able to tell them apart.
[`scripts/model_distill.py`](../scripts/model_distill.py) carries this through
to the export, where it is one row per model.

## Node identity

A node's **delta** is what it adds to its parent: one message, together with
the conditions that message was seen under. Three questions come up over and
over, and the graph keeps a hash for each one:

| Hash | Covers | Answers |
| --- | --- | --- |
| `message_hash` | One message, on its own | "Is this the same message?" |
| `delta_hash` | That message, and the tools and model it was seen with | "Is this the same message under the same conditions?" |
| `context_hash` | Every delta from the root down to here | "Is this the same conversation?" |

A node is identified by `(trajectory_id, parent_node_id, delta_hash)` — the
same place in the tree, plus the same delta arriving at it.

### Why identical messages can land on different nodes

`delta_hash` covers the message, the tool set the call was made with, and the
model that answered. It has to cover everything that makes two generations
different, or the graph merges them and reports one call where there were two.

**Tools**, because a chat template renders their schemas into the prompt. The
same message under a different tool set is a different context, even though the
message list is identical — which is why
[a changed tool set starts a new root](#changed-tool-set).

**The model**, because two models answering a prompt the same way are still two
samples, and [a dataset that distils one into the other](#changed-model) has to
be able to tell them apart.

Neither is a special case in the matching rule. They are part of the hash, so
the ordinary "longest exact prefix" walk fails at the first message and the
call lands on a root of its own.

### Why the conversation needs its own hash

`context_hash` is the running chain down the path:

```
context_hash = sha256(parent's context_hash + ":" + this node's delta_hash)
```

A root has no parent, so it chains from the empty string. Each node's hash
therefore stands for the entire conversation ending at it:

```
- system         d0  ->  c0 = sha256("" + ":" + d0)
  - user         d1  ->  c1 = sha256(c0 + ":" + d1)
    * assistant  d2  ->  c2 = sha256(c1 + ":" + d2)
```

The identity tuple cannot do this job, because it names a parent node, and node
IDs are local to one trajectory. `context_hash` is just content, so equal
hashes mean equal conversations *across* trajectories and across runs. That is
what makes it usable for deduplicating training samples between runs, or as a
prefix-cache key.

Inserts are idempotent, so replaying the same exchange converges on the same
nodes instead of duplicating them.

### In tokens mode

`delta_hash` is computed over the exact token IDs and the sampled boundary
rather than over the message text. Two identical messages that tokenized
differently stay distinct nodes rather than being silently merged, and the
rendered tools are already in those tokens. `context_hash` chains those deltas,
so a client repair that changes nothing in text still diverges in tokens.

## Context that is not a message

The graph's unit is a message, but not every provider sends the whole context
as a list of messages. Anything that occupies a position in the context gets a
node anyway, materialized into the message list where it belongs.

**Anthropic** carries the system prompt in a top-level `system` field rather
than in `messages`. It becomes a leading `system` node, ahead of the rest.

**OpenAI Responses** carries the same thing in `instructions`, and its input
and output items are not messages either. Each becomes a node in the position
it occupies.

Every materialized node says in its derivation evidence where it came from, so
you can tell one from a node that arrived as an ordinary message:

| `materialized_from` | Came from |
| --- | --- |
| `system_field` | Anthropic's top-level `system` |
| `instructions` | Responses `instructions` |
| `input_item` | A non-message Responses input item |
| `response_output` | A Responses output item |

The practical effect is that two calls differing only in their system prompt
are two different contexts, which is what they are.

Beyond this, messages are not canonicalized across providers: an OpenAI message
is hashed as an OpenAI message and an Anthropic one as an Anthropic message.
Prefix matching only ever compares messages inside a single trajectory, so
shapes from two providers never meet and there is nothing for a normalization
step to gain.
