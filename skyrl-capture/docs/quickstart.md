# Quickstart

From a fresh checkout to a captured, annotated, exported trajectory. It uses
the bundled mock provider, so it needs **no API key and no network egress**.
Swap one URL at the end to point at a real provider.

Requirements: Python 3.11 or newer. Nothing else — there is no database to
install, start or migrate. Capture writes every trajectory into a directory you
name, as it runs.

## 1. Start the service

First start the mock provider, because the capture process is told which
inference server it captures when it starts:

```bash
uv sync --all-extras
uv run python -m tools.mock_server --port 9188 &

uv run skyrl-capture serve --record-dir ./traces \
  --upstream-type openai --upstream-url http://127.0.0.1:9188/v1
```

You should see:

```
skyrl-capture 0.1.0 on http://127.0.0.1:8080  api=/v1  health=/healthz
  upstream   openai -> http://127.0.0.1:9188/v1 (text mode)
  record     traces
  exports    traces/exports
  viewer     serving reads and bulk exports
```

One process captures one inference server. Changing it means restarting, which
is what makes the rest of this page short: there is nothing to register.

`--record-dir` is required: every trajectory is persisted as it runs, so a
process with nowhere to write refuses to start. `./traces` is the whole result
— copy it, and you have the run.

Startup is immediate in text mode; in tokens mode it is a few seconds, nearly
all of it loading the tokenizer.

Leave it running. In a second shell, confirm it is healthy:

```bash
curl -s localhost:8080/healthz | python -m json.tool
```

`status` should be `ok`. See [operations.md](operations.md) for what the other
states mean.

The mock provider you started speaks OpenAI Chat Completions and Responses,
Anthropic Messages, and a token-in/token-out `/generate` endpoint, streaming
and non-streaming.

The upstream credential comes from `UPSTREAM_API_KEY` rather than a flag, so it
never reaches a shell history or a process listing. The mock ignores it.

## 2. Capture an unchanged workload

Save this as `agent.py`. Note there is no tracing code in it — it reads only
the two environment variables the OpenAI SDK already reads.

```python
import json, os, urllib.request

base = os.environ["OPENAI_BASE_URL"].rstrip("/")
key = os.environ["OPENAI_API_KEY"]

def chat(messages):
    body = json.dumps({"model": "gpt-4o-mini", "messages": messages, "max_tokens": 64}).encode()
    request = urllib.request.Request(
        f"{base}/chat/completions", data=body,
        headers={"authorization": f"Bearer {key}", "content-type": "application/json"})
    with urllib.request.urlopen(request) as response:
        return json.load(response)["choices"][0]["message"]["content"]

history = [{"role": "system", "content": "You are a terse assistant."}]
for prompt in ["list the files", "now read config.yaml", "summarize what you found"]:
    history.append({"role": "user", "content": prompt})
    reply = chat(history)
    history.append({"role": "assistant", "content": reply})
    print(reply)
```

Run it as one trial:

```bash
uv run skyrl-capture run \
  --project terminal-bench \
  --tag task-17 \
  --annotate rlvr_reward=1 \
  -- python agent.py
```

`skyrl-capture run` creates a trajectory, injects its base URL through the
environment variables the upstream's client protocol uses, runs your command
with normal stdin/stdout/stderr, finishes the trajectory, and exits with your
command's exit code.

The key variable is set to a placeholder, because the OpenAI SDK will not build
a client without one and capture authenticates nothing on the way in. If your
deployment puts authentication in front of capture, its credential goes there
instead. A command killed by a signal exits `128 + N`, the way a
shell reports it.

## 3. Look at what was captured

```bash
uv run skyrl-capture list --project terminal-bench
```

```
id                             mode  status    calls  nodes  project         labels
tr_01M22KM5XT26F1CMHEW9K64XJ6  text  finished      3      7  terminal-bench  task-17
```

One trajectory is read in the viewer:

```bash
uv run skyrl-capture view
```

Its **Calls** tab is the chronological timeline — one row per model call, with
the kind, the model, the status, whether it streamed, and its timings:

```
 seq  kind              model        status  stream  dur ms  ttft ms  gap ms  chunks  flags
 0    chat_completions  gpt-4o-mini  200     no      12.6    12.3     -       1       -
 1    chat_completions  gpt-4o-mini  200     no      1.1     0.8      0.5     1       -
 2    chat_completions  gpt-4o-mini  200     no      0.8     0.6      0.4     1       -
```

`gap ms` is the wait before each call — the time your agent spent doing
something other than inference. It is signed, and it is measured from the call
this one *continued from* rather than from whichever call arrived before it:
once a trajectory branches, the previous arrival is a sibling rather than a
predecessor, and the wait between siblings never happened.

Its **Tree** view is the context graph:

```
7 nodes, 1 leaves, 0 branch points
- system        26ch  WXHAN9PJ
  - user          14ch  Q1BBK75S
    * assistant     23ch  MCGC1B2G
      - user          20ch  V4CT0A31
        * assistant     23ch  TG46WPEB
          - user          24ch  68ESYDFZ
            * assistant     23ch  5A1A17S6
```

`-` is a message the client sent, `*` one the model produced. One node per
message, one parent per node, so this path *is* the conversation.

`skyrl-capture view` shows the same data as a filterable list of attempts, a
trajectory table with each attempt's mask drawn as a strip, and a drawer
carrying the tree, the calls and the decoded text. It needs Node 18 or newer,
and reads nothing but `/v1`.

## 4. See a branch

Run the same trial again with a divergent second turn, and the tree shows the
fork directly — a node with more than one child:

```
7 nodes, 2 leaves, 1 branch points
- system        26ch  WXHAN9PJ
  - user          14ch  Q1BBK75S
    * assistant     23ch  MCGC1B2G <branch>
      - user          20ch  V4CT0A31
        * assistant     23ch  TG46WPEB
      - user          24ch  68ESYDFZ
        * assistant     23ch  5A1A17S6
```

The fork is at `MCGC1B2G`, and the shared prefix above it is stored once,
however many branches extend it. What counts as the same prefix, and what the
system refuses to infer from one, is [graph.md](graph.md).

## 5. Annotate after the fact

Nothing is sealed by finishing. Labels (bare tags) and annotations (key/value)
belong to the whole trajectory and stay editable for its life:

```bash
uv run skyrl-capture annotate $TR --annotate critique='"used the wrong tool on turn 2"'
uv run skyrl-capture annotate $TR --tag success --untag draft
```

```
labels      success
annotations {"rlvr_reward": 1, "critique": "used the wrong tool on turn 2"}
```

The reward is an annotation like any other, and it is on the trajectory rather
than on a node: a branched run produces several training samples from one
trial, and they share one outcome.
Only the latest value of a key is kept — there is no revision history.

## 6. Export

One trajectory, one run, or a whole project:

```bash
uv run skyrl-capture export --trajectory $TR --format replay --output turns.jsonl
uv run skyrl-capture export --run-id run-2026-09-17a --format token-samples \
  --output run.jsonl
uv run skyrl-capture export --project terminal-bench --format text-samples \
  --mask-abandoned --output project.jsonl
```

Output is plain JSONL, so `head` and `jq` work on it directly; pass
`--compression zst` (or `gzip`) for the compressed form. A project export is a
concatenation of its trajectories in the same shape.

There are four formats, one per purpose:

| Format | For |
| --- | --- |
| `graph` | The lossless tree — every node, every edge, every timing |
| `replay` | Reissuing the captured traffic, at a scale factor |
| `text-samples` | Distillation: one row per root-to-leaf conversation |
| `token-samples` | RL on the exact token IDs the endpoint produced |

[exports.md](exports.md) has the schemas, the flags, and worked examples of
each.

## 7. The run is already a directory

`./traces` is the result, and it has been since step 1. Each trajectory is
written as it runs — a journal while it is in flight, a compiled record once it
finishes — into a portable directory you can copy, hand to somebody with no
access to your cluster, and read with no server at all.

```bash
ls traces
# manifest.json  active/  committed/  exports/
```

Reading it needs nothing running:

```bash
uv run skyrl-capture view   --record ./traces
uv run skyrl-capture list   --run-id <run-id> --step 0 --record ./traces
uv run skyrl-capture export --record ./traces --run-id <run-id> \
  --format token-samples --output rl.jsonl

uv run skyrl-capture view --record ./traces      # the same UI, over the directory
```

An RL run is where this pays off: give each attempt a `--run-id`, a `--task-id`
and a `--step`, which is how a run is read: one flat list, narrowed.

```bash
uv run skyrl-capture run --project terminal-bench \
  --run-id nightly-01 --task-id task-17 --step 3 \
  --annotate reward=0.75 -- python agent.py
```

[record.md](record.md) and [viewer.md](viewer.md) have the rest.

## 8. Point at a real provider

Nothing above changes. Restart the service against a different upstream:

```bash
UPSTREAM_API_KEY=$OPENAI_API_KEY uv run skyrl-capture serve --record-dir ./traces \
  --upstream-type openai --upstream-url https://api.openai.com/v1

UPSTREAM_API_KEY=$ANTHROPIC_API_KEY uv run skyrl-capture serve --record-dir ./traces \
  --upstream-type anthropic --upstream-url https://api.anthropic.com
```

For an Anthropic upstream the injected base URL deliberately omits `/v1`,
because the Anthropic SDK appends it itself, and the placeholder goes in
`ANTHROPIC_API_KEY`.

To capture two providers at once, run two capture processes. They may share one
`--record-dir` — each trajectory's files are named by its id — so a single
`skyrl-capture list --record ./traces` sees both. Give one of them
`--disable-viewer`, because there is one indexer per directory.

## Next

- [graph.md](graph.md) — how prefix matching decides what is a branch
- [exports.md](exports.md) — the four formats and what each is for
- [tokens.md](tokens.md) — capturing exact token IDs for training
- [record.md](record.md) — a run as a directory, read with no database
- [viewer.md](viewer.md) — the run's attempts and the loss mask over the text
- [verification.md](verification.md) — checking that the record is what the engine saw
- [architecture.md](architecture.md) — what runs where, and why
- [operations.md](operations.md) — deploying it somewhere real
