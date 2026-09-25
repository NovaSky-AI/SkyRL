"""Every captured exchange, sorted into three buckets.

The baseline never tokenizes: `collect_rollout_details=True` records whatever
`prompt_token_ids` vLLM returned, and vLLM built those from the model's Jinja
template. So rendering the same messages through the engine reproduces what the
baseline would have trained on, and comparing that against the prompt capture
actually sent is the parity question.

Not every difference is a defect, and lumping them together would hide the one
that matters:

  identical        the two integrations would train on the same tokens.

  capture better   the model emitted a token whose text form cannot survive a
                   round trip -- `renderers.parse_response` ends with
                   `content=text.strip()`, so a trailing-whitespace token is
                   gone from the message the harness sees. capture bridges in
                   token space and keeps it; the baseline reconstructs from
                   that text and cannot. Scoring these as failures would
                   penalise capture for the reason it exists.

  differs          anything else. These are the ones to explain.

Detected on the condition itself -- the completion's text is not its own
stripped form -- rather than on `stop_reason`. Trailing whitespace is likelier
after a length cut, because the model is trained to end flush against
`<|im_end|>`, but it is neither confined to length stops nor implied by one.
"""

from __future__ import annotations

import json
import sys
import urllib.request

import orjson
from skyrl_capture.persistence.committed import DiskCommittedStore
from transformers import AutoTokenizer

RECORD = sys.argv[1] if len(sys.argv) > 1 else "/tmp/demo-record"
ENGINE = "http://127.0.0.1:9500"
for index, argument in enumerate(sys.argv):
    if argument == "--engine":
        ENGINE = sys.argv[index + 1]


def served() -> set[str]:
    with urllib.request.urlopen(f"{ENGINE}/v1/models", timeout=30) as reply:
        return {entry["id"] for entry in json.load(reply)["data"]}


def engine_prompt(messages, tools, model):
    body = {"model": model, "messages": messages, "add_generation_prompt": True}
    if tools:
        body["tools"] = tools
    request = urllib.request.Request(
        f"{ENGINE}/tokenize",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as reply:
        return list(json.load(reply)["tokens"])


def main() -> int:
    store = DiskCommittedStore(RECORD)
    available = served()
    tokenizer = None

    identical = better = 0
    differs = []
    skipped = set()

    for identifier in store.committed_ids():
        record = store.get_sync(identifier)
        run = record.trajectory.run_id
        # Which turns of this trajectory produced text that cannot round trip.
        lossy_turns = set()
        for exchange in record.exchanges:
            tokens = exchange.tokens or {}
            completion = list(tokens.get("completion_ids") or [])
            if not completion:
                continue
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(tokens.get("tokenizer") or "Qwen/Qwen3-4B-Instruct-2507")
            text = tokenizer.decode(completion)
            if text != text.rstrip():
                lossy_turns.add(exchange.sequence)

        for exchange in record.exchanges:
            tokens = exchange.tokens or {}
            sent = tokens.get("prompt_token_ids")
            if not sent or not exchange.request_body:
                continue
            payload = orjson.loads(exchange.request_body)
            model = tokens.get("model") or payload.get("model")
            if not payload.get("messages"):
                continue
            if model not in available:
                skipped.add(model)
                continue
            got = list(sent)
            want = engine_prompt(payload["messages"], payload.get("tools"), model)
            if got == want:
                identical += 1
            elif any(turn < exchange.sequence for turn in lossy_turns):
                # An earlier turn of this trajectory emitted a token the text
                # form drops, so this prompt inherits the difference.
                better += 1
            else:
                first = next(
                    (i for i in range(min(len(got), len(want))) if got[i] != want[i]),
                    min(len(got), len(want)),
                )
                differs.append((run, identifier, exchange.sequence, first, len(want), len(got)))

    total = identical + better + len(differs)
    print(f"{total} exchanges compared against the engine's own rendering\n")
    print(f"  identical                        : {identical}")
    print(f"  capture strictly better          : {better}   (token the text form cannot carry)")
    print(f"  differs                          : {len(differs)}")
    if skipped:
        print(f"  skipped, engine does not serve   : {sorted(skipped)}")
    for run, identifier, sequence, first, want, got in differs[:12]:
        print(f"    {run:<20} {identifier[:14]} seq {sequence}: at token {first} ({want} engine, {got} capture)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
