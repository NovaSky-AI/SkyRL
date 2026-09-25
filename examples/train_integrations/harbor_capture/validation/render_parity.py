"""Do capture's rendered prompts equal what the baseline would have trained on?

The harness-side TITO baseline never tokenizes anything itself. It records
whatever `prompt_token_ids` vLLM returns:

    # harbor/llms/lite_llm.py
    prompt_token_ids = getattr(response, "prompt_token_ids", None)

and vLLM builds those by applying the model's Jinja chat template. capture
instead renders the same messages with Prime Intellect's `renderers` and sends
token IDs straight to the engine, so the two never share a line of code.

This replays every captured exchange: take the OpenAI request capture received,
ask vLLM to render exactly those messages and tools, and compare against the
prompt capture actually sent. A difference here is a difference in what the two
integrations would train on, for the same conversation.

Usage:  render_parity.py [record-dir] [--engine URL]
"""

from __future__ import annotations

import json
import sys
import urllib.request

import orjson
from skyrl_capture.domain.hashing import normalize_json
from skyrl_capture.persistence.committed import DiskCommittedStore

RECORD = sys.argv[1] if len(sys.argv) > 1 else "/tmp/demo-record"
ENGINE = "http://127.0.0.1:9500"
for index, argument in enumerate(sys.argv):
    if argument == "--engine":
        ENGINE = sys.argv[index + 1]


def served_models() -> set[str]:
    with urllib.request.urlopen(f"{ENGINE}/v1/models", timeout=30) as reply:
        return {entry["id"] for entry in json.load(reply)["data"]}


def baseline_prompt(messages, tools, model):
    """What the engine renders for these messages.

    `/tokenize` rather than a real completion: the two are byte-identical --
    checked for plain, tool-declaring and tool-result requests -- and this one
    generates nothing. A one-token completion makes the tool-call parser try
    to parse a truncated `<tool_call>` and fill the server log with decode
    errors that are the probe's fault, not the integration's.
    """
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
    available = served_models()
    identifiers = store.committed_ids()
    print(f"{len(identifiers)} trajectories in {RECORD}")
    print(f"engine {ENGINE} serving {sorted(available)}\n")

    checked = exact = key_order_only = 0
    other = []
    skipped_model = set()

    for identifier in identifiers:
        record = store.get_sync(identifier)
        for exchange in record.exchanges:
            tokens = exchange.tokens or {}
            sent = tokens.get("prompt_token_ids")
            if not sent or not exchange.request_body:
                continue
            payload = orjson.loads(exchange.request_body)
            messages = payload.get("messages")
            model = tokens.get("model") or payload.get("model")
            if not messages:
                continue
            if model not in available:
                # An example that named another engine. Nothing to compare to.
                skipped_model.add(model)
                continue
            checked += 1
            got = list(sent)
            tools = payload.get("tools")
            if got == baseline_prompt(messages, tools, model):
                exact += 1
                continue
            # Capture canonicalises tools for node identity -- `normalize_json`
            # sorts keys so two specs that differ only in key order hash the
            # same -- and then renders that canonical form. The engine renders
            # the caller's original order. Sorting the baseline's copy the same
            # way separates that known difference from any other.
            if tools and got == baseline_prompt(messages, normalize_json(tools), model):
                key_order_only += 1
                continue
            want = baseline_prompt(messages, tools, model)
            first = next(
                (i for i in range(min(len(got), len(want))) if got[i] != want[i]),
                min(len(got), len(want)),
            )
            other.append((identifier, exchange.sequence, first, len(want), len(got)))

    print(f"checked {checked} exchanges")
    print(f"  identical to the engine's own rendering : {exact}")
    print(f"  identical once tool keys are sorted     : {key_order_only}")
    print(f"  differing for some other reason         : {len(other)}")
    if skipped_model:
        print(f"  skipped (engine does not serve)         : {sorted(skipped_model)}")
    for identifier, sequence, first, want, got in other[:10]:
        print(f"    {identifier[:18]} seq {sequence}: diverges at {first} ({want} engine, {got} capture)")
    return 1 if other else 0


if __name__ == "__main__":
    raise SystemExit(main())
