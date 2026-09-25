"""An agent that compacts its history, which is the long-horizon default.

Every agent that runs long enough eventually runs out of context. The usual
answer is compaction: ask the model to summarize the conversation so far, throw
the transcript away, and carry on from the system prompt plus that summary.
This script does it twice over a seven-turn migration task.

What capture makes of it is worth seeing, because compaction is *not* a special
case here and is not labelled as one. The rebuilt context still begins with the
same system message, so the longest exact prefix is one node deep, and the run
forks at the root. Two compactions give the root three children: the original
opening, and one per rebuilt context. Long-horizon agents produce a fan of
shallow branches, not a deep spine.

The summary text lands in the graph twice, and the two copies are not the same
kind of thing. On the branch that produced it, it is an assistant node the
model generated. On the branch that consumed it, it is a user node the client
composed. `author` is the field that separates them -- `model` against
`client` -- which is the same distinction that tells a repaired assistant
message apart from a second sample.

That matters for what comes out of an export. The pre-compaction path is a
complete conversation ending in a generation, so it is one trainable sample,
and what it teaches is summarization. Each rebuilt context is another, and what
those teach is the work. They share no tokens with each other beyond the system
prompt, and no generation is trained on twice.

    skyrl-capture run --project compaction \
      -- python scripts/compaction.py

    skyrl-capture view
    skyrl-capture export --trajectory <id> --format text-samples --output samples.jsonl

Replay sees it as separate sessions. A rebuilt context's prefix ends on the
system node, which the client wrote rather than the model, so there is no
parent output to hang it from and no parent session to be relative to -- it is
scheduled by `arrival_ms` from the start of the trace instead. That is the
honest answer: the system knows when the compacted segment started, not that it
started *because* the previous one ended.

The mock upstream honours ``x-mock-reply``, so every reply below is fixed and
the graph is reproducible. Point capture at a real provider and the same
shape appears with whatever the model actually says.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request

BASE = os.environ["OPENAI_BASE_URL"].rstrip("/")
KEY = os.environ["OPENAI_API_KEY"]

SYSTEM = "You are a migration assistant. Work one file at a time."

# Summarizing is a pause in the workload, not in the model. Sleeping across the
# boundary keeps that visible in the capture, where it is the interval between
# the last call of one segment and the first of the next.
COMPACT_SECONDS = float(os.environ.get("COMPACTION_PAUSE_SECONDS", "0.15"))

COMPACT_PROMPT = (
    "We are running out of context. Summarize what is done and what remains, "
    "in a form that can stand in for this conversation."
)

# Each segment is the work done under one context window. The compaction
# between them is what the script is about.
SEGMENTS = [
    [
        (
            "Start the migration. What still calls the legacy billing client?",
            "Four call sites: invoices.py, refunds.py, webhooks.py, reports.py.",
        ),
        ("Port invoices.py.", "Done. invoices.py now builds a BillingClient.v2."),
        (
            "Port refunds.py.",
            "Ported, but v2 requires an idempotency key that refunds.py does not send.",
        ),
    ],
    [
        ("Add the idempotency key to refunds.py.", "Added, keyed on refund_id."),
        ("Port webhooks.py.", "Done. Signature verification now uses the v2 helper."),
    ],
    [
        ("Port reports.py.", "Done. reports.py reads the v2 aggregation endpoint."),
        ("Anything left?", "No. All four call sites are on v2."),
    ],
]

# What the model "returns" from each compaction call, in order.
SUMMARIES = [
    "Migrating four call sites off the legacy billing client. invoices.py and "
    "refunds.py are ported; refunds.py still needs an idempotency key. "
    "webhooks.py and reports.py are untouched.",
    "invoices.py, refunds.py and webhooks.py are on BillingClient.v2, with "
    "refunds keyed on refund_id. reports.py is the last one left.",
]


def chat(messages: list[dict[str, str]], reply: str) -> str:
    body = json.dumps(
        {"model": "gpt-4o-mini", "messages": messages, "max_tokens": 256, "temperature": 0.2}
    ).encode()
    request = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=body,
        headers={
            "authorization": f"Bearer {KEY}",
            "content-type": "application/json",
            # Mock-only: pins the reply so the shape of the graph is fixed.
            "x-mock-reply": reply,
        },
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)["choices"][0]["message"]["content"]


def main() -> None:
    context = [{"role": "system", "content": SYSTEM}]

    for index, segment in enumerate(SEGMENTS):
        for prompt, reply in segment:
            context.append({"role": "user", "content": prompt})
            answer = chat(context, reply)
            context.append({"role": "assistant", "content": answer})
            print(f"  {answer}")

        if index == len(SEGMENTS) - 1:
            break

        # -- compact. The summarization call carries the whole transcript, so
        #    it continues this branch and ends it: nothing is appended to
        #    `context` afterwards, because `context` is about to be replaced.
        summary = chat(context + [{"role": "user", "content": COMPACT_PROMPT}], SUMMARIES[index])
        print(f"\ncompacted {len(context)} messages -> a summary of {len(summary)} chars\n")
        time.sleep(COMPACT_SECONDS)

        # -- and rebuild. The system message is unchanged, which is the whole
        #    of the prefix the next call will share with anything before it.
        context = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Summary of the work so far:\n{summary}"},
        ]

    print(
        f"\n{len(SEGMENTS)} segments, {len(SEGMENTS) - 1} compactions."
        f"\nthe graph should show {len(SEGMENTS)} branches off the system node."
    )


if __name__ == "__main__":
    main()
