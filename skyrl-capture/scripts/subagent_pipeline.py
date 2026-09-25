"""A branching agent workload, for looking at what capture does with one.

An orchestrator plans, three sub-agents inherit that plan verbatim and do
multi-turn work, then the orchestrator synthesizes. Seven model calls that fan
out from a single reply.

There is no tracing code here. It reads ``OPENAI_BASE_URL`` and
``OPENAI_API_KEY`` -- the two variables the OpenAI SDK already reads -- which
is everything the capture proxy needs to see every call and reconstruct which
of them branched from which.

    skyrl-capture run --project audit --tag security-audit \
      -- python scripts/subagent_pipeline.py

The mock upstream honours ``x-mock-reply``, so the replies below are fixed and
the shape of the resulting graph is reproducible. Point capture at a real
provider and the same structure appears with whatever the model actually says.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request

BASE = os.environ["OPENAI_BASE_URL"].rstrip("/")
KEY = os.environ["OPENAI_API_KEY"]

# Sub-agents are given a moment of non-inference work before each call, so the
# gaps in the capture are not all zero and a replay has something to scale.
THINK_SECONDS = float(os.environ.get("PIPELINE_THINK_SECONDS", "0.08"))


def chat(messages: list[dict[str, str]], reply: str, *, think: float = 0.0) -> str:
    if think:
        time.sleep(think)
    body = json.dumps(
        {"model": "gpt-4o-mini", "messages": messages, "max_tokens": 128, "temperature": 0.2}
    ).encode()
    request = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=body,
        headers={
            "authorization": f"Bearer {KEY}",
            "content-type": "application/json",
            # Mock-only: makes each branch identifiable in the graph.
            "x-mock-reply": reply,
        },
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)["choices"][0]["message"]["content"]


SUBAGENTS = [
    (
        "handlers",
        [
            ("Inspect the request handlers.", "Found: token check missing on /refresh."),
            ("Confirm it is reachable unauthenticated.", "Confirmed: reachable without a token."),
        ],
    ),
    (
        "tests",
        [
            ("Review the auth tests.", "Coverage gap: no test for expired tokens."),
            ("Would a regression be caught?", "No. Expiry path is untested."),
        ],
    ),
    (
        "deps",
        [("Check dependencies for CVEs.", "pyjwt 2.1.0 has a known signature bypass.")],
    ),
]


def main() -> None:
    # -- the orchestrator plans; every sub-agent will inherit this ----------
    root = [
        {"role": "system", "content": "You are an orchestrator. Delegate, then synthesize."},
        {"role": "user", "content": "Audit the auth module for security problems."},
    ]
    plan = chat(root, "Plan: inspect handlers, review tests, check dependencies.", think=0.05)
    root.append({"role": "assistant", "content": plan})
    print(f"plan: {plan}")

    # -- fan out. Each sub-agent replays `root` unchanged and appends its own
    #    turns, which is what makes the capture fork at the plan rather than
    #    somewhere above it.
    findings = []
    for name, turns in SUBAGENTS:
        context = list(root)
        for prompt, reply in turns:
            context.append({"role": "user", "content": f"[{name}] {prompt}"})
            answer = chat(context, reply, think=THINK_SECONDS)
            context.append({"role": "assistant", "content": answer})
            print(f"  {name}: {answer}")
        findings.append(f"{name}: {context[-1]['content']}")

    # -- the orchestrator continues from its own plan, not from a sub-agent --
    summary = chat(
        root + [{"role": "user", "content": "Synthesize: " + " | ".join(findings)}],
        "Three issues; the missing token check on /refresh is critical.",
        think=0.12,
    )
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
