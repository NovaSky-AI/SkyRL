"""An agent loop for token-in/token-out capture.

No tracing code, same as any other workload: it reads `OPENAI_BASE_URL` and
`OPENAI_API_KEY` and speaks Chat Completions. It cannot tell that the proxy is
rendering to token IDs and calling a token-in/token-out endpoint underneath.

Setup, from a fresh checkout. Three terminals, or use `&`.

1. The dependencies. The `tokens` extra brings the renderer:

       uv sync --extra dev --extra tokens

2. A stand-in for the inference engine. It exposes `/generate`, the
   token-in/token-out contract a SkyRL-style router speaks: it takes
   `prompt_token_ids` and returns `response_ids` with per-token logprobs.
   Its completions are deterministic pseudo-tokens derived from the prompt,
   so the text is nonsense while the token accounting is exact:

       uv run python -m skyrl_capture.bench.mock_server --port 9188

3. Capture pointed at it, and this workload. Capture is also the API, and it
   starts its own PostgreSQL. `--tokenizer` is a Hugging Face name; the proxy
   renders with it and stores the exact IDs, and the first call pays for
   loading it:

       uv run skyrl-capture serve --mode tokens --upstream-type tokens \
         --upstream-url http://127.0.0.1:9188/generate \
         --model demo-policy --tokenizer Qwen/Qwen3-0.6B --max-model-len 8192

       uv run skyrl-capture run --project tokens-demo -- python examples/agents/tokens.py

Then look at what was captured:

       TR=$(uv run skyrl-capture list --project tokens-demo --ids | head -1)
       uv run skyrl-capture view
       uv run skyrl-capture export --trajectory $TR --format token-samples --output rl.jsonl
       uv run skyrl-capture export --trajectory $TR --format graph --output trace.jsonl

Two things this workload does on purpose. It feeds the assistant message back
verbatim, which is what lets the proxy reuse the exact token prefix instead of
re-rendering the conversation -- replay anything else and the trajectory forks.
And it branches once, asking a second question from the same point, so the
export has more than one path to show.

The other examples here share `_capture.chat`, which returns the reply's text
and asks the bundled mock provider for a fixed one. This example keeps its own
`chat` because neither fits: it runs against a token endpoint rather than that
mock, and it needs the whole assistant message back -- `reasoning_content` and
all -- since that is exactly what prefix reuse depends on.
"""

from __future__ import annotations

import json
import os
import urllib.request

BASE = os.environ["OPENAI_BASE_URL"].rstrip("/")
KEY = os.environ["OPENAI_API_KEY"]


def chat(messages: list[dict[str, str]]) -> dict[str, str]:
    """One Chat Completions call. Returns the assistant message as given."""
    body = json.dumps({"model": "demo-policy", "messages": messages, "max_tokens": 12}).encode()
    request = urllib.request.Request(
        f"{BASE}/chat/completions",
        data=body,
        headers={"authorization": f"Bearer {KEY}", "content-type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        message = json.load(response)["choices"][0]["message"]
    # Send back exactly what came out, minus the nulls the API pads with.
    return {key: value for key, value in message.items() if value is not None}


def main() -> None:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": "You are a migration assistant."},
        {"role": "user", "content": "Which call sites still use the legacy client?"},
    ]

    for turn, question in enumerate(["Port the first one.", "Now port the second."], start=1):
        reply = chat(messages)
        print(f"  turn {turn}: {reply['content'][:60]!r}")
        messages = messages + [reply, {"role": "user", "content": question}]

    final = chat(messages)
    print(f"  turn 3: {final['content'][:60]!r}")

    # Branch: go back to the state after turn 1 and ask something else. Same
    # prefix, different continuation -- a fork in the graph, two paths out.
    branch = messages[:3] + [{"role": "user", "content": "Actually, list the risks first."}]
    other = chat(branch)
    print(f"  branch: {other['content'][:60]!r}")


if __name__ == "__main__":
    main()
