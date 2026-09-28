"""Does the trainer train on exactly what the engine sampled from?

That is the whole claim. The harness talks messages, so capture re-renders the
conversation each turn; if re-rendering perturbs anything the model already
saw, the training sequence contains tokens that were never in any prompt the
engine sampled from, and training diverges from inference by exactly that.

Not tested by walking exchanges in order: a captured trajectory is a graph, and
resampling, retrying or rewriting history branches it on purpose. The property
is stated over the exported training row instead.

For every turn of every root-to-leaf path:

    row's prompt for that turn  ==  the prompt_token_ids capture sent
    row's response for that turn ==  the completion_ids the engine returned

`stepwise_rows` cuts a path at its trainable spans, which is one turn each, so
each turn is directly comparable to the exchange that produced it.
"""

from __future__ import annotations

import sys

from skyrl_capture.export import formats
from skyrl_capture.export.view import view_of
from skyrl_capture.persistence.committed import DiskCommittedStore

RECORD = sys.argv[1] if len(sys.argv) > 1 else "/tmp/demo-record"


def main() -> int:
    store = DiskCommittedStore(RECORD)
    trajectories = paths = turns = exact = 0
    unmatched = []
    mismatched = []

    for identifier in store.committed_ids():
        record = store.get_sync(identifier)
        # What the engine was given, and what it returned, keyed by the node
        # that turn produced. Keyed by node rather than by completion: two
        # turns of one trajectory can sample identical text, and a
        # completion-keyed lookup silently compares a turn against its twin.
        sent = {}
        for exchange in record.exchanges:
            tokens = exchange.tokens or {}
            node = tokens.get("assistant_node_id")
            if node and tokens.get("prompt_token_ids"):
                sent[node] = (
                    list(tokens["prompt_token_ids"]),
                    list(tokens.get("completion_ids") or []),
                )
        if not sent:
            continue
        trajectories += 1

        for row in formats.token_sample_records(view_of(record)):
            paths += 1
            input_ids = list(row["input_ids"])
            spans = row.get("node_spans") or []
            for node, (begin, end) in zip(row.get("node_ids") or [], spans):
                if node not in sent:
                    continue  # a client node: nothing was sampled for it
                turns += 1
                prompt, completion = sent[node]
                # The assistant node holds the generation scaffold and then
                # the sampled tokens, so the engine's prompt ends where the
                # sampled run begins -- `end - len(completion)`.
                cut = end - len(completion)
                trained_prompt = input_ids[:cut]
                trained_response = input_ids[cut:end]
                if trained_prompt == prompt and trained_response == completion:
                    exact += 1
                    continue
                where = "prompt" if trained_prompt != prompt else "response"
                a, b = (prompt, trained_prompt) if where == "prompt" else (completion, trained_response)
                first = next(
                    (i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b))
                )
                mismatched.append((identifier, where, first, len(a), len(b)))

    print(f"{trajectories} trajectories, {paths} paths, {turns} sampled turns")
    print(f"  trained tokens == tokens the engine sampled from : {exact}/{turns}")
    print(f"  sampled turns with no matching exchange          : {len(unmatched)}")
    print(f"  turns that differed                              : {len(mismatched)}")
    for identifier, where, first, was, now in mismatched[:8]:
        print(f"    {identifier[:18]} {where}: diverges at {first} ({was} engine, {now} trained)")
    for identifier, node in unmatched[:8]:
        print(f"    {identifier[:18]}: node {node} has no exchange")
    return 1 if (mismatched or unmatched) else 0


if __name__ == "__main__":
    raise SystemExit(main())
