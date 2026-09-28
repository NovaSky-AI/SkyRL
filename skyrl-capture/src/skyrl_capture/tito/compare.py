"""Say *what* diverged, not just that something did.

The prefix check on the normal path is structural and cheap: the reused prefix
length has to equal the bridge transition's `prompt + completion`, and the
tokens behind that are the ones those nodes were committed from, so agreeing on
where the prefix ends is agreeing on the prefix. `TOKENS_AUDIT_PREFIX` re-proves
it token by token, and it used to return a boolean: all or nothing, and nothing
to act on.

This classifies the disagreement instead. Borrowed in spirit from Miles's
`TokenSeqComparator` -- the idea that "the special tokens moved" and "an
assistant string retokenized differently" are separate signals -- but not in
substance, because the two systems differ on what is tolerable:

* Miles compares a spliced sequence against a **canonical re-render** and must
  tolerate `ASSISTANT_TEXT` differences, since the canonical retokenization of
  sampled text is the thing that is wrong.
* We never re-render a committed turn. Both sides here are tokens that passed
  through this process, so **every class is a bug**. The classes exist to point
  at which part of the machinery, not to be forgiven.

| Class | Where it lands | What it means |
| --- | --- | --- |
| `length` | totals disagree | the graph holds a different number of tokens than the prefix claims -- a commit or a bridge accounting bug |
| `scaffold` | before a node's `sampled_start` | the template's generation prefix moved: a renderer or a `trim_to_turn_close` change |
| `sampled` | at or after `sampled_start` on a model node | the tokens the engine returned are not the tokens stored: a commit bug, and the most serious class |
| `given` | inside a client-authored node | the client's own message retokenized differently, which a bridge is supposed to make impossible |

A report names the class, the node, the absolute token offset, and a decoded
window either side -- because the offset is what you seek to and the text is
what tells you which of the two is wrong.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

# How many tokens either side of the divergence to decode. Enough to see the
# marker it sits at and the text it sits in, short enough to fit in a log line.
WINDOW = 12


@dataclass(frozen=True)
class PrefixComparison:
    """What the stored deltas and the sent prompt disagree about, if anything."""

    ok: bool
    #: None when ok. Otherwise one of length / scaffold / sampled / given.
    kind: str | None = None
    node_id: str | None = None
    author: str | None = None
    role: str | None = None
    #: Absolute offset into the prompt where the two first differ.
    offset: int | None = None
    #: Offset within the node's own delta, which is what a renderer bug indexes.
    node_offset: int | None = None
    expected: tuple[int, ...] = ()
    observed: tuple[int, ...] = ()
    expected_text: str | None = None
    observed_text: str | None = None
    stored_total: int | None = None
    claimed_total: int | None = None

    def describe(self) -> str:
        """One line for an exception or a log."""
        if self.ok:
            return "prefix matches"
        if self.kind == "length":
            return (
                f"prefix length disagrees: the graph holds {self.stored_total} tokens "
                f"across these nodes, the prompt claims {self.claimed_total} were reused"
            )
        where = f"node {self.node_id} ({self.author}/{self.role})"
        detail = f"at prompt offset {self.offset}, {self.node_offset} into {where}"
        if self.expected_text is not None:
            return (
                f"{self.kind} tokens diverge {detail}: "
                f"stored {self.expected_text!r} vs sent {self.observed_text!r}"
            )
        return f"{self.kind} tokens diverge {detail}: {self.expected} vs {self.observed}"


def compare_prefix(
    nodes: Sequence[Any],
    prompt_token_ids: Sequence[int],
    reused: int,
    *,
    decode: Any = None,
) -> PrefixComparison:
    """Compare the stored node deltas against the prompt's reused prefix.

    ``nodes`` are the trace's nodes along the path, in order, each carrying
    ``token_ids``, ``sampled_start``, ``author`` and ``role``. ``decode`` is
    optional: without it the report carries ids rather than text, which is
    still enough to seek to.

    Stops at the first disagreement. There is no value in enumerating the rest:
    everything after a divergence is offset by it.
    """
    stored_total = sum(len(node.token_ids) for node in nodes)
    if stored_total != reused:
        return PrefixComparison(
            ok=False, kind="length", stored_total=stored_total, claimed_total=reused
        )

    cursor = 0
    for node in nodes:
        tokens = node.token_ids
        end = cursor + len(tokens)
        sent = tuple(prompt_token_ids[cursor:end])
        if sent == tuple(tokens):
            cursor = end
            continue
        offset = next(
            (index for index, pair in enumerate(zip(tokens, sent, strict=False)) if pair[0] != pair[1]),
            min(len(tokens), len(sent)),
        )
        return _report(node, tokens, sent, cursor, offset, decode)
    return PrefixComparison(ok=True)


def _report(
    node: Any,
    stored: Sequence[int],
    sent: Sequence[int],
    base: int,
    offset: int,
    decode: Any,
) -> PrefixComparison:
    sampled_start = getattr(node, "sampled_start", None)
    author = getattr(node, "author", None)
    if author == "model" and sampled_start is not None and offset >= sampled_start:
        kind = "sampled"
    elif author == "model":
        # Everything before `sampled_start` on a model node is the template's
        # generation prefix, which is exactly the boundary that moves when a
        # renderer or its turn-close trimming changes.
        kind = "scaffold"
    else:
        kind = "given"

    low, high = max(0, offset - WINDOW), offset + WINDOW
    expected = tuple(stored[low:high])
    observed = tuple(sent[low:high])
    return PrefixComparison(
        ok=False,
        kind=kind,
        node_id=getattr(node, "node_id", None),
        author=author,
        role=getattr(node, "role", None),
        offset=base + offset,
        node_offset=offset,
        expected=expected,
        observed=observed,
        expected_text=decode(list(expected)) if decode else None,
        observed_text=decode(list(observed)) if decode else None,
    )
