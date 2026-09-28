"""A recursive agent: decompose, recurse, consolidate.

Sub-agents that *inherit their parent's history* are the case this directory
did not have. Each sub-agent sends the parent's context plus its own
instruction, so it is a branch from the node it forked at -- and a sub-agent
that spawns its own children is that same shape one level down. Nothing in
capture special-cases recursion; it falls out of prefix matching.

The consolidation is the part worth looking at. When the parent gathers its
children's findings it continues from the node it forked at, so it is a
*sibling* of the sub-agents rather than a descendant of the last one -- which
is what the tree shows, and what a flat transcript cannot.

Three levels deep with two sub-questions each: 32 calls, 66 nodes, 22
root-to-leaf paths, 10 branch points. The paths total about 5,300 tokens and
the distinct nodes about 2,300, so storing each path whole would be 2.3x the
tokens. That ratio *is* how much the harness rewrites its history.
"""

from _capture import chat

MAX_DEPTH = 2
SUBQUESTIONS = {
    0: ["Is it safe under load?", "What does it cost?"],
    1: ["Name one failure mode.", "Name one mitigation."],
}


def solve(history: list[dict], question: str, depth: int) -> str:
    """Answer one question, splitting it first while recursion is allowed."""
    answer = chat([*history, {"role": "user", "content": question}], f"On {question}: it depends.")
    if depth >= MAX_DEPTH:
        return answer

    # This agent's own context. Its children inherit it, and so does the
    # consolidation below -- which is why they are siblings.
    mine = [
        *history,
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    findings = [solve(mine, sub, depth + 1) for sub in SUBQUESTIONS[depth]]
    combine = "Combine these into one sentence:\n" + "\n".join(f"- {f}" for f in findings)
    return chat([*mine, {"role": "user", "content": combine}], "Taken together: it depends.")


root = [
    {"role": "system", "content": "You answer in one short sentence."},
    {"role": "user", "content": "Compare retry strategies for a flaky API."},
]
plan_question = "List the three strategies you will compare."
plan = chat([*root, {"role": "user", "content": plan_question}], "Backoff, jitter, circuit breaker.")

# The node every sub-agent forks from, and the one the consolidation returns to.
shared = [*root, {"role": "user", "content": plan_question}, {"role": "assistant", "content": plan}]

reports = [
    solve(shared, f"Research: {strategy}", 0)
    for strategy in ("Exponential backoff.", "Backoff with jitter.", "A circuit breaker.")
]

verdict = "Given these findings, recommend one:\n" + "\n".join(f"- {r}" for r in reports)
print(chat([*shared, {"role": "user", "content": verdict}], "Use jitter."))
