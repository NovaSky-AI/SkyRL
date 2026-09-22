"""Rewritten history: compaction branches at the last unchanged message.

The agent runs out of context, asks for a summary, throws the transcript away,
and carries on from the system prompt plus that summary. The rebuilt history
still opens with the same system message, so the match gets exactly one node
deep and branches there.

Everything before the rewrite is genuinely shared and is kept once. Everything
after it is a different conversation, and is stored as one.

`scripts/compaction.py` is the same idea at full length, with two compactions
and the export and replay consequences worked through.
"""

from _capture import chat

SYSTEM = "You are a migration assistant."

messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": "What still calls the legacy client?"},
]
messages.append({"role": "assistant", "content": chat(messages, "Two call sites: invoices, refunds.")})
messages.append({"role": "user", "content": "Port invoices."})
messages.append({"role": "assistant", "content": chat(messages, "Done. invoices uses v2 now.")})

# -- compact: one more call carrying the whole transcript, then throw it away
summary = chat(
    messages + [{"role": "user", "content": "Summarize what is done and what remains."}],
    "invoices is ported to v2. refunds is not.",
)
print(f"summary: {summary}")

messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": f"Where we are: {summary}"},
    {"role": "user", "content": "Port refunds."},
]
print(f"after compaction: {chat(messages, 'Done. refunds uses v2 now.')}")
