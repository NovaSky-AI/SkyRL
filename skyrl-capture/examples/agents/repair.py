"""Repair: the client edits the model's reply before continuing from it.

The model answers, the harness decides the answer is not quite right -- a
malformed tool call, a bad format, a wrong number -- fixes it, and sends the
*fixed* version back as history on the next turn.

The first message matches. The edited assistant message does not, so it becomes
a new node, and it is marked `author: client`, because the client wrote it. The
model's original stays where it was, and the user node now has two children
that are the same role and different authors.

That distinction is the reason `author` exists. The repaired text was never
sampled from any model, so training on it would be training on the harness. The
graph keeps both and lets the export decide.
"""

from _capture import chat

messages = [{"role": "user", "content": "How many retries does the client make?"}]

sampled = chat(messages, "It retries four times.")
repaired = "It retries three times."  # the harness knows better
print(f"model said:   {sampled}")
print(f"client wrote: {repaired}")

messages.append({"role": "assistant", "content": repaired})
messages.append({"role": "user", "content": "And the backoff?"})
print(f"model said:   {chat(messages, 'Doubling from 100ms.')}")
