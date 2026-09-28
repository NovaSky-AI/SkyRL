"""Re-sampled reply: one context, sampled twice, is a fork.

The agent asks the same question twice from the same history -- to vote on the
answers, or because it did not like the first one. The request messages match
both times, but the two replies differ, so the shared user node ends up with
two `model`-authored children.

That is a fork, and both children are real generations from the same context.
An export gives two training samples here, which is the correct count: the
model produced two.
"""

from _capture import chat

messages = [{"role": "user", "content": "Name one cause of the flaky auth test."}]

first = chat(messages, "The token expiry check is inverted.")
second = chat(messages, "The test shares a fixture with the session test.")

print(f"sample 1: {first}")
print(f"sample 2: {second}")
