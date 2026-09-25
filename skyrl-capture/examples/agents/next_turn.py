"""Ordinary next turn: the second call continues the first.

The agent asks, gets a reply, appends the reply to its history, and asks again.
The history the second call sends *is* the first call's path plus a tail, so
the match runs the whole way down and only the two new messages are committed.

This is the common case, and it is the one that makes a shared prefix worth
having: the four nodes below are stored once, not twice.
"""

from _capture import chat

messages = [{"role": "user", "content": "What does the retry decorator do?"}]
messages.append({"role": "assistant", "content": chat(messages, "It retries on 5xx, three times.")})

messages.append({"role": "user", "content": "Is the backoff exponential?"})
messages.append({"role": "assistant", "content": chat(messages, "Yes, doubling from 100ms.")})

for message in messages:
    print(f"{message['role']:>9}: {message['content']}")
