"""Identical retry: the same call, sent twice, is not a branch.

A transport hiccup makes the agent resend a request it has already sent. Every
request message matches, and the reply that comes back is one the graph already
has, so nothing new is committed -- the second call adds no message, and a
message is the only thing a node can be.

The retry is still recorded, as a second *exchange* carrying
`is_duplicate_retry: true`. Both exchanges resolve to the same
`input_leaf_node_id`, which is what makes the detection exact rather than a
guess about two calls that looked similar.
"""

from _capture import chat

messages = [{"role": "user", "content": "Which port does the proxy bind?"}]

first = chat(messages, "8080, from CAPTURE_PORT.")
second = chat(messages, "8080, from CAPTURE_PORT.")  # the same call, resent

print(f"first:  {first}")
print(f"second: {second}")
print(f"\nidentical: {first == second}. two calls, two exchanges, two nodes total.")
