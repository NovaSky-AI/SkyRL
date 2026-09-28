"""Nothing matches: a context with no shared opening starts a second root.

Two calls in one trajectory that share no prefix at all -- here because the
system prompt differs, which is what happens when one harness run drives two
differently-configured agents.

The match fails at the very first message, so there is nothing to attach to and
the second call starts a root of its own. A trajectory is free to hold several
roots; the graph is a forest, not a tree, and a root is just a node with no
parent.
"""

from _capture import chat

reviewer = [
    {"role": "system", "content": "You are a code reviewer. Be blunt."},
    {"role": "user", "content": "Review the retry decorator."},
]
print(f"reviewer: {chat(reviewer, 'The backoff is unbounded.')}")

writer = [
    {"role": "system", "content": "You are a technical writer. Be gentle."},
    {"role": "user", "content": "Review the retry decorator."},
]
print(f"writer:   {chat(writer, 'It reads clearly; consider naming the ceiling.')}")
