"""Changing tools mid-conversation re-roots the whole thing.

`tool_change.py` changes the tool set on the very first
message, where starting a second root is unsurprising -- there was no history
to share. This is the case that surprises people: the agent has already had a
turn, and widens its tools for the next one.

The tool set is folded into *every* message of a call, not just the new ones,
so the match fails at the first message rather than at the turn where the tools
changed. The earlier messages are committed again under the new tool set, and
the whole prefix is duplicated.

That is deliberate. Tool schemas render at the top of the prompt, so the second
call's context differed from its first token -- there is no prefix the two
calls genuinely shared, and claiming one would misstate what the model saw. The
cost is that a conversation which changes tools every turn stores a triangle
rather than a path.
"""

from _capture import EDIT_TOOL, SEARCH_TOOL, chat

messages = [{"role": "user", "content": "Find the expiry bug."}]
messages.append(
    {"role": "assistant", "content": chat(messages, "It is in verify_expiry.", tools=[SEARCH_TOOL])}
)

# The same history, one turn further on -- but the agent has widened its tools.
messages.append({"role": "user", "content": "Now fix it."})
print(chat(messages, "Patched the comparison.", tools=[SEARCH_TOOL, EDIT_TOOL]))
