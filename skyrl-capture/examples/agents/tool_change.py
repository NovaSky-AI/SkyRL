"""Changed tool set: the same question under different tools is a new root.

The agent asks with read-only tools, the model answers without calling
anything, so the agent widens the tool set and asks *the same question again*.
The messages are byte-for-byte identical. Only `tools` differs.

It still starts a second root, and that is not an accident. A chat template
renders tool schemas into the prompt, so the second call did not see the first
call's context -- it saw a different one, differing from its very first token.
The tool set is part of what identifies a node for exactly this reason; without
it the graph would merge the two and report one generation where there were
two.

The replies are pinned identical here so that nothing *except* the tool set can
account for the split. `scripts/tool_change.py` is the same case as a longer
workload.
"""

from _capture import EDIT_TOOL, SEARCH_TOOL, chat

question = [{"role": "user", "content": "Fix the expiry check in auth.py."}]

narrow = chat(question, "I would need to edit the file.", tools=[SEARCH_TOOL])
wide = chat(question, "I would need to edit the file.", tools=[SEARCH_TOOL, EDIT_TOOL])

print(f"with search:        {narrow}")
print(f"with search + edit: {wide}")
print(f"\nreplies identical: {narrow == wide}. the graph should still show two roots.")
