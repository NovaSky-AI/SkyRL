import re
from typing import Any

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command in the terminal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Mark the task as complete. Call this when you have verified the task is solved.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

_SYSTEM_PROMPT_REGISTRY = {
    "hermes": (
        "You are a highly capable Linux terminal agent. "
        "Complete the user's task by running commands and verifying the result. "
        "When the task is complete, call done."
    ),
    "qwen35": (
        "You are a highly capable Linux terminal agent. "
        "Complete the user's task by running commands and verifying the result. "
        "When the task is complete, call done."
    ),
}


def get_augmented_system_content(tokenizer: Any, parser_name: str = "hermes") -> str:
    system_msg = _SYSTEM_PROMPT_REGISTRY.get(parser_name, "")
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_msg}, {"role": "user", "content": ""}],
        tools=TOOL_SCHEMAS,
        tokenize=False,
        add_generation_prompt=False,
    )
    match = re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", rendered, re.DOTALL)
    if match:
        return match.group(1)
    return system_msg
