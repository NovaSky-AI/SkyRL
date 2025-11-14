"""Reusable message templates for ReAct agent guidance."""

TOOL_CALL_PARSE_ERROR_GUIDANCE = (
    "Tool call parsing failed with error:\n"
    "{error}\n\n"
    "Please use the correct format:\n"
    "<function=tool_name>\n"
    "<parameter=param_name>value</parameter>\n"
    "</function>\n\n"
    "Remember: Always end with the finish tool using \\boxed{} format."
)


NO_TOOL_CALL_DETECTED_GUIDANCE = (
    "No valid tool call detected. Please use the correct format:\n"
    "<function=tool_name>\n"
    "<parameter=param_name>value</parameter>\n"
    "</function>\n\n"
    "Remember to call the finish tool with \\boxed{} format:\n"
    "<function=finish>\n"
    "<parameter=answer>\\boxed{YOUR_ANSWER}</parameter>\n"
    "</function>"
)


TOOL_INVOCATION_ERROR_GUIDANCE = (
    "Tool call failed due to invalid arguments. Please correct and retry.\n"
    "Examples:\n"
    '- search_engine: {"query": ["term1", "term2"]}\n'
    '- web_browser: {"url": ["https://example.com"], "goal": "extract key facts"}'
)


__all__ = [
    "TOOL_CALL_PARSE_ERROR_GUIDANCE",
    "NO_TOOL_CALL_DETECTED_GUIDANCE",
    "TOOL_INVOCATION_ERROR_GUIDANCE",
]
