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


STEP_REMAINING_REMINDER = "\nSteps remaining: {remaining_steps}."


STEP_REMAINING_EARLY_REMINDER = "\nDo not stop now, conduct at least {min_steps} more actions."


STEP_LAST_REMINDER = "\nThis is your last step, make sure to use the finish tool to submit your final answer."


def get_turn_reminder_text(
    step_count: int,
    remaining_steps: int,
    early_step_threshold: int = 0,
) -> str:
    """Generate turn reminder text based on step count and remaining steps.

    Args:
        step_count: Current step count (1-indexed)
        remaining_steps: Number of steps remaining (max_iterations - step_count + 1)
        early_step_threshold: Step count threshold for early reminder (default: 0)

    Returns:
        Formatted reminder text to append to the last message
    """
    if remaining_steps > 1 and step_count > early_step_threshold:
        return STEP_REMAINING_REMINDER.format(remaining_steps=remaining_steps)
    elif remaining_steps > 1 and step_count <= early_step_threshold:
        min_steps = early_step_threshold - step_count
        return STEP_REMAINING_EARLY_REMINDER.format(min_steps=min_steps)
    else:
        return STEP_LAST_REMINDER


__all__ = [
    "TOOL_CALL_PARSE_ERROR_GUIDANCE",
    "NO_TOOL_CALL_DETECTED_GUIDANCE",
    "TOOL_INVOCATION_ERROR_GUIDANCE",
    "STEP_REMAINING_REMINDER",
    "STEP_REMAINING_EARLY_REMINDER",
    "STEP_LAST_REMINDER",
    "get_turn_reminder_text",
]
