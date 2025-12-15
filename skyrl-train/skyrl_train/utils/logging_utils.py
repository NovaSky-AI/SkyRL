from typing import Any, Optional, Union, List

# Colors for log messages. "Positive" color is used to print responses with positive rewards,
# and "negative" color is used for non-negative rewards.
BASE_PROMPT_COLOR = "blue"
POSITIVE_RESPONSE_COLOR = "green"
NEGATIVE_RESPONSE_COLOR = "yellow"


def log_example(
    logger: Any,
    prompt: str,
    response: str,
    reward: Optional[Union[float, List[float]]] = None,
) -> None:
    """
    Log a single example prompt and response with formatting and colors.

    Args:
        logger: The logger instance to use (expected to be loguru logger or compatible).
        prompt: The input prompt string.
        response: The output response string.
        reward: The reward value(s) associated with the response.
    """
    # Determine reward value for display and color logic
    reward_val = 0.0
    reward_str = "N/A"

    if reward is not None:
        if isinstance(reward, list):
            # If reward is a list (e.g. token-level rewards), sum them up for the total reward
            # This assumes the list contains floats
            reward_val = sum(reward)
        else:
            reward_val = float(reward)
        reward_str = f"{reward_val:.4f}"

    # Determine response color based on reward
    # If reward is <= 0, use yellow (warning/caution)
    # If reward > 0, use green (success)
    # If reward is None (N/A), default to yellow
    if reward is not None and reward_val > 0:
        response_color = f"<{POSITIVE_RESPONSE_COLOR}>"
        response_end_color = f"</{POSITIVE_RESPONSE_COLOR}>"
    else:
        response_color = f"<{NEGATIVE_RESPONSE_COLOR}>"
        response_end_color = f"</{NEGATIVE_RESPONSE_COLOR}>"

    prompt_color = f"<{BASE_PROMPT_COLOR}>"
    prompt_end_color = f"</{BASE_PROMPT_COLOR}>"

    # Escape tags in content to prevent loguru from interpreting them as color directives
    # We only need to escape the opening bracket '<'
    # Ensure inputs are strings
    if not isinstance(prompt, str):
        prompt = str(prompt)
    if not isinstance(response, str):
        response = str(response)

    safe_prompt = prompt.replace("<", "\\<")
    safe_response = response.replace("<", "\\<")

    def colorize_lines(text: str, start_tag: str, end_tag: str) -> str:
        # Apply color tags to each line individually to handle Ray's line prefixing
        return "\n".join(f"{start_tag}{line}{end_tag}" for line in text.splitlines())

    colored_prompt = colorize_lines(safe_prompt, prompt_color, prompt_end_color)
    colored_response = colorize_lines(safe_response, response_color, response_end_color)

    log_msg = f"Example:\n" f"  Input: {colored_prompt}\n" f"  Output (Reward: {reward_str}):\n" f"{colored_response}"

    logger.opt(colors=True).info(log_msg)
