from typing import Any, Optional, Union, List
from loguru import logger as loguru_logger

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
        response_color = "<green>"
        response_end_color = "</green>"
    else:
        response_color = "<yellow>"
        response_end_color = "</yellow>"

    prompt_color = "<blue>"
    prompt_end_color = "</blue>"

    log_msg = (
        f"AKRENTSEL Example:\n"
        f"  Input: {prompt_color}{prompt}{prompt_end_color}\n"
        f"  Output (Reward: {reward_str}):\n"
        f"{response_color}{response}{response_end_color}"
    )

    logger.opt(colors=True).info(log_msg)
