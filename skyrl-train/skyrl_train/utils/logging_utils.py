from typing import Any, List, Optional, Union

POSITIVE_RESPONSE_COLOR = "green"
NEGATIVE_RESPONSE_COLOR = "yellow"
BASE_PROMPT_COLOR = "cyan"


def _color_block_format_and_args(
    text: str,
    color: str,
    field_prefix: str,
) -> tuple[str, dict]:
    """
    Build a format string and kwargs for a multi-line colored block.

    The format string will look like:
        "<color>{p0}</color>\n<color>{p1}</color>\n..."

    where "p0", "p1", ... are placeholder names starting with `field_prefix`.
    """
    # Ensure at least one line
    lines = text.splitlines() or [""]

    fmt_lines = []
    args: dict[str, str] = {}

    for i, line in enumerate(lines):
        key = f"{field_prefix}{i}"
        # NOTE: double braces {{ }} so that {key} survives into str.format
        fmt_lines.append(f"<{color}>{{{key}}}</{color}>")
        args[key] = line

    fmt = "\n".join(fmt_lines)
    return fmt, args


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
    try:
        # Normalize to strings
        if not isinstance(prompt, str):
            prompt = str(prompt)
        if not isinstance(response, str):
            response = str(response)

        # --- Reward handling ---
        reward_val = 0.0
        reward_str = "N/A"
        if reward is not None:
            if isinstance(reward, list):
                reward_val = float(sum(reward))
            else:
                reward_val = float(reward)
            reward_str = f"{reward_val:.4f}"

        # --- Color selection ---
        if reward is not None and reward_val > 0:
            response_color = POSITIVE_RESPONSE_COLOR
        else:
            response_color = NEGATIVE_RESPONSE_COLOR

        # --- Build per-line colored blocks in the *format string* ---
        prompt_fmt, prompt_args = _color_block_format_and_args(prompt, BASE_PROMPT_COLOR, "p")
        response_fmt, response_args = _color_block_format_and_args(response, response_color, "r")

        # Single format string with only our own markup and placeholders
        log_format = "Example:\n" f"  Input: {prompt_fmt}\n" "  Output (Reward: {reward}):\n" f"{response_fmt}"

        # Merge all args for str.format
        format_args = {}
        format_args.update(prompt_args)
        format_args.update(response_args)
        format_args["reward"] = reward_str

        # Let Loguru parse tags in log_format and then substitute arguments.
        logger.opt(colors=True).info(log_format, **format_args)
    except Exception as e:
        logger.info(f"Error pretty printing example, debug printing instead: {e}")
        print(f"Example:\n  Input: {prompt}\n  Output (Reward: {reward_str}):\n{response}")
