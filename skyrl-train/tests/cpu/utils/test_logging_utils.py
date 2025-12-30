"""
uv run --isolated --extra dev pytest tests/cpu/utils/test_logging_utils.py
"""

import pytest

from skyrl_train.utils.logging_utils import (
    BASE_PROMPT_COLOR,
    NEGATIVE_RESPONSE_COLOR,
    POSITIVE_RESPONSE_COLOR,
    _color_block_format_and_kwargs,
    log_example,
)


class StubLogger:
    """Minimal logger stub capturing calls to .opt().info()."""

    def __init__(self) -> None:
        self.last_message = None
        self.last_args = None
        self.last_kwargs = None

    def opt(self, **kwargs):
        # In real loguru this returns a new logger; here we just ignore options.
        return self

    def info(self, msg, *args, **kwargs):
        self.last_message = msg
        self.last_args = args
        self.last_kwargs = kwargs


def test_color_block_format_and_kwargs_single_line():
    fmt, kwargs = _color_block_format_and_kwargs("hello", "red", "p")

    assert fmt == "<red>{p0}</red>"
    assert kwargs == {"p0": "hello"}


def test_color_block_format_and_kwargs_multi_line():
    text = "line1\nline2"
    fmt, kwargs = _color_block_format_and_kwargs(text, "blue", "x")

    assert fmt == "<blue>{x0}</blue>\n<blue>{x1}</blue>"
    assert kwargs == {"x0": "line1", "x1": "line2"}


@pytest.mark.parametrize(
    "reward,expected_color",
    [
        (None, NEGATIVE_RESPONSE_COLOR),
        (0.0, NEGATIVE_RESPONSE_COLOR),
        (-1.0, NEGATIVE_RESPONSE_COLOR),
        (0.1, POSITIVE_RESPONSE_COLOR),
        ([0.1, 0.2], POSITIVE_RESPONSE_COLOR),
    ],
)
def test_log_example_uses_expected_colors_and_reward_string(reward, expected_color):
    logger = StubLogger()

    prompt = [{"role": "user", "content": "line1\nline2"}]
    response = "out1\nout2"

    log_example(logger, prompt=prompt, response=response, reward=reward)

    # Basic structure checks
    assert logger.last_message.startswith("Example:\n  Input: ")
    assert "Output (Total Reward: {reward}):" in logger.last_message

    # Placeholder keys from helper should be present
    assert "p0" in logger.last_kwargs
    assert "r0" in logger.last_kwargs

    # Prompt lines kept as-is
    assert logger.last_kwargs["p0"] == "[{'role': 'user', 'content': 'line1\\nline2'}]"

    # Response lines kept as-is
    assert logger.last_kwargs["r0"] == "out1"
    assert logger.last_kwargs["r1"] == "out2"

    # Reward formatting
    reward_str = logger.last_kwargs["reward"]
    if reward is None:
        assert reward_str == "N/A"
    else:
        # log_example normalizes rewards to a single float
        if isinstance(reward, list):
            expected_val = float(sum(reward))
        else:
            expected_val = float(reward)
        assert pytest.approx(float(reward_str), rel=1e-6) == expected_val

    # Color tags should appear in the format string with the correct colors
    assert f"<{BASE_PROMPT_COLOR}>" in logger.last_message
    assert f"</{BASE_PROMPT_COLOR}>" in logger.last_message
    assert f"<{expected_color}>" in logger.last_message
    assert f"</{expected_color}>" in logger.last_message


def test_log_example_handles_exceptions_gracefully(monkeypatch, capsys):
    """Force an exception inside log_example and ensure the fallback path prints."""

    def broken_color_block(*args, **kwargs):
        raise RuntimeError("boom")

    # Patch the helper to raise
    monkeypatch.setattr(
        "skyrl_train.utils.logging_utils._color_block_format_and_kwargs",
        broken_color_block,
    )

    logger = StubLogger()
    log_example(logger, prompt=[{"role": "user", "content": "p"}], response="r", reward=None)

    # And the plain-text fallback should be printed to stdout
    captured = capsys.readouterr()
    assert "Error pretty printing example" in captured.out
    assert "Example:" in captured.out
    assert "Input: [{'role': 'user', 'content': 'p'}]" in captured.out
    assert "Output (Total Reward: N/A):" in captured.out
    assert "r" in captured.out
