"""The SWE-bench Multimodal agent config's action regex: one command, and only after the thinking closes.

mini-swe-agent runs ``re.findall(action_regex, reply, re.DOTALL)`` and takes the reply only when it finds
exactly one action; anything else is a format error.
"""

import re
from pathlib import Path

import pytest
import yaml

CONFIG = (
    Path(__file__).parents[3]
    / "examples/train_integrations/harbor_skycap/swebench_multimodal/mini_swe_agent_textbased.yaml"
)


def actions(reply: str) -> list:
    regex = yaml.safe_load(CONFIG.read_text())["model"]["action_regex"]
    return [action.strip() for action in re.findall(regex, reply, re.DOTALL)]


@pytest.mark.parametrize("tag", ["bash", "sh", "mswea_bash_command"])
def test_one_command_after_the_thinking_is_the_action(tag):
    reply = f"I should look around.\n</think>\n\nListing files.\n\n```{tag}\nls -la\n```"
    assert actions(reply) == ["ls -la"]


def test_a_fence_inside_the_thinking_does_not_count():
    reply = "Maybe:\n```bash\nrm -rf /\n```\nNo, better to look first.\n</think>\n\n```bash\nls\n```"
    assert actions(reply) == ["ls"]


@pytest.mark.parametrize(
    "reply",
    [
        # The turn ended still thinking, command included.
        "Let me list the files.\n\n```bash\nls -la\n```",
        # Two commands.
        "Done thinking.\n</think>\n\n```bash\nls\n```\n\n```bash\npwd\n```",
        # A garbled tag.
        "Done thinking.\n</think>\n\n```mswea_bash_comand\nls\n```",
        # No command at all.
        "Done thinking.\n</think>\n\nI think the fix is in utils.js.",
        # A fence inside the command: it must not run cut short at the inner fence.
        "Done thinking.\n</think>\n\n```bash\ncat > notes.md <<'EOF'\n```python\nprint(1)\n```\nEOF\nrm -rf build\n```",
        # A second command after a later </think>: not just the last one.
        "Done.\n</think>\n\n```bash\nls\n```\n<think>\nMore.\n</think>\n\n```bash\npwd\n```",
    ],
    ids=["open-think", "two-commands", "misspelled-tag", "no-command", "inner-fence", "second-think-block"],
)
def test_malformed_replies_are_format_errors(reply):
    assert len(actions(reply)) != 1


def test_the_prompts_show_the_fence_the_regex_takes():
    config = CONFIG.read_text()
    assert "```bash" in config
    assert "```mswea_bash_command\n" not in config.split("action_regex")[0]
