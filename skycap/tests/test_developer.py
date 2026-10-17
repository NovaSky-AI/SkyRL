"""The ``developer`` role folded into ``system`` for a template without it, by vLLM's rule."""

from __future__ import annotations

import copy

import pytest

from skycap.tokens.developer import fold_developer, template_knows_developer

PLAIN = "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}"
KNOWS = "{% if m.role == 'developer' %}<|developer|>{% endif %}"


def dev(text: object) -> dict:
    return {"role": "developer", "content": text}


def system(text: str) -> dict:
    return {"role": "system", "content": text}


def user(text: str) -> dict:
    return {"role": "user", "content": text}


@pytest.mark.parametrize(
    "template, knows",
    [
        (KNOWS, True),
        ('{% if m.role == "developer" %}', True),
        (PLAIN, False),
        ("{# a developer note, unquoted #}", False),
        ("", False),
    ],
)
def test_the_template_knows_the_role_when_it_names_it_in_quotes(template: str, knows: bool) -> None:
    assert template_knows_developer(template) is knows


def test_without_a_template_or_with_one_that_knows_the_role_nothing_changes() -> None:
    messages = [user("q"), dev("d")]

    assert fold_developer(messages, None) == messages
    assert fold_developer(messages, KNOWS) == messages


def test_without_a_developer_message_nothing_changes_even_a_late_system_message() -> None:
    messages = [user("q"), system("late")]

    assert fold_developer(messages, PLAIN) == messages


def test_one_leading_developer_message_becomes_the_system_message() -> None:
    assert fold_developer([dev("d"), user("q")], PLAIN) == [system("d"), user("q")]


def test_system_and_developer_text_merge_into_one_leading_message() -> None:
    messages = [system("s"), dev("d1"), user("q"), dev("d2")]

    assert fold_developer(messages, PLAIN) == [system("s\n\nd1\n\nd2"), user("q")]


def test_a_developer_message_after_the_conversation_starts_moves_to_the_front() -> None:
    messages = [user("q"), {"role": "assistant", "content": "a"}, dev("d"), user("q2")]

    assert fold_developer(messages, PLAIN) == [
        system("d"),
        user("q"),
        {"role": "assistant", "content": "a"},
        user("q2"),
    ]


def test_parts_join_with_a_newline_and_empty_text_is_skipped_when_merging() -> None:
    messages = [dev([{"type": "text", "text": "a"}, "b"]), dev(""), dev(None), user("q"), dev("c")]

    assert fold_developer(messages, PLAIN) == [system("a\nb\n\nc"), user("q")]


def test_a_lone_leading_message_keeps_its_parts_and_loses_its_tools() -> None:
    parts = [{"type": "text", "text": "a"}]

    assert fold_developer([{"role": "developer", "content": parts, "tools": []}, user("q")], PLAIN) == [
        {"role": "system", "content": parts},
        user("q"),
    ]


def test_the_request_is_not_modified() -> None:
    messages = [system("s"), dev("d"), user("q")]
    before = copy.deepcopy(messages)

    fold_developer(messages, PLAIN)

    assert messages == before
