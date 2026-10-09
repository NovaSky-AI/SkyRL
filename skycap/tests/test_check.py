"""`skycap check`: each kind of fork a harness can cause, read back from a written record."""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest
import zstandard

from skycap import cli, record
from skycap.check import check_dir, format_reports
from skycap.graph import CallInfo
from skycap.trajectory import Trajectory
from tests.conftest import Stack, openai_client
from tests.fake_renderer import END, NL, encode
from tests.test_tokens import token_stack

SYS = {"role": "system", "content": "You are terse."}
SEARCH = [{"type": "function", "function": {"name": "search", "parameters": {}}}]


def user(text: str) -> dict:
    return {"role": "user", "content": text}


def assistant(text: str, **extra) -> dict:
    return {"role": "assistant", "content": text, **extra}


class Harness:
    """Drives one text-mode trajectory's graph a call at a time, then writes it."""

    def __init__(self, trajectory_id: str) -> None:
        self.trajectory = Trajectory(id=trajectory_id)
        self.clock = 0.0

    def call(self, messages, reply, *, tools=None, model="policy", bridged=None, **sampling) -> None:
        self.clock += 1.0
        info = CallInfo(t_start=self.clock, t_end=self.clock + 0.5, model=model, sampling=sampling, bridged=bridged)
        self.trajectory.graph.commit_text(messages, reply, tools=tools, model=model, call=info)

    def write(self, record_dir: Path) -> None:
        self.trajectory.seal("finished", {"reward": 1.0})
        record.write(record_dir, self.trajectory)


def report_lines(record_dir: Path, *harnesses: Harness) -> list[str]:
    for harness in harnesses:
        harness.write(record_dir)
    return format_reports(check_dir(record_dir)).splitlines()


def test_an_append_only_history_is_linear(tmp_path: Path) -> None:
    h = Harness("tr")
    h.call([SYS, user("q1")], assistant("a1"))
    h.call([SYS, user("q1"), assistant("a1"), user("q2")], assistant("a2"))

    assert report_lines(tmp_path, h) == ["tr  finished  linear", "1 of 1 trajectories linear"]


def test_a_trajectory_without_calls_is_empty(tmp_path: Path) -> None:
    assert report_lines(tmp_path, Harness("tr")) == ["tr  finished  empty", "1 of 1 trajectories linear"]


def test_stripped_reasoning_is_an_edited_reply(tmp_path: Path) -> None:
    h = Harness("tr")
    h.call([user("q")], assistant("answer", reasoning_content="let me think"))
    h.call([user("q"), assistant("answer"), user("more")], assistant("more answer"))

    assert report_lines(tmp_path, h) == [
        "tr  finished  2 paths",
        "  node 2 (under node 0, vs node 1): edited reply: "
        "the harness sent back the model's reply with reasoning_content dropped",
        "0 of 1 trajectories linear; forks: 1 edited reply",
    ]


def test_an_edit_is_compared_with_the_reply_it_came_from(tmp_path: Path) -> None:
    h = Harness("tr")
    h.call([user("q")], assistant("A", reasoning_content="r1"))
    h.call([user("q")], assistant("B", reasoning_content="r2"))
    h.call([user("q"), assistant("A"), user("more")], assistant("ok"))

    assert report_lines(tmp_path, h)[1:3] == [
        "  node 2 (under node 0, vs node 1): resample: a second reply to the same history",
        "  node 3 (under node 0, vs node 1): edited reply: "
        "the harness sent back the model's reply with reasoning_content dropped",
    ]


def test_the_same_reply_under_other_sampling_is_a_resample(tmp_path: Path) -> None:
    h = Harness("tr")
    h.call([user("q")], assistant("a"), top_p=1.0)
    h.call([user("q")], assistant("a"), top_p=0.5)

    assert report_lines(tmp_path, h)[1] == (
        "  node 2 (under node 0, vs node 1): resample: "
        "the same reply again, under other sampling parameters or tokenized differently"
    )


def test_a_sample_equal_to_a_harness_written_message_is_a_resample(tmp_path: Path) -> None:
    h = Harness("tr")
    h.call([user("q"), assistant("x"), user("again")], assistant("y"))
    h.call([user("q")], assistant("x"))

    assert report_lines(tmp_path, h)[1] == (
        "  node 4 (under node 0, vs node 1): resample: the model sampled a message the harness had already written"
    )


def test_a_reply_to_a_shortened_history_is_a_resample(tmp_path: Path) -> None:
    h = Harness("tr")
    h.call([SYS, user("task")], assistant("a"))
    h.call([SYS], assistant("b"))

    assert report_lines(tmp_path, h)[1] == (
        "  node 3 (under node 0, vs node 1): resample: "
        "a reply to a history the other path continues with a user message"
    )


def test_compaction_is_a_different_message(tmp_path: Path) -> None:
    h = Harness("tr")
    h.call([SYS, user("task")], assistant("step1"))
    h.call([SYS, user("task"), assistant("step1"), user("summarize")], assistant("summary"))
    h.call([SYS, user("summary"), user("continue")], assistant("step2"))

    assert report_lines(tmp_path, h)[1] == (
        "  node 5 (under node 0, vs node 1): different message: a different user message: content changed"
    )


def test_a_subagent_with_its_own_system_prompt_is_a_new_root(tmp_path: Path) -> None:
    h = Harness("tr")
    h.call([SYS, user("task")], assistant("delegating"))
    h.call([{"role": "system", "content": "You are a searcher."}, user("find x")], assistant("found"))

    assert report_lines(tmp_path, h)[1] == (
        "  node 3 (new root, vs node 0): different message: a different system message: content changed"
    )


def test_a_changed_tool_set_is_compared_with_the_root_it_repeats(tmp_path: Path) -> None:
    h = Harness("tr")
    h.call([SYS, user("task")], assistant("plan"))
    h.call([{"role": "system", "content": "You are a searcher."}, user("find x")], assistant("found"))
    h.call([SYS, user("task"), assistant("plan"), user("go")], assistant("done"), tools=SEARCH)

    assert report_lines(tmp_path, h)[2] == (
        "  node 6 (new root, vs node 0): tools or model: the same message, sent with a different tool set or model"
    )


def test_a_message_of_another_role_names_both(tmp_path: Path) -> None:
    h = Harness("tr")
    h.call([SYS, user("task")], assistant("plan"))
    h.call([SYS, {"role": "assistant", "content": "prefill"}], assistant("x"))

    assert report_lines(tmp_path, h)[1] == (
        "  node 3 (under node 0, vs node 1): different message: "
        "an assistant message where the other path has a user message"
    )


def test_unbridged_calls_are_counted(tmp_path: Path) -> None:
    h = Harness("tr")
    h.call([user("q1")], assistant("a1"))
    h.call([user("q1"), assistant("a1"), user("q2")], assistant("a2"), bridged=False)

    assert report_lines(tmp_path, h) == [
        "tr  finished  linear",
        "  1 unbridged call(s): prompt re-rendered, not extended",
        "1 of 1 trajectories linear",
    ]


async def test_a_reply_whose_re_render_changes_its_tokens_is_re_rendered(tmp_path: Path) -> None:
    # Without END the bridge can't extend the completion, and the full render of
    # the replayed reply ends it with END, so the same message gets new tokens.
    def completion(prompt, sampling):
        return [*encode("ans"), NL, *encode("wer")]

    async with token_stack(completion=completion, record_dir=tmp_path) as stack:
        created = await stack.create()
        llm = openai_client(created["base_url"])
        reply = (await llm.chat.completions.create(model="policy", messages=[user("q")])).choices[0].message
        replayed = reply.model_dump(exclude_none=True)
        await llm.chat.completions.create(model="policy", messages=[user("q"), replayed, user("more")])
        await stack.finish(created["id"])

    assert format_reports(check_dir(tmp_path)).splitlines() == [
        f"{created['id']}  finished  2 paths",
        "  node 2 (under node 0, vs node 1): re-rendered: the same message, rendered to different tokens",
        "  1 unbridged call(s): prompt re-rendered, not extended",
        "0 of 1 trajectories linear; forks: 1 re-rendered",
    ]


async def test_a_resample_after_a_re_render_is_compared_with_the_model_reply(tmp_path: Path) -> None:
    def completion(prompt, sampling):
        return [*encode("ans"), NL, *encode("wer")]

    async with token_stack(completion=completion, record_dir=tmp_path) as stack:
        created = await stack.create()
        llm = openai_client(created["base_url"])
        reply = (await llm.chat.completions.create(model="policy", messages=[user("q")], top_p=1.0)).choices[0]
        replayed = reply.message.model_dump(exclude_none=True)
        await llm.chat.completions.create(model="policy", messages=[user("q"), replayed, user("more")])
        # The same tokens again under another top_p: a model sibling, not the re-rendered copy.
        await llm.chat.completions.create(model="policy", messages=[user("q")], top_p=0.5)
        await stack.finish(created["id"])

    lines = format_reports(check_dir(tmp_path)).splitlines()
    assert lines[1:3] == [
        "  node 2 (under node 0, vs node 1): re-rendered: the same message, rendered to different tokens",
        "  node 5 (under node 0, vs node 1): resample: "
        "the same reply again, under other sampling parameters or tokenized differently",
    ]


async def test_token_mode_names_only_fields_the_renderer_uses(tmp_path: Path) -> None:
    def completion(prompt, sampling):
        return [*encode("THINK:hmm|CALL:search:{}"), END]

    async with token_stack(completion=completion, record_dir=tmp_path) as stack:
        created = await stack.create()
        llm = openai_client(created["base_url"])
        first = await llm.chat.completions.create(model="policy", messages=[user("q")], tools=SEARCH)
        reply = first.choices[0].message.model_dump(exclude_none=True)
        # Dropping reasoning forks. Client metadata such as provider_specific_fields
        # isn't rendered, so adding it doesn't, and it mustn't be named.
        replay = {key: value for key, value in reply.items() if key != "reasoning_content"}
        replay["provider_specific_fields"] = {"x": 1}
        tool_result = {"role": "tool", "tool_call_id": reply["tool_calls"][0]["id"], "content": "found"}
        body = {"model": "policy", "messages": [user("q"), replay, tool_result], "tools": SEARCH}
        async with stack.http.post(f"{created['base_url']}/chat/completions", json=body) as response:
            assert response.status == 200
        await stack.finish(created["id"])

    assert format_reports(check_dir(tmp_path)).splitlines()[1] == (
        "  node 2 (under node 0, vs node 1): edited reply: "
        "the harness sent back the model's reply with reasoning_content dropped"
    )


async def test_check_reads_what_a_capture_server_wrote(recorded_stack: Stack, capsys: pytest.CaptureFixture) -> None:
    async def run(edit_reply: bool) -> str:
        created = await recorded_stack.create()
        llm = openai_client(created["base_url"])
        first = await llm.chat.completions.create(model="policy", messages=[user("q")])
        replayed = "edited" if edit_reply else first.choices[0].message.content
        await llm.chat.completions.create(model="policy", messages=[user("q"), assistant(replayed), user("next")])
        await recorded_stack.finish(created["id"])
        return created["id"]

    appended, edited = await run(edit_reply=False), await run(edit_reply=True)
    record_dir = recorded_stack.server.record_dir
    assert record_dir is not None

    assert cli.main(["check", str(record_dir), appended]) == 0
    assert capsys.readouterr().out.splitlines() == [f"{appended}  finished  linear", "1 of 1 trajectories linear"]
    assert cli.main(["check", str(record_dir), edited, edited]) == 1
    assert capsys.readouterr().out.splitlines() == [
        f"{edited}  finished  2 paths",
        "  node 2 (under node 0, vs node 1): edited reply: "
        "the harness sent back the model's reply with content changed",
        "0 of 1 trajectories linear; forks: 1 edited reply",
    ]


def test_an_unreadable_record_is_reported_and_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    Harness("tr_ok").write(tmp_path)
    (tmp_path / "tr_bad.json.zst").write_bytes(b"not zstd")
    future = {"format_version": 99, "id": "tr_new"}
    (tmp_path / "tr_new.json.zst").write_bytes(zstandard.ZstdCompressor().compress(orjson.dumps(future)))
    (tmp_path / "tr_list.json.zst").write_bytes(zstandard.ZstdCompressor().compress(orjson.dumps([1, 2, 3])))

    assert cli.main(["check", str(tmp_path)]) == 2
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("tr_bad  unreadable: ZstdError: ")
    assert lines[1:] == [
        "tr_list  unreadable: the document is a JSON list, not an object",
        "tr_new  unreadable: format_version 99, this skycap reads 1",
        "tr_ok  finished  empty",
        "1 of 1 trajectories linear; 3 unreadable",
    ]


@pytest.mark.parametrize(
    ("record_dir", "ids", "message"),
    [
        ("absent", [], "absent is not a directory"),
        ("record", [], "no trajectories in"),
        ("record", ["tr_x"], "no trajectory tr_x in"),
        ("record", ["../tr_x"], "no trajectory ../tr_x in"),
    ],
)
def test_usage_errors_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture, record_dir: str, ids: list[str], message: str
) -> None:
    (tmp_path / "record").mkdir()
    Harness("tr_x").write(tmp_path)  # beside the record directory, not in it

    with pytest.raises(SystemExit) as raised:
        cli.main(["check", str(tmp_path / record_dir), *ids])
    assert raised.value.code == 2
    assert message in capsys.readouterr().err
