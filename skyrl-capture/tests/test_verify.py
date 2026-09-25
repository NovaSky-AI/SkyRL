"""The second verification layer: re-feed the record to the engine.

The prefix audit proves the graph and the prompt agree with each other. That is
the class of bug it was built for, and the class it cannot see past: a record
can be perfectly self-consistent and still not be what the engine saw. Nothing
inside the capture process can tell, because the thing being checked is the
boundary with something outside it.

So this hands the recorded prompt back to the engine, greedily, and compares
what comes back with the recorded completion. Against a real model that is the
GPU check; against the mock engine, whose completion is a deterministic
function of the prompt, it is the same check at CI cost.
"""

from __future__ import annotations

import pytest

from skyrl_capture.export.view import view_of_active
from skyrl_capture.tito import upstream as tito
from skyrl_capture.tito.engine import TokenEngine
from tools.verify.report import split_at_first_sampled, verify_view


async def a_turn(stack, created, messages, *, max_tokens=6):
    return await stack.client.post(
        f"{created['base_url']}/chat/completions",
        json={"model": "mock-tokens-model", "messages": messages, "max_tokens": max_tokens},
        headers={"authorization": "Bearer client-key"},
    )


async def run_verify(stack, trajectory_id, **kwargs):
    await stack.settle()
    view = view_of_active(stack.aggregate(trajectory_id))
    return await verify_view(
        view,
        engine=TokenEngine(stack.runtime.transport),
        protocol=tito.get("tokens"),
        url=f"{stack.upstream_url}/generate",
        model="mock-tokens-model",
        **kwargs,
    )


# -- the split ---------------------------------------------------------------
def test_the_prompt_is_everything_before_the_first_sampled_token():
    """Not a stored prompt/response pair: a multi-turn path has several sampled
    spans, and only the first is reachable from a prompt the engine has never
    seen."""
    prompt, completion = split_at_first_sampled(
        [10, 11, 12, 20, 21, 30, 31, 40, 41],
        [0, 0, 0, 1, 1, 0, 0, 1, 1],
    )
    assert prompt == [10, 11, 12]
    assert completion == [20, 21], "the second sampled span depends on the turn before it"


def test_a_path_with_nothing_to_re_feed_is_not_verifiable():
    # Nothing sampled at all: a fully replayed path.
    assert split_at_first_sampled([1, 2, 3], [0, 0, 0]) is None
    # Sampled from position zero: no prompt to hand back.
    assert split_at_first_sampled([1, 2, 3], [1, 1, 1]) is None


# -- against the engine ------------------------------------------------------
async def test_a_faithful_record_reproduces_its_completion(tokens_stack):
    created = await tokens_stack.create_trajectory()
    await a_turn(tokens_stack, created, [{"role": "user", "content": "verify me"}])
    await tokens_stack.settle()

    report = await run_verify(tokens_stack, created["id"])
    assert report.ok
    assert [verdict.outcome for verdict in report.verdicts] == ["accepted"]
    verdict = report.verdicts[0]
    assert verdict.prompt_tokens > 0
    assert verdict.observed_completion == verdict.expected_completion


async def test_a_prompt_that_is_not_what_the_engine_saw_is_caught(tokens_stack, monkeypatch):
    """The whole point. The record is internally consistent -- the audit would
    pass it -- and it is still not the prompt that produced the completion."""
    created = await tokens_stack.create_trajectory()
    await a_turn(tokens_stack, created, [{"role": "user", "content": "verify me"}])
    await tokens_stack.settle()

    view = view_of_active(tokens_stack.aggregate(created["id"]))
    # Rewrite one prompt token in the record, exactly as a silent corruption
    # between the renderer and the engine would leave it.
    from skyrl_capture.export import formats

    original = formats.token_sample_records

    def corrupted(target_view, **kwargs):
        rows = original(target_view, **kwargs)
        for row in rows:
            row["input_ids"][2] = 9_999
        return rows

    monkeypatch.setattr(formats, "token_sample_records", corrupted)

    report = await verify_view(
        view,
        engine=TokenEngine(tokens_stack.runtime.transport),
        protocol=tito.get("tokens"),
        url=f"{tokens_stack.upstream_url}/generate",
        model="mock-tokens-model",
    )
    assert not report.ok
    verdict = report.verdicts[0]
    assert verdict.outcome == "diverged"
    assert verdict.diverged_at is not None
    assert "not the prompt that produced" in verdict.describe()


async def test_a_sampled_run_can_be_checked_for_acceptance_alone(tokens_stack, monkeypatch):
    """Real rollouts run with temperature, so their completions do not
    reproduce and a mismatch would mean nothing. The prompt is still worth
    checking: a malformed scaffold is a prompt a served model rejects."""
    created = await tokens_stack.create_trajectory()
    await a_turn(tokens_stack, created, [{"role": "user", "content": "sampled"}])
    await tokens_stack.settle()

    from skyrl_capture.export import formats

    original = formats.token_sample_records

    def corrupted(target_view, **kwargs):
        rows = original(target_view, **kwargs)
        for row in rows:
            row["input_ids"][2] = 9_999
        return rows

    monkeypatch.setattr(formats, "token_sample_records", corrupted)
    view = view_of_active(tokens_stack.aggregate(created["id"]))
    report = await verify_view(
        view,
        engine=TokenEngine(tokens_stack.runtime.transport),
        protocol=tito.get("tokens"),
        url=f"{tokens_stack.upstream_url}/generate",
        model="mock-tokens-model",
        tolerate_divergence=True,
    )
    assert report.ok, "the prompt was accepted, which is the half that still holds"
    assert report.verdicts[0].outcome == "diverged"


async def test_an_engine_that_refuses_the_prompt_is_a_failure_not_a_crash(tokens_stack):
    created = await tokens_stack.create_trajectory()
    await a_turn(tokens_stack, created, [{"role": "user", "content": "unreachable"}])
    await tokens_stack.settle()

    view = view_of_active(tokens_stack.aggregate(created["id"]))
    report = await verify_view(
        view,
        engine=TokenEngine(tokens_stack.runtime.transport),
        protocol=tito.get("tokens"),
        # Nothing is listening here.
        url="http://127.0.0.1:1/generate",
        model="mock-tokens-model",
    )
    assert not report.ok
    assert report.verdicts[0].outcome == "rejected"
    assert "refused the stored prompt" in report.verdicts[0].describe()


async def test_a_text_trajectory_has_nothing_to_verify(stack):
    """Text mode records no token ids, so there is no prompt to hand back."""
    created = await stack.create_trajectory()
    await stack.chat(created, [{"role": "user", "content": "text"}])
    await stack.settle()

    view = view_of_active(stack.aggregate(created["id"]))
    report = await verify_view(
        view,
        engine=TokenEngine(stack.runtime.transport),
        protocol=tito.get("tokens"),
        url=f"{stack.upstream_url}/generate",
    )
    # Reported as skipped rather than passed: nothing was checked.
    assert all(verdict.outcome == "skipped" for verdict in report.verdicts) or not report.verdicts


# -- the audit's classification ---------------------------------------------
@pytest.mark.parametrize(
    ("author", "sampled_start", "at", "expected"),
    [
        ("client", None, 1, "given"),
        ("model", 2, 0, "scaffold"),
        ("model", 2, 3, "sampled"),
    ],
)
def test_the_comparator_names_which_part_of_the_machinery(author, sampled_start, at, expected):
    """"It diverged" is detectable; the class is what makes it diagnosable."""
    from skyrl_capture.tito.compare import compare_prefix

    class Node:
        node_id = "nd_1"
        role = "assistant"

        def __init__(self):
            self.token_ids = (1, 2, 3, 4)
            self.author = author
            self.sampled_start = sampled_start

    sent = [1, 2, 3, 4]
    sent[at] = 99
    report = compare_prefix([Node()], sent, 4)
    assert report.kind == expected
    assert report.offset == at
    assert report.node_id == "nd_1"


def test_the_comparator_decodes_when_it_can():
    """The offset is what you seek to; the text is what tells you which side is
    wrong."""
    from skyrl_capture.tito.compare import compare_prefix
    from skyrl_capture.tito.renderer import ByteTokenizer

    tokenizer = ByteTokenizer()

    class Node:
        node_id = "nd_1"
        role = "user"
        author = "client"
        sampled_start = None
        token_ids = tuple(tokenizer.encode("hello"))

    sent = list(tokenizer.encode("hellp"))
    report = compare_prefix([Node()], sent, len(sent), decode=tokenizer.decode)
    assert report.kind == "given"
    assert report.expected_text == "hello"
    assert report.observed_text == "hellp"
    assert "hello" in report.describe() and "hellp" in report.describe()


# -- the report must not overclaim ------------------------------------------
def test_a_trajectory_with_nothing_to_re_feed_is_not_reported_as_verified():
    """"3 of 3 verified" for three text-mode trajectories would be the exact
    overclaim this command exists to prevent elsewhere."""
    from tools.verify.report import PathVerdict, Report, render_report

    nothing = Report(
        trajectory_id="tr_text",
        verdicts=[PathVerdict("p0", 0, (), outcome="skipped", detail="text mode")],
    )
    real = Report(
        trajectory_id="tr_tokens",
        verdicts=[PathVerdict("p0", 10, (1, 2), observed_completion=(1, 2),
                              outcome="accepted", ok=True)],
    )
    assert nothing.ok and not nothing.checked
    assert real.ok and real.checked

    text = render_report([nothing, real])
    assert "1 of 2 trajectories verified" in text
    assert "1 had nothing to re-feed" in text
    # And the line for it is not marked ok either.
    assert "-      tr_text" in text
    assert "ok     tr_tokens" in text


def test_a_failure_is_counted_in_the_summary():
    from tools.verify.report import PathVerdict, Report, render_report

    bad = Report(
        trajectory_id="tr_bad",
        verdicts=[PathVerdict("p0", 10, (1, 2), observed_completion=(1, 9),
                              outcome="diverged", ok=False, diverged_at=1)],
    )
    text = render_report([bad])
    assert "0 of 1 trajectories verified" in text
    assert "1 FAILED." in text
    assert "not the prompt that produced" in text
