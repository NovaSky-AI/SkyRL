"""Tests for the CI digest classifier.

Run by the CI-Slack-Digest workflow itself before it posts, so a broken
classifier turns that job red instead of quietly mis-reporting CI health:

    uv run --isolated --no-project --python 3.12 --with pytest \
      pytest .github/scripts/test_ci_slack_digest.py

Lives next to the script rather than under `tests/` so that editing it does not
trip the `tests/**` path filters that gate the GPU workflows.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("ci_slack_digest", Path(__file__).parent / "ci_slack_digest.py")
assert _spec and _spec.loader
digest = importlib.util.module_from_spec(_spec)
# `@dataclass` resolves annotations through sys.modules, so register before exec.
sys.modules["ci_slack_digest"] = digest
_spec.loader.exec_module(digest)


NOW = datetime(2026, 7, 31, 14, 0, tzinfo=timezone.utc)
WINDOW_START = NOW - timedelta(hours=25)


def raw_run(conclusion: str | None, hours_ago: float, *, number: int = 1, **overrides) -> dict:
    """A workflow run as the GitHub API returns it."""
    started = NOW - timedelta(hours=hours_ago)
    run = {
        "run_number": number,
        "html_url": f"https://github.com/o/r/actions/runs/{number}",
        "head_sha": f"{number:040x}",
        "display_title": f"commit {number}",
        "actor": {"login": "someone"},
        "event": "push",
        "head_branch": "main",
        "status": "completed",
        "conclusion": conclusion,
        "created_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    run.update(overrides)
    return run


def report_for(raw_runs: list[dict], *, history_truncated: bool = False):
    report = digest.WorkflowReport(name="Some-Workflow")
    report.history_truncated = history_truncated
    report.runs, report.in_progress = digest.parse_runs(raw_runs)
    digest.classify(report, WINDOW_START)
    return report


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("conclusion", ["cancelled", "skipped", "neutral", "stale"])
def test_non_verdict_conclusions_are_dropped(conclusion):
    """`cancelled` and friends are neither failures nor successes.

    `main` cancels a meaningful share of runs via `cancel-in-progress`, so
    counting them either way would produce constant false alerts.
    """
    report = report_for([raw_run(conclusion, hours_ago=1, number=2)])
    assert report.runs == []
    assert report.state == "unknown"


def test_cancelled_run_does_not_break_a_failure_streak():
    report = report_for(
        [
            raw_run("failure", hours_ago=1, number=4),
            raw_run("cancelled", hours_ago=2, number=3),
            raw_run("failure", hours_ago=3, number=2),
            raw_run("success", hours_ago=4, number=1),
        ]
    )
    assert report.state == "failing"
    assert report.streak == 2


def test_pull_request_runs_are_excluded():
    """PR runs carry the PR head branch, and must never drive a main alert."""
    report = report_for(
        [
            raw_run("failure", hours_ago=1, number=2, event="pull_request_target", head_branch="feat"),
            raw_run("failure", hours_ago=2, number=1, event="pull_request", head_branch="feat"),
        ]
    )
    assert report.runs == []


def test_non_main_branch_is_excluded():
    report = report_for([raw_run("failure", hours_ago=1, head_branch="rc/1.2")])
    assert report.runs == []


def test_in_progress_runs_are_counted_but_not_classified():
    report = report_for(
        [
            raw_run(None, hours_ago=0.2, number=3, status="in_progress"),
            raw_run("failure", hours_ago=2, number=2),
        ]
    )
    assert report.in_progress == 1
    assert report.state == "failing"
    assert report.streak == 1


def test_runs_are_ordered_newest_first_regardless_of_api_order():
    report = report_for(
        [
            raw_run("failure", hours_ago=10, number=1),
            raw_run("success", hours_ago=1, number=2),
        ]
    )
    assert report.state == "recovered"


# --------------------------------------------------------------------------- #
# Failure state
# --------------------------------------------------------------------------- #


def test_failing_streak_stops_at_the_last_success():
    report = report_for(
        [
            raw_run("failure", hours_ago=1, number=4),
            raw_run("failure", hours_ago=5, number=3),
            raw_run("success", hours_ago=9, number=2),
            raw_run("failure", hours_ago=13, number=1),
        ]
    )
    assert report.state == "failing"
    assert report.streak == 2
    assert report.broken_since == NOW - timedelta(hours=5)


def test_timed_out_and_startup_failure_count_as_failures():
    report = report_for(
        [
            raw_run("timed_out", hours_ago=1, number=2),
            raw_run("startup_failure", hours_ago=2, number=1),
        ]
    )
    assert report.state == "failing"
    assert report.streak == 2


def test_stale_failure_outside_the_window_still_reports_as_failing():
    """State-based, so a missed digest never drops a still-broken workflow."""
    report = report_for([raw_run("failure", hours_ago=200, number=1)])
    assert report.state == "failing"
    assert report.window_runs == 0
    assert report.window_failures == 0


def test_streak_is_marked_truncated_only_when_history_ran_out():
    runs = [raw_run("failure", hours_ago=i + 1, number=100 - i) for i in range(100)]
    assert report_for(runs, history_truncated=True).streak_truncated is True
    assert report_for(runs, history_truncated=False).streak_truncated is False


# --------------------------------------------------------------------------- #
# Recovery
# --------------------------------------------------------------------------- #


def test_first_success_after_failure_is_a_recovery():
    report = report_for(
        [
            raw_run("success", hours_ago=1, number=3),
            raw_run("failure", hours_ago=5, number=2),
            raw_run("failure", hours_ago=9, number=1),
        ]
    )
    assert report.state == "recovered"
    assert report.prior_failures == 2


def test_recovery_outside_the_window_is_not_re_reported():
    """Otherwise every green workflow would be announced as recovered forever."""
    report = report_for(
        [
            raw_run("success", hours_ago=40, number=2),
            raw_run("failure", hours_ago=44, number=1),
        ]
    )
    assert report.state == "green"


def test_second_consecutive_success_is_not_a_recovery():
    report = report_for(
        [
            raw_run("success", hours_ago=1, number=3),
            raw_run("success", hours_ago=5, number=2),
            raw_run("failure", hours_ago=9, number=1),
        ]
    )
    assert report.state == "green"


def test_green_after_a_failure_and_recovery_in_the_same_window_counts_failures():
    report = report_for(
        [
            raw_run("success", hours_ago=1, number=4),
            raw_run("failure", hours_ago=3, number=3),
            raw_run("success", hours_ago=5, number=2),
            raw_run("failure", hours_ago=7, number=1),
        ]
    )
    assert report.state == "recovered"
    assert report.prior_failures == 1
    assert report.window_failures == 2


def test_no_runs_at_all_is_unknown():
    assert report_for([]).state == "unknown"


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def build(reports):
    return digest.build_blocks(reports, "o/r", NOW, WINDOW_START, [])


def test_headline_reflects_the_worst_state():
    failing = report_for([raw_run("failure", hours_ago=1)])
    recovered = report_for([raw_run("success", hours_ago=1, number=2), raw_run("failure", hours_ago=5, number=1)])
    green = report_for([raw_run("success", hours_ago=40)])

    assert "1 workflow failing" in build([failing, green])[1]
    assert "back to green" in build([recovered, green])[1]
    assert "all green" in build([green])[1]
    # A failure outranks a recovery in the headline.
    assert "failing" in build([failing, recovered])[1]


def test_sections_stay_within_slack_limits():
    reports = [report_for([raw_run("failure", hours_ago=1, number=i)]) for i in range(60)]
    for report in reports:
        report.name = "Workflow-" + "x" * 60
    blocks, _ = build(reports)
    assert len(blocks) <= 50
    for block in blocks:
        if block["type"] == "section":
            assert len(block["text"]["text"]) <= 3000


def test_mrkdwn_control_characters_are_escaped():
    report = report_for([raw_run("failure", hours_ago=1, display_title="fix a<b & c>d")])
    blocks, _ = build([report])
    body = "\n".join(b["text"]["text"] for b in blocks if b["type"] == "section")
    assert "a&lt;b &amp; c&gt;d" in body


def test_payload_disables_unfurling():
    """The digest is almost entirely GitHub links; previews would bury it."""
    payload = digest.message_payload("#skyrl-ci-alerts", "headline", [{"type": "divider"}])
    assert payload["unfurl_links"] is False
    assert payload["unfurl_media"] is False
    assert payload["channel"] == "#skyrl-ci-alerts"
    assert payload["text"] == "headline"
    assert payload["blocks"] == [{"type": "divider"}]


def test_payload_omits_blocks_when_plain_text():
    assert "blocks" not in digest.message_payload("#c", "just text")


def test_post_failure_names_the_slack_error(monkeypatch):
    """An unattended daily job needs the reason in the log, not just a 200."""

    class FakeResponse:
        def read(self):
            return b'{"ok": false, "error": "not_in_channel"}'

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(digest.urllib.request, "urlopen", lambda *a, **k: FakeResponse())
    with pytest.raises(RuntimeError) as caught:
        digest.post_to_slack("xoxb-token", "#c", "headline", [])
    assert "not_in_channel" in str(caught.value)
    assert "/invite" in str(caught.value)


def test_scheduled_run_title_is_not_echoed():
    """GitHub sets display_title to the workflow name for scheduled runs."""
    report = report_for([raw_run("failure", hours_ago=1, display_title="Some-Workflow")])
    line = digest.failing_line(report, NOW)
    assert "\n" not in line
