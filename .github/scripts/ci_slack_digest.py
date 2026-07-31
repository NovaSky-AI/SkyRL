#!/usr/bin/env python3
"""Post a daily digest of SkyRL CI health on `main` to Slack.

Answers two questions per monitored workflow:

1. Is it broken right now, and for how long?
2. Did it go green again since the last digest (a recovery)?

Design notes
------------
* "Currently failing" is **state-based**: it is derived from the latest
  meaningful run, regardless of the digest window. A missed or delayed digest
  therefore never drops a still-broken workflow off the report.
* "Recovered" is **event-based**: the latest run is a success, the run before it
  failed, and that success landed inside the window. `LOOKBACK_HOURS` defaults
  to 25 rather than 24 because GitHub's scheduled triggers drift by tens of
  minutes; the extra hour of overlap risks re-reporting a recovery once, which
  is much cheaper than silently missing one.
* Only `main` is considered. `pull_request` / `pull_request_target` runs carry
  the PR's head branch, so the `branch=main` filter plus
  `exclude_pull_requests=true` already drops them; the event check is a third
  belt-and-braces guard.
* `cancelled`, `skipped` and friends are ignored completely -- they are neither
  failures nor successes, and must not break a failure/recovery chain. This
  matters on `main`, where `cancel-in-progress` concurrency groups cancel a
  meaningful fraction of runs.

Lives under `.github/scripts/` rather than `ci/` on purpose: `ci/**` is in the
`paths:` filter of almost every GPU workflow, so editing this file there would
trigger the whole Anyscale GPU suite.

Posts through `chat.postMessage` with a bot token (`SLACK_BOT_TOKEN` +
`SLACK_CHANNEL`) rather than an incoming webhook. Slack's original webhooks are
legacy custom integrations that they have deprecated; the app-scoped
replacement is supported, but it is opaque to misconfiguration, whereas
`chat.postMessage` names the problem (`not_in_channel`, `invalid_auth`) -- which
matters for a job that runs unattended once a day.

Stdlib only, so it runs anywhere without an install step.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

API_ROOT = "https://api.github.com"

# Conclusions that count as a red run.
FAILURE_CONCLUSIONS = {"failure", "timed_out", "startup_failure"}
# Conclusions that count as a green run.
SUCCESS_CONCLUSIONS = {"success"}
# Everything else (cancelled, skipped, neutral, stale, action_required) is
# ignored: it neither alerts nor resets a streak.

# Events that can put a run on `main`. Anything else is PR-triggered.
MAIN_EVENTS = {"push", "schedule", "workflow_dispatch", "workflow_run", "repository_dispatch"}

# How many runs of history to pull per workflow. One page is plenty to cover a
# 24h window plus enough predecessors to measure a failure streak.
RUNS_PER_WORKFLOW = 100

DEFAULT_WORKFLOWS = """
H100-GPU-CI
Megatron-Model-GPU-CI
SkyRL-CPU
SkyRL-GPU
SkyRL-GPU-E2E-CI
SkyRL-GPU-E2E-CI-Fully-Async
SkyRL-GPU-E2E-CI-Megatron
SkyRL-GPU-E2E-CI-SFT
SkyRL-GPU-E2E-CI-Tinker
SkyRL-GPU-E2E-CI-Tinker-Fully-Async
SkyRL-Train-CPU
SkyRL-Train-GPU
SkyRL-Train-GPU-Megatron
Tinker-SkyRL-Train-Backend-GPU
"""


# --------------------------------------------------------------------------- #
# GitHub API
# --------------------------------------------------------------------------- #


class GitHub:
    def __init__(self, token: str, repo: str) -> None:
        self.token = token
        self.repo = repo

    def get(self, path: str, **params: object) -> dict:
        url = f"{API_ROOT}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "skyrl-ci-digest",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.load(resp)

    def workflow_ids(self) -> dict[str, int]:
        """Map workflow display name -> id.

        The API also lists workflows whose file has been deleted, and synthetic
        ones under `dynamic/` (Copilot review, Dependency Graph). Restricting to
        `.github/workflows/` drops the synthetic entries; deleted-but-listed
        workflows simply never match a configured name.
        """
        data = self.get(f"/repos/{self.repo}/actions/workflows", per_page=100)
        return {
            wf["name"]: wf["id"]
            for wf in data.get("workflows", [])
            if str(wf.get("path", "")).startswith(".github/workflows/")
        }

    def main_runs(self, workflow_id: int) -> list[dict]:
        data = self.get(
            f"/repos/{self.repo}/actions/workflows/{workflow_id}/runs",
            branch="main",
            exclude_pull_requests="true",
            per_page=RUNS_PER_WORKFLOW,
        )
        return data.get("workflow_runs", [])


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


@dataclass
class Run:
    number: int
    url: str
    sha: str
    title: str
    actor: str
    event: str
    conclusion: str
    started: datetime

    @property
    def failed(self) -> bool:
        return self.conclusion in FAILURE_CONCLUSIONS

    @property
    def short_sha(self) -> str:
        return self.sha[:7]


@dataclass
class WorkflowReport:
    name: str
    runs: list[Run] = field(default_factory=list)  # newest first, meaningful only
    in_progress: int = 0
    missing: bool = False  # configured but no such workflow in the repo
    history_truncated: bool = False  # the run history filled a whole API page

    # Filled in by `classify`.
    state: str = "unknown"  # failing | recovered | green | unknown
    streak: int = 0  # consecutive failures at the head (state == failing)
    streak_truncated: bool = False  # streak ran off the end of the fetched history
    broken_since: datetime | None = None
    prior_failures: int = 0  # failures cleared by a recovery
    window_failures: int = 0
    window_runs: int = 0

    @property
    def latest(self) -> Run | None:
        return self.runs[0] if self.runs else None


def classify(report: WorkflowReport, window_start: datetime) -> None:
    runs = report.runs
    report.window_runs = sum(1 for r in runs if r.started >= window_start)
    report.window_failures = sum(1 for r in runs if r.started >= window_start and r.failed)

    latest = report.latest
    if latest is None:
        report.state = "unknown"
        return

    if latest.failed:
        report.state = "failing"
        streak = 0
        for run in runs:
            if not run.failed:
                break
            streak += 1
            report.broken_since = run.started
        report.streak = streak
        # Every run we know about failed *and* there is more history we did not
        # fetch, so the real streak may be longer and `broken_since` is only a
        # lower bound on the age.
        report.streak_truncated = streak == len(runs) and report.history_truncated
        return

    # Latest run is green. It is a *recovery* only if the previous meaningful
    # run failed and this success landed inside the window.
    leading_failures = 0
    for run in runs[1:]:
        if not run.failed:
            break
        leading_failures += 1

    if leading_failures and latest.started >= window_start:
        report.state = "recovered"
        report.prior_failures = leading_failures
    else:
        report.state = "green"


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #


def parse_ts(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def humanize(delta: timedelta) -> str:
    minutes = int(delta.total_seconds() // 60)
    if minutes < 60:
        return f"{max(minutes, 1)}m"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h" if not minutes else f"{hours}h {minutes}m"
    days, hours = divmod(hours, 24)
    return f"{days}d" if not hours else f"{days}d {hours}h"


def plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def link(url: str, text: str) -> str:
    return f"<{url}|{text}>"


def escape(text: str) -> str:
    """Escape Slack's three mrkdwn control characters."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def truncate(text: str, limit: int = 80) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def failing_line(report: WorkflowReport, now: datetime) -> str:
    run = report.latest
    assert run is not None
    at_least = "≥" if report.streak_truncated else ""
    bits = [f"*{escape(report.name)}*"]
    if report.broken_since is not None:
        bits.append(f"red for {at_least}{humanize(now - report.broken_since)}")
    if report.streak > 1:
        bits.append(f"{at_least}{report.streak} runs in a row")
    bits.append(link(run.url, f"run #{run.number}"))
    bits.append(f"`{run.short_sha}`")
    if report.in_progress:
        bits.append(f"_{plural(report.in_progress, 'rerun')} in progress_")
    line = " · ".join(bits)
    # Scheduled runs have no meaningful title -- GitHub just repeats the
    # workflow name -- so only show the commit subject when it adds something.
    if run.title and run.title.strip() != report.name:
        line += f"\n ↳ _{escape(truncate(run.title))}_ — {escape(run.actor)}"
    return line


def recovered_line(report: WorkflowReport) -> str:
    run = report.latest
    assert run is not None
    bits = [
        f"*{escape(report.name)}*",
        f"green again after {plural(report.prior_failures, 'failure')}",
        link(run.url, f"run #{run.number}"),
        f"`{run.short_sha}`",
    ]
    return " · ".join(bits)


def build_blocks(
    reports: list[WorkflowReport],
    repo: str,
    now: datetime,
    window_start: datetime,
    unmonitored: list[str],
) -> tuple[list[dict], str]:
    failing = [r for r in reports if r.state == "failing"]
    recovered = [r for r in reports if r.state == "recovered"]
    green = [r for r in reports if r.state == "green"]
    unknown = [r for r in reports if r.state == "unknown" and not r.missing]
    missing = [r for r in reports if r.missing]

    failing.sort(key=lambda r: (r.broken_since or now))
    for group in (recovered, green, unknown, missing):
        group.sort(key=lambda r: r.name)

    if failing:
        icon, mood = ":rotating_light:", f"{plural(len(failing), 'workflow')} failing"
    elif recovered:
        icon, mood = ":large_green_circle:", "back to green"
    else:
        icon, mood = ":white_check_mark:", "all green"

    headline = f"{icon} SkyRL CI on main — {mood}"
    blocks: list[dict] = [
        {"type": "header", "text": {"type": "plain_text", "text": truncate(headline, 150)}},
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        f"`{repo}` · window "
                        f"{window_start:%b %d %H:%M} → {now:%b %d %H:%M} UTC "
                        f"· {len(reports)} workflows monitored"
                    ),
                }
            ],
        },
    ]

    def add_section(title: str, lines: list[str], joiner: str = "\n") -> None:
        if not lines:
            return
        blocks.append({"type": "divider"})
        body = joiner.join(lines)
        chunks: list[str] = []
        current = ""
        for line in body.split("\n"):
            # 2900 leaves room under Slack's 3000-char section limit.
            if len(current) + len(line) + 1 > 2900:
                chunks.append(current)
                current = line
            else:
                current = f"{current}\n{line}" if current else line
        if current:
            chunks.append(current)
        for index, chunk in enumerate(chunks):
            text = f"{title}\n{chunk}" if index == 0 and title else chunk
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": text}})

    add_section(
        f"*:red_circle: Failing now ({len(failing)})*",
        [f"• {failing_line(r, now)}" for r in failing],
    )
    add_section(
        f"*:large_green_circle: Recovered ({len(recovered)})*",
        [f"• {recovered_line(r)}" for r in recovered],
    )

    if green:
        flaky = [r for r in green if r.window_failures]
        steady = [r for r in green if not r.window_failures]
        lines = []
        if steady:
            lines.append("*:white_check_mark: Green* · " + ", ".join(escape(r.name) for r in steady))
        for r in flaky:
            lines.append(
                f"*:warning: {escape(r.name)}* · green now, but "
                f"{plural(r.window_failures, 'failure')} in this window"
            )
        add_section("", lines)

    if unknown or missing:
        lines = []
        if unknown:
            lines.append(
                "*:white_circle: No completed runs on main yet* · " + ", ".join(escape(r.name) for r in unknown)
            )
        if missing:
            lines.append(
                "*:question: Configured but not found in this repo* · "
                + ", ".join(escape(r.name) for r in missing)
                + " — renamed or deleted?"
            )
        add_section("", lines)

    total_runs = sum(r.window_runs for r in reports)
    total_failures = sum(r.window_failures for r in reports)
    footer = f"{plural(total_runs, 'run')} in window · {total_failures} failed · " + link(
        f"https://github.com/{repo}/actions?query=branch%3Amain", "all main runs"
    )
    if unmonitored:
        footer += " · :eyes: not monitored: " + ", ".join(escape(n) for n in unmonitored)
    blocks.append({"type": "divider"})
    blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": footer}]})

    return blocks, headline


# --------------------------------------------------------------------------- #
# Drift detection
# --------------------------------------------------------------------------- #

NAME_RE = re.compile(r"^name:\s*(.+?)\s*$", re.MULTILINE)


def unmonitored_workflows(monitored: set[str], self_path: str) -> list[str]:
    """Workflow files in the checkout whose name is not in the monitored set.

    Keeps the hardcoded list honest as workflows are added, without letting
    auto-discovery silently pull in unrelated jobs.
    """
    root = Path(".github/workflows")
    if not root.is_dir():
        return []
    found = []
    for path in sorted(root.iterdir()):
        if path.suffix not in {".yaml", ".yml"} or path.name == Path(self_path).name:
            continue
        match = NAME_RE.search(path.read_text(encoding="utf-8"))
        if not match:
            continue
        name = match.group(1).strip().strip("'\"")
        if name not in monitored:
            found.append(name)
    return found


# --------------------------------------------------------------------------- #
# Slack
# --------------------------------------------------------------------------- #


POST_MESSAGE_URL = "https://slack.com/api/chat.postMessage"


def message_payload(channel: str, fallback: str, blocks: list[dict] | None = None) -> dict:
    """Body for `chat.postMessage`.

    `text` is the notification/fallback string shown in the sidebar and in push
    notifications; `blocks` is the rendered message. Link unfurling is off
    because the digest is mostly GitHub links and Slack would otherwise expand
    each one into a preview card several times the height of the digest itself.
    """
    payload: dict = {
        "channel": channel,
        "text": fallback,
        "unfurl_links": False,
        "unfurl_media": False,
    }
    if blocks:
        payload["blocks"] = blocks
    return payload


def post_to_slack(token: str, channel: str, fallback: str, blocks: list[dict] | None = None) -> None:
    """Post via `chat.postMessage`.

    Uses a bot token rather than an incoming webhook: webhooks come in a legacy
    custom-integration flavour that Slack has deprecated, and even the
    app-scoped kind cannot be introspected, so a misconfigured one fails with an
    opaque `invalid_payload`. `chat.postMessage` returns a named error
    (`channel_not_found`, `not_in_channel`, `invalid_auth`) that says what to fix.
    """
    req = urllib.request.Request(
        POST_MESSAGE_URL,
        data=json.dumps(message_payload(channel, fallback, blocks)).encode(),
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "Authorization": f"Bearer {token}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.load(resp)
    if not body.get("ok"):
        error = body.get("error", "unknown_error")
        hint = {
            "not_in_channel": "invite the bot to the channel: /invite @<app name>",
            "channel_not_found": "check SLACK_CHANNEL, and that the bot can see a private channel",
            "invalid_auth": "check SLACK_BOT_TOKEN (needs the chat:write scope)",
            "missing_scope": "add the chat:write scope and reinstall the app",
        }.get(error)
        raise RuntimeError(f"chat.postMessage failed: {error}" + (f" -- {hint}" if hint else ""))


def post_plain(token: str, channel: str, text: str) -> None:
    """Best-effort plain-text post, used to report digest failures."""
    try:
        post_to_slack(token, channel, text)
    except Exception as exc:  # pragma: no cover - notification of a notification
        print(f"could not report failure to Slack: {exc}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def parse_runs(raw_runs: list[dict]) -> tuple[list[Run], int]:
    """Turn raw API runs into meaningful `main` runs, newest first.

    Returns the runs plus a count of still-running runs, which are reported as
    "rerun in progress" but never participate in a streak.
    """
    runs: list[Run] = []
    in_progress = 0
    for raw in raw_runs:
        if raw.get("event") not in MAIN_EVENTS or raw.get("head_branch") != "main":
            continue
        if raw.get("status") != "completed":
            in_progress += 1
            continue
        conclusion = raw.get("conclusion") or ""
        if conclusion not in FAILURE_CONCLUSIONS | SUCCESS_CONCLUSIONS:
            continue
        runs.append(
            Run(
                number=raw["run_number"],
                url=raw["html_url"],
                sha=raw.get("head_sha", ""),
                title=raw.get("display_title") or "",
                actor=(raw.get("triggering_actor") or raw.get("actor") or {}).get("login", "?"),
                event=raw["event"],
                conclusion=conclusion,
                started=parse_ts(raw["created_at"]),
            )
        )
    runs.sort(key=lambda r: (r.started, r.number), reverse=True)
    return runs, in_progress


def collect(gh: GitHub, names: list[str], window_start: datetime) -> list[WorkflowReport]:
    available = gh.workflow_ids()
    reports = []
    for name in names:
        report = WorkflowReport(name=name)
        workflow_id = available.get(name)
        if workflow_id is None:
            report.missing = True
            reports.append(report)
            continue
        raw_runs = gh.main_runs(workflow_id)
        report.history_truncated = len(raw_runs) >= RUNS_PER_WORKFLOW
        report.runs, report.in_progress = parse_runs(raw_runs)
        classify(report, window_start)
        reports.append(report)
    return reports


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN is required", file=sys.stderr)
        return 2

    repo = os.environ.get("GITHUB_REPOSITORY") or "NovaSky-AI/SkyRL"
    slack_token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    channel = os.environ.get("SLACK_CHANNEL", "").strip()
    can_post = bool(slack_token and channel)
    dry_run = env_flag("DRY_RUN", not can_post)
    post_when_green = env_flag("POST_WHEN_ALL_GREEN", True)
    lookback = float(os.environ.get("LOOKBACK_HOURS") or 25)
    self_path = os.environ.get("SELF_WORKFLOW_FILE", "ci_slack_digest.yaml")

    names = [
        line.strip()
        for chunk in (os.environ.get("MONITORED_WORKFLOWS") or DEFAULT_WORKFLOWS).splitlines()
        for line in chunk.split(",")
        if line.strip()
    ]

    now = datetime.now(timezone.utc)
    window_start = now - timedelta(hours=lookback)

    try:
        reports = collect(GitHub(token, repo), names, window_start)
    except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError) as exc:
        message = f":warning: SkyRL CI digest could not read the GitHub API: `{exc}`"
        print(message, file=sys.stderr)
        if can_post and not dry_run:
            post_plain(slack_token, channel, message)
        return 1

    blocks, headline = build_blocks(reports, repo, now, window_start, unmonitored_workflows(set(names), self_path))

    quiet = all(r.state in {"green", "unknown"} or r.missing for r in reports)
    if quiet and not post_when_green:
        print("nothing to report and POST_WHEN_ALL_GREEN is off; skipping post")
        return 0

    for report in reports:
        print(
            f"{report.name}: {report.state} "
            f"(streak={report.streak}, window_failures={report.window_failures}, "
            f"window_runs={report.window_runs})"
        )

    if dry_run or not can_post:
        print("\n--- payload (dry run) ---")
        print(json.dumps(message_payload(channel or "#unset", headline, blocks), indent=2))
        return 0

    post_to_slack(slack_token, channel, headline, blocks)
    print(f"posted to {channel}: {headline}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
