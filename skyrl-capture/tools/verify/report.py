"""End-to-end verification: re-feed the record to the engine.

The prefix audit is a *token-level* check. It proves the graph and the prompt
agree with each other, which is exactly the class of bug it was built for and
exactly the class it cannot see past: a record can be perfectly self-consistent
and still not be what the engine saw. Nothing inside this process can tell the
difference, because the thing being checked is the boundary with something
outside it.

So this check goes outside. It takes a finished trajectory's recorded
`input_ids` and `loss_mask`, hands the **prompt** back to the engine, and
compares what comes back with the **completion** the record says was produced:

    input_ids  = [ ......... prompt ......... | .... completion .... ]
    loss_mask  = [ 0 0 0 0 0 0 0 0 0 0 0 0 0  | 1 1 1 1 1 1 1 1 1 1  ]
                                              ^
                                     the first sampled token

Greedy, so the engine is a function of its input. Then:

* **The completion matches.** The stored prompt is the prompt that produced the
  stored completion. If the record held a prompt the engine never saw, a greedy
  re-run diverges -- usually within a few tokens, and the report says where.
* **The engine accepts the prompt at all.** A prompt with a malformed
  scaffold, a doubled turn marker or a truncated tail is a prompt a served
  model will reject or mangle, and nothing on the CPU side would notice.

Against a real engine and a real model this is the GPU layer. Against the mock
engine, whose completion is a deterministic function of the prompt, it is the
same check at CI cost -- which is the point: the same command, the same claim,
run in both places.

What it cannot check: a sampled trajectory. Real rollouts run with temperature,
so their completions are not reproducible, and a mismatch would mean nothing.
Verification needs a greedy run, and `--tolerate-divergence` exists for the
case where you want the *acceptance* half without the *equality* half.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import orjson

from skyrl_capture.export.view import TrajectoryView
from skyrl_capture.transport.http import TransportError

logger = logging.getLogger(__name__)


@dataclass
class PathVerdict:
    """What re-feeding one path's prompt produced."""

    path_id: str
    prompt_tokens: int
    expected_completion: tuple[int, ...]
    observed_completion: tuple[int, ...] = ()
    ok: bool = False
    #: accepted / diverged / rejected / skipped
    outcome: str = "skipped"
    #: First position where the two completions differ, if they do.
    diverged_at: int | None = None
    detail: str | None = None

    def describe(self) -> str:
        if self.outcome == "accepted":
            return f"{self.path_id}: {len(self.expected_completion)} completion tokens reproduced"
        if self.outcome == "diverged":
            return (
                f"{self.path_id}: the engine produced a different completion from token "
                f"{self.diverged_at} of {len(self.expected_completion)}. The stored prompt "
                "is not the prompt that produced the stored completion"
            )
        if self.outcome == "rejected":
            return f"{self.path_id}: the engine refused the stored prompt -- {self.detail}"
        return f"{self.path_id}: skipped -- {self.detail}"


@dataclass
class Report:
    trajectory_id: str
    verdicts: list[PathVerdict] = field(default_factory=list)

    @property
    def checked(self) -> bool:
        """Was anything actually re-fed? Text mode and fully replayed paths
        have no prompt to hand back, so a report can be all skips."""
        return any(verdict.outcome != "skipped" for verdict in self.verdicts)

    @property
    def ok(self) -> bool:
        return all(verdict.ok for verdict in self.verdicts if verdict.outcome != "skipped")

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for verdict in self.verdicts:
            counts[verdict.outcome] = counts.get(verdict.outcome, 0) + 1
        return {"trajectory_id": self.trajectory_id, "ok": self.ok, "paths": counts}


def split_at_first_sampled(
    input_ids: list[int], loss_mask: list[int]
) -> tuple[list[int], list[int]] | None:
    """The prompt and the completion a greedy re-run should reproduce.

    The split is the first sampled position, not a stored prompt/response pair:
    a multi-turn path has several sampled spans, and only the first one is
    reachable from a prompt the engine has never seen. Everything after it
    depends on the turn before, so re-feeding the whole thing would be
    verifying the harness rather than the record.
    """
    try:
        start = loss_mask.index(1)
    except ValueError:
        return None
    if start == 0:
        # Nothing was given to the model, so there is no prompt to re-feed.
        return None
    end = start
    while end < len(loss_mask) and loss_mask[end]:
        end += 1
    return input_ids[:start], input_ids[start:end]


async def verify_view(
    view: TrajectoryView,
    *,
    engine: Any,
    protocol: Any,
    url: str,
    credential: str | None = None,
    model: str | None = None,
    tolerate_divergence: bool = False,
) -> Report:
    """Re-feed every path's first prompt and compare the completion."""
    from skyrl_capture.export import formats

    report = Report(trajectory_id=view.trajectory.id)
    for row in formats.token_sample_records(view):
        split = split_at_first_sampled(row["input_ids"], row["loss_mask"])
        if split is None:
            report.verdicts.append(
                PathVerdict(
                    path_id=row["path_id"],
                    prompt_tokens=len(row["input_ids"]),
                    expected_completion=(),
                    outcome="skipped",
                    detail="no sampled span reachable from a prompt",
                )
            )
            continue
        prompt, completion = split
        verdict = PathVerdict(
            path_id=row["path_id"],
            prompt_tokens=len(prompt),
            expected_completion=tuple(completion),
        )
        try:
            output = await engine.generate(
                protocol=protocol,
                url=url,
                credential=credential,
                prompt_token_ids=prompt,
                # Greedy, and exactly as many tokens as were stored: the engine
                # has to be a function of its input for this to mean anything.
                sampling_params={
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": len(completion),
                },
                model=model,
                session_id=f"verify-{row['path_id']}",
            )
        except (TransportError, Exception) as error:  # noqa: BLE001 - reported, not raised
            verdict.outcome = "rejected"
            verdict.detail = f"{type(error).__name__}: {error}"
            report.verdicts.append(verdict)
            continue

        observed = tuple(output["completion_ids"])
        verdict.observed_completion = observed
        if observed == tuple(completion):
            verdict.outcome, verdict.ok = "accepted", True
        else:
            verdict.outcome = "diverged"
            verdict.ok = tolerate_divergence
            verdict.diverged_at = next(
                (
                    index
                    for index, pair in enumerate(zip(completion, observed, strict=False))
                    if pair[0] != pair[1]
                ),
                min(len(completion), len(observed)),
            )
        report.verdicts.append(verdict)
    return report


def render_report(reports: list[Report]) -> str:
    """One line per trajectory, then the failures in full."""
    lines: list[str] = []
    failed = [report for report in reports if not report.ok]
    skipped = [report for report in reports if report.ok and not report.checked]
    for report in reports:
        counts = report.summary()["paths"]
        state = "FAILED" if not report.ok else ("-" if not report.checked else "ok")
        lines.append(
            f"{state:6} {report.trajectory_id}  "
            + "  ".join(f"{name}={count}" for name, count in sorted(counts.items()))
        )
    for report in failed:
        for verdict in report.verdicts:
            if not verdict.ok and verdict.outcome != "skipped":
                lines.append(f"       {verdict.describe()}")

    # A trajectory with nothing to re-feed is not a trajectory that passed, and
    # saying "verified" of it would be the overclaim this command exists to
    # prevent elsewhere.
    verified = len(reports) - len(failed) - len(skipped)
    summary = f"\n{verified} of {len(reports)} trajectories verified against the engine."
    if skipped:
        summary += (
            f" {len(skipped)} had nothing to re-feed -- text mode, or no sampled span "
            "reachable from a prompt -- and were checked against nothing."
        )
    if failed:
        summary += f" {len(failed)} FAILED."
    return "\n".join(lines) + summary


def report_json(reports: list[Report]) -> bytes:
    return orjson.dumps(
        [
            {
                **report.summary(),
                "checked": report.checked,
                "verdicts": [
                    {
                        "path_id": verdict.path_id,
                        "outcome": verdict.outcome,
                        "ok": verdict.ok,
                        "prompt_tokens": verdict.prompt_tokens,
                        "completion_tokens": len(verdict.expected_completion),
                        "diverged_at": verdict.diverged_at,
                        "detail": verdict.detail,
                    }
                    for verdict in report.verdicts
                ],
            }
            for report in reports
        ],
        option=orjson.OPT_INDENT_2,
    )
