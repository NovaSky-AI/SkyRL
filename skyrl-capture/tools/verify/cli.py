"""Re-feed a record's prompts to the engine and compare the completions.

Development tooling, not product surface: this is a capture-validation
workflow, run against a finished record by whoever is changing the capture
path, and it is the reason it lives here rather than on ``skyrl-capture``.

    uv run python -m tools.verify.cli --record ./traces --run-id run-a \
      --engine-url http://127.0.0.1:8000/generate --model policy
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import typer
from rich.console import Console

from skyrl_capture.cli.main import RecordOption, fail, open_record

console = Console()
app = typer.Typer(
    name="verify",
    add_completion=False,
    help="Re-feed a record's prompts to the engine and compare the completions.",
)


@app.command()
def verify(
    record: str = RecordOption,
    trajectory: str = typer.Option(None, "--trajectory", help="Verify one trajectory."),
    run: str = typer.Option(None, "--run-id", help="Verify a whole run."),
    engine_url: str = typer.Option(
        ..., "--engine-url", envvar="UPSTREAM_URL", help="Token-in/token-out endpoint."
    ),
    engine_type: str = typer.Option(
        "tokens", "--engine-type", envvar="UPSTREAM_TYPE", help="Registered upstream kind."
    ),
    model: str = typer.Option(None, "--model", envvar="UPSTREAM_MODEL"),
    limit: int = typer.Option(20, "--limit", help="Trajectories to verify."),
    tolerate_divergence: bool = typer.Option(
        False,
        "--tolerate-divergence",
        help=(
            "Accept a different completion. Use when the run was sampled rather "
            "than greedy: the prompt is still checked for acceptance."
        ),
    ),
    output: Path = typer.Option(None, "--output", help="Write the JSON report here."),
) -> None:
    """Re-feed a record's prompts to the engine and compare the completions.

    The second verification layer. The prefix audit proves the graph and the
    prompt agree with each other; this proves the prompt is the one the engine
    saw, which nothing inside the capture process can know.

    Greedy, so the engine is a function of its input. Against a real engine and
    a real model this is the GPU check; against the mock engine, whose
    completion is a deterministic function of the prompt, it is the same check
    at CI cost.

        uv run python -m tools.verify.cli --record ./traces --run-id run-a \
          --engine-url http://127.0.0.1:8000/generate --model policy
    """

    from skyrl_capture.config import ProxyConfig
    from skyrl_capture.tito import upstream as tito
    from skyrl_capture.tito.engine import TokenEngine
    from skyrl_capture.transport.http import UpstreamTransport
    from tools.verify.report import render_report, report_json, verify_view

    if not record:
        fail("--record is required: verification reads a finished record")
    if bool(trajectory) == bool(run):
        fail("pass exactly one of --trajectory or --run-id")
    reader = open_record(record)

    async def collect() -> list:
        if trajectory:
            if not await reader.trajectory_exists(trajectory):
                fail(f"no trajectory {trajectory!r} in {record}")
            selected = [trajectory]
        else:
            selected = (await reader.finished_trajectory_ids(project=None, run_id=run))[:limit]
        return [await reader.view(identifier) for identifier in selected]

    views = asyncio.run(collect())
    if not views:
        fail("nothing to verify in that scope")
    try:
        protocol = tito.get(engine_type)
    except tito.UnknownTitoProtocol as failure:
        fail(str(failure))

    async def run_all() -> list:
        transport = UpstreamTransport(ProxyConfig())
        engine = TokenEngine(transport)
        try:
            return [
                await verify_view(
                    view,
                    engine=engine,
                    protocol=protocol,
                    url=engine_url,
                    credential=os.environ.get("UPSTREAM_API_KEY"),
                    model=model,
                    tolerate_divergence=tolerate_divergence,
                )
                for view in views
            ]
        finally:
            await transport.aclose()

    reports = asyncio.run(run_all())
    console.print(render_report(reports))
    if output:
        output.write_bytes(report_json(reports))
        console.print(f"[green]wrote[/green] {output}")
    if not all(report.ok for report in reports):
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
