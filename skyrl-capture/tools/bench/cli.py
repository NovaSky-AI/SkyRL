"""Compare capture-on against capture-off through the same proxy.

Development tooling, not product surface: this measures the capture path while
it is being changed, which is a methodology rather than something the product
does for a user, and that is why it lives here rather than on
``skyrl-capture``.

    uv run python -m tools.bench.cli --requests 4000 --concurrency 8 --processes 3
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

console = Console()
app = typer.Typer(
    name="bench",
    add_completion=False,
    help="Compare capture-on against capture-off through the same proxy.",
)


@app.command()
def bench(
    requests: int = typer.Option(4000, "--requests", help="Requests per arm, split across load processes."),
    concurrency: int = typer.Option(8, "--concurrency", help="Concurrent requests per load process."),
    processes: int = typer.Option(None, "--processes", help="Load generator processes (default: cpus/3)."),
    streaming: bool = typer.Option(False, "--streaming", help="Benchmark streaming responses."),
    duration: float = typer.Option(0.0, "--duration", help="Run for this many seconds instead of a fixed count."),
    target_rps: float = typer.Option(0.0, "--target-rps", help="Open-loop arrival rate; 0 means closed-loop."),
    skip_calibration: bool = typer.Option(False, "--skip-calibration", help="Skip the no-proxy ceiling arm."),
    output: Path = typer.Option(None, "--output", help="Write the JSON report here."),
) -> None:
    """Compare capture-on against capture-off through the same proxy.

    Runs three arms: the upstream directly (the harness ceiling), capture off,
    and capture on. See docs/benchmarks.md.
    """
    import asyncio

    from tools.bench.harness import render_report, run_benchmark

    report = asyncio.run(
        run_benchmark(
            requests=requests,
            concurrency=concurrency,
            processes=processes,
            streaming=streaming,
            duration=duration,
            target_rps=target_rps,
            skip_calibration=skip_calibration,
        )
    )
    render_report(report, console)
    if output:
        output.write_text(json.dumps(report, indent=2))
        console.print(f"[green]wrote[/green] {output}")


if __name__ == "__main__":
    app()
