"""The ``skyrl-capture`` command line.

Two commands carry the design. ``skyrl-capture serve`` names the inference
server this process captures, once, on its own command line -- there is no
target to register afterwards and no key to present. ``skyrl-capture run``
creates a trajectory, injects the provider environment variables the upstream's
protocol uses, runs the unchanged child command with normal
stdin/stdout/stderr, finishes the trajectory, and exits with the child's exit
code. Termination signals are trapped so an interrupted trial finishes as
aborted rather than waiting for the expiry sweep.
"""

from __future__ import annotations

import gzip
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.json import JSON
from rich.table import Table

from skyrl_capture.sdk import CaptureClient, CaptureError, create_trajectory
from skyrl_capture.version import __version__

console = Console()
error_console = Console(stderr=True)

app = typer.Typer(
    name="skyrl-capture",
    add_completion=False,
    no_args_is_help=True,
    help="Trajectory-scoped inference capture.",
)

EndpointOption = typer.Option(None, "--endpoint", envvar="CAPTURE_ENDPOINT", help="Control-plane URL.")
RecordOption = typer.Option(
    None,
    "--record",
    envvar="CAPTURE_RECORD_DIR",
    help="Read a record directory instead of a running service. No database, no server.",
)


def open_record(path: str):
    """A reader over a record directory, or a clear failure.

    Used by every `--record` read. The index is built here, in full, rather
    than progressively: a command that is about to print a page wants the whole
    answer, and there is no user waiting on a first paint.
    """
    import asyncio

    from skyrl_capture.persistence import RecordNotFound
    from skyrl_capture.reader.records import RecordReader

    try:
        reader = RecordReader(path)
    except RecordNotFound as failure:
        fail(str(failure))
    asyncio.run(reader.refresh())
    return reader


def fail(message: str) -> None:
    error_console.print(f"[red]error[/red] {message}")
    raise typer.Exit(code=1)


def parse_annotations(items: list[str] | None) -> dict[str, Any]:
    """Parse ``--annotate key=value`` pairs, decoding JSON values when possible."""
    values: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            fail(f"annotation {item!r} must be key=value; a bare tag is --tag")
        key, _, raw = item.partition("=")
        try:
            values[key] = json.loads(raw)
        except json.JSONDecodeError:
            values[key] = raw
    return values


def show_version(value: bool) -> None:
    """``--version``, rather than a verb of its own: it answers a question
    about the tool instead of doing anything to a capture."""
    if value:
        console.print(f"skyrl-capture {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        callback=show_version,
        is_eager=True,
        help="Print the client version and exit.",
    ),
) -> None:
    """Trajectory-scoped inference capture."""


def parse_tags(items: list[str] | None) -> list[str]:
    """Validate ``--tag NAME`` values. Tags are bare strings, never pairs."""
    tags: list[str] = []
    for item in items or []:
        if "=" in item:
            fail(f"tag {item!r} looks like key=value; use --annotate for a value")
        if item not in tags:
            tags.append(item)
    return tags


# -- serve ------------------------------------------------------------------
@app.command()
def serve(
    host: str = typer.Option(None, help="Bind address (default PROXY_HOST)."),
    port: int = typer.Option(None, help="Bind port (default PROXY_PORT)."),
    upstream_module: list[str] = typer.Option(
        None,
        "--upstream-module",
        "-u",
        help=(
            "Import this module before serving, to register an upstream kind it "
            "contributes. Repeatable. This process only knows the types it has "
            "imported, so an upstream kind defined in your package needs this."
        ),
    ),
    mode: str = typer.Option(
        None,
        "--mode",
        help=(
            "What this process captures, for its whole life: text | tokens "
            "(default CAPTURE_MODE, else text)."
        ),
    ),
    upstream_type: str = typer.Option(
        None,
        "--upstream-type",
        help=(
            "A registered upstream kind: openai | anthropic | tokens | vllm, or one "
            "--upstream-module contributes (default UPSTREAM_TYPE)."
        ),
    ),
    upstream_url: str = typer.Option(None, "--upstream-url", help="Upstream base URL."),
    model: str = typer.Option(None, "--model", help="Model name to send upstream."),
    tokenizer: str = typer.Option(None, "--tokenizer", help="Required for a tokens upstream."),
    max_model_len: int = typer.Option(None, "--max-model-len"),
    record_dir: Path = typer.Option(
        None,
        "--record-dir",
        envvar="CAPTURE_RECORD_DIR",
        help=(
            "Required. Where the record goes: one journal per trajectory in "
            "flight and one compiled record per finished one, readable with no "
            "database and no server."
        ),
    ),
    viewer: bool = typer.Option(
        True,
        "--viewer/--disable-viewer",
        help=(
            "Serve the read API and bulk exports from this replica. Exactly one "
            "process per record directory should: there is one indexer per "
            "record root."
        ),
    ),
) -> None:
    """Run the proxy, the lifecycle API and the writer against one upstream.

    The upstream is fixed for the life of the process. There is nothing to
    register afterwards: changing it means relaunching. The credential is read
    from ``UPSTREAM_API_KEY`` rather than taken as a flag, so it never reaches
    a shell history or a process listing.
    """
    from skyrl_capture.config import StartupError, load_config
    from skyrl_capture.service import CaptureService
    from skyrl_capture.upstream.plugins import load_modules

    # Before the configuration is built, so a contributed upstream kind is
    # registered by the time `validate()` below resolves its name.
    if upstream_module:
        load_modules(tuple(upstream_module), source="--upstream-module")
    config = load_config()
    if mode is not None and mode != config.upstream.mode:
        # The mode decides which shape the upstream is, so it is applied first
        # and the flags below land on the right one.
        from skyrl_capture.config import upstream_from_env

        try:
            config = config.with_overrides(upstream=upstream_from_env(mode))
        except StartupError as error:
            fail(str(error))
    chosen = {
        key: value
        for key, value in (
            ("type", upstream_type),
            ("url", upstream_url),
            ("model", model),
            ("tokenizer", tokenizer),
            ("max_model_len", max_model_len),
        )
        # `--tokenizer` and `--max-model-len` exist only on a token upstream:
        # passing one in text mode is ignored rather than a crash.
        if value is not None and hasattr(config.upstream, key)
    }
    if chosen:
        from dataclasses import replace

        config = config.with_overrides(upstream=replace(config.upstream, **chosen))
    if record_dir is not None:
        config = config.with_overrides(record_dir=record_dir)
    config = config.with_overrides(viewer=viewer)
    # After --record-dir, because that is what decides where the payloads go.
    try:
        config.require_record_dir()
    except StartupError as error:
        fail(str(error))
    try:
        config.upstream.validate()
    except ValueError as error:
        fail(str(error))
    if upstream_module:
        # Kept on the config as well, so an embedder that is handed one -- and
        # `build_runtime`, which loads them again -- sees the same list.
        config = config.with_overrides(upstream_modules=tuple(upstream_module))

    # host/port go to `CaptureService` rather than being merged here: it also
    # moves `public_url`, which is what a trajectory's base URL is built from.
    # Overriding the bind address without it hands every caller a URL pointing
    # at the old port.

    try:
        service = CaptureService(config=config, host=host, port=port)
    except StartupError as error:
        # A configuration mistake, not a crash. `fail` prints it and exits 1.
        fail(str(error))
    # Read back from the service: it is what resolved host, port and the
    # advertised `public_url`, so printing its view is printing what callers get.
    config = service.config

    console.print(
        f"[green]skyrl-capture[/green] {__version__} on "
        f"http://{config.proxy.host}:{config.proxy.port}  api=/v1  health=/healthz"
    )
    upstream = config.upstream
    described = f"{upstream.type} -> {upstream.url}"
    if upstream.model:
        described += f"  model={upstream.model}"
    if upstream.tokenizer:
        described += f"  tokenizer={upstream.tokenizer}"
    console.print(f"  upstream   {described} ({upstream.mode} mode)")
    console.print(f"  record     {config.record_dir}")
    console.print(f"  exports    {config.export_dir}")
    console.print(
        "  viewer     serving reads and bulk exports"
        if config.viewer
        else "  viewer     disabled (lifecycle and capture only)"
    )
    from skyrl_capture.service import CaptureServiceError

    try:
        # The same call an embedded caller makes; only `blocking` differs.
        service.start(blocking=True)
    except CaptureServiceError as error:
        # A bound port or a bad upstream is an operator's problem, not a crash.
        fail(str(error))


@app.command("view")
def view(
    api: str = typer.Option(
        "http://127.0.0.1:8080", "--api", help="The capture API to read: a serve or a view."
    ),
    record: Path = typer.Option(
        None,
        "--record",
        help="Read this record directory instead, serving /v1 for the viewer in-process.",
    ),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8750, "--port"),
) -> None:
    """Open the viewer: projects, runs, and the trajectories inside them.

    The viewer is a Node program under `viewer/`, and it is only ever an HTTP
    client of `/v1`. That is the whole reason it reads a run in progress and a
    record directory alike: `--api` points it at a live capture process, and
    `--record` starts the read-only API for a directory in this process and
    points it at that. One UI, one client, two sources it cannot tell apart.
    """
    import shutil
    import threading

    node = shutil.which("node")
    if node is None:
        fail(
            "the viewer needs Node (18 or newer) on PATH, and there is none.\n"
            "  Install it from https://nodejs.org, or read the API directly: "
            "every number the viewer shows comes from /v1."
        )

    root = _viewer_root()
    if root is None:
        fail("the viewer's files are missing from this install (expected a `viewer` directory).")

    if record is not None:
        import uvicorn

        from skyrl_capture.control_plane.viewer import record_viewer_app
        from skyrl_capture.persistence import RecordNotFound

        api_port = _free_port()
        api = f"http://127.0.0.1:{api_port}"
        try:
            # The same viewer app a capture replica mounts, over the same
            # reader. One set of handlers, whichever the source is.
            served = record_viewer_app(str(record), public_url=api)
        except RecordNotFound as failure:
            fail(str(failure))
        server = uvicorn.Server(
            uvicorn.Config(served, host="127.0.0.1", port=api_port, log_level="warning", access_log=False)
        )
        # A daemon thread, so ctrl-c on the viewer takes the API with it rather
        # than leaving a reader holding the record.
        threading.Thread(target=server.run, daemon=True).start()

    console.print(
        f"[green]skyrl-capture[/green] {__version__} viewer\n"
        f"  viewer     http://{host}:{port}\n"
        f"  reading    {api}" + (f"  (record {record})" if record is not None else "")
    )
    command = [node, str(root / "server.mjs"), "--api", api, "--host", host, "--port", str(port)]
    try:
        raise SystemExit(subprocess.call(command))
    except KeyboardInterrupt:
        raise SystemExit(130) from None


def _viewer_root() -> Path | None:
    """Where the viewer's files landed.

    Inside the package in a wheel, and beside it in a checkout; nothing else is
    worth guessing at.
    """
    import skyrl_capture

    package = Path(skyrl_capture.__file__).parent
    for candidate in (package / "viewer", package.parent.parent / "viewer"):
        if (candidate / "server.mjs").is_file():
            return candidate
    return None


def _free_port() -> int:
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def run(
    ctx: typer.Context,
    project: str = typer.Option(..., "--project", help="Grouping key for querying and export."),
    run_id: str = typer.Option(
        None, "--run-id", help="The run this attempt belongs to. Created on first use."
    ),
    task_id: str = typer.Option(None, "--task-id", help="Which task this attempt attempted."),
    step: int = typer.Option(None, "--step", help="Which training step produced it."),
    tag: list[str] = typer.Option(None, "--tag", help="Bare string tag, repeatable."),
    annotate_pairs: list[str] = typer.Option(None, "--annotate", help="key=value, repeatable."),
    bodies: str = typer.Option("full", "--bodies", help="full | sampled"),
    endpoint: str = EndpointOption,
) -> None:
    """Run one unchanged command as one trial.

    Everything after ``--`` is the child command.
    """
    command = list(ctx.args)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        fail("no command given; put the command after --")

    labels = parse_tags(tag)
    annotations = parse_annotations(annotate_pairs)
    capture = CaptureClient(endpoint)
    try:
        trajectory = create_trajectory(
            project=project,
            run_id=run_id,
            task_id=task_id,
            step=step,
            labels=labels,
            annotations=annotations,
            bodies=bodies,
            source_metadata={
                "launcher": "skyrl-capture run",
                "command": command,
                "cwd": str(Path.cwd()),
            },
            client=capture,
        )
    except CaptureError as failure:
        fail(str(failure))

    environment = trajectory.env()
    error_console.print(
        f"[green]skyrl-capture[/green] trajectory {trajectory.id} "
        f"({trajectory.protocol}/{trajectory.mode}) -> {trajectory.base_url}"
    )

    child_environment = dict(os.environ)
    child_environment.update(environment)
    result_kind = "success"
    code = 0
    process: subprocess.Popen[bytes] | None = None

    def forward(signal_number: int, _frame: Any) -> None:
        """Pass termination on to the child and remember that we were stopped."""
        nonlocal result_kind
        result_kind = "aborted"
        if process is not None and process.poll() is None:
            try:
                process.send_signal(signal_number)
            except ProcessLookupError:
                pass

    previous_handlers = {}
    for signal_name in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            previous_handlers[signal_name] = signal.signal(signal_name, forward)
        except (ValueError, OSError):
            continue

    try:
        # stdin/stdout/stderr are inherited so the wrapped command behaves
        # exactly as it would without the wrapper.
        process = subprocess.Popen(command, env=child_environment)
        code = process.wait()
        if code < 0:
            # Killed by a signal. `Popen.wait` reports -N; a shell reports
            # 128+N, and 128+N is what a caller's `$?`, `set -e` and CI runner
            # are written against. SIGINT becomes 130, not -2.
            code = 128 - code
        if code != 0 and result_kind == "success":
            result_kind = "failed"
    except FileNotFoundError:
        fail(f"command not found: {command[0]}")
    except KeyboardInterrupt:
        result_kind = "aborted"
        code = 130
    finally:
        for signal_name, handler in previous_handlers.items():
            try:
                signal.signal(signal_name, handler)
            except (ValueError, OSError):
                continue
        try:
            result = trajectory.finish(
                labels=labels,
                annotations=annotations,
                command_result=result_kind,
            )
            error_console.print(
                f"[green]skyrl-capture[/green] finished {trajectory.id} status={result.get('status')} result={result_kind}"
            )
        except CaptureError as failure:
            # The trial already ran; do not mask the child's exit code.
            # The trial ran and its journal is on disk. Nothing expires it:
            # finishing it again, by id, is what turns it into a record.
            error_console.print(
                f"[yellow]warning[/yellow] finish failed ({failure}); "
                f"{trajectory.id} is unfinished in the record and can be finished again"
            )
        capture.close()
    raise typer.Exit(code=code)


# -- reads ------------------------------------------------------------------
@app.command("list")
def list_trajectories(
    project: str = typer.Option(None, "--project"),
    run_id: str = typer.Option(None, "--run-id", help="--run-id, not --run: `run` is a command."),
    task_id: str = typer.Option(None, "--task-id"),
    step: int = typer.Option(None, "--step"),
    status: str = typer.Option(None, "--status"),
    limit: int = typer.Option(50, "--limit"),
    cursor: str = typer.Option(None, "--cursor", help="Continue a previous page."),
    ids_only: bool = typer.Option(False, "--ids", help="Print only trajectory IDs, one per line."),
    as_json: bool = typer.Option(False, "--json"),
    endpoint: str = EndpointOption,
    record: str = RecordOption,
) -> None:
    """List trajectories."""
    if record:
        from skyrl_capture.reader.records import TrajectoryQuery

        found = open_record(record).list_trajectories(
            TrajectoryQuery(
                project=project,
                run_id=run_id,
                task_id=task_id,
                step=step,
                status=status,
                limit=limit,
                cursor=cursor,
            )
        )
        page = {
            "data": found.items,
            "next_cursor": found.next_cursor,
            "has_more": found.next_cursor is not None,
            "total": found.total,
        }
    else:
        with CaptureClient(endpoint) as capture:
            try:
                page = capture.list_trajectories(
                    project=project,
                    run_id=run_id,
                    task_id=task_id,
                    step=step,
                    status=status,
                    limit=limit,
                    cursor=cursor,
                )
            except CaptureError as failure:
                fail(str(failure))
    if as_json:
        console.print_json(json.dumps(page))
        return
    if ids_only:
        for item in page["data"]:
            print(item["id"])
        return

    # The ID is the field a reader needs to copy, so it never wraps or
    # truncates; the wide, low-value columns give up space instead. This is a
    # scripting listing, not an inspector: detail is `--json`, or the viewer.
    table = Table(box=None, pad_edge=False)
    table.add_column("id", no_wrap=True)
    table.add_column("mode", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("calls", justify="right", no_wrap=True)
    table.add_column("nodes", justify="right", no_wrap=True)
    table.add_column("task", no_wrap=True, overflow="ellipsis", max_width=20)
    table.add_column("step", justify="right", no_wrap=True)
    table.add_column("project", overflow="ellipsis", max_width=16)
    table.add_column("labels", overflow="ellipsis", max_width=24)
    incomplete = []
    for item in page["data"]:
        capture = item["capture"]
        status = item["status"]
        if capture["calls_missing"]:
            status = f"[red]{status}![/red]"
            incomplete.append((item["id"], capture["calls_missing"]))
        table.add_row(
            item["id"],
            item["mode"],
            status,
            str(capture["exchange_count"]),
            str(capture["node_count"]),
            item["task_id"] or "-",
            "-" if item["step"] is None else str(item["step"]),
            item["project"],
            ", ".join(item["labels"] or []) or "-",
        )
    console.print(table)
    for identifier, dropped in incomplete:
        console.print(f"[red]![/red] {identifier} dropped {dropped} capture events")
    if page.get("has_more"):
        # The cursor is opaque: it is meant to be handed back, not read.
        more = (
            f"pass --cursor {page['next_cursor']}"
            if page.get("next_cursor")
            else "raise --limit"
        )
        total = page.get("total")
        scope = f" ({total} match)" if total else ""
        console.print(f"[dim]more available{scope}; {more}[/dim]")


@app.command("annotate")
def annotate(
    trajectory: str,
    annotate_pairs: list[str] = typer.Option(None, "--annotate", help="key=value, repeatable."),
    tag: list[str] = typer.Option(None, "--tag", help="Add a bare string tag, repeatable."),
    untag: list[str] = typer.Option(None, "--untag", help="Remove a tag, repeatable."),
    unset: list[str] = typer.Option(None, "--unset", help="Remove an annotation key, repeatable."),
    endpoint: str = EndpointOption,
) -> None:
    """Edit a trajectory's labels and annotations.

    Both are mutable for the life of the trajectory and hold only their latest
    value; finishing does not seal them and no history is kept.
    """
    annotations = parse_annotations(annotate_pairs)
    add = parse_tags(tag)
    remove = parse_tags(untag)
    if not (annotations or add or remove or unset):
        fail("nothing to do; pass --annotate, --tag, --untag or --unset")
    with CaptureClient(endpoint) as capture:
        try:
            result = capture.annotate_trajectory(
                trajectory,
                annotations=annotations,
                remove_annotations=list(unset or []),
                labels=add,
                remove_labels=remove,
            )
        except CaptureError as failure:
            fail(str(failure))
    console.print(
        f"[green]labels[/green] {', '.join(result['labels']) or '-'}\n"
        f"[green]annotations[/green] {json.dumps(result['annotations'])}"
    )


# -- exports ----------------------------------------------------------------
COMPRESSIONS = ("none", "gzip", "zst")
_WRAPPER = {"none": "", "gzip": ".gz", "zst": ".zst"}


def artifact_extension(compression: str) -> str:
    """What ``--output`` should be called for this compression."""
    if compression not in COMPRESSIONS:
        raise ValueError(
            f"unknown --compression {compression!r}; expected one of {', '.join(COMPRESSIONS)}"
        )
    return ".jsonl" + _WRAPPER[compression]


def check_output_name(destination: Path, compression: str) -> None:
    """Refuse a filename that would lie about its contents.

    Checked before the export runs: doing the work and then declining to write
    it wastes the run and reads like a failure of the export rather than of the
    name.
    """
    expected = artifact_extension(compression)
    written = "".join(destination.suffixes)
    if not written or written.endswith(expected):
        return
    fail(
        f"--output {destination.name} ends in {written!r} but --compression {compression} "
        f"writes {expected!r}. Rename the file or change --compression."
    )


def encode_artifact(data: bytes, compression: str) -> bytes:
    """Re-encode a downloaded artifact for local use.

    Stored artifacts are compressed, which is right for the object store and
    for delivery, and useless at a terminal.
    """
    from skyrl_capture.compression import compress, decompress

    raw = decompress(data)
    if compression == "none":
        return raw
    if compression == "gzip":
        return gzip.compress(raw)
    return compress(raw)


@app.command("export")
def export(
    project: str = typer.Option(None, "--project"),
    run: str = typer.Option(None, "--run-id", help="Export one run's trajectories."),
    trajectory: str = typer.Option(None, "--trajectory"),
    format: str = typer.Option(..., "--format", help="graph | replay | text-samples | token-samples"),
    output: Path = typer.Option(None, "--output", help="Write the artifact here."),
    compression: str = typer.Option(
        "none", "--compression", help="none | gzip | zst. How to write --output."
    ),
    allow_repeated_targets: bool = typer.Option(
        False,
        "--allow-repeated-targets",
        help=(
            "text-samples, token-samples: let one sampled message be a training target in "
            "several rows. Off by default, because branches share their ancestors."
        ),
    ),
    mask_abandoned: bool = typer.Option(
        False,
        "--mask-abandoned",
        help="text-samples, token-samples: rows whose branch lost a race train on nothing.",
    ),
    overlong_filtering: bool = typer.Option(
        False,
        "--overlong-filtering",
        help="token-samples: mask rollouts that stopped at the context limit.",
    ),
    timeout: float = typer.Option(300.0, "--timeout"),
    endpoint: str = EndpointOption,
    record: str = RecordOption,
) -> None:
    """Create an export, wait for it, and download the result.

    With ``--record`` the export runs here, against a record directory, with no
    service and no database. The four exporters are the same code either way --
    they take a trajectory view and nothing else -- so the artifact is
    identical.
    """
    if sum(1 for scope in (project, run, trajectory) if scope) != 1:
        fail("pass exactly one of --project, --run-id or --trajectory")
    if output is not None:
        try:
            check_output_name(output, compression)
        except ValueError as error:
            fail(str(error))
    normalized = format.strip().lower().replace("-", "_")
    options: dict[str, Any] = {}
    if normalized in ("text_samples", "token_samples"):
        # Rows are never dropped; these decide what `trainable` says.
        options = {
            "allow_repeated_targets": allow_repeated_targets,
            "mask_abandoned": mask_abandoned,
        }
        if normalized == "token_samples":
            options["overlong_filtering"] = overlong_filtering
    if record:
        _export_from_record(
            record,
            project=project,
            run=run,
            trajectory=trajectory,
            export_format=normalized,
            options=options,
            output=output,
            compression=compression,
        )
        return

    with CaptureClient(endpoint) as capture:
        try:
            job = capture.create_export(
                format=format,
                project=project,
                run=run,
                trajectory=trajectory,
                options=options,
            )
            console.print(f"[dim]export {job['id']} accepted ({len(job['selected_trajectory_ids'])} trajectories)[/dim]")
            job = capture.wait_for_export(job["id"], timeout=timeout)
            if job["status"] == "failed":
                fail(f"export failed: {job.get('error')}")
            if output is None:
                # Nothing asked for the bytes, so they are not fetched. The job
                # record carries the download URL and the checksum, which is
                # what a caller without --output came for.
                console.print(JSON.from_data(job))
                return
            data = capture.download_export(job["id"])
        except CaptureError as failure:
            fail(str(failure))

    payload = encode_artifact(data, compression)
    output.write_bytes(payload)
    console.print(
        f"[green]wrote[/green] {output} ({len(payload)} bytes, {job['record_count']} records, "
        f"{job['checksum']})"
    )


def _compress_artifact(raw: bytes, compression: str) -> bytes:
    """Compress a freshly rendered artifact. The mirror of `encode_artifact`,
    which starts from a stored one."""
    from skyrl_capture.compression import compress

    if compression == "none":
        return raw
    if compression == "gzip":
        return gzip.compress(raw)
    return compress(raw)


def _export_from_record(
    path: str,
    *,
    project: str | None,
    run: str | None,
    trajectory: str | None,
    export_format: str,
    options: dict[str, Any],
    output: Path | None,
    compression: str,
) -> None:
    """Run an export against a record directory, in this process.

    The same selection rule as the service -- every committed trajectory in the
    scope -- and the same origin rule: arrivals are relative to the first call
    in the export, not to each trajectory's own start.
    """
    import asyncio

    from skyrl_capture.export.service import render_artifact

    reader = open_record(path)

    async def render() -> tuple[list[str], bytes]:
        if trajectory:
            if not await reader.trajectory_exists(trajectory):
                fail(f"no trajectory {trajectory!r} in {path}")
            selected = [trajectory]
        else:
            selected = await reader.finished_trajectory_ids(project=project, run_id=run)
        views = [await reader.view(identifier) for identifier in selected]
        body = render_artifact(
            views,
            export_format=export_format,
            options=options,
            origin=await reader.earliest_request(selected),
        )
        return selected, body

    selected, body = asyncio.run(render())
    if output is None:
        console.print_json(
            json.dumps(
                {
                    "record": str(reader.root),
                    "format": export_format,
                    "selected_trajectory_ids": selected,
                    "record_count": len(body.splitlines()),
                }
            )
        )
        return
    payload = _compress_artifact(body, compression)
    output.write_bytes(payload)
    console.print(
        f"[green]wrote[/green] {output} ({len(payload)} bytes, "
        f"{len(body.splitlines())} records from {len(selected)} trajectories)"
    )


@app.command("reindex")
def reindex(record: str = RecordOption) -> None:
    """Rebuild the trajectory headers from the committed records.

    Every listing reads the headers rather than the records, so they are
    authoritative, and nothing on the read path quietly falls back to
    decompressing records instead. This is the deliberate way back: for a
    record directory written by an older build, one whose headers were
    deleted, or one assembled by copying records in from somewhere else.

    It reads every committed record, so it costs what a listing would cost
    without headers. Running it against a directory that is already correct is
    harmless.
    """
    from skyrl_capture.persistence.committed import DiskCommittedStore
    from skyrl_capture.persistence.layout import open_record as open_root

    try:
        root = open_root(record)
    except Exception as failure:
        fail(str(failure))
    store = DiskCommittedStore(root)
    result = store.headers.rebuild(store)
    console.print(
        f"listed [bold]{result['listed']}[/bold] trajectories, "
        f"dropped [bold]{result['dropped']}[/bold] stale headers in {root}/committed"
    )


@app.command("health")
def health(endpoint: str = EndpointOption) -> None:
    """Show service and capture health."""
    with CaptureClient(endpoint) as capture:
        try:
            console.print(JSON.from_data(capture.health()))
        except Exception as failure:
            fail(str(failure))


def app_main() -> None:
    """Entry point.

    Every command that reaches the control plane can fail for reasons that are
    the operator's business rather than a defect -- the service is not running,
    the endpoint is wrong, the key is wrong. Those are reported as one line.
    A traceback here would say the tool is broken when it is not.
    """
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)
    except CaptureError as failure:
        error_console.print(f"[red]error[/red] {failure}")
        sys.exit(1)


if __name__ == "__main__":
    app_main()
