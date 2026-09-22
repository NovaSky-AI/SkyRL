"""One `/v1`, whichever source answers it.

The viewer, the CLI and anything scripted against the API read a live capture
process and a finished record through the same routes, and are meant to be
unable to tell which they got. This holds both to the same documents: capture a
run with a record directory configured, finish it, open the record, and fetch
every read route from each side.

They agree everywhere but one place, and that place is listed rather than
hidden. `KNOWN_DRIFT` was a Phase 0 fact about the codebase -- a dozen fields
the record answered with defaults -- and the refactor replaced the two read
paths with one reader over one reducer, which is what emptied it.

Also here: the whole loop through public surfaces alone -- create, turns,
finish, reopen the record from another process's point of view, export -- and
the requirement that the two exports are the same bytes.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from conftest import record_app
from golden import diff, normalise
from test_golden import text_forked, text_late_metadata, tokens_repaired


async def _read_everything(get: Any, trajectory_id: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "runs": await get("/v1/runs"),
        "trajectories": await get("/v1/trajectories?project=golden"),
    }
    for name, route in (("detail", ""), ("exchanges", "/exchanges"), ("graph", "/graph"), ("paths", "/paths")):
        out[name] = await get(f"/v1/trajectories/{trajectory_id}{route}")
    return out


async def _live_and_record(stack, root, trajectory_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    async def live(path: str) -> Any:
        response = await stack.get(path)
        assert response.status_code == 200, (path, response.text)
        return response.json()

    transport = httpx.ASGITransport(app=record_app(root))
    async with httpx.AsyncClient(transport=transport, base_url="http://record") as client:
        # This app was mounted on a transport with no lifespan, so its index
        # has not been built. One manual refresh is what a viewer sends.
        await client.get("/v1/trajectories?refresh=true")

        async def recorded(path: str) -> Any:
            response = await client.get(path)
            assert response.status_code == 200, (path, response.text)
            return response.json()

        return await _read_everything(live, trajectory_id), await _read_everything(recorded, trajectory_id)


def _assert_no_drift(live: dict[str, Any], recorded: dict[str, Any]) -> None:
    # The same ids appear in both, so one normaliser numbers them identically
    # only if it sees them in the same order; normalise each on its own and
    # compare structure, which is what a reader cares about.
    entries = diff(normalise(live), normalise(recorded))
    assert not entries, "live and record disagree:\n  " + "\n  ".join(entries)


async def test_a_text_record_answers_the_same_v1_as_the_live_process(stack_builder, tmp_path):
    root = tmp_path / "traces"
    stack = await stack_builder(record_dir=root)
    await text_forked(stack)
    trajectory_id = await text_late_metadata(stack)
    live, recorded = await _live_and_record(stack, root, trajectory_id)
    _assert_no_drift(live, recorded)


async def test_a_tokens_record_answers_the_same_v1_as_the_live_process(stack_builder, tmp_path):
    root = tmp_path / "traces"
    stack = await stack_builder(tokens=True, record_dir=root)
    trajectory_id = await tokens_repaired(stack)
    live, recorded = await _live_and_record(stack, root, trajectory_id)
    _assert_no_drift(live, recorded)
    # The one thing that must not drift: the blocks a reader is shown.
    assert [p["blocks"] for p in live["paths"]["paths"]] == [p["blocks"] for p in recorded["paths"]["paths"]]


# -- the whole loop, through public surfaces only -------------------------------
async def test_the_record_a_run_writes_exports_the_same_bytes_as_the_live_process(stack_builder, tmp_path):
    """create -> turns -> finish -> reopen the record -> export.

    Nothing here reaches inside: the record is written by `finish`, reopened by
    `skyrl-capture export --record` as a later process would, and compared to
    what the live process serves from `/v1/exports`. Byte for byte, for every
    format, because an export is a training artifact and "equivalent" is not a
    property a trainer can check.
    """
    from typer.testing import CliRunner

    from skyrl_capture.cli.main import app
    from skyrl_capture.compression import decompress

    root = tmp_path / "traces"
    stack = await stack_builder(tokens=True, record_dir=root)
    trajectory_id = await tokens_repaired(stack)

    for export_format in ("graph", "replay", "text_samples", "token_samples"):
        job = await stack.run_export(trajectory=trajectory_id, format=export_format)
        assert job["status"] == "ready", job
        download = await stack.get(f"/v1/exports/{job['id']}/download")
        assert download.status_code == 200
        live_bytes = decompress(download.content)

        output = tmp_path / f"{export_format}.jsonl"
        # The CLI is a synchronous program with its own event loop.
        result = await asyncio.to_thread(
            CliRunner().invoke,
            app,
            ["export", "--record", str(root), "--trajectory", trajectory_id,
             "--format", export_format, "--output", str(output)],
        )
        assert result.exit_code == 0, result.output
        assert output.read_bytes() == live_bytes, export_format
