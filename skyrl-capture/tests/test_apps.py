"""Which application answers what, and what `--disable-viewer` takes away.

A capture replica always mounts the lifecycle app: create, finish, annotate,
health. It mounts the viewer -- the reads and the bulk exports -- only when it
is the replica serving them, because there is one indexer per record
directory and several processes may share one.

What must not happen is that turning the viewer off changes capture. A replica
with `--disable-viewer` captures exactly as much, writes exactly the same
files, and is read by whichever process does serve the viewer.
"""

from __future__ import annotations

import pytest

from skyrl_capture.application import reads_captured_data


@pytest.mark.parametrize(
    ("method", "path", "viewer"),
    [
        ("GET", "/v1/trajectories", True),
        ("GET", "/v1/trajectories/tr_1", True),
        ("GET", "/v1/trajectories/tr_1/graph", True),
        ("GET", "/v1/runs", True),
        ("POST", "/v1/exports", True),
        ("GET", "/v1/exports/exp_1/download", True),
        ("POST", "/v1/trajectories", False),
        ("POST", "/v1/trajectories/tr_1/finish", False),
        ("PATCH", "/v1/trajectories/tr_1/metadata", False),
        ("GET", "/healthz", False),
        ("GET", "/metrics", False),
    ],
)
def test_the_split_is_by_what_a_request_does(method, path, viewer):
    """Reads of captured data and bulk exports on one side; the lifecycle of a
    trajectory and the health of this process on the other."""
    assert reads_captured_data(method, path) is viewer


async def test_a_replica_without_a_viewer_still_captures(stack_builder, tmp_path):
    root = tmp_path / "traces"
    stack = await stack_builder(record_dir=root, viewer=False)

    created = await stack.create_trajectory(project="headless")
    assert (await stack.chat(created, [{"role": "user", "content": "hi"}])).status_code == 200
    await stack.settle()
    assert (await stack.finish(created["id"], format="graph")).status_code == 200

    # Everything is on disk, and the record is complete.
    assert stack.record(created["id"]).trajectory.status == "finished"
    assert len(stack.record(created["id"]).exchanges) == 1
    assert stack.runtime.reader is None and stack.runtime.exports is None

    # Health and metrics still answer: they are about this process, not about
    # what it captured.
    assert (await stack.get("/healthz")).json()["status"] == "ok"
    assert (await stack.get("/metrics")).status_code == 200


async def test_a_replica_without_a_viewer_serves_no_reads_or_exports(stack_builder, tmp_path):
    stack = await stack_builder(record_dir=tmp_path / "traces", viewer=False)
    created = await stack.create_trajectory(project="headless")

    for path in (
        "/v1/trajectories",
        f"/v1/trajectories/{created['id']}",
        f"/v1/trajectories/{created['id']}/graph",
        f"/v1/trajectories/{created['id']}/exchanges",
        f"/v1/trajectories/{created['id']}/paths",
        "/v1/runs",
    ):
        assert (await stack.get(path)).status_code in (404, 405), path
    assert (await stack.post("/v1/exports", {"format": "graph", "project": "headless"})).status_code in (
        404,
        405,
    )


async def test_another_process_reads_what_a_headless_replica_wrote(stack_builder, tmp_path):
    """The deployment the split is for: capture replicas with the viewer off,
    and one process -- here a standalone viewer -- reading their directory."""
    import httpx
    from conftest import record_app

    root = tmp_path / "traces"
    stack = await stack_builder(record_dir=root, viewer=False)
    created = await stack.create_trajectory(project="headless", run_id="run-a")
    await stack.chat(created, [{"role": "user", "content": "captured blind"}])
    await stack.settle()
    await stack.finish(created["id"])

    transport = httpx.ASGITransport(app=record_app(root))
    async with httpx.AsyncClient(transport=transport, base_url="http://viewer") as client:
        listing = (await client.get("/v1/trajectories?refresh=true")).json()
        assert [row["id"] for row in listing["data"]] == [created["id"]]
        assert (await client.get(f"/v1/trajectories/{created['id']}/graph")).json()["nodes"]
        job = (await client.post(
            "/v1/exports", json={"format": "graph", "run": "run-a"}
        )).json()
        assert job["selected_trajectory_ids"] == [created["id"]]
