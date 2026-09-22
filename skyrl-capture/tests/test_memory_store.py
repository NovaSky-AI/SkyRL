"""The whole stack, with no database.

`test_aggregate.py` covers one trajectory's aggregate in isolation and
`test_persistence.py` the files it is written to. This runs the real thing --
proxy, commit coordinator, lifecycle API, record directory -- and asserts the
things a database used to be responsible for: that a trajectory is captured and
read back, that finishing is idempotent, that a finished route stops working,
and that what outlives the process is on disk.

There is nothing to connect to, nothing to migrate, and no worker to spawn.
"""

from __future__ import annotations

import pytest

# Most of this file drives the running stack; the path tests at the end are
# plain functions, so the mark is applied per test rather than to the module.
pytestmark = pytest.mark.asyncio


async def _captured(stack, **kwargs):
    created = await stack.create_trajectory(**kwargs)
    await stack.client.post(
        f"{created['base_url']}/chat/completions",
        json={"model": "mock-model", "messages": [{"role": "user", "content": "hi"}]},
        headers={"authorization": "Bearer client-key"},
    )
    await stack.settle()
    return created


async def test_a_trajectory_is_captured_with_no_database(stack_builder, tmp_path):
    """The end-to-end claim: a run goes through the proxy and the recorder,
    and comes back out of the control API, with nothing to connect to and no
    schema to migrate."""
    stack = await stack_builder(record_dir=tmp_path / "traces")

    created = await _captured(stack, project="rl", run_id="run-a", task_id="t1", step=0)
    await stack.finish(created["id"])

    body = (await stack.get(f"/v1/trajectories/{created['id']}")).json()
    assert body["status"] == "finished"
    assert body["capture"]["exchange_count"] == 1
    assert body["capture"]["complete"] is True
    assert body["capture"]["node_count"] >= 2, "the graph was built and committed"


async def test_healthz_does_not_claim_a_database_it_does_not_have(stack_builder):
    stack = await stack_builder()
    health = (await stack.get("/healthz")).json()
    assert health["status"] == "ok"
    assert health["persistence"] == "ok"
    assert "database" not in health, "there is none to report on"
    # What there is instead: a directory, and the numbers that say how the
    # writing of it is going.
    assert health["record"] == str(stack.runtime.config.record_dir)
    assert health["commits"]["failures"] == 0


async def test_the_listing_the_graph_and_the_exchanges_all_answer(stack_builder):
    """Every read path the viewer uses, off the map instead of off SQL."""
    stack = await stack_builder()
    created = await _captured(stack, project="rl", run_id="run-a", task_id="t1", step=0)
    await stack.finish(created["id"])

    listing = (await stack.get("/v1/trajectories?run_id=run-a")).json()
    assert [row["id"] for row in listing["data"]] == [created["id"]]
    assert listing["total"] == 1

    runs = (await stack.get("/v1/runs")).json()["data"]
    assert runs[0]["trajectory_count"] == 1
    assert runs[0]["step_counts"] == {"0": 1}

    graph = (await stack.get(f"/v1/trajectories/{created['id']}/graph")).json()
    assert len(graph["nodes"]) >= 2
    assert graph["leaf_node_ids"]

    exchanges = (await stack.get(f"/v1/trajectories/{created['id']}/exchanges")).json()
    assert len(exchanges["data"]) == 1
    # Derived on read in SQL, and derived on read here too.
    assert "gap_ms" in exchanges["data"][0]
    assert "overlapping" in exchanges["data"][0]


async def test_a_second_finish_returns_the_committed_record(stack_builder):
    """Idempotency was an `ON CONFLICT` and a stored response. It is the hash
    of the finish request, persisted beside the record it produced."""
    stack = await stack_builder()
    created = await _captured(stack, project="rl")

    first = await stack.client.post(
        f"{stack.base_url}/v1/trajectories/{created['id']}/finish",
        json={"command_result": "success"},
    )
    second = await stack.client.post(
        f"{stack.base_url}/v1/trajectories/{created['id']}/finish",
        json={"command_result": "success"},
    )
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json(), "the committed record, not a second finish"


async def test_finishing_revokes_the_route(stack_builder):
    """A finished trajectory's route must stop working, or a late call joins
    a closed trace."""
    stack = await stack_builder()
    created = await _captured(stack, project="rl")
    await stack.finish(created["id"])

    response = await stack.client.post(
        f"{created['base_url']}/chat/completions",
        json={"model": "mock-model", "messages": [{"role": "user", "content": "again"}]},
        headers={"authorization": "Bearer client-key"},
    )
    # 410, not 401: the route existed and has closed. A credential that never
    # existed is the 401, and collapsing the two would tell a workload its
    # token was wrong when its trajectory had simply finished.
    assert response.status_code == 410


async def test_the_record_is_written_without_a_database(stack_builder, tmp_path):
    """The record is what outlives the process now, so it has to be written on
    exactly the path that no longer has a database behind it."""
    from skyrl_capture.reader.records import RecordReader, TrajectoryQuery

    root = tmp_path / "traces"
    stack = await stack_builder(record_dir=root)
    created = await _captured(stack, project="rl", run_id="run-a", task_id="t1", step=0)
    await stack.finish(created["id"])
    await stack.settle()

    reader = RecordReader(root)
    await reader.refresh()
    page = reader.list_trajectories(TrajectoryQuery(run_id="run-a", limit=50))
    assert [row["id"] for row in page.items] == [created["id"]]
    view = await reader.view(created["id"])
    assert view.trajectory.status == "finished"
    assert len(view.exchanges) == 1


async def test_ingestion_has_no_worker_knob_left_to_get_wrong():
    """The failure this prevents is silent, which is why the knob is gone
    rather than merely refused.

    Ingestion workers were *spawned*, so a child built its own state and wrote
    into a map the parent cannot see. Run that way and the proxy counts
    accepted exchanges, the children ingest them, and the trajectory sits in
    `finalizing` for ever with nothing written to the record -- observed as
    `accepted=2 exchanges=0 nodes=0`, no error anywhere.

    A refusal left the shape of the mistake in the API. Nothing takes a worker
    count now, on the command line or in Python, and nothing takes `migrate`
    either: there is no schema. Asserted on the signatures, because a parameter
    that still exists is one a caller can still pass.
    """
    import inspect
    import tempfile
    from pathlib import Path

    from skyrl_capture.application import CaptureApplication
    from skyrl_capture.config import Config, TextUpstream
    from skyrl_capture.runtime import build_runtime
    from skyrl_capture.service import CaptureService

    for entry in (CaptureService, CaptureApplication, build_runtime):
        parameters = inspect.signature(entry).parameters
        assert "num_workers" not in parameters, entry
        assert "worker_processes" not in parameters, entry
        assert "migrate" not in parameters, entry
        assert "roles" not in parameters, entry

    # And nothing can hand a runtime in either: there is one way to build one,
    # and it runs inside the application's lifespan.
    assert "runtime" not in inspect.signature(CaptureApplication).parameters
    assert "builder" not in inspect.signature(CaptureApplication).parameters

    config = Config(
        upstream=TextUpstream(type="openai", url="http://127.0.0.1:1/v1", api_key="x"),
        record_dir=Path(tempfile.mkdtemp(prefix="knob-")) / "record",
    )
    CaptureService(config=config)


async def test_a_record_directory_is_required_and_exports_sit_beside_it(tmp_path, monkeypatch):
    """Persistence is not a mode. A process with nowhere to write is a process
    whose whole output is lost, so it refuses to start -- and what it exports
    belongs with what it captured."""
    from skyrl_capture.config import StartupError, load_config

    monkeypatch.delenv("CAPTURE_RECORD_DIR", raising=False)
    monkeypatch.setenv("CAPTURE_DATA_DIR", str(tmp_path / "scratch"))

    plain = load_config()
    assert plain.record_dir is None
    with pytest.raises(StartupError, match="record directory"):
        plain.require_record_dir()

    placed = plain.with_overrides(record_dir=tmp_path / "traces")
    assert placed.export_dir == tmp_path / "traces" / "exports"


