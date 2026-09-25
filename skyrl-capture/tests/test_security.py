"""Credentials: the one this process holds, and the ones it refuses to keep.

Capture authenticates nothing on the way in. One process serves one upstream
and one workload; its control plane is unauthenticated on purpose (see
`control_plane/app.py`) and so is its data plane, where the trajectory id in the
route is correlation rather than authorization. A deployment that needs
callers authenticated puts that in front.

So there is exactly one credential here -- the upstream's, held in memory and
applied outbound -- and two rules about it: it reaches the upstream, and it is
never written down. Whatever a client sends inbound is stripped and is never
written down either.
"""

from __future__ import annotations

import orjson
import pytest


# -- credentials -----------------------------------------------------------
async def test_credentials_are_never_persisted(stack):
    """Not configurable: credentials are dropped whatever the allowlist says."""
    created = await stack.create_trajectory()
    inbound = "sk-the-clients-own-key"
    await stack.chat(
        created, [{"role": "user", "content": "hi"}], headers={"authorization": f"Bearer {inbound}"}
    )
    exchanges = await stack.exchanges(created["id"])

    row = stack.get_exchange(exchanges[0]["id"])
    combined = orjson.dumps([row["request_headers"], row["response_headers"]]).decode()
    assert inbound not in combined, "an inbound credential is stripped, not recorded"
    assert "upstream-secret" not in combined
    for header in ("authorization", "x-api-key", "cookie"):
        assert header not in row["request_headers"]
    # Content-type survived, so the allowlist is doing real work.
    assert row["request_headers"]["content-type"] == "application/json"

    # And neither credential is anywhere in the record.
    for exchange in exchanges:
        bodies = await stack.stored_bodies(exchange["id"])
        text = orjson.dumps(bodies, default=lambda value: value.decode("utf-8", "replace")).decode()
        assert inbound not in text
        assert "upstream-secret" not in text

    # The inbound one never reached the upstream either: it saw this process's.
    assert stack.upstream.requests[-1]["headers"]["authorization"] == "Bearer upstream-secret"


async def test_the_upstream_credential_is_never_stored(stack):
    """It is startup configuration, read from the environment and held in the
    process. Nothing the state holds could leak it."""
    await stack.chat(await stack.create_trajectory(), [{"role": "user", "content": "hi"}])
    await stack.settle()

    everywhere = orjson.dumps(
        [
            [active.document().public() for active in stack.aggregates()],
            [active.exchange_rows() for active in stack.aggregates()],
        ],
        default=str,
    ).decode()
    assert "upstream-secret" not in everywhere
    assert stack.runtime.config.upstream.api_key == "upstream-secret", "held, not stored"


async def test_the_upstream_is_not_settable_over_the_api(stack):
    """Changing it means relaunching, so there is no route that could."""
    for path in ("/v1/targets", "/v1/sinks"):
        assert (await stack.post(path, {"name": "x"})).status_code == 404

    # Nor is there one that describes it. What a caller can see is its own
    # trajectory -- the protocol and mode on creation, the whole snapshot on a
    # read -- and neither may hand back the credential.
    created = await stack.create_trajectory()
    assert created["protocol"] == "openai"
    assert created["mode"] == "text"
    record = (await stack.get(f"/v1/trajectories/{created['id']}")).json()
    assert record["upstream"]["type"] == "openai"
    assert "upstream-secret" not in orjson.dumps([created, record]).decode()


# -- the route is correlation, not a boundary ------------------------------
async def test_the_route_decides_the_trajectory_and_nothing_else_does(stack):
    """What a request is attributed to is the id in its path.

    Written down as a test because it is the security consequence of having no
    ingress credential: a client that can reach this process and knows an
    active trajectory id can write to it. That is what "capture does not
    authenticate ingress" means, and a deployment that cannot accept it puts
    authentication in front -- see `data_plane/app.py`.
    """
    mine = await stack.create_trajectory()
    theirs = await stack.create_trajectory()

    # One client, no credential, two routes: each turn lands where its path
    # said and nowhere else.
    for created, content in ((mine, "mine"), (theirs, "theirs")):
        response = await stack.client.post(
            f"{created['base_url']}/chat/completions",
            json={"model": "mock-model", "messages": [{"role": "user", "content": content}]},
        )
        assert response.status_code == 200

    assert len(await stack.exchanges(mine["id"])) == 1
    assert len(await stack.exchanges(theirs["id"])) == 1


async def test_finishing_revokes_the_route(stack):
    created = await stack.create_trajectory()
    await stack.chat(created, [{"role": "user", "content": "one turn"}])
    await stack.finish(created["id"])

    again = await stack.chat(created, [{"role": "user", "content": "second run"}])
    assert again.status_code == 410, "a finished route must never accept a second trial"


# -- artifact paths --------------------------------------------------------
async def test_an_artifact_path_cannot_escape_the_exports_directory(stack):
    """Export ids and filenames become path segments, so they are sanitized."""
    from skyrl_capture.export.artifacts import ArtifactStore, artifact_key, safe_segment

    # Separators, traversal sequences and empty names are all neutralized, and
    # no ".." survives anywhere -- these names also reach `file://` URLs, where
    # a client may normalize a path.
    assert safe_segment("../../etc/passwd") == "etc_passwd"
    assert safe_segment("") == "_"
    assert safe_segment("..") == "_"
    assert safe_segment("a/b") == "a_b"
    assert safe_segment("v1.2.3") == "v1.2.3", "ordinary dots are preserved"
    assert len(safe_segment("x" * 500)) == 96
    key = artifact_key("../../also-evil", "../x.jsonl")
    assert ".." not in key and "/" not in key.removeprefix(key.split("/")[0] + "/")

    # And the store refuses one anyway, whatever built it.
    store = ArtifactStore(stack.runtime.config.export_dir)
    with pytest.raises(ValueError):
        store.path("../../../etc/passwd")


async def test_a_project_name_with_separators_is_a_value_not_a_path(stack):
    """Project names used to become directories. They are keys in a document
    now, and come back exactly as given.

    Grouping trajectories by project and run is a query the reader answers,
    not a shape on disk -- which is also what keeps an arbitrary user string
    like this one from ever being a path segment.
    """
    project = "../../projects/elsewhere/theirs"
    created = await stack.create_trajectory(project=project)
    await stack.chat(created, [{"role": "user", "content": "traversal attempt"}])
    await stack.settle()
    assert (await stack.finish(created["id"])).status_code == 200

    root = stack.runtime.config.record_dir
    assert sorted(path.name for path in root.iterdir()) == [
        "active", "committed", "exports", "manifest.json"
    ]
    assert not (root.parent / "projects").exists()

    # Every file this trajectory produced is named by its id, which is
    # validated, and sits in a shard computed from that id.
    written = sorted(path.name for path in root.glob("committed/*/*"))
    assert written == [f"{created['id']}.head.json", f"{created['id']}.json.zst"]

    # The project survives intact in the document, which is the only place it
    # was ever meant to live -- on disk and over the API alike.
    import orjson

    header = orjson.loads(next(root.glob("committed/*/*.head.json")).read_bytes())
    assert header["project"] == project
    assert (await stack.get(f"/v1/trajectories/{created['id']}")).json()["project"] == project


