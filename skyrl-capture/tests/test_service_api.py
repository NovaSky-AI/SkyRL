"""The surfaces a training stack drives: named trajectories, and an upstream
that is fixed when the process starts.

A bootstrap runs more than once -- a restarted job, a second node -- which is
why creating a trajectory is the only setup call there is. What a trajectory
cannot do is reconfigure the upstream: deployment and authentication come from
process configuration, per-inference behaviour comes from the inference
request, and what is on the trajectory is capture metadata.
"""

from __future__ import annotations

import pytest

from skyrl_capture.config import TextUpstream, TitoUpstream
from skyrl_capture.domain.models import validate_identifier
from skyrl_capture.domain.records import TrajectoryConflict, TrajectoryError


# -- naming a trajectory ----------------------------------------------------
def test_an_id_must_survive_a_url():
    """It becomes the first path segment of every route this trajectory serves."""
    for good in ("sess-1", "tr_abc", "a.b_c-1", "A1"):
        assert validate_identifier(good) == good
    for bad in ("has space", "has/slash", "", "-leading", "a" * 129, "q?uery"):
        with pytest.raises(TrajectoryError, match="path segment"):
            validate_identifier(bad)


async def test_a_caller_can_name_a_trajectory(stack):
    response = await stack.post("/v1/trajectories", {"project": "rl", "trajectory_id": "sess-7"})
    assert response.status_code == 201
    assert response.json()["id"] == "sess-7"
    # The route carries the name, which is what makes it the engine session key.
    assert "/sess-7/" in response.json()["base_url"]


async def test_a_taken_name_is_a_conflict_only_when_it_means_a_second_trial(stack):
    """Repeating the same creation is a retry; changing it is a collision.

    A create whose response was lost has to be safe to send again, and the id
    is the caller's, so "same id, same body" cannot be a conflict. "Same id,
    different body" is two trials reaching for one route, and it is.
    """
    payload = {"project": "rl", "trajectory_id": "sess-dup"}
    first = await stack.post("/v1/trajectories", payload)
    assert first.status_code == 201
    assert (await stack.post("/v1/trajectories", payload)).json() == first.json()

    second = await stack.post("/v1/trajectories", {**payload, "project": "other"})
    assert second.status_code == 409, "a second trial must not inherit the first's route"


async def test_a_malformed_name_is_a_bad_request(stack):
    response = await stack.post("/v1/trajectories", {"project": "rl", "trajectory_id": "no/slash"})
    assert response.status_code == 400


# -- the upstream is startup configuration ----------------------------------
def test_an_unknown_upstream_type_fails_at_startup():
    with pytest.raises(ValueError, match="unsupported upstream type"):
        TextUpstream(type="not-a-provider", url="http://x/v1").validate()


def test_a_tokens_upstream_must_declare_a_tokenizer():
    with pytest.raises(ValueError, match="tokenizer"):
        TitoUpstream(type="tokens", url="http://x/generate").validate()
    TitoUpstream(type="tokens", url="http://x/generate", tokenizer="builtin").validate()


def test_the_mode_comes_from_the_upstream_type():
    assert TextUpstream(type="openai", url="http://x/v1").mode == "text"
    assert TitoUpstream(type="tokens", url="http://x/generate", tokenizer="b").mode == "tokens"


async def test_a_trajectory_snapshots_the_upstream_it_ran_against(stack):
    """The definition is not stored anywhere else, and the database outlives
    the process that wrote it."""
    created = await stack.create_trajectory()
    record = (await stack.get(f"/v1/trajectories/{created['id']}")).json()
    assert record["upstream"]["url"] == f"{stack.upstream_url}/v1"
    assert record["upstream"]["type"] == "openai"
    assert "api_key" not in record["upstream"]

    # What a launcher needs is on the creation response, not only on a later
    # read: which wire to speak, and whether tokens were captured. So there is
    # no route for asking the process about itself.
    assert created["protocol"] == "openai"
    assert created["mode"] == "text"


# -- a trajectory does not reconfigure the upstream -------------------------
async def test_a_trajectory_carries_capture_metadata_and_nothing_else(stack):
    """What the request path reads off a trajectory is the whole of it.

    Per-trajectory upstream overrides used to live here: headers merged into
    every forwarded request, body fields merged into every token generate
    call. They are gone, and this is the assertion that keeps them gone --
    adding one back means adding a field to the header, in front of a reviewer.
    """
    from dataclasses import fields

    created = await stack.create_trajectory(project="rl")
    header = stack.aggregate(created["id"]).header

    assert {f.name for f in fields(header)} == {
        "id", "project", "run_id", "task_id", "step", "mode", "upstream",
        "labels", "annotations", "bodies", "source_metadata", "created_at",
        "create_request_hash",
    }


async def test_an_upstream_setting_offered_on_create_is_not_stored(stack):
    """A caller that sends one gets a trajectory, not a reconfigured upstream."""
    response = await stack.post(
        "/v1/trajectories",
        {
            "project": "rl",
            "trajectory_id": "tr_overridden",
            "overrides": {"headers": {"X-Route": "pool-a"}},
        },
    )
    assert response.status_code == 201
    record = (await stack.get(f"/v1/trajectories/{response.json()['id']}")).json()
    assert "overrides" not in record
    assert "X-Route" not in str(record)


def test_the_outbound_credential_is_the_process_one(stack):
    """The only header capture adds is the upstream's own credential.

    Which is process configuration: there is no per-trajectory hook left that
    could add a header, so this is the whole of what the proxy injects.
    """
    from skyrl_capture.transport import headers
    from skyrl_capture.upstream import get

    headers = headers.prepare_headers(
        [(b"content-type", b"application/json"), (b"authorization", b"Bearer inbound-secret")],
        protocol=get("openai"),
        credential="sk-process",
    )
    by_name = {name.lower(): value for name, value in headers}
    assert by_name[b"authorization"] == b"Bearer sk-process"
    assert b"inbound-secret" not in b"".join(value for _, value in headers)


# -- conflicts stay distinguishable ----------------------------------------
async def test_a_creation_conflict_survives_a_restart(stack_builder, tmp_path):
    """The reason the creation hash is on disk rather than in a table.

    A process-local idempotency key could only answer for as long as the
    process lived, so a retry after a restart looked like a fresh create and
    got a second trajectory on the same route.
    """
    root = tmp_path / "traces"
    first = await stack_builder(record_dir=root)
    payload = {"project": "a", "trajectory_id": "tr_survives"}
    assert (await first.post("/v1/trajectories", payload)).status_code == 201
    await first.application.runtime.stop()

    second = await stack_builder(record_dir=root)
    assert (await second.post("/v1/trajectories", payload)).status_code == 201, "the same trial"
    clash = await second.post("/v1/trajectories", {**payload, "project": "b"})
    assert clash.status_code == 409
    assert isinstance(TrajectoryConflict("x"), TrajectoryError)


def test_moving_the_port_moves_the_url_trajectories_are_given():
    """`public_url` is what a trajectory's `base_url` is built from.

    `CaptureService(port=...)` used to move the bind port and leave
    `public_url` on the default, so every trajectory came back pointing at
    :8080 whatever the service was actually listening on -- a URL with nothing
    visibly wrong with it that answers nothing. Found by running the service
    in-process on a free port, which is what a training job does.
    """
    from skyrl_capture.service import CaptureService

    service = CaptureService(data_dir="/tmp/skyrl-capture-port-check", port=60459)
    assert service.base_url == "http://127.0.0.1:60459"
    assert service.config.proxy.public_url == "http://127.0.0.1:60459"


def test_an_explicit_public_url_is_not_overwritten(monkeypatch):
    """Behind a load balancer the public address is deliberately not the bind
    address, so only the *default* follows the port."""
    from skyrl_capture.config import load_config
    from skyrl_capture.service import CaptureService

    monkeypatch.setenv("PUBLIC_URL", "https://capture.example.com")
    service = CaptureService(config=load_config(), data_dir="/tmp/skyrl-capture-port-check", port=60459)
    assert service.config.proxy.public_url == "https://capture.example.com"
    # The bind address still moved; only the advertised one is pinned.
    assert service.base_url == "http://127.0.0.1:60459"
