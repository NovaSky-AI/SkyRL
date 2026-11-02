"""daytona.py: a run's sandboxes are labelled and time-bounded, and cleanup reaches only them."""

import asyncio
from types import SimpleNamespace

import pytest

from skyrl.integration.sandboxes import daytona


def test_labels_are_required_and_stringified() -> None:
    for missing in (None, {}):
        with pytest.raises(ValueError, match="need labels"):
            daytona.check_labels(missing)
    assert daytona.check_labels({"owner": "me", "run": 7}) == {"owner": "me", "run": "7"}


def test_label_params_adds_labels_over_the_params_own_and_bounds_its_life() -> None:
    pytest.importorskip("daytona")
    from daytona import CreateSandboxFromImageParams

    params = CreateSandboxFromImageParams(image="x", labels={"kept": "1", "run": "old"})
    assert daytona.label_params(params, {"owner": "me", "run": "r1"}, 90) is params
    assert params.labels == {"kept": "1", "owner": "me", "run": "r1"} and params.ttl_minutes == 90

    untimed = daytona.label_params(CreateSandboxFromImageParams(image="x"), {"owner": "me", "run": "r1"})
    assert untimed.labels == {"owner": "me", "run": "r1"} and untimed.ttl_minutes is None


def test_the_cli_needs_an_owner_and_a_run_label() -> None:
    assert daytona.command_is_unscoped({"owner": "me"})
    assert not daytona.command_is_unscoped({"owner": "me", "run": "r", "extra": "x"})
    for argv in (["cleanup", "--label", "owner=me"], ["cleanup", "--label", "owner"], ["delete", "--label", "a=b"]):
        with pytest.raises(SystemExit):
            daytona.parse_args(argv)
    args = daytona.parse_args(["cleanup", "--label", "owner=me", "--label", "run=a=b"])
    assert (args.command, args.labels) == ("cleanup", {"owner": "me", "run": "a=b"})


class FakeDaytona:
    def __init__(self, sandboxes, fail=()):
        self.sandboxes, self.queries, self.deleted, self.fail = sandboxes, [], [], set(fail)

    async def list(self, query):
        self.queries.append(query)
        for sandbox in self.sandboxes:
            if query.labels.items() <= sandbox.labels.items():
                yield sandbox

    async def delete(self, sandbox):
        if sandbox.id in self.fail:
            raise ConnectionError("network glitch")
        self.deleted.append(sandbox.id)


@pytest.mark.parametrize("command", ["list", "cleanup"])
def test_list_and_cleanup_touch_only_the_labelled_sandboxes(command, capsys) -> None:
    pytest.importorskip("daytona")
    ours = SimpleNamespace(id="a", state="started", labels={"owner": "me", "run": "r1", "kept": "1"})
    other_run = SimpleNamespace(id="b", state="started", labels={"owner": "me", "run": "r2"})
    someone_else = SimpleNamespace(id="c", state="started", labels={"owner": "you", "run": "r1"})
    client = FakeDaytona([ours, other_run, someone_else])

    found = asyncio.run(daytona.run(command, {"owner": "me", "run": "r1"}, client))

    assert found == [ours] and client.queries[0].labels == {"owner": "me", "run": "r1"}
    assert client.deleted == (["a"] if command == "cleanup" else [])
    assert "1 sandboxes labelled" in capsys.readouterr().out


@pytest.mark.parametrize(
    "command, labels",
    [("cleanup", {"owner": "me"}), ("cleanup", {"run": "r1"}), ("lists", {"owner": "me", "run": "r1"})],
)
def test_run_refuses_an_unscoped_or_unknown_command_before_reaching_daytona(command, labels) -> None:
    client = FakeDaytona([SimpleNamespace(id="a", state="started", labels={"owner": "me", "run": "r1"})])
    with pytest.raises(ValueError):
        asyncio.run(daytona.run(command, labels, client))
    assert client.queries == [] and client.deleted == []


def test_cleanup_tries_every_deletion_then_reports_the_failures(capsys) -> None:
    pytest.importorskip("daytona")
    labels = {"owner": "me", "run": "r1"}
    sandboxes = [SimpleNamespace(id=name, state="started", labels=labels) for name in "abc"]
    client = FakeDaytona(sandboxes, fail={"b"})

    with pytest.raises(RuntimeError, match="1 of 3 sandboxes could not be deleted"):
        asyncio.run(daytona.run("cleanup", labels, client))

    assert sorted(client.deleted) == ["a", "c"]
    out = capsys.readouterr().out
    assert "could not delete sandbox b: ConnectionError: network glitch" in out and "deleted 2 of 3" in out
