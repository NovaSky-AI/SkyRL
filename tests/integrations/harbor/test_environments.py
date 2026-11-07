"""environments.py: Harbor's Daytona environment labels every sandbox and bounds its life."""

import asyncio

import pytest

pytest.importorskip("harbor")
pytest.importorskip("daytona")

from daytona import CreateSandboxFromImageParams  # noqa: E402

from skyrl.integration.harbor import environments  # noqa: E402


def test_the_labelled_environment_tags_every_sandbox_and_bounds_its_life(monkeypatch) -> None:
    created = []

    async def create(self, params, daytona=None):
        created.append(params)

    monkeypatch.setattr(environments.DaytonaEnvironment, "_create_sandbox", create)
    environment = object.__new__(environments.LabelledDaytonaEnvironment)
    environment._labels, environment._ttl_minutes = {"owner": "me", "run": "r1"}, 90

    asyncio.run(environment._create_sandbox(CreateSandboxFromImageParams(image="x", labels={"kept": "1"})))

    (params,) = created
    assert params.labels == {"kept": "1", "owner": "me", "run": "r1"} and params.ttl_minutes == 90
    with pytest.raises(ValueError, match="need labels"):
        environments.LabelledDaytonaEnvironment(labels={})
