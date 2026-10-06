"""Per-trajectory API keys: minted at create, required on the harness routes when the server says so."""

from __future__ import annotations

from pathlib import Path

import openai
import pytest

from skycap import CapturePool, record
from skycap.cli import build_parser, build_server
from tests.conftest import Stack, openai_client, running_stack
from tests.mock_openai import API_KEY


async def chat(stack: Stack, trajectory: dict, headers: dict | None = None) -> int:
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    async with stack.http.post(f"{trajectory['base_url']}/chat/completions", json=body, headers=headers) as response:
        return response.status


async def models(stack: Stack, trajectory: dict, headers: dict | None = None) -> int:
    async with stack.http.get(f"{trajectory['base_url']}/models", headers=headers) as response:
        return response.status


def bearer(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"}


async def test_each_trajectory_gets_a_key_of_its_own(stack: Stack) -> None:
    first, second = await stack.create(), await stack.create()
    assert first["api_key"].startswith("sk-skycap-") and first["api_key"] != second["api_key"]
    # Without require_api_key the routes stay open, as before.
    assert await chat(stack, first) == 200


async def test_a_required_key_must_be_the_trajectorys_own() -> None:
    async with running_stack(require_api_key=True) as stack:
        mine, other = await stack.create(), await stack.create()
        for headers in (None, bearer("sk-skycap-guess"), bearer(other["api_key"]), {"Authorization": mine["api_key"]}):
            assert await chat(stack, mine, headers) == 401
            assert await models(stack, mine, headers) == 401
        assert await chat(stack, mine, bearer(mine["api_key"])) == 200
        assert await models(stack, mine, bearer(mine["api_key"])) == 200
        # Refused calls never reached the graph: only the accepted one is in it.
        document = await stack.document(mine["id"])
        assert [n["author"] for n in document["nodes"]] == ["client", "model"]
        # The control plane is not keyed: the trainer reaches it on its own.
        await stack.finish(mine["id"], {"reward": 1.0})


async def test_an_openai_client_sends_the_key_as_its_api_key() -> None:
    async with running_stack(require_api_key=True) as stack:
        trajectory = await stack.create()
        with pytest.raises(openai.AuthenticationError):
            await openai_client(trajectory["base_url"]).chat.completions.create(
                model="m", messages=[{"role": "user", "content": "hi"}]
            )
        llm = openai_client(trajectory["base_url"], api_key=trajectory["api_key"])
        reply = await llm.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
        assert reply.choices[0].message.content
        # skycap's key stays here; the upstream gets its own.
        assert stack.upstream.headers[-1]["Authorization"] == f"Bearer {API_KEY}"


async def test_the_pool_hands_out_the_key_and_the_record_never_holds_it(tmp_path: Path) -> None:
    async with running_stack(require_api_key=True, record_dir=tmp_path) as stack, CapturePool([stack.url]) as pool:
        async with pool.trajectory() as trajectory:
            llm = openai_client(trajectory.base_url, api_key=trajectory.api_key)
            await llm.chat.completions.create(model="m", messages=[{"role": "user", "content": "hi"}])
            await trajectory.finish({"reward": 1.0})
        # Ended, the trajectory answers 410 whatever key is sent.
        assert await chat(stack, {"base_url": trajectory.base_url}, bearer(trajectory.api_key)) == 410

    assert trajectory.api_key and trajectory.api_key.startswith("sk-skycap-")
    raw = b"".join(path.read_bytes() for path in tmp_path.iterdir())
    assert trajectory.api_key.encode() not in raw
    assert "api_key" not in record.read_document(tmp_path, trajectory.id)


def test_the_cli_requires_keys_on_request() -> None:
    serve = ["serve", "--upstream-url", "http://engine:8000/v1"]
    assert not build_server(build_parser().parse_args(serve)).require_api_key
    assert build_server(build_parser().parse_args([*serve, "--require-api-key"])).require_api_key
