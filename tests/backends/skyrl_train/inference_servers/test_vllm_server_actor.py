import asyncio
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from skyrl.backends.skyrl_train.inference_servers import vllm_server_actor


@pytest.mark.asyncio
async def test_server_cancellation_shuts_down_vllm_workers(monkeypatch):
    engine = MagicMock()
    server = MagicMock()
    server.serve = AsyncMock(side_effect=asyncio.CancelledError)
    app = SimpleNamespace(state=SimpleNamespace())

    monkeypatch.setattr(vllm_server_actor, "create_server_socket", lambda _: MagicMock())
    monkeypatch.setattr(vllm_server_actor, "build_app", lambda _: app)
    monkeypatch.setattr(vllm_server_actor.AsyncEngineArgs, "from_cli_args", lambda _: MagicMock())
    monkeypatch.setattr(vllm_server_actor.AsyncLLMEngine, "from_engine_args", lambda **_: engine)
    monkeypatch.setattr(vllm_server_actor.VLLMServerActor, "_add_custom_endpoints", MagicMock())
    monkeypatch.setattr(vllm_server_actor, "init_app_state", AsyncMock())
    monkeypatch.setattr(vllm_server_actor.uvicorn, "Config", MagicMock())
    monkeypatch.setattr(vllm_server_actor.uvicorn, "Server", lambda _: server)

    cli_args = Namespace(
        host="0.0.0.0",
        port=8000,
        uvicorn_log_level="info",
        ssl_keyfile=None,
        ssl_certfile=None,
        ssl_ca_certs=None,
        ssl_cert_reqs=0,
        disable_uvicorn_access_log=False,
    )

    with pytest.raises(asyncio.CancelledError):
        await vllm_server_actor._build_and_serve_vllm_server(cli_args)

    engine.shutdown.assert_called_once_with()
