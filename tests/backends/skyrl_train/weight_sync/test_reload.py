"""The ``reload`` weight-sync client: configuration and the agent protocol, against a local HTTP agent."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from skyrl.backends.skyrl_train.weight_sync.reload import (
    ReloadWeightSync,
    _default_agent_url,
)


@pytest.fixture
def agent():
    """A minimal reload agent that records the files it receives and the reload calls."""
    received = {"files": {}, "reloads": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _reply(self, payload):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_PUT(self):
            data = self.rfile.read(int(self.headers["Content-Length"]))
            received["files"][self.path] = data
            self._reply({"bytes": len(data)})

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received["reloads"].append(data)
            self._reply({"ok": True, "version": data["version"]})

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", received
    server.shutdown()


def test_default_agent_url_is_server_port_plus_one():
    assert _default_agent_url("http://rollout:8000") == "http://rollout:8001"
    with pytest.raises(ValueError):
        _default_agent_url("http://rollout")


def test_from_env(monkeypatch):
    for key in ("SKYRL_RELOAD_AGENT_URLS", "SKYRL_RELOAD_TRANSPORT", "SKYRL_RELOAD_SKIP_INITIAL"):
        monkeypatch.delenv(key, raising=False)
    sync = ReloadWeightSync.from_env(server_urls=["http://a:8000", "http://b:9000"], model_dtype="bfloat16")
    assert sync.agent_urls == ["http://a:8001", "http://b:9001"]
    assert sync.transport == "http" and sync.skip_initial

    monkeypatch.setenv("SKYRL_RELOAD_AGENT_URLS", "http://x:1")
    with pytest.raises(ValueError, match="one agent per server"):
        ReloadWeightSync.from_env(server_urls=["http://a:8000", "http://b:9000"], model_dtype="bfloat16")

    monkeypatch.setenv("SKYRL_RELOAD_TRANSPORT", "nfs")
    with pytest.raises(ValueError, match="SKYRL_RELOAD_TRANSPORT"):
        ReloadWeightSync.from_env(server_urls=["http://a:8000"], model_dtype="bfloat16")


def test_http_transport_uploads_files_then_reloads(agent, tmp_path):
    url, received = agent
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"\x00" * 1024)
    sync = ReloadWeightSync(agent_urls=[url], export_dir=str(tmp_path.parent), version=3)

    result = sync._publish_one(url, str(tmp_path))

    assert result == {"ok": True, "version": 3}
    assert received["files"]["/v1/files/3/model.safetensors"] == b"\x00" * 1024
    assert received["files"]["/v1/files/3/config.json"] == b"{}"
    assert received["reloads"] == [{"version": 3, "files": ["config.json", "model.safetensors"]}]


def test_path_transport_sends_only_the_directory(agent, tmp_path):
    url, received = agent
    sync = ReloadWeightSync(agent_urls=[url], export_dir=str(tmp_path), transport="path", version=5)

    sync._publish_one(url, str(tmp_path / "v000005"))

    assert received["files"] == {}
    assert received["reloads"] == [{"version": 5, "model_dir": str(tmp_path / "v000005")}]


def test_first_send_is_skipped_by_default():
    sync = ReloadWeightSync(agent_urls=["http://unused:1"], export_dir="/nonexistent")
    sync.send(strategy=None, model=None)
    assert sync.calls == 1 and sync.version == 0
