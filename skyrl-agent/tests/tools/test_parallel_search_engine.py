"""Synthetic protocol and registry compatibility checks; no live provider calls."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests

from skyrl_agent.tools import TOOL_REGISTRY
from skyrl_agent.tools.parallel_search_engine import ParallelSearchEngine, _ParallelSession


PASSAGE = "alpha\u0085beta\u2028gamma\u2029delta"


@pytest.mark.parametrize("fail", [False, True])
def test_held_open_transport(monkeypatch, fail):
    requests_seen = []
    release = threading.Event()
    closed = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_DELETE(self):
            requests_seen.append(("DELETE", self.headers))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            method = request["method"]
            requests_seen.append((method, self.headers))
            if "id" not in request:
                self.send_response(202)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            reply = {"jsonrpc": "2.0", "id": request["id"]}
            if method == "initialize":
                reply["result"] = {"protocolVersion": "2025-03-26"}
            elif method == "tools/list":
                reply["result"] = {"tools": [{"name": "web_search"}]}
            elif fail:
                reply["error"] = {"code": -32042, "message": "synthetic failure"}
            else:
                reply["result"] = {
                    "structuredContent": {"results": [{"url": "https://example.org/source", "excerpts": [PASSAGE]}]}
                }
            body = json.dumps(reply, ensure_ascii=False).encode()
            self.send_response(200)
            if method == "initialize":
                self.send_header("Mcp-Session-Id", "synthetic-session")
            if method != "tools/call":
                self.send_header("Content-Type", "Application/JSON; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_header("Content-Type", "Text/Event-Stream; charset=utf-8")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            notice = b'data: {"jsonrpc":"2.0","method":"notifications/tools/list_changed"}\r\n\r\n'
            for event in (notice, b"data: " + body + b"\r\n\r\n"):
                self.wfile.write(f"{len(event):x}\r\n".encode() + event + b"\r\n")
                self.wfile.flush()
            self.close_connection = True
            self.connection.settimeout(3)
            if self.connection.recv(1) == b"":
                closed.set()
            release.wait(5)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    monkeypatch.setattr(_ParallelSession, "endpoint", f"http://127.0.0.1:{server.server_port}/mcp")
    try:
        with ThreadPoolExecutor() as executor:
            result = executor.submit(TOOL_REGISTRY["parallel_search_engine"]().call, {"query": ["test"]}).result(2)
        if fail:
            assert "MCP error -32042: synthetic failure" in result["error"]
        else:
            assert PASSAGE in result["results"]
            assert "https://example.org/source" in result["results"]
        assert closed.wait(1)
        assert [method for method, _ in requests_seen] == [
            "initialize",
            "notifications/initialized",
            "tools/list",
            "tools/call",
            "DELETE",
        ]
        for _, headers in requests_seen:
            assert headers["User-Agent"].startswith("SkyRL-Agent/")
            assert "Authorization" not in headers
            assert "x-api-key" not in headers
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        serving.join()


def test_opt_in_and_credentials(monkeypatch):
    def unexpected_request(*args, **kwargs):
        pytest.fail("Registration and unselected tools must not make network requests")

    monkeypatch.setattr(requests.Session, "request", unexpected_request)
    monkeypatch.delenv("GOOGLE_SEARCH_KEY", raising=False)
    monkeypatch.delenv("YDC_API_KEY", raising=False)
    TOOL_REGISTRY["parallel_search_engine"]()
    TOOL_REGISTRY["finish"]()
    with pytest.raises(ValueError, match="GOOGLE_SEARCH_KEY"):
        TOOL_REGISTRY["search_engine"]()
    with pytest.raises(ValueError, match="YDC_API_KEY"):
        TOOL_REGISTRY["youcom_search_engine"]()
    monkeypatch.setenv("GOOGLE_SEARCH_KEY", "explicit-serper-key")
    monkeypatch.setenv("YDC_API_KEY", "explicit-you-key")
    assert TOOL_REGISTRY["search_engine"]().google_search_key == "explicit-serper-key"
    assert TOOL_REGISTRY["youcom_search_engine"]().you_search_key == "explicit-you-key"


@pytest.mark.parametrize("params", [{}, {"query": []}, {"query": [""]}, {"query": [" "]}, {"query": "test"}])
def test_invalid_query_does_not_connect(monkeypatch, params):
    monkeypatch.setattr(_ParallelSession, "search", lambda *args: pytest.fail("Unexpected network call"))
    assert "error" in ParallelSearchEngine().call(params)


@pytest.mark.parametrize("structured", [True, False])
def test_leakage_filters(monkeypatch, structured):
    payload = {
        "results": [
            {"url": "https://example.org/safe", "title": "Allowed", "excerpts": ["Passage"]},
            {"url": "https://github.com/data", "title": "Blocked domain", "excerpts": ["Text"]},
            {"url": "https://example.org/leak", "title": "GPQA answers", "excerpts": ["Text"]},
            {"url": "https://example.org/leak2", "title": "Leak", "excerpts": ["gpqa-diamond"]},
        ]
    }
    seen = []

    def synthetic_search(self, queries):
        seen.extend(queries)
        return (
            {"structuredContent": payload}
            if structured
            else {"content": [{"type": "text", "text": json.dumps(payload)}]}
        )

    monkeypatch.setattr(_ParallelSession, "search", synthetic_search)
    tool = ParallelSearchEngine()
    result = tool.call('{"query": ["test"]}')
    assert "Passage" in result["results"]
    assert "Blocked domain" not in result["results"]
    assert "Leak" not in result["results"]
    assert "-site:github.com" in seen.pop()
    tool._use_negative_filters = False
    tool.call({"query": ["test"]})
    assert seen == ["test"]


def test_tool_error_is_not_search_results(monkeypatch):
    monkeypatch.setattr(
        _ParallelSession,
        "search",
        lambda *args: {"isError": True, "content": [{"type": "text", "text": "synthetic rate limit"}]},
    )
    assert "synthetic rate limit" in ParallelSearchEngine().call({"query": ["test"]})["error"]


@pytest.mark.parametrize(
    "provider,env_key,method",
    [
        ("search_engine", "GOOGLE_SEARCH_KEY", "post"),
        ("youcom_search_engine", "YDC_API_KEY", "get"),
    ],
)
def test_explicit_bad_credentials_do_not_fall_back(monkeypatch, provider, env_key, method):
    from types import SimpleNamespace

    monkeypatch.setenv(env_key, "synthetic-invalid-key")
    seen = []

    def reject(url, **kwargs):
        seen.append((url, kwargs["headers"]["X-API-KEY"]))
        return SimpleNamespace(status_code=401, text="synthetic unauthorized", json=lambda: {})

    monkeypatch.setattr(requests, method, reject)
    monkeypatch.setattr(_ParallelSession, "search", lambda *args: pytest.fail("Unexpected provider fallback"))
    result = TOOL_REGISTRY[provider]().call({"query": ["test"]})
    assert "401" in result["results"]
    assert len(seen) == 1
    assert seen[0][1] == "synthetic-invalid-key"


def test_agent_tool_selection_is_unchanged(monkeypatch):
    from types import SimpleNamespace
    from skyrl_agent.agents.react import ReActAgent

    monkeypatch.setattr(requests.Session, "request", lambda *args, **kwargs: pytest.fail("Unexpected request"))
    agent = SimpleNamespace(tools={}, tool_params=[])
    ReActAgent._register_tools(agent, ["finish"])
    assert set(agent.tools) == {"finish"}
    ReActAgent._register_tools(agent, ["parallel_search_engine"])
    assert set(agent.tools) == {"finish", "parallel_search_engine"}
    assert agent.tool_params[-1]["function"]["name"] == "parallel_search_engine"
    with pytest.raises(ValueError, match="Unknown tool"):
        ReActAgent._register_tools(agent, ["unregistered_search"])
