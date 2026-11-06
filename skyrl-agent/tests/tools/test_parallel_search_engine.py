"""Synthetic protocol and registry compatibility checks; no live provider calls."""

import json
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import requests

from skyrl_agent.tools import TOOL_REGISTRY
from skyrl_agent.tools.parallel_search_engine import ParallelSearchEngine, _ParallelSession


PASSAGE = "alpha\u0085beta\u2028gamma\u2029delta"


@pytest.mark.parametrize("chunked", [True, False])
@pytest.mark.parametrize("failure", [None, "rpc", "http", "connection"])
@pytest.mark.parametrize("teardown_failure", [None, "http", "connection"])
def test_held_open_transport(monkeypatch, caplog, failure, teardown_failure, chunked):
    requests_seen = []
    local_closes = []
    release = threading.Event()
    closed = threading.Event()
    original_close = requests.Session.close

    def close_session(session):
        local_closes.append(session)
        original_close(session)

    monkeypatch.setattr(requests.Session, "close", close_session)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_DELETE(self):
            requests_seen.append(("DELETE", self.headers))
            if teardown_failure == "connection":
                self.close_connection = True
                self.connection.shutdown(socket.SHUT_RDWR)
                return
            self.send_response(500 if teardown_failure == "http" else 200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            method = request["method"]
            requests_seen.append((method, self.headers))
            if method == "tools/call" and failure == "connection":
                self.close_connection = True
                self.connection.shutdown(socket.SHUT_RDWR)
                return
            if method == "tools/call" and failure == "http":
                self.send_response(502)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
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
            elif failure == "rpc":
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
            notice = b'data: {"jsonrpc":"2.0","method":"notifications/tools/list_changed"}\r\n\r\n'
            events = (notice, b"data: " + body + b"\r\n\r\n")
            if chunked:
                self.send_header("Transfer-Encoding", "chunked")
            else:
                # More body bytes are promised but withheld until the client closes.
                self.send_header("Content-Length", str(sum(map(len, events)) + 8192))
            self.end_headers()
            for event in events:
                self.wfile.write(f"{len(event):x}\r\n".encode() + event + b"\r\n" if chunked else event)
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
        tool = TOOL_REGISTRY["parallel_search_engine"]()
        tool.blocklist_domains = set()
        tool.blocklist_keywords = set()
        tool._use_negative_filters = False
        with ThreadPoolExecutor() as executor:
            result = executor.submit(tool.call, {"query": ["test"]}).result(2)
        if failure == "rpc":
            assert "MCP error -32042: synthetic failure" in result["error"]
        elif failure == "http":
            assert "502" in result["error"]
        elif failure == "connection":
            assert "Connection aborted" in result["error"]
        else:
            assert PASSAGE in result["results"]
            assert "https://example.org/source" in result["results"]
        if failure in (None, "rpc"):
            assert closed.wait(1)
        assert len(local_closes) == 1
        if teardown_failure:
            assert caplog.messages == ["Parallel MCP session termination failed; preserving the search outcome."]
        else:
            assert not caplog.messages
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


@pytest.mark.parametrize("params", [{}, {"query": []}, {"query": [""]}, {"query": [" "]}])
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
    tool.blocklist_domains = {"github.com"}
    tool.blocklist_keywords = {"gpqa"}
    tool._use_negative_filters = True
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


@pytest.mark.parametrize(
    "provider,method",
    [("search_engine", "google_search"), ("youcom_search_engine", "you_search"), ("parallel_search_engine", None)],
)
@pytest.mark.parametrize(
    "params,expected",
    [
        ({"query": "NASA Webb"}, ["NASA Webb"]),
        ("NASA Webb", ["NASA Webb"]),
        ('{query: "NASA Webb"}', ["NASA Webb"]),
        ('```json\n{"query": "NASA Webb"}\n```', ["NASA Webb"]),
        ({"query": '["NASA", "Webb"]'}, ["NASA", "Webb"]),
        ({"query": [["NASA", "Webb"], 42]}, ["NASA", "Webb", "42"]),
        (["NASA", "Webb"], ["NASA", "Webb"]),
    ],
)
def test_search_normalization_contract(monkeypatch, provider, method, params, expected):
    monkeypatch.setenv("GOOGLE_SEARCH_KEY", "synthetic-serper-key")
    monkeypatch.setenv("YDC_API_KEY", "synthetic-you-key")
    seen = []
    tool = TOOL_REGISTRY[provider]()
    tool._use_negative_filters = False
    if method:

        def synthetic_search(query):
            seen.append(query)
            return "synthetic result"

        monkeypatch.setattr(tool, method, synthetic_search)
    else:

        def synthetic_search(self, queries):
            seen.extend(queries)
            return {"structuredContent": {"results": []}}

        monkeypatch.setattr(_ParallelSession, "search", synthetic_search)
    result = tool.call(params)
    assert "error" not in result
    assert sorted(seen) == sorted(expected)


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"structuredContent": {}},
        {"structuredContent": []},
        {"content": []},
        {"structuredContent": {"results": [{}]}},
        None,
    ],
)
def test_malformed_response_returns_error(monkeypatch, response):
    monkeypatch.setattr(_ParallelSession, "search", lambda *args: response)
    assert "error" in ParallelSearchEngine().call({"query": ["test"]})


def test_filter_environment_is_shared_with_sibling():
    import os
    from skyrl_agent.tools.youcom_search_engine import YouComSearchEngine

    tool = ParallelSearchEngine()
    assert tool.blocklist_domains == YouComSearchEngine.blocklist_domains
    assert tool.blocklist_keywords == YouComSearchEngine.blocklist_keywords
    assert tool._use_negative_filters == (os.getenv("SEARCH_NEGATIVE_FILTERS", "true").lower() == "true")
    if "SEARCH_BLOCKLIST_DOMAINS" in os.environ:
        assert tool.blocklist_domains == {
            d.strip().lower() for d in os.environ["SEARCH_BLOCKLIST_DOMAINS"].split(",") if d.strip()
        }
    if "SEARCH_BLOCKLIST_KEYWORDS" in os.environ:
        assert tool.blocklist_keywords == {
            k.strip().lower() for k in os.environ["SEARCH_BLOCKLIST_KEYWORDS"].split(",") if k.strip()
        }
    tool._use_negative_filters = False
    pages = [{"url": f"https://{domain}/data", "title": "", "snippets": []} for domain in tool.blocklist_domains] + [
        {"url": "https://example.org", "title": "", "snippets": [keyword]} for keyword in tool.blocklist_keywords
    ]
    assert tool._process_results(pages) == []


@pytest.mark.parametrize("newline", [b"\r", b"\n", b"\r\n"])
@pytest.mark.parametrize("read1_available", [False, True])
def test_sse_chunk_boundaries(newline, read1_available):
    from types import SimpleNamespace

    message = {"jsonrpc": "2.0", "id": 1, "result": {"passage": PASSAGE}}
    body = b"data: " + json.dumps(message, ensure_ascii=False).encode() + newline + newline
    chunks = iter(body[i : i + 3] for i in range(0, len(body), 3))
    raw = SimpleNamespace(read1=lambda *args, **kwargs: next(chunks, b"")) if read1_available else object()
    response = SimpleNamespace(raw=raw, iter_content=lambda **kwargs: chunks)
    assert list(_ParallelSession._events(response)) == [message]
