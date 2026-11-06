"""Opt-in anonymous web search through Parallel's Streamable HTTP MCP endpoint."""

import json
import logging

import requests

from skyrl_agent.tools.base import BaseTool, register_tool, normalize_search_params
from skyrl_agent.tools.youcom_search_engine import YouComSearchEngine, _SKYRL_AGENT_VERSION


logger = logging.getLogger(__name__)


class _ParallelSession:
    """A request-scoped MCP session; only web_search is exposed to the agent."""

    endpoint = "https://mcp.parallel.ai/mcp"

    def __init__(self):
        self.http = requests.Session()
        self.http.headers.update(
            {"User-Agent": f"SkyRL-Agent/{_SKYRL_AGENT_VERSION}", "Accept": "application/json, text/event-stream"}
        )
        self.request_id = 0

    @staticmethod
    def _events(response):
        # Parse bytes before UTF-8 decoding: Unicode separators inside JSON are not SSE line endings.
        line = bytearray()
        data = []
        after_cr = False
        # read1 returns available data without waiting to fill a buffer on non-chunked SSE.
        read1 = getattr(response.raw, "read1", None)
        chunks = iter(lambda: read1(4096, decode_content=True), b"") if read1 else response.iter_content(chunk_size=1)
        for chunk in chunks:
            for byte in chunk:
                if after_cr and byte == 10:
                    after_cr = False
                    continue
                after_cr = byte == 13
                if byte not in (10, 13):
                    line.append(byte)
                    continue
                if not line and data:
                    yield json.loads(b"\n".join(data))
                    data = []
                elif line.startswith(b"data:"):
                    data.append(bytes(line[5:]).removeprefix(b" "))
                line.clear()

    def rpc(self, method, params=None, notify=False):
        self.request_id += 1
        message = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        if not notify:
            message["id"] = self.request_id
        with self.http.post(self.endpoint, json=message, stream=True, timeout=(10, 60)) as response:
            response.raise_for_status()
            if method == "initialize" and response.headers.get("Mcp-Session-Id"):
                self.http.headers["Mcp-Session-Id"] = response.headers["Mcp-Session-Id"]
            if notify:
                return None
            media_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if media_type == "application/json":
                messages = [response.json()]
            elif media_type == "text/event-stream":
                messages = self._events(response)
            else:
                raise ValueError(f"Unsupported MCP response type: {media_type}")
            for reply in messages:
                if reply.get("id") != message["id"]:
                    continue
                if "error" in reply:
                    error = reply["error"]
                    raise ValueError(f"MCP error {error['code']}: {error['message']}")
                return reply["result"]
            raise ValueError("MCP response ended without a matching result")

    def search(self, queries):
        try:
            initialized = self.rpc(
                "initialize",
                {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "SkyRL-Agent", "version": _SKYRL_AGENT_VERSION},
                },
            )
            self.http.headers["MCP-Protocol-Version"] = initialized["protocolVersion"]
            self.rpc("notifications/initialized", notify=True)
            tools = self.rpc("tools/list")
            if not any(tool["name"] == "web_search" for tool in tools["tools"]):
                raise ValueError("Parallel MCP does not expose web_search")
            return self.rpc(
                "tools/call",
                {"name": "web_search", "arguments": {"objective": "\n".join(queries), "search_queries": queries}},
            )
        finally:
            try:
                if "Mcp-Session-Id" in self.http.headers:
                    with self.http.delete(self.endpoint, timeout=10) as response:
                        # Some Streamable HTTP servers do not support explicit session termination.
                        if response.status_code not in (404, 405):
                            response.raise_for_status()
            except Exception:
                # Remote cleanup must not replace the search result or its original error.
                logger.warning("Parallel MCP session termination failed; preserving the search outcome.")
            finally:
                self.http.close()


@register_tool("parallel_search_engine")
class ParallelSearchEngine(BaseTool):
    """Select with tools: [parallel_search_engine, finish]. No API key or inference is needed."""

    name = "parallel_search_engine"
    description = "Search the web anonymously using Parallel. Supply an array of related query strings."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1},
        },
        "required": ["query"],
    }

    # Keep the same leakage protections and environment controls as the adjacent web search tool.
    blocklist_domains = YouComSearchEngine.blocklist_domains
    blocklist_keywords = YouComSearchEngine.blocklist_keywords
    _use_negative_filters = YouComSearchEngine._use_negative_filters
    _process_results = YouComSearchEngine._process_results

    def call(self, params, **kwargs):
        try:
            params = self._verify_json_format_args(normalize_search_params(params))
            queries = params["query"]
            if any(not query.strip() for query in queries):
                raise ValueError("Queries must not be blank")
        except Exception as error:
            return {"error": f"Invalid parameters: {error}"}

        filtered_queries = queries
        if self._use_negative_filters and self.blocklist_domains:
            exclusions = " ".join(f"-site:{domain}" for domain in sorted(self.blocklist_domains))
            filtered_queries = [f"{query} {exclusions}" for query in queries]
        try:
            result = _ParallelSession().search(filtered_queries)
            if result.get("isError"):
                raise ValueError("; ".join(item.get("text", "") for item in result.get("content", [])))
            payload = result.get("structuredContent")
            if payload is None:
                payload = json.loads("\n".join(item["text"] for item in result["content"] if item["type"] == "text"))
            pages = [
                {"url": page["url"], "title": page.get("title") or page["url"], "snippets": page["excerpts"]}
                for page in payload["results"]
            ]
            pages = self._process_results(pages)
            snippets = [
                f"{index}. [{page['title']}]({page['url']})\n" + "\n".join(page["snippets"])
                for index, page in enumerate(pages, 1)
            ]
            return {"results": f"Parallel search found {len(pages)} results:\n\n" + "\n\n".join(snippets)}
        except Exception as error:
            return {"error": f"Parallel search failed: {error}"}
