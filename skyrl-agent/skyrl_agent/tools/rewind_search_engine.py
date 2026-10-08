from skyrl_agent.tools.base import BaseTool, register_tool, json_loads
from importlib.metadata import version
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from functools import partial
from typing import Optional, Union
import json
import logging
import os
import requests


try:
    _SKYRL_AGENT_VERSION = version("skyrl_agent")
except Exception:
    _SKYRL_AGENT_VERSION = "0.0.1"  # fallback version


@register_tool("rewind_search_engine")
class RewindSearchEngine(BaseTool):
    """
    A tool that performs batched web searches over the web as it was on a past UTC day, using Linkup Rewind.
    Matching and ranking only use pages crawled by that day, so results cannot reveal later events or
    later copies of a benchmark's answers. This replaces domain/keyword blocklists for leakage control.

    Requires the LINKUP_API_KEY environment variable. The day comes from the instance field named by
    REWIND_AS_OF_FIELD (default "as_of"), falling back to REWIND_AS_OF (YYYY-MM-DD) for the whole run,
    e.g. a date before the evaluation benchmark was published. Example of adding as a tool:
    tools: ["rewind_search_engine", "finish"]
    To run this tool standalone:
    LINKUP_API_KEY="<your_api_key>" REWIND_AS_OF="2025-01-01" python -m skyrl_agent.tools.rewind_search_engine
    """

    name = "rewind_search_engine"
    description = (
        "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call, "
        "including the text of each page.\n\n"
        'For search_engine, query must be JSON array: ["term1", "term2"] NOT [term1, term2] or [["term1", "term2"]]'
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Array of query strings. Include multiple complementary search queries in a single call.\n\n"
                    'For search_engine, query must be JSON array: ["term1", "term2"] NOT [term1, term2] or [["term1", "term2"]]'
                ),
            },
        },
        "required": ["query"],
    }

    def __init__(self):
        super().__init__()
        self.linkup_api_key = os.getenv("LINKUP_API_KEY")
        if not self.linkup_api_key:
            raise ValueError("LINKUP_API_KEY environment variable is required")
        self.default_as_of = os.getenv("REWIND_AS_OF")
        self.as_of_field = os.getenv("REWIND_AS_OF_FIELD", "as_of")
        self.num_results = int(os.getenv("REWIND_NUM_RESULTS", "10"))
        self.content_chars = int(os.getenv("REWIND_CONTENT_CHARS", "2000"))

    def _resolve_as_of(self, agent) -> Optional[str]:
        instance = getattr(agent, "instance", None)
        value = instance.get(self.as_of_field) if isinstance(instance, dict) else None
        value = value or self.default_as_of
        if not value:
            return None
        return date.fromisoformat(str(value)[:10]).isoformat()

    def rewind_search(self, query: str, as_of: str):
        """
        Performs a search using the Linkup Rewind API.

        Args:
            query (str): The search query string
            as_of (str): The UTC day (YYYY-MM-DD) to search the web as of

        Returns:
            str: Formatted search results or error message
        """
        url = "https://api.linkup.so/v1/rewind/search"
        headers = {
            "Authorization": f"Bearer {self.linkup_api_key}",
            "Content-Type": "application/json",
            "user-agent": f"SkyRL-Agent/{_SKYRL_AGENT_VERSION}",
        }
        data = {"q": query, "asOf": as_of}

        for i in range(5):
            try:
                response = requests.post(url, headers=headers, data=json.dumps(data), timeout=30)
                results = response.json()
                break
            except requests.exceptions.RequestException as re:
                if i == 4:
                    return f"Rewind search encountered error {re} for query '{query}'."
                continue
            except ValueError:
                return f"Search API error: {response.status_code} - {response.text}"

        if response.status_code != 200:
            return f"Search API error: {response.status_code} - {response.text}"

        try:
            pages = results.get("results", [])[: self.num_results]
            if not pages:
                return f"No results found for query: '{query}'. Use a less specific query."

            web_snippets = []
            for idx, page in enumerate(pages, start=1):
                # Only include title, link, and page text; omit capture dates and other metadata
                text = (page.get("content") or "")[: self.content_chars]
                web_snippets.append(f"{idx}. [{page.get('name', '')}]({page.get('url', '')})\n\n{text}")

            content = (
                f"A web search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n"
                + "\n\n".join(web_snippets)
            )
            return content
        except Exception as e:
            logging.warning(f"Error parsing Rewind results: {e}", exc_info=True)
            return f"Error parsing search results for '{query}': {str(e)}"

    def call(self, params: dict, **kwargs) -> Union[str, dict]:
        """
        Executes web search queries.

        Args:
            params (dict): Dictionary containing 'query' (array of strings or single string).
            **kwargs: Additional keyword arguments; 'agent' provides the instance used to resolve the date.

        Returns:
            str or dict: The search results or an error message.
        """
        # Normalize and validate parameters robustly
        # 1) Parse to dict (be tolerant of JSON5/markdowny inputs)
        raw: dict
        if isinstance(params, dict):
            raw = dict(params)
        else:
            try:
                raw = json_loads(params) if isinstance(params, str) else {"query": params}
            except Exception:
                raw = {"query": params}
        if not isinstance(raw, dict):
            raw = {"query": raw}

        # 2) Normalize "query" into a flat list[str]
        def _normalize_query(q):
            if q is None:
                return None
            # If it's a JSON-like string representing an array, try to parse
            if isinstance(q, str):
                s = q.strip()
                if s.startswith("[") and s.endswith("]"):
                    try:
                        parsed = json.loads(s)
                        q = parsed
                    except Exception:
                        # fall back to single string query
                        return [q]
                else:
                    return [q]
            # If it's a list, flatten nested lists and stringify items
            if isinstance(q, list):
                flat = []
                for item in q:
                    if isinstance(item, list):
                        flat.extend([str(x) for x in item])
                    else:
                        flat.append(str(item))
                return flat
            # Any other type → cast to single string element
            return [str(q)]

        normalized_query = _normalize_query(raw.get("query"))
        if not normalized_query:
            return {
                "error": "Query parameter is required.",
                "hint": "Provide a JSON array of search strings.",
                "example": {"query": ["term1", "term2"]},
            }

        raw["query"] = normalized_query

        # 3) Schema-validate, but catch any schema error and return actionable hint
        try:
            params = self._verify_json_format_args(raw)
        except Exception as e:
            return {
                "error": f"Invalid parameters: {str(e)}",
                "hint": "query must be an array of strings (no nested arrays).",
                "example": {"query": ["term1", "term2"]},
            }

        # 4) Never fall back to the live web: a missing or invalid date is a configuration error
        try:
            as_of = self._resolve_as_of(kwargs.get("agent"))
        except ValueError as e:
            return {"error": f"Invalid Rewind date: {str(e)}"}
        if not as_of:
            return {"error": f"Set REWIND_AS_OF or the instance field '{self.as_of_field}' to a YYYY-MM-DD date."}

        query = params.get("query")

        try:
            with ThreadPoolExecutor(max_workers=3) as executor:
                response = list(executor.map(partial(self.rewind_search, as_of=as_of), query))
            response = "\n=======\n".join(response)
            return {"results": response}

        except Exception as e:
            return {"error": f"Search failed: {str(e)}"}


if __name__ == "__main__":
    # Example usage for testing
    tool = RewindSearchEngine()
    test_params = {"query": ["python programming", "machine learning"]}
    result = tool.call(test_params)
    print("Test Result:", result)
