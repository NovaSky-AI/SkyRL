from skyrl_agent.tools.base import BaseTool, register_tool, json_loads
from importlib.metadata import version
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Union
import requests
import os
import logging


try:
    _SKYRL_AGENT_VERSION = version("skyrl_agent")
except Exception:
    _SKYRL_AGENT_VERSION = "0.0.1"  # fallback version


@register_tool("youcom_search_engine")
class YouComSearchEngine(BaseTool):
    """
    A tool that performs batched web searches using the YouCom API.
    In order to use this tool, you need to set the YDC_API_KEY environment variable. Example of adding to as a tool:
    tools: ["youcom_search_engine", "web_browser", "finish"]
    Also, if you want to run this tool standalone, run it as: YDC_API_KEY="<your_api_key>" python -m skyrl_agent.tools.youcom_search_engine
    """

    def __init__(self):
        super().__init__()
        self.you_search_key = os.getenv("YDC_API_KEY")
        if not self.you_search_key:
            raise ValueError("YDC_API_KEY environment variable is required")

    name = "youcom_search_engine"
    description = (
        "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call.\n\n"
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

    # Optional blocklists to prevent data leakage (e.g., excluding benchmark/dataset sites)
    # Configure via env vars:
    #   SEARCH_BLOCKLIST_DOMAINS: comma-separated domains to exclude (e.g., "huggingface.co,github.com")
    #   SEARCH_BLOCKLIST_KEYWORDS: comma-separated keywords to filter out (e.g., "gpqa,web_research_hle")
    #   SEARCH_NEGATIVE_FILTERS: if "true" (default), append -site:domain to the query for blocklisted domains
    _default_block_domains = os.getenv(
        "SEARCH_BLOCKLIST_DOMAINS",
        # Sensible defaults to avoid benchmark leakage and paywalled homework sites
        "huggingface.co,github.com,gitlab.com,chegg.com,coursehero.com,studocu.com,brainly.com,quizlet.com",
    ).strip()
    _default_block_keywords = os.getenv(
        "SEARCH_BLOCKLIST_KEYWORDS",
        # Common benchmark/dataset keywords observed in HLE/GPQA scenarios
        "gpqa,Chemistry-GPQA,gpqa-diamond",
    ).strip()
    _use_negative_filters = os.getenv("SEARCH_NEGATIVE_FILTERS", "true").lower() == "true"

    # Normalize to sets for faster checks
    blocklist_domains = {d.strip().lower() for d in _default_block_domains.split(",") if d.strip()}
    blocklist_keywords = {k.strip().lower() for k in _default_block_keywords.split(",") if k.strip()}

    def _process_results(self, results: list):
        filtered_pages = []
        for p in results:
            try:
                link = str(p.get("url", ""))
                title = str(p.get("title", ""))
                snippet = str(p.get("snippets", ""))
                link_l = link.lower()
                combined_l = f"{title} {snippet} {link}".lower()
                # Domain filtering based on URL substring match
                blocked_domain = any(d in link_l for d in self.blocklist_domains) if self.blocklist_domains else False
                # Keyword filtering anywhere in title/snippet/link
                blocked_keyword = (
                    any(k in combined_l for k in self.blocklist_keywords) if self.blocklist_keywords else False
                )
                if not blocked_domain and not blocked_keyword:
                    filtered_pages.append(p)
            except Exception as e:
                # If anything goes wrong during filtering, conservatively keep the page
                logging.warning(f"Error processing a search result: {e}", exc_info=True)
                filtered_pages.append(p)
        return filtered_pages

    def you_search(self, query: str):
        """
        Performs a search using the YouCom API.

        Args:
            query (str): The search query string

        Returns:
            str: Formatted search results or error message
        """
        url = "https://api.ydc-index.io/v1/search"
        headers = {
            "X-API-KEY": self.you_search_key,
            "Content-Type": "application/json",
            "user-agent": f"SkyRL-Agent/{_SKYRL_AGENT_VERSION}",
        }
        # Optionally add negative site filters directly to the query
        q = query
        if self._use_negative_filters and self.blocklist_domains:
            try:
                q = query + " " + " ".join([f"-site:{d}" for d in sorted(self.blocklist_domains)])
            except Exception:
                q = query

        request_params = {
            "query": q,
            "count": 10,
        }

        for i in range(5):
            try:
                response = requests.get(url, headers=headers, params=request_params, timeout=10)
                results = response.json()
                break
            except requests.exceptions.RequestException as re:
                if i == 4:
                    return f"YouSearchEngine search encountered error {re} for query '{query}'."
                continue

        if response.status_code != 200:
            return f"Search API error: {response.status_code} - {response.text}"

        try:
            if "results" not in results:
                return f"No results found for query: '{query}'. Use a less specific query."

            # Filter results by blocklists (domains/keywords) to reduce leakage
            results = results.get("results", [])
            filtered_pages = self._process_results(results.get("web", []))
            filtered_pages.extend(self._process_results(results.get("news", [])))

            web_snippets = []
            idx = 0
            for page in filtered_pages:
                idx += 1
                # Only include title, link, and snippet; omit publish date/source/citations metadata
                snippet = ""
                if "snippets" in page:
                    snippet = "\n" + "\n".join(page["snippets"])

                redacted_version = f"{idx}. [{page['title']}]({page['url']})\n{snippet}"
                redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                web_snippets.append(redacted_version)

            content = f"YDC search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(
                web_snippets
            )
            return content
        except Exception as e:
            return f"Error parsing search results for '{query}': {str(e)}"

    def call(self, params: dict, **kwargs) -> Union[str, dict]:
        """
        Executes web search queries.

        Args:
            params (dict): Dictionary containing 'query' (array of strings or single string).
            **kwargs: Additional keyword arguments.

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

        query = params.get("query")

        try:
            with ThreadPoolExecutor(max_workers=3) as executor:
                response = list(executor.map(self.you_search, query))
            response = "\n=======\n".join(response)
            return {"results": response}

        except Exception as e:
            return {"error": f"Search failed: {str(e)}"}


if __name__ == "__main__":
    # Example usage for testing
    tool = YouComSearchEngine()
    test_params = {"query": ["python programming", "machine learning"]}
    result = tool.call(test_params)
    print("Test Result:", result)
