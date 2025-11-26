# From https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/llm_agent/generation.py
# Erro Handling from https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl-gym/skyrl_gym/tools/search.py

from skyrl_agent.tools.base import BaseTool, register_tool
import requests
import os
import time
import uuid
import json
from typing import Union, List, Tuple, Optional, Dict, Any

# Constants for retry logic
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 1


@register_tool("local_search")
class LocalSearchTool(BaseTool):
    name = "local_search"
    description = "Performs a single query with a local search engine, use this if you need more information."
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "The query to search for."}},
        "required": ["query"],
    }

    def __init__(self):
        super().__init__()
        # Check for required environment variable
        assert "LOCAL_SEARCH_URL" in os.environ, "Environment variable LOCAL_SEARCH_URL must be set."
        self.search_url = os.getenv("LOCAL_SEARCH_URL")
        self.topk = int(os.getenv("LOCAL_SEARCH_TOP_K", "3"))
        self.timeout = DEFAULT_TIMEOUT
        self.log_requests = True

    def _call_search_api(self, query: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Calls the search API with a single query using improved error handling.

        Args:
            query: The query to search for.

        Returns:
            response: The response from the search API (json if successful, None otherwise)
            error_msg: The error message if the request failed.
        """
        request_id = str(uuid.uuid4())
        log_prefix = f"[Search Request ID: {request_id}] "

        payload = {"queries": [query], "topk": self.topk, "return_scores": True}
        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                if self.log_requests:
                    print(f"{log_prefix}Attempt {attempt + 1}/{MAX_RETRIES}: Calling search API at {self.search_url}")

                response = requests.post(
                    self.search_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )

                # Check for server errors that should trigger retries
                if response.status_code in [500, 502, 503, 504]:
                    last_error = f"{log_prefix}API Request Error: Server Error ({response.status_code}) on attempt {attempt + 1}/{MAX_RETRIES}"
                    print(f"Warning: {last_error}")
                    if attempt < MAX_RETRIES - 1:
                        delay = INITIAL_RETRY_DELAY * (attempt + 1)
                        print(f"{log_prefix}Retrying after {delay} seconds...")
                        time.sleep(delay)
                    continue

                # Check for other HTTP errors
                response.raise_for_status()

                # If successful (status code 2xx)
                if self.log_requests:
                    print(f"{log_prefix}Search API call successful on attempt {attempt + 1}")

                return response.json(), None

            except requests.exceptions.ConnectionError as e:
                last_error = f"{log_prefix}Connection Error: {e}"
                print(f"Warning: {last_error}")
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    print(f"{log_prefix}Retrying after {delay} seconds...")
                    time.sleep(delay)
                continue
            except requests.exceptions.Timeout as e:
                last_error = f"{log_prefix}Timeout Error: {e}"
                print(f"Warning: {last_error}")
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    print(f"{log_prefix}Retrying after {delay} seconds...")
                    time.sleep(delay)
                continue
            except requests.exceptions.RequestException as e:
                last_error = f"{log_prefix}API Request Error: {e}"
                break  # Exit retry loop on other request errors
            except json.JSONDecodeError as e:
                raw_response_text = response.text if "response" in locals() else "N/A"
                last_error = f"{log_prefix}API Response JSON Decode Error: {e}, Response: {raw_response_text[:200]}"
                break  # Exit retry loop on JSON decode errors
            except Exception as e:
                last_error = f"{log_prefix}Unexpected Error: {e}"
                break  # Exit retry loop on other unexpected errors

        # If we reach here, all attempts failed
        print(f"Error: {log_prefix}API Request Failed after {MAX_RETRIES} attempts: {last_error}")
        return None, last_error

    def _passages2string(self, retrieval_result: List[dict]) -> str:
        """
        Convert search results to formatted string representation.

        Args:
            retrieval_result: List of document results from search API

        Returns:
            Formatted string with document titles and content
        """
        if not retrieval_result:
            return "No search results found."

        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item["document"]["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference.strip()

    def call(self, params: dict, **kwargs) -> Union[str, list, dict]:
        """
        Main entry point for the search tool with improved error handling.

        Args:
            params (dict): Dictionary containing 'query' parameter.
            **kwargs: Additional keyword arguments.

        Returns:
            Formatted search results as string or error message.
        """
        # Verify required parameters
        try:
            params = self._verify_json_format_args(params)
        except ValueError as e:
            return {"error": f"Invalid parameters: {str(e)}"}

        query = params.get("query")

        # Execute search
        try:
            api_response, error_msg = self._call_search_api(query)

            if error_msg:
                return f"Search error: {error_msg}"

            if not api_response:
                return "No response from search API"

            if "result" not in api_response:
                return "Unexpected response format from search API"

            # Convert results to formatted string
            assert len(api_response["result"]) == 1, "Expected 1 result, got " + str(len(api_response["result"]))
            formatted_result = self._passages2string(api_response["result"][0])
            return formatted_result

        except Exception as e:
            error_msg = f"Unexpected error processing query '{query}': {str(e)}"
            return error_msg


if __name__ == "__main__":
    # Example usage for testing
    tool = LocalSearchTool()

    # 10 random test parameters
    test_queries = [
        "artificial intelligence machine learning",
        "quantum computing applications",
        "climate change renewable energy",
        "blockchain cryptocurrency bitcoin",
        "space exploration mars mission",
        "genetic engineering CRISPR technology",
        "virtual reality augmented reality",
        "cybersecurity data protection",
        "robotics automation manufacturing",
        "neural networks deep learning",
    ]

    # Import tokenizer
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
        print("Successfully loaded Qwen2.5-7B tokenizer")
    except ImportError:
        print("transformers library not available, cannot count tokens")
        tokenizer = None
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        tokenizer = None

    total_tokens = 0
    results = []

    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test {i}/10 ---")
        print(f"Query: {query}")

        test_params = {"query": query}
        result = tool.call(test_params)

        if tokenizer:
            # Count tokens in the result
            tokens = tokenizer.encode(str(result), add_special_tokens=False)
            token_count = len(tokens)
            total_tokens += token_count
            print(f"Result tokens: {token_count}")
        else:
            print("Cannot count tokens - tokenizer not available")

        results.append({"query": query, "result": result, "tokens": token_count if tokenizer else 0})

        print(f"Result preview: {str(result)[:200]}...")

    print("\n=== SUMMARY ===")
    print(f"Total queries tested: {len(test_queries)}")
    if tokenizer:
        print(f"Total tokens used: {total_tokens}")
        print(f"Average tokens per result: {total_tokens / len(test_queries):.2f}")

        # Show token distribution
        print("\nToken distribution per query:")
        for i, result in enumerate(results, 1):
            print(f"Query {i}: {result['tokens']} tokens")
    else:
        print("Token counting not available")
