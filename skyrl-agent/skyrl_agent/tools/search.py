"""
Dense retrieval search tool using FAISS.
Ported from BrowseComp-Plus/searcher/searchers/faiss_searcher.py
"""

import glob
import json
import logging
import os
import pickle
import threading
from itertools import chain
from typing import Any, Dict, List, Optional, Union

import faiss
import numpy as np
import requests
from datasets import load_dataset
from tevatron.retriever.searcher import FaissFlatSearcher
from tqdm import tqdm

from skyrl_agent.tools.base import BaseTool, register_tool

logger = logging.getLogger(__name__)


@register_tool("faiss_search")
class FaissSearch(BaseTool):
    name = "faiss_search"
    description = "Performs dense retrieval search using FAISS: supply a 'query' string; the tool retrieves the top 5 most relevant documents from a local corpus. <IMPORTANT>: You may need to adjust your queries or split them into sub-queries for better results."
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search query string for semantic retrieval"}},
        "required": ["query"],
    }

    # Class-level cache for shared resources
    _retriever = None
    _lookup = None
    _docid_to_text = None
    _snippet_tokenizer = None
    _initialized = False
    _init_lock = threading.Lock()  # Thread-safe initialization

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)

        # Required configuration from environment variables
        self.embedding_api_url = os.getenv("FAISS_EMBEDDING_API_URL")  # e.g., "http://remote-server:8000/v1"
        self.index_path = os.getenv("FAISS_INDEX_PATH")

        # Use hardcoded defaults (matching BrowseComp-Plus)
        self.dataset_name = "Tevatron/browsecomp-plus-corpus"
        self.task_prefix = (
            "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
        )
        self.k = 5  # Fixed k value as in BrowseComp-Plus
        self.snippet_max_tokens = 512

        # Validate required environment variables
        if not self.embedding_api_url:
            raise ValueError(
                "FAISS_EMBEDDING_API_URL environment variable is required (e.g., 'http://remote-server:8000/v1')"
            )
        if not self.index_path:
            raise ValueError("FAISS_INDEX_PATH environment variable is required")

        logger.info(f"FAISS search tool created (lazy init, remote API: {self.embedding_api_url})")

    def _ensure_initialized(self):
        """Lazy initialization - load resources only once on first use (thread-safe)."""
        # Fast path: check without lock first (double-checked locking pattern)
        if FaissSearch._initialized:
            return

        # Slow path: acquire lock and check again
        with FaissSearch._init_lock:
            # Another thread might have initialized while we were waiting for the lock
            if FaissSearch._initialized:
                logger.info("FAISS resources already initialized by another thread")
                return

            logger.info("Initializing FAISS resources (first use)...")

            # Load FAISS index
            self._load_faiss_index()

            # Load dataset
            self._load_dataset()

            # Skip tokenizer loading - using character-based truncation instead for speed
            FaissSearch._snippet_tokenizer = None
            logger.info("Using character-based snippet truncation (faster than tokenizer)")

            FaissSearch._initialized = True
            logger.info("FAISS search tool initialized successfully")

    def _load_faiss_index(self) -> None:
        """Load FAISS index from pickle files into class-level cache."""

        def pickle_load(path):
            with open(path, "rb") as f:
                reps, lookup = pickle.load(f)
            return np.array(reps), lookup

        index_files = glob.glob(self.index_path)
        logger.info(f"Found {len(index_files)} index files matching pattern: {self.index_path}")

        if not index_files:
            raise ValueError(f"No files found matching pattern: {self.index_path}")

        # Load first shard
        p_reps_0, p_lookup_0 = pickle_load(index_files[0])
        FaissSearch._retriever = FaissFlatSearcher(p_reps_0)

        # Load remaining shards
        shards = chain([(p_reps_0, p_lookup_0)], map(pickle_load, index_files[1:]))
        if len(index_files) > 1:
            shards = tqdm(shards, desc="Loading index shards", total=len(index_files))

        FaissSearch._lookup = []
        for p_reps, p_lookup in shards:
            FaissSearch._retriever.add(p_reps)
            FaissSearch._lookup += p_lookup

        self._setup_gpu()

    def _setup_gpu(self) -> None:
        """Setup GPU for FAISS if available."""
        num_gpus = faiss.get_num_gpus()
        if num_gpus == 0:
            logger.info("No GPU found. Using CPU for FAISS.")
        else:
            logger.info(f"Using {num_gpus} GPU(s) for FAISS")
            if num_gpus == 1:
                co = faiss.GpuClonerOptions()
                co.useFloat16 = True
                res = faiss.StandardGpuResources()
                FaissSearch._retriever.index = faiss.index_cpu_to_gpu(res, 0, FaissSearch._retriever.index, co)
            else:
                co = faiss.GpuMultipleClonerOptions()
                co.shard = True
                co.useFloat16 = True
                FaissSearch._retriever.index = faiss.index_cpu_to_all_gpus(
                    FaissSearch._retriever.index, co, ngpu=num_gpus
                )

    def _load_dataset(self) -> None:
        """Load the document dataset into class-level cache."""
        logger.info(f"Loading dataset: {self.dataset_name}")

        try:
            dataset_cache = os.getenv("HF_DATASETS_CACHE")
            cache_dir = dataset_cache if dataset_cache else None

            ds = load_dataset(self.dataset_name, split="train", cache_dir=cache_dir)
            FaissSearch._docid_to_text = {row["docid"]: row["text"] for row in ds}
            logger.info(f"Loaded {len(FaissSearch._docid_to_text)} documents from dataset")
        except Exception as e:
            if "doesn't exist on the Hub or cannot be accessed" in str(e):
                logger.error(f"Dataset '{self.dataset_name}' access failed. This is likely an authentication issue.")
                logger.error("Solutions:")
                logger.error("1. Run: huggingface-cli login")
                logger.error("2. Set: export HF_TOKEN=your_token_here")
                logger.error(f"Current HF_TOKEN: {'Set' if os.getenv('HF_TOKEN') else 'Not set'}")
            raise RuntimeError(f"Failed to load dataset '{self.dataset_name}': {e}")

    def _get_embedding_from_api(self, text: str) -> np.ndarray:
        """Get embedding from remote API using OpenAI-compatible interface."""
        import time

        start_time = time.time()

        url = f"{self.embedding_api_url}/embeddings"
        headers = {"Content-Type": "application/json"}

        # Get the actual model name from environment or use the served model
        # vLLM requires the actual model name (e.g., "Qwen/Qwen3-Embedding-8B")
        model_name = os.getenv("FAISS_EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-8B")

        payload = {"input": text, "model": model_name}

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)  # Increased to 120s
            response.raise_for_status()

            result = response.json()
            embedding = result["data"][0]["embedding"]

            elapsed = time.time() - start_time
            logger.info(f"[TIMING] Embedding API request took {elapsed:.3f}s (query length: {len(text)} chars)")

            return np.array(embedding, dtype=np.float32)

        except requests.exceptions.Timeout as e:
            logger.error(f"Embedding API request timed out after 120s: {e}")
            logger.error(f"API URL: {self.embedding_api_url}")
            raise RuntimeError("Embedding API timeout (120s) - server may be overloaded or model is slow")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Cannot connect to embedding API: {e}")
            logger.error(f"API URL: {self.embedding_api_url}")
            logger.error("Check if vLLM server is running and accessible")
            raise RuntimeError(f"Embedding API connection failed: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Embedding API request failed: {e}")
            logger.error(f"API URL: {self.embedding_api_url}")
            raise RuntimeError(f"Embedding API request failed: {e}")

    def _search(self, query: str) -> List[Dict[str, Any]]:
        """Perform FAISS search using remote embedding API."""
        import time

        total_start = time.time()

        # Lazy initialization on first use
        self._ensure_initialized()

        if not FaissSearch._retriever or not FaissSearch._lookup:
            raise RuntimeError("Search tool not properly initialized")

        # Get embedding from remote API
        query_with_prefix = self.task_prefix + query
        q_reps = self._get_embedding_from_api(query_with_prefix)
        # Reshape to match expected format [1, embedding_dim]
        q_reps = q_reps.reshape(1, -1)

        # Search with fixed k
        faiss_start = time.time()
        all_scores, psg_indices = FaissSearch._retriever.search(q_reps, self.k)
        faiss_elapsed = time.time() - faiss_start
        logger.info(f"[TIMING] FAISS search took {faiss_elapsed:.3f}s")

        # Format results
        format_start = time.time()
        results = []
        for score, index in zip(all_scores[0], psg_indices[0]):
            passage_id = FaissSearch._lookup[index]
            passage_text = FaissSearch._docid_to_text.get(passage_id, "Text not found")
            # Simple character-based truncation instead of tokenizer (much faster)
            snippet = passage_text[:2000] if len(passage_text) > 2000 else passage_text

            results.append(
                {
                    "docid": passage_id,
                    "score": float(score),
                    "snippet": snippet,
                    "text": passage_text,  # Include full text as well
                }
            )

        format_elapsed = time.time() - format_start
        total_elapsed = time.time() - total_start
        logger.info(f"[TIMING] Result formatting took {format_elapsed:.3f}s")
        logger.info(f"[TIMING] Total search took {total_elapsed:.3f}s")

        return results

    def call(self, params: dict, **kwargs) -> Union[str, dict]:
        """
        Execute search query.

        Args:
            params (dict): Dictionary containing 'query' (string).
            **kwargs: Additional keyword arguments.

        Returns:
            dict: Search results or error message.
        """
        try:
            params = self._verify_json_format_args(params)
        except ValueError as e:
            return {"error": f"Invalid parameters: {str(e)}"}

        query = params.get("query")
        if not query:
            return {"error": "Query parameter is required"}

        logger.info(
            f"[SEARCH] Starting search for query: {query[:100]}..."
            if len(query) > 100
            else f"[SEARCH] Starting search for query: {query}"
        )

        try:
            results = self._search(query)

            # Format for display
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append(
                    {"rank": i, "docid": result["docid"], "score": result["score"], "snippet": result["snippet"]}
                )

            logger.info(f"[SEARCH] Completed search successfully, returned {len(formatted_results)} results")

            return {"query": query, "num_results": len(formatted_results), "results": formatted_results}

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return {"error": f"Search failed: {str(e)}"}


if __name__ == "__main__":
    # Example usage for testing
    # Make sure to set environment variables first:
    # export FAISS_MODEL_NAME="your-model-name"
    # export FAISS_INDEX_PATH="/path/to/index/*.pkl"

    tool = FaissSearch()
    test_params = {"query": "What is machine learning?"}
    result = tool.call(test_params)
    print("Test Result:")
    print(json.dumps(result, indent=2))
