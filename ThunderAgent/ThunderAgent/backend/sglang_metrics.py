"""SGLang metrics parsing, storage, and client.

Handles HTTP communication with SGLang endpoints, including:
- Server info lookup for capacity
- Prometheus metrics parsing
- Metrics history management
"""
import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional
import re
import time

import httpx

from .metrics_base import MetricsClient

logger = logging.getLogger(__name__)


@dataclass
class SGLangCacheConfig:
    """Static capacity configuration from SGLang server info."""
    total_tokens_capacity: int = 0


@dataclass
class SGLangMetrics:
    """Parsed metrics from SGLang /metrics endpoint."""
    # Request stats
    num_requests_running: int = 0
    num_requests_waiting: int = 0

    # Cache / usage
    token_usage: float = 0.0
    cache_hit_rate: float = 0.0
    num_used_tokens: int = 0

    # Tokens (cumulative)
    prompt_tokens_total: int = 0
    generation_tokens_total: int = 0

    # Timestamp when metrics were fetched
    timestamp: float = 0.0

    @classmethod
    def from_prometheus_text(cls, text: str) -> "SGLangMetrics":
        """Parse Prometheus text format into SGLangMetrics."""
        metrics = cls(timestamp=time.time())

        def extract_value(pattern: str) -> Optional[float]:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except (ValueError, IndexError):
                    return None
            return None

        # Current request counts
        val = extract_value(r"sglang:num_running_reqs\{[^}]*\}\s+([\d.eE+]+)")
        if val is not None:
            metrics.num_requests_running = int(val)

        val = extract_value(r"sglang:num_queue_reqs\{[^}]*\}\s+([\d.eE+]+)")
        if val is not None:
            metrics.num_requests_waiting = int(val)

        # Cache / usage
        val = extract_value(r"sglang:token_usage\{[^}]*\}\s+([\d.eE+]+)")
        if val is not None:
            metrics.token_usage = val

        val = extract_value(r"sglang:cache_hit_rate\{[^}]*\}\s+([\d.eE+]+)")
        if val is not None:
            metrics.cache_hit_rate = val

        val = extract_value(r"sglang:num_used_tokens\{[^}]*\}\s+([\d.eE+]+)")
        if val is not None:
            metrics.num_used_tokens = int(val)

        # Token counts
        val = extract_value(r"sglang:prompt_tokens_total\{[^}]*\}\s+([\d.eE+]+)")
        if val is not None:
            metrics.prompt_tokens_total = int(val)

        val = extract_value(r"sglang:generation_tokens_total\{[^}]*\}\s+([\d.eE+]+)")
        if val is not None:
            metrics.generation_tokens_total = int(val)

        return metrics


# Keep only the most recent N metrics samples
METRICS_HISTORY_SIZE = 12


class SGLangMetricsClient(MetricsClient):
    """Client for fetching and managing metrics from a SGLang backend.

    Handles HTTP communication with SGLang endpoints,
    metrics history management, and capacity discovery.
    """

    def __init__(self, url: str):
        super().__init__(url)
        self.healthy = True
        self.metrics_history: List[SGLangMetrics] = []
        self.cache_config: Optional[SGLangCacheConfig] = None

        # HTTP client and monitoring state
        self._client: Optional[httpx.AsyncClient] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._monitor_stop = False

    @property
    def metrics_url(self) -> str:
        """Prometheus metrics endpoint."""
        return f"{self.url}/metrics"

    @property
    def server_info_url(self) -> str:
        """SGLang server info endpoint (capacity information)."""
        return f"{self.url}/get_server_info"

    @property
    def latest_metrics(self) -> Optional[SGLangMetrics]:
        """Get the most recent metrics sample."""
        return self.metrics_history[-1] if self.metrics_history else None

    @property
    def is_monitoring(self) -> bool:
        """Check if monitoring is active."""
        return self._monitor_task is not None

    # ---------------------------------------------------------------------
    # Metrics Monitoring
    # ---------------------------------------------------------------------

    async def start_monitoring(self, interval: float = 5.0):
        """Start background metrics monitoring."""
        if self._monitor_task is not None:
            return

        self._monitor_stop = False
        self._client = httpx.AsyncClient(timeout=10.0)

        # Fetch cache config once at startup
        await self.fetch_cache_config()

        self._monitor_task = asyncio.create_task(self._monitor_loop(interval))
        logger.info(
            "Started metrics monitoring for %s (interval: %ss, capacity: %s tokens)",
            self.url,
            interval,
            self.cache_config.total_tokens_capacity if self.cache_config else "unknown",
        )

    async def stop_monitoring(self):
        """Stop background metrics monitoring."""
        if self._monitor_task is None:
            return

        self._monitor_stop = True
        self._monitor_task.cancel()
        try:
            await self._monitor_task
        except asyncio.CancelledError:
            pass
        self._monitor_task = None

        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("Stopped metrics monitoring for %s", self.url)

    async def _monitor_loop(self, interval: float):
        """Background loop to periodically fetch metrics."""
        while not self._monitor_stop:
            try:
                await self.fetch_metrics()
            except Exception as exc:
                logger.debug("Error fetching metrics from %s: %s", self.url, exc)
            await asyncio.sleep(interval)

    # ---------------------------------------------------------------------
    # Metrics Fetching
    # ---------------------------------------------------------------------

    async def fetch_cache_config(self) -> bool:
        """Fetch static capacity config from SGLang server info."""
        client = self._client
        close_client = False

        if client is None:
            client = httpx.AsyncClient(timeout=10.0)
            close_client = True

        try:
            resp = await client.get(self.server_info_url)
            if resp.status_code != 200:
                return False
            data = resp.json()
            capacity = self._extract_capacity(data)
            if capacity is None:
                return False
            self.cache_config = SGLangCacheConfig(total_tokens_capacity=capacity)
            logger.info(
                "Fetched server info for %s: total_capacity=%s tokens",
                self.url,
                self.cache_config.total_tokens_capacity,
            )
            return True
        except Exception as exc:
            logger.warning("Failed to fetch server info from %s: %s", self.url, exc)
            return False
        finally:
            if close_client:
                await client.aclose()

    async def fetch_metrics(self) -> bool:
        """Fetch and update metrics from SGLang /metrics endpoint."""
        if not self._client:
            return False
        try:
            resp = await self._client.get(self.metrics_url)
            if resp.status_code == 200:
                metrics = SGLangMetrics.from_prometheus_text(resp.text)
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > METRICS_HISTORY_SIZE:
                    self.metrics_history = self.metrics_history[-METRICS_HISTORY_SIZE:]
                self.healthy = True
                return True
            self.healthy = False
            return False
        except Exception as exc:
            logger.debug("Failed to fetch metrics from %s: %s", self.url, exc)
            self.healthy = False
            return False

    # ---------------------------------------------------------------------
    # Calculations
    # ---------------------------------------------------------------------

    def calculate_shared_tokens(self, reasoning_program_tokens: int) -> int:
        """Calculate shared tokens from token usage.

        shared_tokens = reasoning_program_tokens - used_tokens
        used_tokens = token_usage * total_tokens_capacity
        """
        if not self.latest_metrics or not self.cache_config:
            return 0
        used_tokens = int(self.latest_metrics.token_usage * self.cache_config.total_tokens_capacity)
        return max(0, reasoning_program_tokens - used_tokens)

    def to_dict(self) -> dict:
        """Convert metrics state to dict for API response."""
        result = {
            "healthy": self.healthy,
            "monitoring": self.is_monitoring,
        }
        if self.cache_config:
            result["cache_config"] = {
                "total_tokens_capacity": self.cache_config.total_tokens_capacity,
            }
        if self.metrics_history:
            latest = self.latest_metrics
            result["metrics"] = {
                "num_requests_running": latest.num_requests_running,
                "num_requests_waiting": latest.num_requests_waiting,
                "token_usage": round(latest.token_usage, 4),
                "cache_hit_rate": round(latest.cache_hit_rate, 4),
                "num_used_tokens": latest.num_used_tokens,
                "prompt_tokens_total": latest.prompt_tokens_total,
                "generation_tokens_total": latest.generation_tokens_total,
                "last_updated": latest.timestamp,
                "history_size": len(self.metrics_history),
            }
        return result

    def _extract_capacity(self, data: object) -> Optional[int]:
        """Extract total token capacity from server info payload."""
        if isinstance(data, dict):
            capacity = self._extract_capacity_from_dict(data)
            if capacity is not None:
                return capacity
            nested = data.get("server_info")
            if isinstance(nested, dict):
                capacity = self._extract_capacity_from_dict(nested)
                if capacity is not None:
                    return capacity
        return None

    @staticmethod
    def _extract_capacity_from_dict(data: dict) -> Optional[int]:
        """Extract capacity from a server info dict."""
        val = data.get("max_total_num_tokens")
        if isinstance(val, (int, float)) and val > 0:
            return int(val)
        if isinstance(val, str):
            try:
                parsed = int(float(val))
                if parsed > 0:
                    return parsed
            except ValueError:
                pass

        internal_states = data.get("internal_states")
        if isinstance(internal_states, list):
            capacities: List[int] = []
            for state in internal_states:
                if not isinstance(state, dict):
                    continue
                memory = state.get("memory_usage")
                if not isinstance(memory, dict):
                    continue
                cap = memory.get("token_capacity")
                if isinstance(cap, (int, float)) and cap > 0:
                    capacities.append(int(cap))
            if capacities:
                return max(capacities)
        return None
