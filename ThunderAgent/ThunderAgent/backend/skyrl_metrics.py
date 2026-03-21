"""SkyRL metrics client.

Parses the JSON format returned by SkyRL's /metrics endpoint, which aggregates
per-engine vLLM metrics. Used when ThunderAgent proxies to a SkyRL HTTP endpoint
(single-backend mode) rather than directly to vLLM /metrics (Prometheus format).

SkyRL /metrics response format:
{
  "timestamp": 1770532792.69,
  "engines": [
    {
      "engine_id": 0,
      "kv_cache_usage_pct": 5.9,
      "num_running_reqs": 2,
      "num_waiting_reqs": 0,
      "num_cumulative_preemption": 0
    },
    ...
  ]
}
"""
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import List, Optional

import httpx

from .metrics_base import MetricsClient

logger = logging.getLogger(__name__)


@dataclass
class SkyRLCacheConfig:
    """Aggregate KV cache config across all SkyRL engines."""
    total_tokens_capacity: int = 0
    num_engines: int = 0


@dataclass
class SkyRLMetrics:
    """Aggregated metrics from SkyRL /metrics endpoint."""
    # Aggregate across all engines
    num_requests_running: int = 0
    num_requests_waiting: int = 0
    kv_cache_usage_perc: float = 0.0  # weighted average across engines (0-100)
    num_preemptions: int = 0  # sum across engines
    num_engines: int = 0

    timestamp: float = 0.0


METRICS_HISTORY_SIZE = 12


class SkyRLMetricsClient(MetricsClient):
    """Client for fetching metrics from a SkyRL HTTP endpoint.

    SkyRL's /metrics returns JSON with per-engine data. This client
    aggregates it into a single-backend view for ThunderAgent's scheduler.
    """

    def __init__(self, url: str):
        super().__init__(url)
        self.healthy = True
        self.metrics_history: List[SkyRLMetrics] = []
        self.cache_config: Optional[SkyRLCacheConfig] = None

        self._client: Optional[httpx.AsyncClient] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._monitor_stop = False

    @property
    def metrics_url(self) -> str:
        return f"{self.url}/metrics"

    @property
    def latest_metrics(self) -> Optional[SkyRLMetrics]:
        return self.metrics_history[-1] if self.metrics_history else None

    @property
    def is_monitoring(self) -> bool:
        return self._monitor_task is not None

    # -------------------------------------------------------------------------
    # Monitoring
    # -------------------------------------------------------------------------

    async def start_monitoring(self, interval: float = 5.0):
        if self._monitor_task is not None:
            return
        self._monitor_stop = False
        self._client = httpx.AsyncClient(timeout=10.0)
        await self.fetch_cache_config()
        self._monitor_task = asyncio.create_task(self._monitor_loop(interval))
        cap = self.cache_config.total_tokens_capacity if self.cache_config else "unknown"
        logger.info(
            "Started SkyRL metrics monitoring for %s (interval: %ss, total_capacity: %s tokens)",
            self.url, interval, cap,
        )

    async def stop_monitoring(self):
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
        logger.info("Stopped SkyRL metrics monitoring for %s", self.url)

    async def _monitor_loop(self, interval: float):
        while not self._monitor_stop:
            try:
                await self.fetch_metrics()
            except Exception as e:
                logger.debug("Error fetching SkyRL metrics from %s: %s", self.url, e)
            await asyncio.sleep(interval)

    # -------------------------------------------------------------------------
    # Fetching
    # -------------------------------------------------------------------------

    async def fetch_cache_config(self) -> bool:
        """Estimate total KV cache capacity from SkyRL /metrics.

        SkyRL doesn't expose block_size/num_gpu_blocks directly.
        We estimate total capacity from kv_cache_usage_pct and running requests.
        For now, we use a conservative fixed estimate per engine based on
        typical vLLM configs (will be refined on first metrics fetch).
        """
        client = self._client
        close_client = False
        if client is None:
            client = httpx.AsyncClient(timeout=10.0)
            close_client = True

        try:
            resp = await client.get(self.metrics_url)
            if resp.status_code != 200:
                return False
            data = resp.json()
            engines = data.get("engines", [])
            num_engines = len(engines)
            if num_engines == 0:
                return False

            # We cannot determine exact token capacity from SkyRL's JSON.
            # Use a configurable default per engine. The scheduler will still
            # work because it primarily uses kv_cache_usage_perc for decisions.
            # Default: 200,000 tokens per engine (typical for 14B with 0.8 gpu_mem_util).
            DEFAULT_TOKENS_PER_ENGINE = 200_000
            total_cap = num_engines * DEFAULT_TOKENS_PER_ENGINE
            self.cache_config = SkyRLCacheConfig(
                total_tokens_capacity=total_cap,
                num_engines=num_engines,
            )
            logger.info(
                "SkyRL cache config: %d engines, estimated total_capacity=%d tokens",
                num_engines, total_cap,
            )
            return True
        except Exception as e:
            logger.warning("Failed to fetch SkyRL cache config from %s: %s", self.url, e)
            return False
        finally:
            if close_client:
                await client.aclose()

    async def fetch_metrics(self) -> bool:
        if not self._client:
            return False
        try:
            resp = await self._client.get(self.metrics_url)
            if resp.status_code != 200:
                self.healthy = False
                return False

            data = resp.json()
            engines = data.get("engines", [])
            if not engines:
                self.healthy = False
                return False

            # Aggregate per-engine metrics
            total_running = 0
            total_waiting = 0
            total_preemptions = 0
            kv_usages = []
            for eng in engines:
                if "error" in eng:
                    continue
                total_running += eng.get("num_running_reqs", 0)
                total_waiting += eng.get("num_waiting_reqs", 0)
                total_preemptions += eng.get("num_cumulative_preemption", 0)
                kv_pct = eng.get("kv_cache_usage_pct", 0.0)
                kv_usages.append(kv_pct)

            # Average KV cache usage across engines (0-100 scale)
            avg_kv = sum(kv_usages) / len(kv_usages) if kv_usages else 0.0

            metrics = SkyRLMetrics(
                num_requests_running=total_running,
                num_requests_waiting=total_waiting,
                kv_cache_usage_perc=avg_kv,
                num_preemptions=total_preemptions,
                num_engines=len(engines),
                timestamp=time.time(),
            )
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > METRICS_HISTORY_SIZE:
                self.metrics_history = self.metrics_history[-METRICS_HISTORY_SIZE:]
            self.healthy = True
            return True
        except Exception as e:
            logger.debug("Failed to fetch SkyRL metrics from %s: %s", self.url, e)
            self.healthy = False
            return False

    # -------------------------------------------------------------------------
    # Calculations
    # -------------------------------------------------------------------------

    def calculate_shared_tokens(self, reasoning_program_tokens: int) -> int:
        """Calculate shared tokens (prefix cache savings).

        Uses average KV cache usage percentage across all engines.
        """
        if not self.latest_metrics or not self.cache_config:
            return 0
        # kv_cache_usage_perc is 0-100 scale
        vllm_actual_used = int(
            (self.latest_metrics.kv_cache_usage_perc / 100.0)
            * self.cache_config.total_tokens_capacity
        )
        return max(0, reasoning_program_tokens - vllm_actual_used)

    def to_dict(self) -> dict:
        result = {
            "healthy": self.healthy,
            "monitoring": self.is_monitoring,
            "backend_type": "skyrl",
        }
        if self.cache_config:
            result["cache_config"] = {
                "total_tokens_capacity": self.cache_config.total_tokens_capacity,
                "num_engines": self.cache_config.num_engines,
            }
        if self.metrics_history:
            latest = self.latest_metrics
            result["metrics"] = {
                "num_requests_running": latest.num_requests_running,
                "num_requests_waiting": latest.num_requests_waiting,
                "kv_cache_usage_perc": round(latest.kv_cache_usage_perc, 4),
                "num_preemptions": latest.num_preemptions,
                "num_engines": latest.num_engines,
                "last_updated": latest.timestamp,
                "history_size": len(self.metrics_history),
            }
        return result
