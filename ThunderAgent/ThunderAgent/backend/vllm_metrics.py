"""vLLM metrics parsing, storage, and client."""
import asyncio
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
import re
import time

import httpx

from .metrics_base import MetricsClient

logger = logging.getLogger(__name__)


@dataclass
class VLLMCacheConfig:
    """Static KV cache configuration from vLLM (fetched once at startup)."""
    block_size: int = 0          # Tokens per block
    num_gpu_blocks: int = 0      # Total GPU blocks
    
    @property
    def total_tokens_capacity(self) -> int:
        """Total KV cache capacity in tokens."""
        return self.block_size * self.num_gpu_blocks
    
    @classmethod
    def from_prometheus_text(cls, text: str) -> "VLLMCacheConfig":
        """Parse cache_config_info from Prometheus metrics text."""
        config = cls()
        
        # Extract block_size and num_gpu_blocks from labels
        # vllm:cache_config_info{block_size="16",...,num_gpu_blocks="27283",...} 1.0
        match = re.search(r'vllm:cache_config_info\{([^}]+)\}', text)
        if match:
            labels = match.group(1)
            
            # Extract block_size
            bs_match = re.search(r'block_size="(\d+)"', labels)
            if bs_match:
                config.block_size = int(bs_match.group(1))
            
            # Extract num_gpu_blocks
            ngb_match = re.search(r'num_gpu_blocks="(\d+)"', labels)
            if ngb_match:
                config.num_gpu_blocks = int(ngb_match.group(1))
        
        return config


@dataclass
class VLLMMetrics:
    """Parsed metrics from vLLM /metrics endpoint."""
    # Request stats
    num_requests_running: int = 0
    num_requests_waiting: int = 0
    
    # KV Cache
    kv_cache_usage_perc: float = 0.0
    
    # Prefix cache (cumulative)
    prefix_cache_queries: int = 0
    prefix_cache_hits: int = 0
    
    # Tokens (cumulative)
    prompt_tokens_total: int = 0
    generation_tokens_total: int = 0
    
    # Preemptions
    num_preemptions: int = 0
    
    # Request success counts
    request_success_stop: int = 0
    request_success_length: int = 0
    request_success_abort: int = 0
    request_success_error: int = 0
    
    # Timestamp when metrics were fetched
    timestamp: float = 0.0
    
    @property
    def prefix_cache_hit_rate(self) -> float:
        """Calculate prefix cache hit rate."""
        if self.prefix_cache_queries == 0:
            return 0.0
        return self.prefix_cache_hits / self.prefix_cache_queries
    
    @property
    def total_requests_completed(self) -> int:
        """Total completed requests."""
        return (self.request_success_stop + self.request_success_length + 
                self.request_success_abort + self.request_success_error)
    
    @classmethod
    def from_prometheus_text(cls, text: str) -> "VLLMMetrics":
        """Parse Prometheus text format into VLLMMetrics."""
        metrics = cls(timestamp=time.time())
        
        # Helper to extract gauge/counter value
        def extract_value(pattern: str) -> Optional[float]:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except (ValueError, IndexError):
                    return None
            return None
        
        # Current request counts
        val = extract_value(r'vllm:num_requests_running\{[^}]*\}\s+([\d.]+)')
        if val is not None:
            metrics.num_requests_running = int(val)
        
        val = extract_value(r'vllm:num_requests_waiting\{[^}]*\}\s+([\d.]+)')
        if val is not None:
            metrics.num_requests_waiting = int(val)
        
        # KV cache usage
        val = extract_value(r'vllm:kv_cache_usage_perc\{[^}]*\}\s+([\d.]+)')
        if val is not None:
            metrics.kv_cache_usage_perc = val
        
        # Prefix cache (counters)
        val = extract_value(r'vllm:prefix_cache_queries_total\{[^}]*\}\s+([\d.eE+]+)')
        if val is not None:
            metrics.prefix_cache_queries = int(val)
        
        val = extract_value(r'vllm:prefix_cache_hits_total\{[^}]*\}\s+([\d.eE+]+)')
        if val is not None:
            metrics.prefix_cache_hits = int(val)
        
        # Token counts
        val = extract_value(r'vllm:prompt_tokens_total\{[^}]*\}\s+([\d.eE+]+)')
        if val is not None:
            metrics.prompt_tokens_total = int(val)
        
        val = extract_value(r'vllm:generation_tokens_total\{[^}]*\}\s+([\d.eE+]+)')
        if val is not None:
            metrics.generation_tokens_total = int(val)
        
        # Preemptions
        val = extract_value(r'vllm:num_preemptions_total\{[^}]*\}\s+([\d.eE+]+)')
        if val is not None:
            metrics.num_preemptions = int(val)
        
        # Request success counts by finish reason
        for reason in ['stop', 'length', 'abort', 'error']:
            val = extract_value(rf'vllm:request_success_total\{{[^}}]*finished_reason="{reason}"[^}}]*\}}\s+([\d.]+)')
            if val is not None:
                setattr(metrics, f'request_success_{reason}', int(val))
        
        return metrics


# Keep only the most recent N metrics samples
METRICS_HISTORY_SIZE = 12


class VLLMMetricsClient(MetricsClient):
    """Client for fetching and managing metrics from a vLLM backend.
    
    Handles HTTP communication with vLLM /metrics endpoint,
    metrics history management, and shared_tokens calculation.
    """
    
    def __init__(self, url: str):
        super().__init__(url)
        self.healthy = True
        self.metrics_history: List[VLLMMetrics] = []
        self.cache_config: Optional[VLLMCacheConfig] = None
        
        # HTTP client and monitoring state
        self._client: Optional[httpx.AsyncClient] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._monitor_stop = False
    
    @property
    def metrics_url(self) -> str:
        """Prometheus metrics endpoint."""
        return f"{self.url}/metrics"
    
    @property
    def latest_metrics(self) -> Optional[VLLMMetrics]:
        """Get the most recent metrics sample."""
        return self.metrics_history[-1] if self.metrics_history else None
    
    @property
    def is_monitoring(self) -> bool:
        """Check if monitoring is active."""
        return self._monitor_task is not None
    
    # -------------------------------------------------------------------------
    # Metrics Monitoring
    # -------------------------------------------------------------------------
    
    async def start_monitoring(self, interval: float = 5.0):
        """Start background metrics monitoring."""
        if self._monitor_task is not None:
            return  # Already running
        
        self._monitor_stop = False
        self._client = httpx.AsyncClient(timeout=10.0)
        
        # Fetch cache config once at startup
        await self.fetch_cache_config()
        
        self._monitor_task = asyncio.create_task(self._monitor_loop(interval))
        logger.info(f"Started metrics monitoring for {self.url} (interval: {interval}s, kv_capacity: {self.cache_config.total_tokens_capacity if self.cache_config else 'unknown'} tokens)")
    
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
        logger.info(f"Stopped metrics monitoring for {self.url}")
    
    async def _monitor_loop(self, interval: float):
        """Background loop to periodically fetch metrics."""
        while not self._monitor_stop:
            try:
                await self.fetch_metrics()
            except Exception as e:
                logger.debug(f"Error fetching metrics from {self.url}: {e}")
            await asyncio.sleep(interval)
    
    # -------------------------------------------------------------------------
    # Metrics Fetching
    # -------------------------------------------------------------------------
    
    async def fetch_cache_config(self) -> bool:
        """Fetch static cache config from vLLM.
        
        Can be called independently of start_monitoring().
        Uses existing client if monitoring, otherwise creates a temporary one.
        """
        client = self._client
        close_client = False
        
        if client is None:
            client = httpx.AsyncClient(timeout=10.0)
            close_client = True
        
        try:
            resp = await client.get(self.metrics_url)
            if resp.status_code == 200:
                self.cache_config = VLLMCacheConfig.from_prometheus_text(resp.text)
                logger.info(f"Fetched cache config for {self.url}: block_size={self.cache_config.block_size}, num_gpu_blocks={self.cache_config.num_gpu_blocks}, total_capacity={self.cache_config.total_tokens_capacity}")
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to fetch cache config from {self.url}: {e}")
            return False
        finally:
            if close_client:
                await client.aclose()
    
    async def fetch_metrics(self) -> bool:
        """Fetch and update metrics from vLLM /metrics endpoint."""
        if not self._client:
            return False
        try:
            resp = await self._client.get(self.metrics_url)
            if resp.status_code == 200:
                metrics = VLLMMetrics.from_prometheus_text(resp.text)
                self.metrics_history.append(metrics)
                # Keep only the most recent samples
                if len(self.metrics_history) > METRICS_HISTORY_SIZE:
                    self.metrics_history = self.metrics_history[-METRICS_HISTORY_SIZE:]
                self.healthy = True
                return True
            else:
                self.healthy = False
                return False
        except Exception as e:
            logger.debug(f"Failed to fetch metrics from {self.url}: {e}")
            self.healthy = False
            return False
    
    # -------------------------------------------------------------------------
    # Calculations
    # -------------------------------------------------------------------------
    
    def calculate_shared_tokens(self, reasoning_program_tokens: int) -> int:
        """Calculate shared tokens (prefix cache savings).
        
        shared_tokens = reasoning_program_tokens - vllm_actual_used_tokens
        
        Args:
            reasoning_program_tokens: Current total tokens from REASONING programs
        
        Returns:
            Shared tokens (0 if no metrics available or negative)
        """
        if not self.latest_metrics or not self.cache_config:
            return 0
        vllm_actual_used = int(
            self.latest_metrics.kv_cache_usage_perc 
            * self.cache_config.total_tokens_capacity
        )
        return max(0, reasoning_program_tokens - vllm_actual_used)
    
    def to_dict(self) -> dict:
        """Convert metrics state to dict for API response."""
        result = {
            "healthy": self.healthy,
            "monitoring": self.is_monitoring,
        }
        # Include cache config (static)
        if self.cache_config:
            result["cache_config"] = {
                "block_size": self.cache_config.block_size,
                "num_gpu_blocks": self.cache_config.num_gpu_blocks,
                "total_tokens_capacity": self.cache_config.total_tokens_capacity,
            }
        # Include latest metrics (dynamic)
        if self.metrics_history:
            latest = self.latest_metrics
            result["metrics"] = {
                "num_requests_running": latest.num_requests_running,
                "num_requests_waiting": latest.num_requests_waiting,
                "kv_cache_usage_perc": round(latest.kv_cache_usage_perc, 4),
                "prefix_cache_hit_rate": round(latest.prefix_cache_hit_rate, 4),
                "prefix_cache_queries": latest.prefix_cache_queries,
                "prefix_cache_hits": latest.prefix_cache_hits,
                "prompt_tokens_total": latest.prompt_tokens_total,
                "generation_tokens_total": latest.generation_tokens_total,
                "num_preemptions": latest.num_preemptions,
                "requests_completed": latest.total_requests_completed,
                "last_updated": latest.timestamp,
                "history_size": len(self.metrics_history),
            }
        return result
