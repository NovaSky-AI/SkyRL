"""Metrics client interface.

Provides a shared contract for:
- Health status
- Cache capacity
- Metrics polling and serialization
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class MetricsClient(ABC):
    """Abstract interface for backend metrics clients."""

    def __init__(self, url: str):
        self.url = url

    # Implementations are expected to expose the following attributes:
    # - healthy: bool
    # - cache_config: Any (must expose total_tokens_capacity)
    # - metrics_history: List[Any]
    # - metrics_url: str
    # - is_monitoring: bool
    # - latest_metrics: Any

    @abstractmethod
    async def start_monitoring(self, interval: float = 5.0) -> None:
        """Start background metrics monitoring."""

    @abstractmethod
    async def stop_monitoring(self) -> None:
        """Stop background metrics monitoring."""

    @abstractmethod
    async def fetch_metrics(self) -> bool:
        """Fetch metrics from backend."""

    @abstractmethod
    async def fetch_cache_config(self) -> bool:
        """Fetch cache configuration from backend."""

    @abstractmethod
    def calculate_shared_tokens(self, reasoning_program_tokens: int) -> int:
        """Calculate shared tokens (prefix cache savings)."""

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize metrics client state for API response."""
