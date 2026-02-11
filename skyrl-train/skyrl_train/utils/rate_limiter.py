"""
Rate limiter for controlling trajectory submission rates and concurrency.

This module provides a token bucket rate limiter and concurrency limiter for async code,
allowing users to express "N trajectories per second" and "max M concurrent trajectories".

Note: Fractional rates >= 1.0 are supported (e.g., 1.5 trajectories/second).
Rates < 1.0 are not supported due to the token bucket implementation.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

from loguru import logger


@dataclass
class RateLimiterConfig:
    """Configuration for rate and concurrency limiting.

    Attributes:
        enabled: Whether limiting is enabled. If False, no limits are applied.
        trajectories_per_second: Maximum trajectories per second (rate limiting).
            Must be >= 1.0 if set. None means no rate limiting.
        max_concurrency: Maximum concurrent trajectories allowed.
            Must be >= 1 if set. None means no concurrency limiting.
    """

    enabled: bool = False
    trajectories_per_second: Optional[float] = None
    max_concurrency: Optional[int] = None

    def __post_init__(self):
        if self.trajectories_per_second is not None and self.trajectories_per_second < 1.0:
            raise ValueError(
                f"trajectories_per_second must be >= 1.0, got {self.trajectories_per_second}. "
                "Rates < 1.0 are not supported due to the token bucket implementation."
            )
        if self.max_concurrency is not None and self.max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {self.max_concurrency}")


class RateLimiterInterface(ABC):
    """Abstract base class for rate limiters."""

    @abstractmethod
    async def acquire(self) -> None:
        """Acquire permission to proceed. May block if rate/concurrency limited."""
        pass

    @abstractmethod
    def release(self) -> None:
        """Release a concurrency slot. Must be called after operation completes."""
        pass

    async def __aenter__(self) -> "RateLimiterInterface":
        """Context manager entry: acquire permission."""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit: release concurrency slot."""
        self.release()


class NoOpRateLimiter(RateLimiterInterface):
    """A no-op rate limiter that never blocks."""

    async def acquire(self) -> None:
        """Immediately returns without blocking."""
        pass

    def release(self) -> None:
        """No-op release."""
        pass


class AsyncRateLimiter(RateLimiterInterface):
    """Combined rate limiter and concurrency limiter for async code.

    Rate limiting (token bucket algorithm):
    - Bucket holds tokens, max capacity = rate (trajectories_per_second)
    - Tokens refill at rate N per second
    - Each acquire() call consumes 1 token
    - If no tokens available, caller waits until refill

    Concurrency limiting (semaphore):
    - Limits how many operations can run simultaneously
    - acquire() waits if max_concurrency operations are already running
    - release() must be called when operation completes

    Note: Fractional rates >= 1.0 are supported (e.g., 1.5 means 1.5 ops/second).
    Rates < 1.0 are not supported because the bucket capacity equals the rate,
    so the bucket could never hold a full token to allow an acquire().
    """

    def __init__(
        self,
        rate: Optional[float] = None,
        max_concurrency: Optional[int] = None,
    ):
        """Initialize the rate limiter.

        Args:
            rate: Maximum operations per second (tokens per second).
                  Must be >= 1.0 if provided. None disables rate limiting.
            max_concurrency: Maximum concurrent operations allowed.
                  Must be >= 1 if provided. None disables concurrency limiting.
        """
        if rate is not None and rate < 1.0:
            raise ValueError(
                f"Rate must be >= 1.0, got {rate}. "
                "Rates < 1.0 are not supported due to the token bucket implementation."
            )
        if max_concurrency is not None and max_concurrency < 1:
            raise ValueError(f"max_concurrency must be >= 1, got {max_concurrency}")

        # Rate limiting state (token bucket)
        self._rate = rate
        if rate is not None:
            self._max_tokens = rate  # bucket capacity = rate
            self._tokens = rate  # start with a full bucket
            self._last_refill = time.monotonic()
            self._rate_lock = asyncio.Lock()

        # Concurrency limiting state (semaphore)
        self._max_concurrency = max_concurrency
        if max_concurrency is not None:
            self._semaphore = asyncio.Semaphore(max_concurrency)

    async def acquire(self) -> None:
        """Acquire permission to proceed, waiting if necessary.

        First applies rate limiting (controls how fast operations start),
        then acquires concurrency slot (controls how many run simultaneously).
        """
        # Rate limit first (controls start rate)
        if self._rate is not None:
            await self._acquire_rate_token()

        # Then concurrency limit (controls concurrent execution)
        if self._max_concurrency is not None:
            await self._semaphore.acquire()

    def release(self) -> None:
        """Release a concurrency slot.

        Must be called after the operation completes to allow other
        operations to proceed. Safe to call even if concurrency limiting
        is disabled.
        """
        if self._max_concurrency is not None:
            self._semaphore.release()

    async def _acquire_rate_token(self) -> None:
        """Acquire a rate limit token, waiting if necessary."""
        async with self._rate_lock:
            while True:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                # Calculate wait time for next token
                wait_time = (1.0 - self._tokens) / self._rate
                # Release lock while sleeping so _refill time stays accurate
                self._rate_lock.release()
                try:
                    await asyncio.sleep(wait_time)
                finally:
                    await self._rate_lock.acquire()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._rate)
        self._last_refill = now


def create_rate_limiter(
    config: Union[RateLimiterConfig, dict, None]
) -> RateLimiterInterface:
    """Factory function to create a rate limiter from config.

    Args:
        config: Rate limiter configuration. Can be:
            - RateLimiterConfig dataclass
            - dict with 'enabled', 'trajectories_per_second', and/or 'max_concurrency' keys
            - None (returns NoOpRateLimiter)

    Returns:
        AsyncRateLimiter if enabled with at least one limit configured,
        NoOpRateLimiter otherwise.
    """
    if config is None:
        return NoOpRateLimiter()

    if isinstance(config, dict):
        config = RateLimiterConfig(
            enabled=config.get("enabled", False),
            trajectories_per_second=config.get("trajectories_per_second"),
            max_concurrency=config.get("max_concurrency"),
        )

    if not config.enabled:
        return NoOpRateLimiter()

    # Log what's enabled
    limits = []
    if config.trajectories_per_second is not None:
        limits.append(f"{config.trajectories_per_second} trajectories/second")
    if config.max_concurrency is not None:
        limits.append(f"max {config.max_concurrency} concurrent")
    if limits:
        logger.info(f"Rate limiter enabled: {', '.join(limits)}")
    else:
        # enabled=True but no limits configured, treat as no-op
        return NoOpRateLimiter()

    return AsyncRateLimiter(
        rate=config.trajectories_per_second,
        max_concurrency=config.max_concurrency,
    )
