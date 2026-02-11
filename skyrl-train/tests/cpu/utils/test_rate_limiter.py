"""
uv run --isolated --extra dev pytest tests/cpu/utils/test_rate_limiter.py -v
"""

import asyncio
import time

import pytest
from omegaconf import OmegaConf

from skyrl_train.utils.rate_limiter import (
    AsyncRateLimiter,
    NoOpRateLimiter,
    RateLimiterConfig,
    create_rate_limiter,
)


class TestNoOpRateLimiter:
    @pytest.mark.asyncio
    async def test_acquire_returns_immediately(self):
        limiter = NoOpRateLimiter()
        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start
        assert elapsed < 0.01  # Should be nearly instant

    @pytest.mark.asyncio
    async def test_release_is_noop(self):
        limiter = NoOpRateLimiter()
        # Should not raise
        limiter.release()
        limiter.release()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        limiter = NoOpRateLimiter()
        async with limiter:
            pass  # Should not block or raise


class TestAsyncRateLimiterRateLimiting:
    """Tests for rate limiting functionality."""

    def test_invalid_rate_raises_error(self):
        with pytest.raises(ValueError, match="Rate must be >= 1.0"):
            AsyncRateLimiter(rate=0)
        with pytest.raises(ValueError, match="Rate must be >= 1.0"):
            AsyncRateLimiter(rate=-1)
        with pytest.raises(ValueError, match="Rate must be >= 1.0"):
            AsyncRateLimiter(rate=0.5)

    @pytest.mark.asyncio
    async def test_initial_burst_allowed(self):
        """Initial burst up to bucket capacity should not block."""
        rate = 5.0
        limiter = AsyncRateLimiter(rate=rate)

        start = time.monotonic()
        # Acquire up to the bucket capacity
        for _ in range(int(rate)):
            await limiter.acquire()
        elapsed = time.monotonic() - start

        # Should complete quickly (no waiting)
        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_rate_limiting_enforced(self):
        """After burst, additional acquires should be rate limited."""
        rate = 10.0
        limiter = AsyncRateLimiter(rate=rate)

        # Exhaust the initial burst
        for _ in range(int(rate)):
            await limiter.acquire()

        # Now the next acquire should wait ~0.1 seconds (1/rate)
        start = time.monotonic()
        await limiter.acquire()
        elapsed = time.monotonic() - start

        # Should have waited approximately 1/rate seconds
        expected_wait = 1.0 / rate
        assert elapsed >= expected_wait * 0.8  # Allow 20% tolerance
        assert elapsed < expected_wait * 2.0  # But not too long

    @pytest.mark.asyncio
    async def test_concurrent_rate_acquires(self):
        """Multiple concurrent acquires should be properly serialized."""
        rate = 5.0
        limiter = AsyncRateLimiter(rate=rate)

        # Exhaust initial burst
        for _ in range(int(rate)):
            await limiter.acquire()

        # Launch 3 concurrent acquires
        async def timed_acquire():
            start = time.monotonic()
            await limiter.acquire()
            return time.monotonic() - start

        start = time.monotonic()
        await asyncio.gather(*[timed_acquire() for _ in range(3)])
        total_elapsed = time.monotonic() - start

        # With rate=5, each token takes 0.2s to refill
        # 3 concurrent acquires should take roughly 0.6s total
        expected_total = 3 * (1.0 / rate)
        assert total_elapsed >= expected_total * 0.8
        assert total_elapsed < expected_total * 2.0

    @pytest.mark.asyncio
    async def test_token_refill_after_wait(self):
        """Tokens should refill over time."""
        rate = 10.0
        limiter = AsyncRateLimiter(rate=rate)

        # Exhaust initial burst
        for _ in range(int(rate)):
            await limiter.acquire()

        # Wait for tokens to refill (0.5 seconds = 5 tokens at rate 10)
        await asyncio.sleep(0.5)

        # Should be able to acquire 5 more without significant delay
        start = time.monotonic()
        for _ in range(5):
            await limiter.acquire()
        elapsed = time.monotonic() - start

        assert elapsed < 0.005  # Should be quick since tokens refilled

    @pytest.mark.asyncio
    async def test_fractional_rate_above_one(self):
        """Fractional rates >= 1.0 should work correctly."""
        rate = 1.5
        limiter = AsyncRateLimiter(rate=rate)

        # First acquire should be instant (bucket starts with 1.5 tokens)
        start = time.monotonic()
        await limiter.acquire()
        first_elapsed = time.monotonic() - start
        assert first_elapsed < 0.05

        # Second acquire should wait ~0.33s ((1.0 - 0.5) / 1.5)
        start = time.monotonic()
        await limiter.acquire()
        second_elapsed = time.monotonic() - start
        expected_wait = (1.0 - 0.5) / rate  # ~0.33s
        assert second_elapsed >= expected_wait * 0.8
        assert second_elapsed < expected_wait * 2.0


class TestAsyncRateLimiterConcurrency:
    """Tests for concurrency limiting functionality."""

    def test_invalid_max_concurrency_raises_error(self):
        with pytest.raises(ValueError, match="max_concurrency must be >= 1"):
            AsyncRateLimiter(max_concurrency=0)
        with pytest.raises(ValueError, match="max_concurrency must be >= 1"):
            AsyncRateLimiter(max_concurrency=-1)

    @pytest.mark.asyncio
    async def test_concurrency_limit_enforced(self):
        """Concurrency limit should block when max concurrent operations reached."""
        max_conc = 2
        limiter = AsyncRateLimiter(max_concurrency=max_conc)
        running = []
        completed = []

        async def task(task_id):
            async with limiter:
                running.append(task_id)
                # Verify we never exceed max_concurrency
                assert len(running) <= max_conc
                await asyncio.sleep(0.1)
                running.remove(task_id)
                completed.append(task_id)

        # Launch 5 tasks, only 2 should run at a time
        await asyncio.gather(*[task(i) for i in range(5)])
        assert len(completed) == 5

    @pytest.mark.asyncio
    async def test_release_allows_next_task(self):
        """Releasing a slot should allow a waiting task to proceed."""
        limiter = AsyncRateLimiter(max_concurrency=1)

        order = []

        async def task1():
            await limiter.acquire()
            order.append("task1_acquired")
            await asyncio.sleep(0.1)
            order.append("task1_releasing")
            limiter.release()

        async def task2():
            await asyncio.sleep(0.05)  # Start slightly after task1
            order.append("task2_waiting")
            await limiter.acquire()
            order.append("task2_acquired")
            limiter.release()

        await asyncio.gather(task1(), task2())

        # task2 should wait for task1 to release
        assert order == [
            "task1_acquired",
            "task2_waiting",
            "task1_releasing",
            "task2_acquired",
        ]

    @pytest.mark.asyncio
    async def test_context_manager_releases_on_exception(self):
        """Context manager should release slot even if exception occurs."""
        limiter = AsyncRateLimiter(max_concurrency=1)

        with pytest.raises(RuntimeError):
            async with limiter:
                raise RuntimeError("test error")

        # Should be able to acquire again since slot was released
        await limiter.acquire()
        limiter.release()

    @pytest.mark.asyncio
    async def test_concurrency_only_no_rate_limit(self):
        """With only concurrency limit, no rate limiting should occur."""
        limiter = AsyncRateLimiter(max_concurrency=100)

        # Should be able to acquire 100 immediately
        start = time.monotonic()
        for _ in range(100):
            await limiter.acquire()
        elapsed = time.monotonic() - start

        assert elapsed < 0.01  # Should be fast, no rate limiting


class TestAsyncRateLimiterCombined:
    """Tests for combined rate and concurrency limiting."""

    @pytest.mark.asyncio
    async def test_both_limits_applied(self):
        """Both rate and concurrency limits should be applied."""
        rate = 10.0
        max_conc = 2
        limiter = AsyncRateLimiter(rate=rate, max_concurrency=max_conc)
        running_count = 0
        max_running = 0

        async def task():
            nonlocal running_count, max_running
            async with limiter:
                running_count += 1
                max_running = max(max_running, running_count)
                await asyncio.sleep(0.05)
                running_count -= 1

        # Run 5 tasks
        await asyncio.gather(*[task() for _ in range(5)])

        # Concurrency should have been limited to 2
        assert max_running <= max_conc

    @pytest.mark.asyncio
    async def test_rate_limit_applies_before_concurrency(self):
        """Rate limiting should control start rate, concurrency limits running."""
        rate = 5.0
        max_conc = 10  # High concurrency, rate should be the bottleneck
        limiter = AsyncRateLimiter(rate=rate, max_concurrency=max_conc)

        # Exhaust rate limit burst
        for _ in range(int(rate)):
            await limiter.acquire()
            limiter.release()

        # Next acquire should be rate limited
        start = time.monotonic()
        await limiter.acquire()
        limiter.release()
        elapsed = time.monotonic() - start

        expected_wait = 1.0 / rate
        assert elapsed >= expected_wait * 0.8


class TestRateLimiterConfig:
    def test_default_values(self):
        config = RateLimiterConfig()
        assert config.enabled is False
        assert config.trajectories_per_second is None
        assert config.max_concurrency is None

    def test_custom_values(self):
        config = RateLimiterConfig(enabled=True, trajectories_per_second=5.0, max_concurrency=10)
        assert config.enabled is True
        assert config.trajectories_per_second == 5.0
        assert config.max_concurrency == 10

    def test_invalid_trajectories_per_second_raises_error(self):
        with pytest.raises(ValueError, match="trajectories_per_second must be >= 1.0"):
            RateLimiterConfig(enabled=True, trajectories_per_second=0.5)
        with pytest.raises(ValueError, match="trajectories_per_second must be >= 1.0"):
            RateLimiterConfig(enabled=True, trajectories_per_second=0.0)
        with pytest.raises(ValueError, match="trajectories_per_second must be >= 1.0"):
            RateLimiterConfig(trajectories_per_second=-1.0)

    def test_invalid_max_concurrency_raises_error(self):
        with pytest.raises(ValueError, match="max_concurrency must be >= 1"):
            RateLimiterConfig(max_concurrency=0)
        with pytest.raises(ValueError, match="max_concurrency must be >= 1"):
            RateLimiterConfig(max_concurrency=-1)

    def test_fractional_rate_above_one_accepted(self):
        config = RateLimiterConfig(trajectories_per_second=1.5)
        assert config.trajectories_per_second == 1.5

    def test_rate_exactly_one_accepted(self):
        config = RateLimiterConfig(trajectories_per_second=1.0)
        assert config.trajectories_per_second == 1.0


class TestCreateRateLimiter:
    def test_none_config_returns_noop(self):
        limiter = create_rate_limiter(None)
        assert isinstance(limiter, NoOpRateLimiter)

    def test_disabled_config_returns_noop(self):
        config = RateLimiterConfig(enabled=False)
        limiter = create_rate_limiter(config)
        assert isinstance(limiter, NoOpRateLimiter)

    def test_enabled_with_rate_returns_async_limiter(self):
        config = RateLimiterConfig(enabled=True, trajectories_per_second=5.0)
        limiter = create_rate_limiter(config)
        assert isinstance(limiter, AsyncRateLimiter)

    def test_enabled_with_concurrency_returns_async_limiter(self):
        config = RateLimiterConfig(enabled=True, max_concurrency=10)
        limiter = create_rate_limiter(config)
        assert isinstance(limiter, AsyncRateLimiter)

    def test_enabled_with_both_returns_async_limiter(self):
        config = RateLimiterConfig(enabled=True, trajectories_per_second=5.0, max_concurrency=10)
        limiter = create_rate_limiter(config)
        assert isinstance(limiter, AsyncRateLimiter)

    def test_enabled_with_no_limits_returns_noop(self):
        """enabled=True but no limits configured should return NoOp."""
        config = RateLimiterConfig(enabled=True)
        limiter = create_rate_limiter(config)
        assert isinstance(limiter, NoOpRateLimiter)

    def test_dict_config_disabled(self):
        config = {"enabled": False, "trajectories_per_second": 10.0}
        limiter = create_rate_limiter(config)
        assert isinstance(limiter, NoOpRateLimiter)

    def test_dict_config_with_rate(self):
        config = {"enabled": True, "trajectories_per_second": 20.0}
        limiter = create_rate_limiter(config)
        assert isinstance(limiter, AsyncRateLimiter)

    def test_dict_config_with_concurrency(self):
        config = {"enabled": True, "max_concurrency": 128}
        limiter = create_rate_limiter(config)
        assert isinstance(limiter, AsyncRateLimiter)

    def test_dict_config_with_both(self):
        config = {"enabled": True, "trajectories_per_second": 10.0, "max_concurrency": 64}
        limiter = create_rate_limiter(config)
        assert isinstance(limiter, AsyncRateLimiter)

    def test_dict_config_missing_keys_uses_defaults(self):
        config = {}
        limiter = create_rate_limiter(config)
        assert isinstance(limiter, NoOpRateLimiter)

        # enabled=True but no limits
        config = {"enabled": True}
        limiter = create_rate_limiter(config)
        assert isinstance(limiter, NoOpRateLimiter)

    def test_dict_config_invalid_rate_raises_error(self):
        with pytest.raises(ValueError, match="trajectories_per_second must be >= 1.0"):
            create_rate_limiter({"enabled": True, "trajectories_per_second": 0.5})

    def test_omegaconf_dictconfig_with_partial_keys(self):
        """DictConfig (from OmegaConf) should be handled like a dict, with missing keys defaulting.
        """
        cfg = OmegaConf.create({"enabled": True, "trajectories_per_second": 5.0})
        limiter = create_rate_limiter(cfg)
        assert isinstance(limiter, AsyncRateLimiter)

    def test_omegaconf_dictconfig_with_all_keys(self):
        cfg = OmegaConf.create(
            {"enabled": True, "trajectories_per_second": 10.0, "max_concurrency": 64}
        )
        limiter = create_rate_limiter(cfg)
        assert isinstance(limiter, AsyncRateLimiter)
