"""Generator-side survival of an inference-engine death (fault tolerance, Part 1).

Two behaviours, tested against the real ``SkyRLGymGenerator`` methods with
``agent_loop`` stubbed — the trajectory body is irrelevant here; what matters is
what the harness around it does when one raises.

  * ``_gather_trajectories`` cancels the siblings when a trajectory fails.
    UNCONDITIONAL: today the first exception out of ``tqdm.gather`` propagates
    while every other trajectory keeps running against a fleet that is already in
    trouble, and their ``finish_session`` cleanup races the driver's unwind.
  * ``_agent_loop_with_retry`` re-runs a whole trajectory on a transport failure,
    and only on a transport failure.

Run:
    uv run --extra dev --extra fsdp pytest tests/train/generators/test_generator_fault_tolerance.py
"""

import asyncio
from unittest.mock import MagicMock

import aiohttp
import pytest

from skyrl.train.config import GeneratorConfig
from skyrl.train.config.config import InferenceFaultToleranceConfig
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator


def _generator(max_retries: int = 0) -> SkyRLGymGenerator:
    """A generator with only the fields these two methods touch.

    ``__init__`` builds a tokenizer, env executor and chat templates that neither
    ``_gather_trajectories`` nor ``_agent_loop_with_retry`` reads, so it is
    bypassed rather than mocked into existence.
    """
    gen = SkyRLGymGenerator.__new__(SkyRLGymGenerator)
    gen.generator_cfg = GeneratorConfig()
    gen._max_trajectory_retries = max_retries
    gen.inference_engine_client = MagicMock()
    return gen


class TestMaxRetriesWiring:
    """``max_trajectory_retries`` only applies when fault tolerance is on, so a
    config that carries the field but leaves the switch off keeps today's
    fail-fast behaviour."""

    @staticmethod
    def _resolve(ft):
        cfg = GeneratorConfig()
        cfg.inference_engine.fault_tolerance = ft
        _ft = cfg.inference_engine.fault_tolerance
        return _ft.max_trajectory_retries if (_ft is not None and _ft.enabled) else 0

    def test_disabled_means_no_retries(self):
        assert self._resolve(InferenceFaultToleranceConfig(enabled=False, max_trajectory_retries=5)) == 0

    def test_enabled_uses_the_configured_count(self):
        assert self._resolve(InferenceFaultToleranceConfig(enabled=True, max_trajectory_retries=3)) == 3

    def test_default_config_is_off(self):
        assert self._resolve(InferenceFaultToleranceConfig()) == 0


class TestTrajectoryRetry:
    @pytest.mark.asyncio
    async def test_a_transport_failure_is_re_run_from_scratch(self):
        gen = _generator(max_retries=2)
        attempts = []

        async def _loop(*args, attempt=0, **kwargs):
            attempts.append(attempt)
            if attempt < 2:
                raise aiohttp.ClientConnectionError("engine died")
            return "ok"

        gen.agent_loop = _loop
        assert await gen._agent_loop_with_retry("prompt") == "ok"
        assert attempts == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_retries_are_bounded(self):
        gen = _generator(max_retries=2)
        attempts = []

        async def _loop(*args, attempt=0, **kwargs):
            attempts.append(attempt)
            raise asyncio.TimeoutError()

        gen.agent_loop = _loop
        with pytest.raises(asyncio.TimeoutError):
            await gen._agent_loop_with_retry("prompt")
        assert attempts == [0, 1, 2], "max_trajectory_retries=2 means three attempts total"

    @pytest.mark.asyncio
    async def test_non_transport_failures_fail_fast(self):
        """An env bug or a bad config would just reproduce itself; re-running it
        wastes a rollout and hides the error behind retries."""
        gen = _generator(max_retries=2)
        attempts = []

        async def _loop(*args, attempt=0, **kwargs):
            attempts.append(attempt)
            raise ValueError("bad env")

        gen.agent_loop = _loop
        with pytest.raises(ValueError):
            await gen._agent_loop_with_retry("prompt")
        assert attempts == [0], "a non-transport error must not be retried"

    @pytest.mark.asyncio
    async def test_the_fleet_floor_error_is_not_retried(self):
        """``EngineFleetError`` means there is nothing left to retry against."""
        from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
            EngineFleetError,
        )

        gen = _generator(max_retries=2)
        attempts = []

        async def _loop(*args, attempt=0, **kwargs):
            attempts.append(attempt)
            raise EngineFleetError("fleet below floor")

        gen.agent_loop = _loop
        with pytest.raises(EngineFleetError):
            await gen._agent_loop_with_retry("prompt")
        assert attempts == [0]

    @pytest.mark.asyncio
    async def test_zero_retries_is_a_single_pass_through(self):
        """With fault tolerance off this wrapper must be behaviourally invisible."""
        gen = _generator(max_retries=0)
        attempts = []

        async def _loop(*args, attempt=0, **kwargs):
            attempts.append(attempt)
            raise aiohttp.ClientConnectionError("engine died")

        gen.agent_loop = _loop
        with pytest.raises(aiohttp.ClientConnectionError):
            await gen._agent_loop_with_retry("prompt")
        assert attempts == [0]

    @pytest.mark.asyncio
    async def test_the_attempt_ordinal_reaches_agent_loop(self):
        """It salts the session id, so a re-run does not inherit the sticky routing
        of the attempt that just failed."""
        gen = _generator(max_retries=1)
        seen = []

        async def _loop(*args, attempt=0, **kwargs):
            seen.append(attempt)
            if attempt == 0:
                raise aiohttp.ClientError("transport")
            return attempt

        gen.agent_loop = _loop
        assert await gen._agent_loop_with_retry("prompt") == 1
        assert seen == [0, 1]

    @pytest.mark.asyncio
    async def test_arguments_are_passed_through_unchanged(self):
        gen = _generator(max_retries=1)
        captured = {}

        async def _loop(prompt, env_class, env_extras, max_tokens, max_input_length, attempt=0, **kwargs):
            captured.update(
                prompt=prompt,
                env_class=env_class,
                env_extras=env_extras,
                max_tokens=max_tokens,
                max_input_length=max_input_length,
                kwargs=kwargs,
            )
            return "ok"

        gen.agent_loop = _loop
        await gen._agent_loop_with_retry(
            "p", "gsm8k", {"x": 1}, 32, 512, sampling_params={"temperature": 0.5}, cache_salt="v3"
        )
        assert captured["prompt"] == "p"
        assert captured["env_class"] == "gsm8k"
        assert captured["env_extras"] == {"x": 1}
        assert captured["max_tokens"] == 32 and captured["max_input_length"] == 512
        assert captured["kwargs"]["sampling_params"] == {"temperature": 0.5}
        assert captured["kwargs"]["cache_salt"] == "v3"


class TestGatherCancelsSiblings:
    @pytest.mark.asyncio
    async def test_siblings_are_cancelled_when_one_fails(self):
        """The orphan fix. Without it the survivors keep issuing requests against a
        broken fleet for the whole of the driver's unwind."""
        gen = _generator()
        started = asyncio.Event()
        cancelled = []

        async def _slow(i):
            started.set()
            try:
                await asyncio.sleep(30)
                return i
            except asyncio.CancelledError:
                cancelled.append(i)
                raise

        async def _boom():
            await started.wait()
            raise RuntimeError("engine died")

        with pytest.raises(RuntimeError, match="engine died"):
            await gen._gather_trajectories([_slow(0), _boom(), _slow(2)], disable_tqdm=True)
        assert sorted(cancelled) == [0, 2]

    @pytest.mark.asyncio
    async def test_gather_returns_only_after_the_siblings_are_done(self):
        """Draining matters as much as cancelling: each trajectory's
        ``finally: finish_session`` has to run before the driver tears the client
        down, or the cleanup races the shutdown."""
        gen = _generator()
        started = asyncio.Event()
        finished = []

        async def _slow(i):
            started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                await asyncio.sleep(0)  # stand-in for the finish_session await
                finished.append(i)
                raise

        async def _boom():
            await started.wait()
            raise RuntimeError("engine died")

        with pytest.raises(RuntimeError):
            await gen._gather_trajectories([_slow(0), _boom(), _slow(2)], disable_tqdm=True)
        assert sorted(finished) == [0, 2], "gather must not return before cleanup lands"

    @pytest.mark.asyncio
    async def test_results_keep_input_order(self):
        """The positional contract every downstream consumer of GeneratorOutput
        depends on."""
        gen = _generator()

        async def _work(i, delay):
            await asyncio.sleep(delay)
            return i

        out = await gen._gather_trajectories(
            [_work(0, 0.03), _work(1, 0.0), _work(2, 0.015)],
            disable_tqdm=True,
        )
        assert out == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_an_empty_batch_is_fine(self):
        gen = _generator()
        assert await gen._gather_trajectories([], disable_tqdm=True) == []
