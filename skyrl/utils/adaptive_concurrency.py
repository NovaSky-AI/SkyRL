"""Backend-neutral contracts for adaptive asynchronous sampling admission."""

import asyncio
import math
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Protocol, TypeAlias, runtime_checkable

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

ENGINE_LOADS_METRIC = "engine.loads"


@dataclass(frozen=True)
class VLLMEngineLoad:
    """One coherent vLLM engine-load observation.

    This is deliberately a value object instead of a loose metrics dictionary:
    policies that depend on KV pressure, waiting work, or preemptions must not
    accidentally combine values from different engines or scrape intervals.
    """

    engine_id: str
    role: str | None
    kv_capacity_tokens: int | None
    max_model_len: int | None
    kv_usage: float
    running: int
    waiting: int
    waiting_capacity: int | None
    preemptions_delta: int

    def __post_init__(self) -> None:
        if not self.engine_id:
            raise ValueError("engine_id must not be empty")
        for name in ("kv_capacity_tokens", "max_model_len", "waiting_capacity"):
            value = getattr(self, name)
            if value is not None:
                _require_non_negative_int(value, path=name)
        for name in ("running", "waiting", "preemptions_delta"):
            _require_non_negative_int(getattr(self, name), path=name)
        if isinstance(self.kv_usage, bool) or not isinstance(self.kv_usage, (int, float)):
            raise TypeError(f"kv_usage must be numeric, got {type(self.kv_usage).__name__}")
        if not math.isfinite(self.kv_usage) or self.kv_usage < 0:
            raise ValueError(f"kv_usage must be a finite non-negative value, got {self.kv_usage}")


@dataclass(frozen=True)
class SamplingFeedback:
    """Generic backend feedback for custom integrations.

    Managed vLLM uses :class:`VLLMEngineSamplingFeedback`. Other backends can
    extend this value with a typed subclass once they expose an actionable
    pressure signal; ``metrics`` remains available for experimental/custom
    integrations without requiring a framework-wide producer abstraction.
    """

    metrics: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", dict(self.metrics))


@dataclass(frozen=True)
class VLLMEngineSamplingFeedback(SamplingFeedback):
    """Typed coherent snapshot from a managed vLLM deployment."""

    engine_loads: tuple[VLLMEngineLoad, ...] = ()

    def __post_init__(self) -> None:
        if not self.engine_loads:
            raise ValueError("engine_loads must not be empty")
        metrics = dict(self.metrics)
        if ENGINE_LOADS_METRIC in metrics:
            raise ValueError(f"{ENGINE_LOADS_METRIC} is derived from engine_loads and must not be supplied in metrics")
        metrics[ENGINE_LOADS_METRIC] = [
            {
                "engine_id": load.engine_id,
                "role": load.role,
                "kv_capacity_tokens": load.kv_capacity_tokens,
                "max_model_len": load.max_model_len,
                "kv_usage": load.kv_usage,
                "running": load.running,
                "waiting": load.waiting,
                "waiting_capacity": load.waiting_capacity,
                "preemptions_delta": load.preemptions_delta,
            }
            for load in self.engine_loads
        ]
        object.__setattr__(self, "metrics", metrics)
        super().__post_init__()


@dataclass(frozen=True)
class SamplingCompletion:
    """Lifecycle event emitted after one admitted inference request completes."""

    duration_s: float
    metrics: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.duration_s, bool) or not isinstance(self.duration_s, (int, float)):
            raise TypeError(f"duration_s must be numeric, got {type(self.duration_s).__name__}")
        if not math.isfinite(self.duration_s) or self.duration_s < 0:
            raise ValueError(f"duration_s must be a finite non-negative value, got {self.duration_s}")
        object.__setattr__(self, "metrics", dict(self.metrics))


@dataclass(frozen=True)
class ConcurrencyContext:
    """Inference-request admission state captured for a policy hook."""

    current_limit: int
    in_flight: int
    observed_at_s: float

    def __post_init__(self) -> None:
        if isinstance(self.current_limit, bool) or not isinstance(self.current_limit, int):
            raise TypeError(f"current_limit must be an integer, got {type(self.current_limit).__name__}")
        if self.current_limit < 1:
            raise ValueError(f"current_limit must be at least 1, got {self.current_limit}")
        if isinstance(self.in_flight, bool) or not isinstance(self.in_flight, int):
            raise TypeError(f"in_flight must be an integer, got {type(self.in_flight).__name__}")
        if self.in_flight < 0:
            raise ValueError(f"in_flight must be non-negative, got {self.in_flight}")
        if isinstance(self.observed_at_s, bool) or not isinstance(self.observed_at_s, (int, float)):
            raise TypeError(f"observed_at_s must be numeric, got {type(self.observed_at_s).__name__}")
        if not math.isfinite(self.observed_at_s) or self.observed_at_s < 0:
            raise ValueError(
                f"observed_at_s must be a finite non-negative monotonic timestamp, got {self.observed_at_s}"
            )


@dataclass(frozen=True)
class ConcurrencyDecision:
    """New concurrent inference-request limit requested by a policy."""

    desired_limit: int
    reason: str = "unspecified"

    def __post_init__(self) -> None:
        if isinstance(self.desired_limit, bool) or not isinstance(self.desired_limit, int):
            raise TypeError(f"desired_limit must be an integer, got {type(self.desired_limit).__name__}")
        if self.desired_limit < 1:
            raise ValueError(f"desired_limit must be at least 1, got {self.desired_limit}")
        if not self.reason:
            raise ValueError("reason must not be empty")


@runtime_checkable
class ConcurrencyPolicy(Protocol):
    """Stateful strategy consumed by a sampling admission controller.

    The policy observes feedback and request completions, but never owns the
    queue, inference calls, or generator trajectory tasks.  The long-lived
    admission owner applies the returned decision.
    """

    def on_feedback(self, feedback: SamplingFeedback, context: ConcurrencyContext) -> ConcurrencyDecision | None: ...

    def on_completion(
        self, completion: SamplingCompletion, context: ConcurrencyContext
    ) -> ConcurrencyDecision | None: ...


class ResizableConcurrencyLimiter:
    """An async concurrency limiter whose limit can change at runtime.

    Lowering the limit never cancels work that already holds a slot. New work
    waits until the number of holders falls below the new limit.
    Queued calls are admitted in arrival order, including restarted work.
    """

    def __init__(self, limit: int):
        if limit < 1:
            raise ValueError(f"limit must be at least 1, got {limit}")
        self._limit = limit
        self._in_flight = 0
        self._waiting = 0
        self._waiters: deque[object] = deque()
        self._condition = asyncio.Condition()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def waiting(self) -> int:
        return self._waiting

    async def acquire(self) -> None:
        async with self._condition:
            waiter = object()
            self._waiters.append(waiter)
            self._waiting += 1
            try:
                await self._condition.wait_for(lambda: self._waiters[0] is waiter and self._in_flight < self._limit)
                self._in_flight += 1
            finally:
                self._waiters.remove(waiter)
                self._waiting -= 1
                if self._in_flight < self._limit:
                    self._condition.notify_all()

    async def release(self) -> None:
        async with self._condition:
            if self._in_flight < 1:
                raise RuntimeError("cannot release a concurrency slot when none are held")
            self._in_flight -= 1
            self._condition.notify_all()

    async def resize(self, limit: int) -> None:
        if limit < 1:
            raise ValueError(f"limit must be at least 1, got {limit}")
        async with self._condition:
            self._limit = limit
            self._condition.notify_all()

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Hold one admission slot for one inference request."""

        await self.acquire()
        try:
            yield
        finally:
            await self.release()


class SamplingConcurrencyController:
    """Co-locate one concurrency policy with the limiter it controls.

    One long-lived sampling service owns the controller, acquires a slot around
    every physical sampling request, and publishes backend observations to it.
    Lightweight clients submit work to that service and do not independently
    interpret backend pressure. Policy decisions resize the limiter atomically
    with respect to other policy callbacks.
    """

    def __init__(
        self,
        *,
        policy: ConcurrencyPolicy,
        initial_limit: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._policy = policy
        self._limiter = ResizableConcurrencyLimiter(initial_limit)
        self._clock = clock
        self._policy_lock = asyncio.Lock()

    @property
    def policy(self) -> ConcurrencyPolicy:
        return self._policy

    @property
    def current_limit(self) -> int:
        return self._limiter.limit

    @property
    def in_flight(self) -> int:
        return self._limiter.in_flight

    @property
    def waiting(self) -> int:
        return self._limiter.waiting

    def context(self) -> ConcurrencyContext:
        """Capture the limiter state supplied to the next policy hook."""

        return ConcurrencyContext(
            current_limit=self.current_limit,
            in_flight=self.in_flight,
            observed_at_s=self._clock(),
        )

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Hold one request slot in the shared admission window."""

        async with self._limiter.slot():
            yield

    async def on_feedback(self, feedback: SamplingFeedback) -> ConcurrencyDecision | None:
        """Run the policy feedback hook and apply its requested limit."""

        async with self._policy_lock:
            decision = self._policy.on_feedback(feedback, self.context())
            return await self._apply(decision)

    async def on_completion(self, completion: SamplingCompletion) -> ConcurrencyDecision | None:
        """Run the policy completion hook and apply its requested limit."""

        async with self._policy_lock:
            decision = self._policy.on_completion(completion, self.context())
            return await self._apply(decision)

    async def _apply(self, decision: ConcurrencyDecision | None) -> ConcurrencyDecision | None:
        if decision is not None:
            await self._limiter.resize(decision.desired_limit)
        return decision


class FixedConcurrencyPolicy:
    """Safe fallback for backends with no trustworthy pressure signal."""

    def on_feedback(self, feedback: SamplingFeedback, context: ConcurrencyContext) -> ConcurrencyDecision | None:
        return None

    def on_completion(self, completion: SamplingCompletion, context: ConcurrencyContext) -> ConcurrencyDecision | None:
        return None


class EngineLoadConcurrencyPolicy:
    """Adaptive policy for coherent per-engine vLLM load snapshots.

    The policy grows by pipeline turnover while the most recent scrape is
    clear, soft-trims when KV usage loses headroom, and cuts on preemptions or
    a persistent capacity queue. It is intentionally a state machine only;
    lowering the admission limit lets existing work drain naturally.

    Thresholds are conservative implementation constants rather than user
    hyperparameters.  The public tuning surface is the initial/min/max window;
    exposing every threshold before there is operating data would merely move
    manual concurrency tuning into a larger configuration space.
    """

    TURNOVER_GROWTH = 1.25
    BINDING_FRACTION = 0.9
    KV_USAGE_GROW = 0.6
    KV_USAGE_SOFT_CAP = 0.8
    KV_USAGE_HARD_CAP = 0.9
    KV_USAGE_TARGET = 0.7
    KV_TRIM_COOLDOWN_POLLS = 6
    QUEUE_RATIO = 0.5
    QUEUE_PERSISTENCE_POLLS = 6
    QUEUE_CUT_FRACTION = 0.9
    PREEMPTION_CUT_FRACTION = 0.8
    ESCALATED_CUT_FRACTION = 0.5
    ESCALATION_GRACE_POLLS = 6
    GROWTH_GATE_TTL_S = 15.0

    def __init__(
        self,
        *,
        min_limit: int = 1,
        max_limit: int,
    ) -> None:
        if min_limit < 1:
            raise ValueError(f"min_limit must be at least 1, got {min_limit}")
        if max_limit < min_limit:
            raise ValueError(f"max_limit must be at least min_limit ({min_limit}), got {max_limit}")
        self.min_limit = min_limit
        self.max_limit = max_limit

        self._cap: float | None = None
        self._turnover = 0.0
        self._can_grow = False
        self._can_grow_until_s = 0.0
        self._previous_waiting: dict[str, int] = {}
        self._queue_overload_polls = 0
        self._trim_cooldown_polls = 0
        self._draining = False
        self._escalated = False
        self._escalation_grace_polls = 0

    def on_feedback(self, feedback: SamplingFeedback, context: ConcurrencyContext) -> ConcurrencyDecision | None:
        """Classify one coherent scrape and request a resize when needed."""

        if not isinstance(feedback, VLLMEngineSamplingFeedback):
            return None

        loads = tuple(load for load in feedback.engine_loads if load.role != "prefill")
        if not loads:
            return None
        self._sync_cap(context)

        max_usage = max(load.kv_usage for load in loads)
        total_running = sum(load.running for load in loads)
        total_capacity_waiting = sum(
            load.waiting_capacity if load.waiting_capacity is not None else load.waiting for load in loads
        )
        preempted = any(load.preemptions_delta > 0 for load in loads)

        if total_running > 0 and total_capacity_waiting > self.QUEUE_RATIO * total_running:
            self._queue_overload_polls += 1
        else:
            self._queue_overload_polls = 0
        queue_overload = self._queue_overload_polls >= self.QUEUE_PERSISTENCE_POLLS

        # A queue observed in two successive polls is pressure even before it
        # reaches the hard, ratio-based cut.  It closes the growth gate without
        # overreacting to a single turn-completion burst.
        repeated_waiting = any(load.waiting > 0 and self._previous_waiting.get(load.engine_id, 0) > 0 for load in loads)
        self._previous_waiting = {load.engine_id: load.waiting for load in loads}

        if self._draining and context.in_flight <= context.current_limit:
            self._draining = False
            self._escalation_grace_polls = self.ESCALATION_GRACE_POLLS
        elif not self._draining and self._escalated:
            self._escalation_grace_polls -= 1
            if self._escalation_grace_polls <= 0:
                self._escalated = False

        self._trim_cooldown_polls = max(0, self._trim_cooldown_polls - 1)
        self._can_grow = (
            max_usage <= self.KV_USAGE_GROW
            and total_capacity_waiting == 0
            and not repeated_waiting
            and not preempted
            and not self._draining
        )
        self._can_grow_until_s = context.observed_at_s + self.GROWTH_GATE_TTL_S

        if self._draining:
            return None

        if preempted or queue_overload:
            cut_fraction = (
                self.ESCALATED_CUT_FRACTION
                if self._escalated
                else (self.QUEUE_CUT_FRACTION if queue_overload else self.PREEMPTION_CUT_FRACTION)
            )
            target = self._clamp(math.floor(context.in_flight * cut_fraction))
            reason = "engine_queue_overload" if queue_overload else "engine_preemptions"
            self._queue_overload_polls = 0
            self._draining = True
            self._escalated = True
            return self._resize_down(target, context, reason=reason)

        if max_usage > self.KV_USAGE_SOFT_CAP and context.in_flight > 0 and self._trim_cooldown_polls == 0:
            target = self._clamp(math.floor(context.in_flight * self.KV_USAGE_TARGET / max_usage))
            self._trim_cooldown_polls = self.KV_TRIM_COOLDOWN_POLLS
            return self._resize_down(
                target,
                context,
                reason="kv_hard_trim" if max_usage > self.KV_USAGE_HARD_CAP else "kv_soft_trim",
            )

        return None

    def on_completion(self, completion: SamplingCompletion, context: ConcurrencyContext) -> ConcurrencyDecision | None:
        """Pace multiplicative growth by completed request turnover."""

        self._sync_cap(context)
        completed_in_flight = context.in_flight + 1
        fraction = 1 / completed_in_flight
        self._turnover += fraction
        if not (
            self._can_grow
            and context.observed_at_s < self._can_grow_until_s
            and completed_in_flight >= self.BINDING_FRACTION * context.current_limit
        ):
            return None

        assert self._cap is not None
        self._cap = self._clamp_float(self._cap * self.TURNOVER_GROWTH**fraction)
        desired_limit = int(self._cap)
        if desired_limit == context.current_limit:
            return None
        return ConcurrencyDecision(desired_limit=desired_limit, reason="clear_engine_turnover")

    @property
    def turnover(self) -> float:
        """Completed pipeline turnovers, exposed for observability and tests."""

        return self._turnover

    def _sync_cap(self, context: ConcurrencyContext) -> None:
        if self._cap is None:
            self._cap = float(context.current_limit)

    def _resize_down(
        self,
        target: int,
        context: ConcurrencyContext,
        *,
        reason: str,
    ) -> ConcurrencyDecision | None:
        target = min(target, context.current_limit)
        self._cap = float(target)
        if target == context.current_limit:
            return None
        return ConcurrencyDecision(desired_limit=target, reason=reason)

    def _clamp(self, value: int) -> int:
        return min(self.max_limit, max(self.min_limit, value))

    def _clamp_float(self, value: float) -> float:
        return min(float(self.max_limit), max(float(self.min_limit), value))


def _require_non_negative_int(value: object, *, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be an integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{path} must be non-negative, got {value}")
