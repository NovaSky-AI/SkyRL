import asyncio

import pytest

from skyrl.utils.adaptive_concurrency import (
    ConcurrencyContext,
    ConcurrencyDecision,
    ConcurrencyPolicy,
    EngineLoadConcurrencyPolicy,
    FixedConcurrencyPolicy,
    ResizableConcurrencyLimiter,
    SamplingCompletion,
    SamplingConcurrencyController,
    SamplingFeedback,
    VLLMEngineLoad,
    VLLMEngineSamplingFeedback,
)


def context(*, current_limit: int, in_flight: int | None = None) -> ConcurrencyContext:
    return ConcurrencyContext(
        current_limit=current_limit,
        in_flight=current_limit if in_flight is None else in_flight,
        observed_at_s=123.0,
    )


@pytest.mark.asyncio
async def test_resize_up_admits_waiting_work():
    limiter = ResizableConcurrencyLimiter(2)
    await limiter.acquire()
    await limiter.acquire()

    waiter = asyncio.create_task(limiter.acquire())
    await asyncio.sleep(0)
    assert limiter.in_flight == 2
    assert limiter.waiting == 1

    await limiter.resize(3)
    await asyncio.wait_for(waiter, timeout=1)
    assert limiter.in_flight == 3
    assert limiter.waiting == 0

    for _ in range(3):
        await limiter.release()


@pytest.mark.asyncio
async def test_resize_down_does_not_cancel_in_flight_work():
    limiter = ResizableConcurrencyLimiter(3)
    for _ in range(3):
        await limiter.acquire()

    await limiter.resize(1)
    waiter = asyncio.create_task(limiter.acquire())
    await asyncio.sleep(0)

    await limiter.release()
    await limiter.release()
    await asyncio.sleep(0)
    assert not waiter.done()

    await limiter.release()
    await asyncio.wait_for(waiter, timeout=1)
    assert limiter.in_flight == 1
    await limiter.release()


@pytest.mark.asyncio
async def test_slot_releases_after_exception():
    limiter = ResizableConcurrencyLimiter(1)

    with pytest.raises(RuntimeError, match="sampling failed"):
        async with limiter.slot():
            raise RuntimeError("sampling failed")

    assert limiter.in_flight == 0
    async with limiter.slot():
        assert limiter.in_flight == 1


@pytest.mark.asyncio
async def test_controller_co_locates_policy_and_request_limiter():
    class GrowPolicy:
        def on_feedback(self, feedback, policy_context):
            return ConcurrencyDecision(desired_limit=policy_context.current_limit + 1, reason="backend_clear")

        def on_completion(self, completion, policy_context):
            return None

    policy = GrowPolicy()
    controller = SamplingConcurrencyController(policy=policy, initial_limit=2, clock=lambda: 456.0)

    async with controller.slot():
        assert controller.in_flight == 1
        assert controller.context() == ConcurrencyContext(current_limit=2, in_flight=1, observed_at_s=456.0)
        decision = await controller.on_feedback(SamplingFeedback({"custom.queue_duration_s": 0.1}))

    assert controller.policy is policy
    assert decision == ConcurrencyDecision(desired_limit=3, reason="backend_clear")
    assert controller.current_limit == 3
    assert controller.in_flight == 0


@pytest.mark.asyncio
async def test_controller_forwards_decisions_to_service_owned_effect_handler():
    class ShedPolicy:
        def on_feedback(self, feedback, policy_context):
            return ConcurrencyDecision(desired_limit=1, reason="hard_pressure", shed_count=2)

        def on_completion(self, completion, policy_context):
            return None

    decisions = []
    controller = SamplingConcurrencyController(policy=ShedPolicy(), initial_limit=4, decision_handler=decisions.append)

    decision = await controller.on_feedback(SamplingFeedback())

    assert decision == ConcurrencyDecision(desired_limit=1, reason="hard_pressure", shed_count=2)
    assert decisions == [decision]


@pytest.mark.asyncio
async def test_controller_forwards_request_completion_to_policy():
    class CompletionPolicy:
        completion = None

        def on_feedback(self, feedback, policy_context):
            return None

        def on_completion(self, completion, policy_context):
            self.completion = completion
            return ConcurrencyDecision(
                desired_limit=policy_context.current_limit + 1,
                reason="completion_growth",
            )

    policy = CompletionPolicy()
    controller = SamplingConcurrencyController(policy=policy, initial_limit=4)
    completion = SamplingCompletion(duration_s=2.5)

    decision = await controller.on_completion(completion)

    assert policy.completion is completion
    assert decision == ConcurrencyDecision(desired_limit=5, reason="completion_growth")
    assert controller.current_limit == 5


def test_policy_protocol_supports_feedback_and_completion_hooks():
    policy = FixedConcurrencyPolicy()

    assert isinstance(policy, ConcurrencyPolicy)
    assert policy.on_feedback(SamplingFeedback(), context(current_limit=4)) is None
    assert policy.on_completion(SamplingCompletion(duration_s=1.25), context(current_limit=4)) is None


def test_feedback_accepts_a_coherent_engine_load_snapshot():
    loads = (
        VLLMEngineLoad(
            engine_id="decode-0",
            role="decode",
            kv_capacity_tokens=131_072,
            max_model_len=32_768,
            kv_usage=0.72,
            running=12,
            waiting=2,
            waiting_capacity=1,
            preemptions_delta=0,
        ),
        VLLMEngineLoad(
            engine_id="prefill-0",
            role="prefill",
            kv_capacity_tokens=None,
            max_model_len=32_768,
            kv_usage=0.1,
            running=4,
            waiting=0,
            waiting_capacity=None,
            preemptions_delta=0,
        ),
    )

    feedback = VLLMEngineSamplingFeedback(metrics={"backend.router.requests": 16}, engine_loads=loads)

    assert feedback.engine_loads == loads
    assert feedback.metrics["engine.loads"][0]["engine_id"] == "decode-0"


def engine_feedback(
    *,
    kv_usage: float,
    running: int = 8,
    waiting: int = 0,
    waiting_capacity: int | None = None,
    preemptions_delta: int = 0,
    role: str | None = "decode",
) -> VLLMEngineSamplingFeedback:
    return VLLMEngineSamplingFeedback(
        engine_loads=(
            VLLMEngineLoad(
                engine_id="engine-0",
                role=role,
                kv_capacity_tokens=131_072,
                max_model_len=32_768,
                kv_usage=kv_usage,
                running=running,
                waiting=waiting,
                waiting_capacity=waiting_capacity,
                preemptions_delta=preemptions_delta,
            ),
        )
    )


def test_engine_load_policy_soft_trims_kv_pressure():
    policy = EngineLoadConcurrencyPolicy(min_limit=1, max_limit=128)

    decision = policy.on_feedback(
        engine_feedback(kv_usage=0.85),
        context(current_limit=100, in_flight=90),
    )

    assert decision == ConcurrencyDecision(desired_limit=74, reason="kv_soft_trim")


def test_engine_load_policy_hard_trim_requests_active_shedding():
    policy = EngineLoadConcurrencyPolicy(min_limit=1, max_limit=128)

    decision = policy.on_feedback(
        engine_feedback(kv_usage=0.91),
        context(current_limit=100, in_flight=90),
    )

    assert decision == ConcurrencyDecision(desired_limit=69, reason="kv_hard_trim", shed_count=21)


def test_engine_load_policy_cuts_on_preemption_and_persistent_capacity_queue():
    preemption_policy = EngineLoadConcurrencyPolicy(min_limit=1, max_limit=128)
    preemption = preemption_policy.on_feedback(
        engine_feedback(kv_usage=0.5, preemptions_delta=1),
        context(current_limit=100, in_flight=100),
    )
    assert preemption == ConcurrencyDecision(desired_limit=80, reason="engine_preemptions", shed_count=20)

    queue_policy = EngineLoadConcurrencyPolicy(min_limit=1, max_limit=128)
    for _ in range(queue_policy.QUEUE_PERSISTENCE_POLLS - 1):
        assert (
            queue_policy.on_feedback(
                engine_feedback(kv_usage=0.5, running=10, waiting=6, waiting_capacity=6),
                context(current_limit=100, in_flight=100),
            )
            is None
        )
    queue = queue_policy.on_feedback(
        engine_feedback(kv_usage=0.5, running=10, waiting=6, waiting_capacity=6),
        context(current_limit=100, in_flight=100),
    )
    assert queue == ConcurrencyDecision(desired_limit=90, reason="engine_queue_overload", shed_count=10)


def test_engine_load_policy_escalates_persistent_pressure_after_drain():
    policy = EngineLoadConcurrencyPolicy(min_limit=1, max_limit=128)
    pressure = engine_feedback(kv_usage=0.99, preemptions_delta=10)
    first = policy.on_feedback(pressure, context(current_limit=64, in_flight=64))
    assert first == ConcurrencyDecision(51, "engine_preemptions", 13)
    assert policy.on_feedback(pressure, context(current_limit=51, in_flight=64)) is None
    second = policy.on_feedback(pressure, context(current_limit=51, in_flight=51))
    assert second == ConcurrencyDecision(25, "engine_preemptions", 26)


def test_engine_load_policy_grows_by_turnover_only_while_recent_scrape_is_clear():
    policy = EngineLoadConcurrencyPolicy(min_limit=1, max_limit=16)
    clear_context = ConcurrencyContext(current_limit=4, in_flight=4, observed_at_s=100.0)
    assert policy.on_feedback(engine_feedback(kv_usage=0.5), clear_context) is None

    decision = None
    for offset in range(5):
        decision = policy.on_completion(
            SamplingCompletion(duration_s=1),
            ConcurrencyContext(current_limit=4, in_flight=3, observed_at_s=101.0 + offset),
        )
    assert decision == ConcurrencyDecision(desired_limit=5, reason="clear_engine_turnover")

    stale_policy = EngineLoadConcurrencyPolicy(min_limit=1, max_limit=16)
    assert stale_policy.on_feedback(engine_feedback(kv_usage=0.5), clear_context) is None
    assert (
        stale_policy.on_completion(
            SamplingCompletion(duration_s=1),
            ConcurrencyContext(current_limit=4, in_flight=3, observed_at_s=116.0),
        )
        is None
    )


def test_engine_load_policy_ignores_generic_feedback_and_prefill_only_load():
    policy = EngineLoadConcurrencyPolicy(min_limit=1, max_limit=16)

    assert policy.on_feedback(SamplingFeedback({"custom.queue_ms": 10}), context(current_limit=4)) is None
    assert (
        policy.on_feedback(
            engine_feedback(kv_usage=0.99, preemptions_delta=3, role="prefill"),
            context(current_limit=4),
        )
        is None
    )


def test_generic_feedback_preserves_backend_specific_metrics():
    feedback = SamplingFeedback({"custom.queue_ms": 12.5, "custom.metadata": {"region": "test"}})

    assert feedback.metrics["custom.queue_ms"] == 12.5
    assert feedback.metrics["custom.metadata"] == {"region": "test"}


@pytest.mark.parametrize(
    "value",
    [
        ConcurrencyContext(current_limit=1, in_flight=0, observed_at_s=0),
        SamplingCompletion(duration_s=0),
        ConcurrencyDecision(desired_limit=1),
    ],
)
def test_contracts_accept_boundary_values(value):
    assert value is not None


def test_contracts_reject_invalid_values():
    with pytest.raises(ValueError, match="current_limit"):
        ConcurrencyContext(current_limit=0, in_flight=0, observed_at_s=1)
    with pytest.raises(ValueError, match="in_flight"):
        ConcurrencyContext(current_limit=1, in_flight=-1, observed_at_s=1)
    with pytest.raises(ValueError, match="observed_at_s"):
        ConcurrencyContext(current_limit=1, in_flight=0, observed_at_s=float("inf"))
    with pytest.raises(ValueError, match="duration_s"):
        SamplingCompletion(duration_s=-1)
    with pytest.raises(ValueError, match="shed_count"):
        ConcurrencyDecision(desired_limit=1, shed_count=-1)


@pytest.mark.asyncio
async def test_reacquiring_holder_cannot_barge_ahead_of_queued_work():
    limiter = ResizableConcurrencyLimiter(1)
    order = []
    await limiter.acquire()

    async def older_waiter():
        await limiter.acquire()
        order.append("older")
        await limiter.release()

    older = asyncio.create_task(older_waiter())
    await asyncio.sleep(0)
    await limiter.release()
    # No yield before reacquiring: this reproduces an immediate retry.
    async with asyncio.timeout(1):
        await limiter.acquire()
    order.append("retry")
    await limiter.release()
    await older
    assert order == ["older", "retry"]
    assert limiter.waiting == 0 and limiter.in_flight == 0


@pytest.mark.asyncio
async def test_cancelled_front_waiter_does_not_block_the_queue():
    limiter = ResizableConcurrencyLimiter(1)
    await limiter.acquire()
    first = asyncio.create_task(limiter.acquire())
    await asyncio.sleep(0)
    second = asyncio.create_task(limiter.acquire())
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    await limiter.release()
    await asyncio.wait_for(second, timeout=1)
    assert limiter.waiting == 0 and limiter.in_flight == 1
    await limiter.release()
