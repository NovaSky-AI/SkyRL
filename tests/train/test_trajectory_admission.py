import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from skyrl.train.sampling_service import SamplingClient, SamplingService, TrajectoryShed
from skyrl.utils.adaptive_concurrency import (
    ConcurrencyDecision,
    EngineLoadConcurrencyPolicy,
    SamplingConcurrencyController,
    SamplingFeedback,
    VLLMEngineLoad,
    VLLMEngineSamplingFeedback,
)


class Policy:
    def __init__(self):
        self.completions = []
        self.decision = None

    def on_feedback(self, feedback, context):
        decision, self.decision = self.decision, None
        return decision

    def on_completion(self, completion, context):
        self.completions.append((completion, context))


def setup(limit=1):
    policy = Policy()
    controller = SamplingConcurrencyController(policy=policy, initial_limit=limit)
    backend = MagicMock()
    backend.chat_completion = AsyncMock(return_value={"response_ids": [[1, 2]]})
    backend.finish_session = AsyncMock()
    service = SamplingService(backend, controller=controller, admission_scope="trajectory")
    return service, SamplingClient(service, backend), controller, policy, backend


async def waiting(controller):
    for _ in range(100):
        if controller.waiting:
            return
        await asyncio.sleep(0)
    raise AssertionError("Expected another trajectory to wait for a permit")


@pytest.mark.asyncio
async def test_permit_spans_tool_waits_and_child_requests_do_not_reacquire():
    service, client, controller, policy, backend = setup()
    tool = asyncio.Event()
    release = asyncio.Event()
    entered_second = asyncio.Event()

    async def first():
        async with client.trajectory("first"):
            await client.chat_completion({"json": {"session_id": "child-a"}})
            tool.set()
            await release.wait()
            await client.chat_completion({"json": {"session_id": "child-b"}})
            await client.finish_session("child-a")

    async def second():
        async with client.trajectory("second"):
            entered_second.set()

    a = asyncio.create_task(first())
    await asyncio.wait_for(tool.wait(), 1)
    b = asyncio.create_task(second())
    await waiting(controller)
    assert service.in_flight == 0 and controller.in_flight == 1
    assert not entered_second.is_set() and not policy.completions
    release.set()
    await asyncio.wait_for(asyncio.gather(a, b), 1)
    assert backend.chat_completion.await_count == 2
    assert len(policy.completions) == 2
    assert controller.in_flight == 0 and not service._trajectories


@pytest.mark.asyncio
async def test_sheds_youngest_tool_wait_and_holds_permit_until_caller_cleanup():
    service, _, controller, policy, _ = setup(2)
    old_started, young_started = asyncio.Event(), asyncio.Event()
    old_release, cleanup_started, cleanup_release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def old():
        async with service.trajectory("old"):
            old_started.set()
            await old_release.wait()

    async def young():
        async with service.trajectory("young"):
            try:
                young_started.set()
                await asyncio.Future()
            finally:
                cleanup_started.set()
                await cleanup_release.wait()
                await service.finish_attempt("young")

    a = asyncio.create_task(old())
    await old_started.wait()
    b = asyncio.create_task(young())
    await young_started.wait()
    policy.decision = ConcurrencyDecision(desired_limit=1, shed_count=1, reason="engine_preemptions")
    await controller.on_feedback(SamplingFeedback())
    await asyncio.wait_for(cleanup_started.wait(), 1)
    assert controller.in_flight == 2 and not a.done() and not b.done()
    cleanup_release.set()
    with pytest.raises(TrajectoryShed, match="engine_preemptions"):
        await asyncio.wait_for(b, 1)
    assert controller.in_flight == 1 and "young" not in service._revoked_attempts
    assert not policy.completions
    old_release.set()
    await a
    assert len(policy.completions) == 1


@pytest.mark.asyncio
async def test_preemption_sheds_after_soft_trim_without_a_second_limit_change():
    controller = SamplingConcurrencyController(
        policy=EngineLoadConcurrencyPolicy(min_limit=1, max_limit=16), initial_limit=2
    )
    backend = MagicMock()
    service = SamplingService(backend, controller=controller, admission_scope="trajectory")
    started = [asyncio.Event(), asyncio.Event()]
    release = asyncio.Event()

    async def run(index):
        async with service.trajectory("old" if index == 0 else "young"):
            started[index].set()
            await release.wait()

    def feedback(usage, preemptions=0):
        return VLLMEngineSamplingFeedback(
            engine_loads=(
                VLLMEngineLoad(
                    engine_id="engine",
                    role="decode",
                    kv_capacity_tokens=786432,
                    max_model_len=786432,
                    kv_usage=usage,
                    running=1,
                    waiting=0,
                    waiting_capacity=0,
                    preemptions_delta=preemptions,
                ),
            )
        )

    old = asyncio.create_task(run(0))
    await started[0].wait()
    young = asyncio.create_task(run(1))
    await started[1].wait()
    soft = await controller.on_feedback(feedback(0.85))
    assert soft == ConcurrencyDecision(1, "kv_soft_trim")
    assert controller.current_limit == 1 and controller.in_flight == 2
    assert not old.done() and not young.done()
    hard = await controller.on_feedback(feedback(0.75, 1))
    assert hard == ConcurrencyDecision(1, "engine_preemptions", shed_count=1)
    with pytest.raises(TrajectoryShed, match="engine_preemptions"):
        await asyncio.wait_for(young, 1)
    assert not old.done() and controller.current_limit == controller.in_flight == 1
    release.set()
    await old
    assert controller.in_flight == 0 and not service._trajectories
    await service.aclose()


@pytest.mark.asyncio
async def test_soft_resize_queues_new_trajectories_without_cancelling_existing_ones():
    service, _, controller, policy, _ = setup(2)
    releases = [asyncio.Event(), asyncio.Event()]
    started = [asyncio.Event(), asyncio.Event()]

    async def run(i):
        async with service.trajectory(str(i)):
            started[i].set()
            await releases[i].wait()

    tasks = [asyncio.create_task(run(i)) for i in range(2)]
    await asyncio.gather(*(e.wait() for e in started))
    policy.decision = ConcurrencyDecision(desired_limit=1, reason="kv_soft_trim")
    await controller.on_feedback(SamplingFeedback())
    assert all(not t.done() and not t.cancelling() for t in tasks)
    releases[0].set()
    await tasks[0]
    assert controller.in_flight == 1
    releases[1].set()
    await tasks[1]


@pytest.mark.asyncio
async def test_caller_cancellation_preserves_cancelled_error_and_does_not_pace_growth():
    service, _, controller, policy, _ = setup()
    entered = asyncio.Event()

    async def run():
        async with service.trajectory("a"):
            entered.set()
            await asyncio.Future()

    task = asyncio.create_task(run())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert controller.in_flight == 0 and not policy.completions and not service._trajectories


@pytest.mark.asyncio
async def test_missing_and_nested_leases_fail_before_backend_dispatch():
    service, _, _, _, backend = setup()
    with pytest.raises(ValueError, match="active trajectory lease"):
        await service.chat_completion({}, attempt_id="missing")
    async with service.trajectory("a"):
        with pytest.raises(ValueError, match="Nested"):
            async with service.trajectory("b"):
                pass
    backend.chat_completion.assert_not_awaited()


def test_http_ingress_rejects_trajectory_admission_without_a_lease_transport():
    service, client, _, _, _ = setup()
    with pytest.raises(ValueError, match="requires direct SamplingClient trajectory leases"):
        client.get_endpoint_url()
    assert service._http_proxy is None


@pytest.mark.asyncio
async def test_shutdown_cancels_active_and_pending_trajectory_owners():
    service, _, controller, policy, _ = setup()
    entered = asyncio.Event()

    async def run(name):
        async with service.trajectory(name):
            entered.set()
            await asyncio.Future()

    first = asyncio.create_task(run("a"))
    await entered.wait()
    second = asyncio.create_task(run("b"))
    await waiting(controller)
    await service.aclose()
    assert all(t.cancelled() for t in [first, second])
    assert controller.in_flight == 0 and controller.waiting == 0
    assert not policy.completions and not service._trajectories


@pytest.mark.asyncio
async def test_swallowed_policy_cancellation_cannot_credit_a_successful_trajectory():
    service, _, controller, policy, _ = setup()
    entered = asyncio.Event()

    async def run():
        async with service.trajectory("a"):
            entered.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                pass

    task = asyncio.create_task(run())
    await entered.wait()
    policy.decision = ConcurrencyDecision(desired_limit=1, shed_count=1, reason="pressure")
    await controller.on_feedback(SamplingFeedback())
    with pytest.raises(TrajectoryShed):
        await task
    assert not policy.completions and controller.in_flight == 0


@pytest.mark.asyncio
async def test_parallel_child_requests_share_one_parent_permit():
    service, client, controller, _, backend = setup()
    both_started, release = asyncio.Event(), asyncio.Event()
    active = 0

    async def serve(payload):
        nonlocal active
        active += 1
        if active == 2:
            both_started.set()
        await release.wait()
        return {"response_ids": [[1]]}

    backend.chat_completion = serve

    async def parent():
        async with client.trajectory("parent"):
            await asyncio.gather(*(client.chat_completion({"json": {"session_id": str(i)}}) for i in range(2)))

    task = asyncio.create_task(parent())
    await asyncio.wait_for(both_started.wait(), 1)
    assert controller.in_flight == 1 and service.in_flight == 2
    release.set()
    await task


@pytest.mark.asyncio
async def test_retry_can_reuse_attempt_id_after_shed_cleanup():
    service, _, controller, policy, _ = setup()
    entered = asyncio.Event()

    async def run():
        async with service.trajectory("same"):
            entered.set()
            await asyncio.Future()

    task = asyncio.create_task(run())
    await entered.wait()
    policy.decision = ConcurrencyDecision(desired_limit=1, shed_count=1, reason="pressure")
    await controller.on_feedback(SamplingFeedback())
    with pytest.raises(TrajectoryShed):
        await task
    async with service.trajectory("same"):
        await service.chat_completion({}, attempt_id="same")
    assert len(policy.completions) == 1


@pytest.mark.asyncio
async def test_caller_cancellation_concurrent_with_shedding_is_not_retried():
    service, _, controller, policy, _ = setup()
    entered = asyncio.Event()

    async def run():
        async with service.trajectory("a"):
            entered.set()
            await asyncio.Future()

    task = asyncio.create_task(run())
    await entered.wait()
    policy.decision = ConcurrencyDecision(desired_limit=1, shed_count=1, reason="pressure")
    await controller.on_feedback(SamplingFeedback())
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not policy.completions and controller.in_flight == 0


@pytest.mark.asyncio
async def test_real_generator_retry_boundary_reacquires_after_tool_wait_shedding():
    from types import SimpleNamespace

    from skyrl.train.config import GeneratorConfig
    from skyrl.train.generators.skyrl_gym_generator import (
        SkyRLGymGenerator,
        TrajectoryOutput,
    )

    service, client, controller, policy, _ = setup()
    generator = SkyRLGymGenerator.__new__(SkyRLGymGenerator)
    generator.generator_cfg = GeneratorConfig()
    generator.generator_cfg.use_cache_salt = False
    generator.generator_cfg.step_wise_trajectories = False
    generator.batched = False
    generator.inference_engine_client = client
    generator.tokenizer = MagicMock()
    generator.policy_model_name = None
    entered, cleaned = asyncio.Event(), asyncio.Event()
    attempts = {0: 0, 1: 0}
    calls = []
    victim = []

    async def loop(prompt, env_class, extras, *args, **kwargs):
        index = extras["index"]
        attempts[index] += 1
        calls.append((index, controller.in_flight))
        if not victim:
            victim.append(index)
        await client.chat_completion({"json": {"session_id": f"child-{index}"}})
        if index == victim[0] and attempts[index] == 1:
            try:
                entered.set()
                await asyncio.Future()
            finally:
                cleaned.set()
                await client.finish_session(f"{index}_0")
        return TrajectoryOutput(
            response_ids=[1],
            reward=[1.0],
            stop_reason="stop",
            loss_mask=[1],
            prompt_ids=[2],
            rollout_logprobs=None,
            env_metrics={},
        )

    generator.agent_loop = loop
    task = asyncio.create_task(
        generator.generate(
            dict(
                prompts=[[], []],
                env_classes=["gsm8k", "gsm8k"],
                env_extras=[{"index": 0}, {"index": 1}],
                trajectory_ids=[SimpleNamespace(instance_id=i, repetition_id=0) for i in range(2)],
            )
        )
    )
    await asyncio.wait_for(entered.wait(), 1)
    await waiting(controller)
    assert sum(attempts.values()) == 1
    policy.decision = ConcurrencyDecision(desired_limit=1, shed_count=1, reason="pressure")
    await controller.on_feedback(SamplingFeedback())
    output = await asyncio.wait_for(task, 2)
    assert cleaned.is_set() and output["response_ids"] == [[1], [1]]
    assert attempts[victim[0]] == 2 and attempts[1 - victim[0]] == 1
    assert all(count == 1 for _, count in calls)
    assert len(policy.completions) == 2 and controller.in_flight == 0
