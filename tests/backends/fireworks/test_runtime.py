import asyncio
import re
from types import SimpleNamespace

import pytest

from skyrl.backends.fireworks.runtime import FireworksRuntime
from skyrl.train.config import FireworksConfig


class _Future:
    def __init__(self, value):
        self.value = value
        self.timeouts = []

    def result(self, timeout=None):
        self.timeouts.append(timeout)
        return self.value


class _Sampler:
    def __init__(self):
        self.closed = False

    async def sample_async(self, *, prompt, num_samples, sampling_params):
        assert num_samples == 1
        return (prompt, sampling_params)

    def close(self):
        self.closed = True


class _Service:
    def __init__(self):
        self.samplers = []
        self.hotloads = []
        self.closed = False

    def create_sampling_client(self, *, tokenizer):
        assert tokenizer == "tokenizer"
        sampler = _Sampler()
        self.samplers.append(sampler)
        return sampler

    def hotload_sampler_snapshot(self, path):
        self.hotloads.append(path)

    def close(self):
        self.closed = True


class _TrainingClient:
    def __init__(self):
        self.names = []

    def save_weights_for_sampler(self, name):
        self.names.append(name)
        return _Future(SimpleNamespace(path=f"snapshot://{name}"))


def _runtime(
    *,
    service=None,
    training_client=None,
    config=None,
) -> FireworksRuntime:
    return FireworksRuntime(
        service=service or _Service(),
        training_client=training_client or _TrainingClient(),
        tokenizer="tokenizer",
        config=config or FireworksConfig(),
    )


def test_connect_uses_managed_dedicated_resources(monkeypatch) -> None:
    from fireworks.training.sdk import FiretitanServiceClient

    captured = {}

    class ManagedService(_Service):
        trainer_job_id = "skyrl-smoke-test-trainer"
        deployment_id = "skyrl-smoke-test-rollout"

        def create_training_client(self, **kwargs):
            captured["training_client"] = kwargs
            return _TrainingClient()

    service = ManagedService()

    def _factory(**kwargs):
        captured["service"] = kwargs
        return service

    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setattr(
        FiretitanServiceClient, "from_firetitan_config", staticmethod(_factory)
    )
    config = FireworksConfig(
        base_model="accounts/fireworks/models/qwen3-4b",
        max_seq_len=32768,
        training_shape_id="accounts/fireworks/trainingShapes/qwen3-4b-minimum-lora",
        trainer_job_id="skyrl-smoke-test-trainer",
        deployment_id="skyrl-smoke-test-rollout",
        cleanup_deployment_on_close="delete",
        trainer_replica_count=2,
    )

    runtime = FireworksRuntime.connect(
        config=config,
        tokenizer="tokenizer",
        tokenizer_model="Qwen/Qwen3-4B",
        lora_rank=8,
        learning_rate=1e-5,
    )

    assert captured["service"]["training_shape_id"] == config.training_shape_id
    assert captured["service"]["cleanup_trainer_on_close"] is True
    assert captured["service"]["cleanup_deployment_on_close"] == "delete"
    assert captured["service"]["trainer_replica_count"] == 2
    assert captured["service"]["replica_count"] == 1
    assert captured["training_client"] == {
        "base_model": config.base_model,
        "lora_rank": 8,
    }
    assert runtime.trainer_job_id == config.trainer_job_id
    assert runtime.deployment_id == config.deployment_id
    asyncio.run(runtime.close())
    assert service.closed is True


def test_runtime_exposes_native_inference_endpoint() -> None:
    service = _Service()
    service._managed_handle = SimpleNamespace(
        deployment=SimpleNamespace(
            inference_model="accounts/test/deployments/skyrl-rollout"
        ),
        deployment_manager=SimpleNamespace(inference_url="https://api.fireworks.ai/"),
    )
    runtime = _runtime(service=service)

    endpoint = runtime.inference_endpoint

    assert endpoint.api_base == "https://api.fireworks.ai/inference/v1"
    assert endpoint.model == "accounts/test/deployments/skyrl-rollout"


@pytest.mark.asyncio
async def test_publish_reuses_one_stable_sampler() -> None:
    service = _Service()
    training = _TrainingClient()
    runtime = _runtime(service=service, training_client=training)

    first = await runtime.publish_sampler_weights()
    second = await runtime.publish_sampler_weights()

    assert first.version == 0
    assert second.version == 1
    assert first.snapshot_path != second.snapshot_path
    assert service.hotloads == [first.snapshot_path, second.snapshot_path]
    assert len(service.samplers) == 1
    assert service.samplers[0].closed is False
    assert runtime.weight_version == second.version
    assert runtime.snapshot_path == second.snapshot_path

    await runtime.close()
    assert service.samplers[0].closed is True
    assert service.closed is True

    # Teardown is intentionally idempotent.
    await runtime.close()


@pytest.mark.asyncio
async def test_active_sample_spans_hotload_on_same_client() -> None:
    started = asyncio.Event()
    finish = asyncio.Event()

    class BlockingSampler(_Sampler):
        async def sample_async(self, *, prompt, num_samples, sampling_params):
            started.set()
            await finish.wait()
            return "result"

    class BlockingService(_Service):
        def create_sampling_client(self, *, tokenizer):
            assert tokenizer == "tokenizer"
            sampler = BlockingSampler()
            self.samplers.append(sampler)
            return sampler

    service = BlockingService()
    runtime = _runtime(
        service=service,
        config=FireworksConfig(sampling_timeout_s=1),
    )
    first = await runtime.publish_sampler_weights()
    sample_task = asyncio.create_task(
        runtime.sample_async(prompt="prompt", sampling_params="params")
    )
    await asyncio.wait_for(started.wait(), timeout=1)

    second = await runtime.publish_sampler_weights()

    assert len(service.samplers) == 1
    assert service.samplers[0].closed is False
    assert runtime.weight_version == second.version

    finish.set()
    result, admission_identity = await sample_task
    assert result == "result"
    assert admission_identity == first
    await runtime.close()


@pytest.mark.asyncio
async def test_failed_hotload_preserves_published_identity_and_client() -> None:
    class FailingService(_Service):
        def hotload_sampler_snapshot(self, path):
            if self.hotloads:
                raise RuntimeError("hotload failed")
            super().hotload_sampler_snapshot(path)

    service = FailingService()
    runtime = _runtime(service=service)
    first = await runtime.publish_sampler_weights()

    with pytest.raises(RuntimeError, match="hotload failed"):
        await runtime.publish_sampler_weights()

    assert runtime.weight_version == first.version
    assert runtime.snapshot_path == first.snapshot_path
    assert len(service.samplers) == 1
    assert service.samplers[0].closed is False
    await runtime.close()


def test_snapshot_name_leaves_room_for_provider_suffix() -> None:
    runtime = _runtime(
        config=FireworksConfig(snapshot_prefix="skyrl-smoke-" + "long-prefix-" * 8)
    )

    name = runtime._snapshot_name(42)
    prefix, version, random_suffix = name.rsplit("-", 2)

    assert len(name) <= 54
    assert len(f"{name}-deadbeef") <= 63
    assert prefix
    assert version == "v00000042"
    assert len(random_suffix) == 8


def test_snapshot_name_is_a_lowercase_dns_label() -> None:
    runtime = _runtime(
        config=FireworksConfig(snapshot_prefix="My Run_with.dots / and spaces")
    )

    name = runtime._snapshot_name(0)

    assert re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", name)


@pytest.mark.asyncio
async def test_close_waits_for_active_sample() -> None:
    started = asyncio.Event()
    finish = asyncio.Event()

    class BlockingSampler(_Sampler):
        async def sample_async(self, *, prompt, num_samples, sampling_params):
            started.set()
            await finish.wait()
            return "result"

    class BlockingService(_Service):
        def create_sampling_client(self, *, tokenizer):
            sampler = BlockingSampler()
            self.samplers.append(sampler)
            return sampler

    service = BlockingService()
    runtime = _runtime(
        service=service,
        config=FireworksConfig(sampling_timeout_s=1),
    )
    await runtime.publish_sampler_weights()
    sample_task = asyncio.create_task(
        runtime.sample_async(prompt="prompt", sampling_params="params")
    )
    await asyncio.wait_for(started.wait(), timeout=1)

    close_task = asyncio.create_task(runtime.close())
    await asyncio.sleep(0.01)

    assert close_task.done() is False
    assert service.samplers[0].closed is False

    finish.set()
    result, identity = await sample_task
    assert result == "result"
    assert identity.version == 0
    await close_task
    assert service.samplers[0].closed is True
    assert service.closed is True
