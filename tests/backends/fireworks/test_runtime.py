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
    def __init__(self, path: str):
        self.path = path
        self.closed = False

    def close(self):
        self.closed = True


class _Service:
    def __init__(self):
        self.samplers = []
        self.closed = False

    def create_sampling_client(self, *, model_path, tokenizer):
        assert tokenizer == "tokenizer"
        sampler = _Sampler(model_path)
        self.samplers.append(sampler)
        return sampler

    def close(self):
        self.closed = True


class _TrainingClient:
    def __init__(self):
        self.names = []

    def save_weights_for_sampler(self, name):
        self.names.append(name)
        return _Future(SimpleNamespace(path=f"snapshot://{name}"))


def test_connect_dedicated_uses_managed_resources(monkeypatch) -> None:
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
        infrastructure="dedicated",
        base_model="accounts/fireworks/models/qwen3-4b",
        max_seq_len=32768,
        training_shape_id="accounts/fireworks/trainingShapes/qwen3-4b-minimum-lora",
        trainer_job_id="skyrl-smoke-test-trainer",
        deployment_id="skyrl-smoke-test-rollout",
        cleanup_deployment_on_close="delete",
        trainer_replica_count=2,
    )

    runtime = FireworksRuntime.connect_dedicated(
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


def test_dedicated_runtime_exposes_native_inference_endpoint() -> None:
    service = _Service()
    service._managed_handle = SimpleNamespace(
        deployment=SimpleNamespace(
            inference_model="accounts/test/deployments/skyrl-rollout"
        ),
        deployment_manager=SimpleNamespace(
            inference_url="https://api.fireworks.ai/"
        ),
    )
    runtime = FireworksRuntime(
        service=service,
        training_client=_TrainingClient(),
        tokenizer="tokenizer",
        config=FireworksConfig(infrastructure="dedicated"),
    )

    endpoint = runtime.inference_endpoint

    assert endpoint.api_base == "https://api.fireworks.ai/inference/v1"
    assert endpoint.model == "accounts/test/deployments/skyrl-rollout"


@pytest.mark.asyncio
async def test_publish_uses_rcu_and_retires_sampler_after_active_call() -> None:
    service = _Service()
    training = _TrainingClient()
    runtime = FireworksRuntime(
        service=service,
        training_client=training,
        tokenizer="tokenizer",
        config=FireworksConfig(snapshot_prefix="test run"),
    )

    first = await runtime.publish_sampler_weights()
    lease = runtime.acquire_sampler()
    second = await runtime.publish_sampler_weights()

    assert first.version == 0
    assert second.version == 1
    assert runtime.weight_version == 1
    assert first.snapshot_path != second.snapshot_path
    assert training.names[0].startswith("test-run-v00000000-")
    assert service.samplers[0].closed is False

    runtime.release_sampler(lease)
    assert service.samplers[0].closed is True
    assert service.samplers[1].closed is False

    await runtime.close()
    assert service.samplers[1].closed is True
    assert service.closed is True

    # Teardown is intentionally idempotent.
    await runtime.close()


@pytest.mark.asyncio
async def test_native_async_sample_keeps_lease_across_publish() -> None:
    started = asyncio.Event()
    finish = asyncio.Event()

    class AsyncSampler(_Sampler):
        async def sample_async(self, *, prompt, num_samples, sampling_params):
            assert prompt == "prompt"
            assert num_samples == 1
            assert sampling_params == "params"
            started.set()
            await finish.wait()
            return "result"

    class AsyncService(_Service):
        def create_sampling_client(self, *, model_path, tokenizer):
            assert tokenizer == "tokenizer"
            sampler = AsyncSampler(model_path)
            self.samplers.append(sampler)
            return sampler

    service = AsyncService()
    runtime = FireworksRuntime(
        service=service,
        training_client=_TrainingClient(),
        tokenizer="tokenizer",
        config=FireworksConfig(sampling_timeout_s=1),
    )
    first = await runtime.publish_sampler_weights()

    sample_task = asyncio.create_task(
        runtime.sample_async(prompt="prompt", sampling_params="params")
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    second = await runtime.publish_sampler_weights()

    assert second.version == first.version + 1
    assert service.samplers[0].closed is False

    finish.set()
    result, identity = await sample_task
    assert result == "result"
    assert identity == first
    assert service.samplers[0].closed is True
    assert service.samplers[1].closed is False

    await runtime.close()


@pytest.mark.asyncio
async def test_dedicated_hotload_reuses_one_client_during_active_sample() -> None:
    started = asyncio.Event()
    finish = asyncio.Event()

    class AsyncSampler(_Sampler):
        async def sample_async(self, *, prompt, num_samples, sampling_params):
            started.set()
            await finish.wait()
            return "result"

    class DedicatedService(_Service):
        def __init__(self):
            super().__init__()
            self.hotloads = []

        def hotload_sampler_snapshot(self, model_path):
            self.hotloads.append(model_path)

        def create_sampling_client(self, *, tokenizer):
            assert tokenizer == "tokenizer"
            sampler = AsyncSampler("stable-deployment")
            self.samplers.append(sampler)
            return sampler

    service = DedicatedService()
    runtime = FireworksRuntime(
        service=service,
        training_client=_TrainingClient(),
        tokenizer="tokenizer",
        config=FireworksConfig(infrastructure="dedicated", sampling_timeout_s=1),
    )
    first = await runtime.publish_sampler_weights()
    sample_task = asyncio.create_task(
        runtime.sample_async(prompt="prompt", sampling_params="params")
    )
    await asyncio.wait_for(started.wait(), timeout=1)

    second = await runtime.publish_sampler_weights()

    assert service.hotloads == [first.snapshot_path, second.snapshot_path]
    assert len(service.samplers) == 1
    assert service.samplers[0].closed is False
    assert runtime.weight_version == second.version

    finish.set()
    result, admission_identity = await sample_task
    assert result == "result"
    assert admission_identity == first

    await runtime.close()
    assert service.samplers[0].closed is True
    assert service.closed is True

    # Teardown is intentionally idempotent.
    await runtime.close()


def test_snapshot_name_leaves_room_for_dedicated_provider_suffix() -> None:
    runtime = FireworksRuntime(
        service=_Service(),
        training_client=_TrainingClient(),
        tokenizer="tokenizer",
        config=FireworksConfig(
            snapshot_prefix="skyrl-smoke-" + "very-long-prefix-" * 8
        ),
    )

    name = runtime._snapshot_name(42)
    prefix, version, random_suffix = name.rsplit("-", 2)

    assert len(name) <= 54
    assert len(f"{name}-deadbeef") <= 63
    assert prefix
    assert version == "v00000042"
    assert len(random_suffix) == 8


def test_snapshot_name_is_a_lowercase_dns_label() -> None:
    runtime = FireworksRuntime(
        service=_Service(),
        training_client=_TrainingClient(),
        tokenizer="tokenizer",
        config=FireworksConfig(snapshot_prefix="My Run_with.dots / and spaces"),
    )

    name = runtime._snapshot_name(0)

    assert re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", name)


@pytest.mark.asyncio
async def test_publish_keeps_current_sampler_when_new_sampler_creation_fails() -> None:
    class FailingService(_Service):
        def create_sampling_client(self, *, model_path, tokenizer):
            if self.samplers:
                raise RuntimeError("sampler unavailable")
            return super().create_sampling_client(
                model_path=model_path, tokenizer=tokenizer
            )

    service = FailingService()
    runtime = FireworksRuntime(
        service=service,
        training_client=_TrainingClient(),
        tokenizer="tokenizer",
        config=FireworksConfig(),
    )
    first = await runtime.publish_sampler_weights()

    with pytest.raises(RuntimeError, match="sampler unavailable"):
        await runtime.publish_sampler_weights()

    assert runtime.weight_version == first.version
    assert runtime.snapshot_path == first.snapshot_path
    assert service.samplers[0].closed is False
    await runtime.close()


@pytest.mark.asyncio
async def test_close_waits_for_active_sampler_lease() -> None:
    service = _Service()
    runtime = FireworksRuntime(
        service=service,
        training_client=_TrainingClient(),
        tokenizer="tokenizer",
        config=FireworksConfig(sampling_timeout_s=1),
    )
    await runtime.publish_sampler_weights()
    lease = runtime.acquire_sampler()

    close_task = asyncio.create_task(runtime.close())
    await asyncio.sleep(0.01)

    assert close_task.done() is False
    assert service.samplers[0].closed is False

    runtime.release_sampler(lease)
    await close_task
    assert service.samplers[0].closed is True
    assert service.closed is True
