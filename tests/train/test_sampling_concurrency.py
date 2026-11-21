from unittest.mock import MagicMock

import pytest

from skyrl.train.config import GeneratorConfig
from skyrl.train.sampling_concurrency import (
    build_sampling_client,
    build_sampling_concurrency_controller,
)
from skyrl.train.sampling_service import SamplingClient
from skyrl.utils.adaptive_concurrency import EngineLoadConcurrencyPolicy


def test_builds_service_owned_engine_load_controller():
    config = GeneratorConfig()
    config.batched = False
    config.sampling_concurrency.enabled = True
    config.sampling_concurrency.policy = "engine_load"
    config.sampling_concurrency.initial_limit = 3

    controller = build_sampling_concurrency_controller(config)

    assert controller is not None
    assert controller.current_limit == 3
    assert isinstance(controller.policy, EngineLoadConcurrencyPolicy)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("run_engines_locally", "requires SkyRL-managed local vLLM"),
        ("enable_pd", "does not yet support prefill/decode"),
    ],
)
def test_rejects_engine_load_without_coherent_managed_vllm_metrics(field, message):
    config = GeneratorConfig()
    config.sampling_concurrency.enabled = True
    config.sampling_concurrency.policy = "engine_load"
    if field == "run_engines_locally":
        config.inference_engine.run_engines_locally = False
    else:
        config.inference_engine.enable_pd = True

    with pytest.raises(ValueError, match=message):
        build_sampling_concurrency_controller(config)


def test_engine_load_does_not_depend_on_ray_prometheus_export():
    config = GeneratorConfig()
    config.sampling_concurrency.enabled = True
    config.sampling_concurrency.policy = "engine_load"
    config.inference_engine.enable_ray_prometheus_stats = False

    controller = build_sampling_concurrency_controller(config)

    assert controller is not None
    assert isinstance(controller.policy, EngineLoadConcurrencyPolicy)


def test_builds_lightweight_client_around_service_even_when_adaptive_control_is_disabled():
    config = GeneratorConfig()
    backend = MagicMock()

    client = build_sampling_client(backend, config)

    assert isinstance(client, SamplingClient)
    assert client._service.controller is None
