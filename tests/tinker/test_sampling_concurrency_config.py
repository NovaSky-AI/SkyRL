import pytest
from pydantic import ValidationError

from skyrl.tinker.config import EngineConfig, config_to_argv


def test_sampling_concurrency_round_trips_through_engine_subprocess_argv():
    config = EngineConfig(
        base_model="test-model",
        sampling_concurrency={
            "enabled": True,
            "policy": "engine_load",
            "initial_limit": 12,
            "min_limit": 2,
            "max_limit": 32,
        },
    )

    argv = config_to_argv(config)

    index = argv.index("--sampling-concurrency")
    assert '"policy": "engine_load"' in argv[index + 1]
    assert '"initial_limit": 12' in argv[index + 1]


def test_sampling_concurrency_rejects_an_invalid_window():
    with pytest.raises(ValidationError, match="initial_limit"):
        EngineConfig(
            base_model="test-model",
            sampling_concurrency={"enabled": True, "initial_limit": 0},
        )
