import pytest
from pydantic import ValidationError

from skyrl.tinker.config import EngineConfig, config_to_argv


def test_sampling_concurrency_round_trips_through_engine_subprocess_argv():
    config = EngineConfig(
        base_model="test-model",
        sampling_concurrency={"enabled": True, "initial_limit": 12},
    )

    argv = config_to_argv(config)

    index = argv.index("--sampling-concurrency")
    assert '"enabled": true' in argv[index + 1]
    assert '"initial_limit": 12' in argv[index + 1]


def test_sampling_concurrency_rejects_an_invalid_window():
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        EngineConfig(
            base_model="test-model",
            sampling_concurrency={"enabled": True, "initial_limit": 0},
        )
