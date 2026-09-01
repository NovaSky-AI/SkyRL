import argparse
from unittest.mock import patch

from skyrl.tinker.config import EngineConfig, add_model
from skyrl.tinker.extra.skyrl_train_inference_forwarding import (
    SkyRLTrainInferenceForwardingClient,
)


def test_forwarding_timeout_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("SKYRL_FORWARDING_INFERENCE_TIMEOUT_SEC", "1800")
    parser = argparse.ArgumentParser()
    add_model(parser, EngineConfig)

    args = parser.parse_args(["--base-model", "test-model"])
    config = EngineConfig.model_validate(vars(args))

    assert config.forwarding_inference_timeout_sec == 1800.0


def test_forwarding_client_uses_configured_timeout() -> None:
    config = EngineConfig(
        base_model="test-model",
        forwarding_inference_timeout_sec=1800.0,
    )

    with patch("skyrl.tinker.extra.skyrl_train_inference_forwarding.httpx.AsyncClient") as async_client:
        SkyRLTrainInferenceForwardingClient(config, db_engine=None)

    timeout = async_client.call_args.kwargs["timeout"]
    assert timeout.connect == 10.0
    assert timeout.read == 1800.0
    assert timeout.write == 300.0
    assert timeout.pool == 300.0
