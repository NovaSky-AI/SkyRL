"""Tests for environment variables forwarded to Ray workers."""

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils import utils


def test_prepare_runtime_environment_forwards_nccl_net(monkeypatch):
    monkeypatch.setenv("NCCL_NET", "Socket")
    monkeypatch.setattr(utils, "peer_access_supported", lambda **_: True)

    env_vars = utils.prepare_runtime_environment(SkyRLTrainConfig())

    assert env_vars["NCCL_NET"] == "Socket"
