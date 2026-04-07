"""Tests for VLLMRouter and build_router_args."""

import multiprocessing
from unittest.mock import MagicMock, patch

import pytest
from vllm_router.router_args import RouterArgs

from skyrl.backends.skyrl_train.inference_servers.utils import build_router_args
from skyrl.backends.skyrl_train.inference_servers.vllm_router import VLLMRouter
from skyrl.train.config import SkyRLTrainConfig


def _make_router_args(**overrides) -> RouterArgs:
    """Helper to build a RouterArgs with sensible defaults."""
    defaults = dict(
        worker_urls=["http://localhost:8000"],
        host="0.0.0.0",
        port=9999,
        policy="consistent_hash",
    )
    defaults.update(overrides)
    return RouterArgs(**defaults)


def test_process_exit_raises_runtime_error():
    """start() raises RuntimeError if the router process exits before becoming healthy."""
    router_args = _make_router_args()
    router = VLLMRouter(router_args)

    mock_process = MagicMock(spec=multiprocessing.Process)
    mock_process.is_alive.return_value = False
    mock_process.exitcode = 1

    with patch("skyrl.backends.skyrl_train.inference_servers.vllm_router.get_node_ip", return_value="127.0.0.1"):
        with patch.object(router, "_process", mock_process):
            # Bypass actual process start; just test health check failure path
            router._process = mock_process
            with pytest.raises(RuntimeError, match="process exited with code 1"):
                router._wait_until_healthy("http://127.0.0.1:9999", timeout=1)


def test_shutdown_terminates_process():
    """shutdown() sends SIGTERM to the child process."""
    router_args = _make_router_args()
    router = VLLMRouter(router_args)

    mock_process = MagicMock(spec=multiprocessing.Process)
    # First call (guard check) returns True, second (after join) returns False
    mock_process.is_alive.side_effect = [True, False]
    router._process = mock_process

    router.shutdown()

    mock_process.terminate.assert_called_once()
    mock_process.join.assert_called_once_with(timeout=5)
    mock_process.kill.assert_not_called()


def test_shutdown_kills_on_timeout():
    """shutdown() escalates to SIGKILL if SIGTERM doesn't work."""
    router_args = _make_router_args()
    router = VLLMRouter(router_args)

    mock_process = MagicMock(spec=multiprocessing.Process)
    # is_alive returns True even after terminate+join (simulating stuck process)
    mock_process.is_alive.return_value = True
    router._process = mock_process

    router.shutdown()

    mock_process.terminate.assert_called_once()
    mock_process.kill.assert_called_once()


def test_shutdown_noop_when_not_started():
    """shutdown() is safe to call when process was never started."""
    router_args = _make_router_args()
    router = VLLMRouter(router_args)
    router.shutdown()  # should not raise


def test_pd_router_args():
    """RouterArgs for PD mode has correct fields."""
    router_args = RouterArgs(
        host="0.0.0.0",
        port=9999,
        policy="consistent_hash",
        prefill_urls=[("http://p1:8000", None), ("http://p2:8000", None)],
        decode_urls=["http://d1:8001", "http://d2:8001"],
        vllm_pd_disaggregation=True,
        prefill_policy="consistent_hash",
        decode_policy="consistent_hash",
    )
    assert router_args.vllm_pd_disaggregation is True
    assert len(router_args.prefill_urls) == 2
    assert len(router_args.decode_urls) == 2
    assert router_args.prefill_urls[0] == ("http://p1:8000", None)


class TestBuildRouterArgs:
    """Tests for build_router_args helper."""

    def test_uniform_mode(self):
        cfg = SkyRLTrainConfig()
        ie_cfg = cfg.generator.inference_engine
        urls = ["http://w1:8000", "http://w2:8000"]
        with patch("skyrl.backends.skyrl_train.inference_servers.common.get_open_port", return_value=30000):
            args = build_router_args(ie_cfg, server_urls=urls)
        assert args.worker_urls == urls
        assert args.port == 30000
        assert args.policy == "consistent_hash"
        assert args.vllm_pd_disaggregation is False

    def test_pd_mode(self):
        cfg = SkyRLTrainConfig()
        ie_cfg = cfg.generator.inference_engine
        prefill = ["http://p1:8000"]
        decode = ["http://d1:8001"]
        with patch("skyrl.backends.skyrl_train.inference_servers.common.get_open_port", return_value=30000):
            args = build_router_args(ie_cfg, prefill_urls=prefill, decode_urls=decode)
        assert args.vllm_pd_disaggregation is True
        assert args.prefill_urls == [("http://p1:8000", None)]
        assert args.decode_urls == ["http://d1:8001"]
        assert args.prefill_policy == "consistent_hash"
        assert args.decode_policy == "consistent_hash"

    def test_no_urls_raises(self):
        cfg = SkyRLTrainConfig()
        ie_cfg = cfg.generator.inference_engine
        with patch("skyrl.backends.skyrl_train.inference_servers.common.get_open_port", return_value=30000):
            with pytest.raises(ValueError, match="Either server_urls"):
                build_router_args(ie_cfg)

    def test_router_init_kwargs_override(self):
        cfg = SkyRLTrainConfig()
        ie_cfg = cfg.generator.inference_engine
        cfg.generator.inference_engine.router_init_kwargs = {"policy": "round_robin", "request_timeout_secs": 60}
        urls = ["http://w1:8000"]
        with patch("skyrl.backends.skyrl_train.inference_servers.common.get_open_port", return_value=30000):
            args = build_router_args(ie_cfg, server_urls=urls)
        assert args.policy == "round_robin"
        assert args.request_timeout_secs == 60
