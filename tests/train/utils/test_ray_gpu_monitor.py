"""
uv run --isolated --extra dev --extra skyrl-train  pytest tests/train/utils/test_ray_gpu_monitor.py
"""

from unittest.mock import MagicMock, patch

import pytest

from skyrl.train.utils.ray_gpu_monitor import RayGpuMonitor

# ---------------------------------------------------------------------------
# Minimal Prometheus text payload used across multiple tests
# ---------------------------------------------------------------------------
_PROMETHEUS_TEXT = """\
# HELP ray_node_gpus_utilization GPU utilization
# TYPE ray_node_gpus_utilization gauge
ray_node_gpus_utilization{GpuIndex="0",GpuDeviceName="NVIDIA H100"} 75.5
ray_node_gpus_utilization{GpuIndex="1",GpuDeviceName="NVIDIA H100"} 50.0
# HELP ray_node_gram_used GPU memory used (MB)
# TYPE ray_node_gram_used gauge
ray_node_gram_used{GpuIndex="0",GpuDeviceName="NVIDIA H100"} 81920.0
ray_node_gram_used{GpuIndex="1",GpuDeviceName="NVIDIA H100"} 40960.0
# HELP ray_node_mem_used Host RAM used (bytes)
# TYPE ray_node_mem_used gauge
ray_node_mem_used{InstanceId="n/a"} 107374182400.0
"""

# 81920 MB / 1024 = 80.0 GB, 40960 / 1024 = 40.0 GB
# 107374182400 bytes / 1024^3 = 100.0 GB


class TestRayGpuMonitorInit:
    def test_default_params(self):
        monitor = RayGpuMonitor()
        assert monitor.collection_interval == 5.0
        assert monitor._timeout == 3.0
        assert monitor._buffer == []
        assert monitor._running is False
        assert monitor._thread is None
        assert monitor._urls == []

    def test_custom_params(self):
        monitor = RayGpuMonitor(collection_interval=10.0, request_timeout_s=1.0)
        assert monitor.collection_interval == 10.0
        assert monitor._timeout == 1.0


class TestRayGpuMonitorStart:
    @patch("skyrl.train.utils.ray_gpu_monitor.ray")
    @patch("skyrl.train.utils.ray_gpu_monitor.threading.Thread")
    @patch("skyrl.train.utils.ray_gpu_monitor.discover_ray_metrics_urls")
    def test_start_launches_thread(self, mock_discover, mock_thread, mock_ray):
        mock_ray.is_initialized.return_value = True
        mock_discover.return_value = ["http://10.0.0.1:8080/metrics"]

        monitor = RayGpuMonitor()
        monitor.start()

        mock_thread.assert_called_once()
        mock_thread.return_value.start.assert_called_once()
        assert monitor._running is True
        assert monitor._thread is mock_thread.return_value

    @patch("skyrl.train.utils.ray_gpu_monitor.ray")
    def test_start_noop_when_ray_not_initialized(self, mock_ray):
        mock_ray.is_initialized.return_value = False

        monitor = RayGpuMonitor()
        monitor.start()  # should warn, not raise

        assert monitor._running is False
        assert monitor._thread is None

    @patch("skyrl.train.utils.ray_gpu_monitor.ray")
    @patch("skyrl.train.utils.ray_gpu_monitor.discover_ray_metrics_urls")
    def test_start_noop_when_no_urls(self, mock_discover, mock_ray):
        mock_ray.is_initialized.return_value = True
        mock_discover.return_value = []

        monitor = RayGpuMonitor()
        monitor.start()

        assert monitor._running is False
        assert monitor._thread is None

    @patch("skyrl.train.utils.ray_gpu_monitor.ray")
    @patch("skyrl.train.utils.ray_gpu_monitor.threading.Thread")
    @patch("skyrl.train.utils.ray_gpu_monitor.discover_ray_metrics_urls")
    def test_start_idempotent(self, mock_discover, mock_thread, mock_ray):
        mock_ray.is_initialized.return_value = True
        mock_discover.return_value = ["http://10.0.0.1:8080/metrics"]

        monitor = RayGpuMonitor()
        monitor.start()
        monitor.start()  # second call should be a no-op

        assert mock_thread.call_count == 1


class TestRayGpuMonitorStop:
    @patch("skyrl.train.utils.ray_gpu_monitor.ray")
    @patch("skyrl.train.utils.ray_gpu_monitor.threading.Thread")
    @patch("skyrl.train.utils.ray_gpu_monitor.discover_ray_metrics_urls")
    def test_stop_clears_running_flag(self, mock_discover, mock_thread, mock_ray):
        mock_ray.is_initialized.return_value = True
        mock_discover.return_value = ["http://10.0.0.1:8080/metrics"]

        monitor = RayGpuMonitor()
        monitor.start()
        monitor.stop()

        assert monitor._running is False
        mock_thread.return_value.join.assert_called_once()


class TestRayGpuMonitorCollect:
    def test_collect_parses_gpu_and_ram_metrics(self):
        monitor = RayGpuMonitor()
        monitor._urls = ["http://10.0.0.1:8080/metrics"]

        mock_resp = MagicMock()
        mock_resp.text = _PROMETHEUS_TEXT
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp

        result = monitor._collect(mock_client)

        assert result["node.0.gpu.0.util"] == pytest.approx(75.5)
        assert result["node.0.gpu.1.util"] == pytest.approx(50.0)
        assert result["node.0.gpu.0.mem_used_gb"] == pytest.approx(80.0)
        assert result["node.0.gpu.1.mem_used_gb"] == pytest.approx(40.0)
        assert result["node.0.cpu_ram_used_gb"] == pytest.approx(100.0)

    def test_collect_multi_node(self):
        monitor = RayGpuMonitor()
        monitor._urls = [
            "http://10.0.0.1:8080/metrics",
            "http://10.0.0.2:8080/metrics",
        ]

        mock_resp = MagicMock()
        mock_resp.text = _PROMETHEUS_TEXT
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp

        result = monitor._collect(mock_client)

        # node 0 and node 1 keys should both be present
        assert "node.0.gpu.0.util" in result
        assert "node.1.gpu.0.util" in result

    def test_collect_tolerates_http_error(self):
        import httpx

        monitor = RayGpuMonitor()
        monitor._urls = ["http://10.0.0.1:8080/metrics"]

        mock_client = MagicMock()
        mock_client.get.side_effect = httpx.RequestError("connection refused")

        result = monitor._collect(mock_client)
        assert result == {}


class TestRayGpuMonitorFlush:
    def test_flush_empty_buffer_returns_empty_dict(self):
        monitor = RayGpuMonitor()
        assert monitor.flush() == {}

    def test_flush_returns_ray_prefixed_keys(self):
        monitor = RayGpuMonitor()
        monitor._buffer = [{"node.0.gpu.0.util": 80.0}]

        result = monitor.flush()
        assert "ray/node.0.gpu.0.util" in result

    def test_flush_clears_buffer(self):
        monitor = RayGpuMonitor()
        monitor._buffer = [{"node.0.gpu.0.util": 80.0}]

        monitor.flush()
        assert monitor._buffer == []

    def test_flush_time_averages_across_snapshots(self):
        monitor = RayGpuMonitor()
        monitor._buffer = [
            {"node.0.gpu.0.util": 60.0},
            {"node.0.gpu.0.util": 80.0},
            {"node.0.gpu.0.util": 100.0},
        ]

        result = monitor.flush()
        assert result["ray/node.0.gpu.0.util"] == pytest.approx(80.0)

    def test_flush_handles_missing_keys_across_snapshots(self):
        # GPU 1 only appears in 2 of 3 snapshots — should still average correctly
        monitor = RayGpuMonitor()
        monitor._buffer = [
            {"node.0.gpu.0.util": 60.0, "node.0.gpu.1.util": 40.0},
            {"node.0.gpu.0.util": 80.0},
            {"node.0.gpu.0.util": 100.0, "node.0.gpu.1.util": 60.0},
        ]

        result = monitor.flush()
        assert result["ray/node.0.gpu.0.util"] == pytest.approx(80.0)
        assert result["ray/node.0.gpu.1.util"] == pytest.approx(50.0)  # (40+60)/2

    def test_flush_computes_gpu_util_average(self):
        monitor = RayGpuMonitor()
        monitor._buffer = [
            {
                "node.0.gpu.0.util": 60.0,
                "node.0.gpu.1.util": 80.0,
                "node.1.gpu.0.util": 70.0,
            }
        ]

        result = monitor.flush()
        assert result["ray/gpu.util.avg"] == pytest.approx((60.0 + 80.0 + 70.0) / 3)

    def test_flush_computes_gpu_mem_average(self):
        monitor = RayGpuMonitor()
        monitor._buffer = [
            {
                "node.0.gpu.0.mem_used_gb": 40.0,
                "node.0.gpu.1.mem_used_gb": 80.0,
            }
        ]

        result = monitor.flush()
        assert result["ray/gpu.mem_used_gb.avg"] == pytest.approx(60.0)

    def test_flush_computes_cpu_ram_average(self):
        monitor = RayGpuMonitor()
        monitor._buffer = [
            {
                "node.0.cpu_ram_used_gb": 100.0,
                "node.1.cpu_ram_used_gb": 200.0,
            }
        ]

        result = monitor.flush()
        assert result["ray/cpu_ram_used_gb.avg"] == pytest.approx(150.0)

    def test_flush_cpu_mem_average_excludes_gpu_mem(self):
        # GPU mem keys contain ".gpu." — they must not be counted in CPU RAM avg
        monitor = RayGpuMonitor()
        monitor._buffer = [
            {
                "node.0.gpu.0.mem_used_gb": 80.0,
                "node.0.cpu_ram_used_gb": 120.0,
            }
        ]

        result = monitor.flush()
        assert result["ray/cpu_ram_used_gb.avg"] == pytest.approx(120.0)
        assert result["ray/gpu.mem_used_gb.avg"] == pytest.approx(80.0)

    def test_flush_no_avg_keys_when_no_data(self):
        # If only CPU RAM is present, gpu averages must not appear
        monitor = RayGpuMonitor()
        monitor._buffer = [{"node.0.cpu_ram_used_gb": 50.0}]

        result = monitor.flush()
        assert "ray/gpu.util.avg" not in result
        assert "ray/gpu.mem_used_gb.avg" not in result
        assert "ray/cpu_ram_used_gb.avg" in result

    def test_flush_thread_safe(self):
        """flush() must drain the buffer under the lock."""
        import threading

        monitor = RayGpuMonitor()
        monitor._buffer = [{"node.0.gpu.0.util": 90.0}]

        results = []

        def do_flush():
            results.append(monitor.flush())

        threads = [threading.Thread(target=do_flush) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly one flush should have seen data; the rest return empty dicts
        non_empty = [r for r in results if r]
        assert len(non_empty) == 1
        assert non_empty[0]["ray/node.0.gpu.0.util"] == pytest.approx(90.0)
