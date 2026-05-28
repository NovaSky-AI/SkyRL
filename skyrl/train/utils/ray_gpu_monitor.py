"""Background GPU/RAM monitor that scrapes Ray node Prometheus endpoints.

Collects per-node, per-GPU utilization and memory metrics in a daemon thread and
exposes them via ``flush()``, which returns averaged readings since the last call.
The returned dict uses ``ray/`` prefixed keys and is ready to be merged into
``self.all_metrics`` before the wandb log step.

Per-GPU / per-node metrics (multiple lines on the same wandb chart)
-------------------------------------------------------------------
ray/node.<idx>.gpu.<gpu_idx>.util        GPU utilization (%) — tagged by node+GPU ID
ray/node.<idx>.gpu.<gpu_idx>.mem_used_gb GPU memory used (GB) — tagged by node+GPU ID
ray/node.<idx>.cpu_ram_used_gb           Host CPU RAM used (GB) — tagged by node ID

Cluster-wide averages (single line per metric)
----------------------------------------------
ray/gpu.util.avg                         Average GPU utilization across all GPUs
ray/gpu.mem_used_gb.avg                  Average GPU memory used across all GPUs
ray/cpu_ram_used_gb.avg                  Average CPU RAM used across all nodes
"""

import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional

import httpx
import ray
from loguru import logger

from skyrl.train.utils.vllm_metrics_scraper import (
    discover_ray_metrics_urls,
    parse_metrics_text,
)

# Ray Prometheus metric names published by Ray's node metrics agent.
_GPU_UTIL = "ray_node_gpus_utilization"
_GPU_MEM = "ray_node_gram_used"
_RAM_USED = "ray_node_mem_used"


class RayGpuMonitor:
    """Scrape Ray node GPU/RAM metrics in a background thread.

    Args:
        collection_interval: Seconds between Prometheus scrapes.
        request_timeout_s: Per-request HTTP timeout.
    """

    def __init__(
        self,
        collection_interval: float = 5.0,
        request_timeout_s: float = 3.0,
    ):
        self.collection_interval = collection_interval
        self._timeout = request_timeout_s

        self._buffer: List[Dict[str, float]] = []
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._urls: List[str] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Discover Ray node endpoints and start the collection thread."""
        if self._running:
            return

        if not ray.is_initialized():
            logger.warning("RayGpuMonitor: Ray is not initialized — GPU monitoring disabled.")
            return

        self._urls = discover_ray_metrics_urls()
        if not self._urls:
            logger.warning("RayGpuMonitor: no Ray metrics endpoints found — GPU monitoring disabled.")
            return

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="ray-gpu-monitor")
        self._thread.start()
        logger.info(
            f"RayGpuMonitor started: {len(self._urls)} node(s), " f"collection_interval={self.collection_interval}s"
        )

    def stop(self) -> None:
        """Stop the collection thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=self.collection_interval * 2)
        logger.info("RayGpuMonitor stopped.")

    # ------------------------------------------------------------------
    # Data access
    # ------------------------------------------------------------------

    def flush(self) -> Dict[str, float]:
        """Return averaged metrics collected since the last flush and clear the buffer.

        Returns an empty dict if nothing was collected yet or monitoring is disabled.
        The keys are prefixed with ``ray/``.

        Per-GPU/per-node keys are included as-is so wandb renders them as
        multiple lines on the same chart (grouped by metric name, split by tag).
        Cluster-wide averages are appended as separate scalar keys.
        """
        with self._lock:
            if not self._buffer:
                return {}
            snapshots = self._buffer.copy()
            self._buffer.clear()

        # Average each metric across all collected time-snapshots.
        totals: Dict[str, float] = defaultdict(float)
        counts: Dict[str, int] = defaultdict(int)
        for snap in snapshots:
            for k, v in snap.items():
                totals[k] += v
                counts[k] += 1

        averaged = {k: totals[k] / counts[k] for k in totals}

        # Build output: individual tagged metrics + cluster-wide averages.
        out: Dict[str, float] = {f"ray/{k}": v for k, v in averaged.items()}

        gpu_util_vals = [v for k, v in averaged.items() if k.endswith(".util")]
        gpu_mem_vals = [v for k, v in averaged.items() if ".gpu." in k and k.endswith(".mem_used_gb")]
        cpu_mem_vals = [v for k, v in averaged.items() if k.endswith(".cpu_ram_used_gb")]

        if gpu_util_vals:
            out["ray/gpu.util.avg"] = sum(gpu_util_vals) / len(gpu_util_vals)
        if gpu_mem_vals:
            out["ray/gpu.mem_used_gb.avg"] = sum(gpu_mem_vals) / len(gpu_mem_vals)
        if cpu_mem_vals:
            out["ray/cpu_ram_used_gb.avg"] = sum(cpu_mem_vals) / len(cpu_mem_vals)

        return out

    # ------------------------------------------------------------------
    # Internal collection loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        with httpx.Client(timeout=self._timeout) as client:
            while self._running:
                try:
                    snapshot = self._collect(client)
                    if snapshot:
                        with self._lock:
                            self._buffer.append(snapshot)
                except Exception as exc:
                    logger.debug(f"RayGpuMonitor: collection error: {exc}")
                time.sleep(self.collection_interval)

    def _collect(self, client: httpx.Client) -> Dict[str, float]:
        """Fetch and parse GPU/RAM metrics from every Ray node."""
        result: Dict[str, float] = {}
        for node_idx, url in enumerate(self._urls):
            try:
                resp = client.get(url)
                resp.raise_for_status()
            except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                logger.debug(f"RayGpuMonitor: failed to scrape {url}: {exc}")
                continue

            parsed = parse_metrics_text(resp.text)
            for (name, labels), value in parsed.items():
                labels_dict = dict(labels)
                if name == _GPU_UTIL:
                    gpu_idx = labels_dict.get("GpuIndex", "0")
                    result[f"node.{node_idx}.gpu.{gpu_idx}.util"] = value
                elif name == _GPU_MEM:
                    gpu_idx = labels_dict.get("GpuIndex", "0")
                    # Ray reports gram_used in MB; convert to GB.
                    result[f"node.{node_idx}.gpu.{gpu_idx}.mem_used_gb"] = value / 1024.0
                elif name == _RAM_USED:
                    # Ray reports mem_used in bytes; convert to GB (host CPU RAM utilized).
                    result[f"node.{node_idx}.cpu_ram_used_gb"] = value / (1024**3)
        return result
