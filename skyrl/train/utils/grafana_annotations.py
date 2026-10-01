"""Optional run lifecycle annotations in Grafana's built-in event store."""

import json
import time
import uuid
from pathlib import Path

import ray
from loguru import logger
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from skyrl.utils.grafana import request_from_head


class GrafanaRunAnnotation:
    """Create a start marker and update it to a region on finalization."""

    def __init__(self, config, run_name: str, directory: str):
        self.config = config
        self.run_name = run_name
        self.run_id = uuid.uuid4().hex
        self.start_ms = int(time.time() * 1000)
        self.annotation_id = None
        self._finished = False
        self.path = Path(directory) / f"grafana-run-{self.run_id}.json"

    def _request(self, method, path, payload):
        if not ray.is_initialized():
            raise RuntimeError("Ray must be initialized before publishing run annotations")
        heads = [
            node
            for node in ray.nodes()
            if node.get("Alive") and node.get("Resources", {}).get("node:__internal_head__", 0) > 0
        ]
        if len(heads) != 1:
            raise RuntimeError("Could not identify a unique live Ray head node")
        task = (
            ray.remote(num_cpus=0, max_retries=0)(request_from_head)
            .options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=heads[0]["NodeID"], soft=False))
            .remote(
                method,
                path,
                payload,
                self.config.token_env_var,
                self.config.organization_id,
                self.config.timeout_seconds,
            )
        )
        try:
            return ray.get(task, timeout=self.config.timeout_seconds + 10)
        except ray.exceptions.GetTimeoutError:
            ray.cancel(task, force=True)
            raise

    def _payload(self, end_ms=None):
        payload = {
            "text": self.run_name,
            "time": self.start_ms,
            "tags": ["skyrl-run", f"run-id:{self.run_id}", *self.config.tags],
        }
        if end_ms is not None:
            payload["timeEnd"] = end_ms
        return payload

    def _save(self, status, end_ms=None):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "run_name": self.run_name,
            "run_id": self.run_id,
            "start_ms": self.start_ms,
            "end_ms": end_ms,
            "annotation_id": self.annotation_id,
            "run_status": status,
        }
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(self.path)

    def start(self):
        """Publish a start event when enabled; failures leave training unaffected."""
        if not self.config.enabled:
            return
        try:
            response = self._request("POST", "/api/annotations", self._payload())
            self.annotation_id = response["id"]
        except Exception as error:
            logger.warning(f"Grafana start annotation failed ({type(error).__name__})")
        try:
            self._save("running")
        except Exception as error:
            logger.warning(f"Could not save annotation record ({type(error).__name__})")

    def finish(self, status):
        """Close the existing marker once and save the final run interval."""
        if not self.config.enabled or self._finished:
            return
        self._finished = True
        end_ms = max(self.start_ms, int(time.time() * 1000))
        try:
            if self.annotation_id is not None:
                self._request("PUT", f"/api/annotations/{self.annotation_id}", self._payload(end_ms))
        except Exception as error:
            logger.warning(f"Grafana final annotation failed ({type(error).__name__})")
        try:
            self._save(status, end_ms)
        except Exception as error:
            logger.warning(f"Could not save final annotation record ({type(error).__name__})")
        if self.config.dashboard_url:
            separator = "&" if "?" in self.config.dashboard_url else "?"
            logger.info(f"Grafana run: {self.config.dashboard_url}{separator}from={self.start_ms}&to={end_ms}")
