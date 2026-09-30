"""Optional run lifecycle annotations in Grafana's built-in event store."""

import json
import os
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import httpx
from loguru import logger


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
        parsed = urlparse(self.config.url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("Grafana URL must be an HTTP(S) URL without embedded credentials")
        headers = {}
        token = os.environ.get(self.config.token_env_var)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if self.config.organization_id is not None:
            headers["X-Grafana-Org-Id"] = str(self.config.organization_id)
        with httpx.Client(timeout=self.config.timeout_seconds) as client:
            response = client.request(method, self.config.url.rstrip("/") + path, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

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
