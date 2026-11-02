"""Grafana API requests made from Ray's head node."""

import os
from urllib.parse import urlparse

import httpx


def request_from_head(method, path, payload, token_env_var, organization_id, timeout_seconds):
    """Read the head's backend configuration and send one bounded HTTP request."""
    url = os.environ.get("RAY_GRAFANA_HOST", "http://localhost:3000")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("Ray's Grafana backend must be an HTTP(S) URL without embedded credentials")
    headers = {}
    token = os.environ.get(token_env_var)
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if organization_id is None:
        organization_id = os.environ.get("RAY_GRAFANA_ORG_ID", "1")
    headers["X-Grafana-Org-Id"] = str(organization_id)
    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.request(method, url.rstrip("/") + path, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
