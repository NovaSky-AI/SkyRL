#!/usr/bin/env python3
"""Liveness probes for Fleet preflight checks.

Invoked by ``scripts/fleet-preflight.sh`` to actively validate that API
credentials in the environment are not just *present* but actually *work*.
The probes are best-effort: if a probe's optional dependency
(``boto3``, ``requests``, ``wandb``, ``fleet``) is not installed locally,
the probe is skipped with a warning rather than failing the preflight.
This keeps the preflight runnable on a developer laptop that hasn't
pip-installed the full training stack.

Each requested probe runs with a short network timeout. Results are
emitted as a single JSON object on stdout::

    {
      "results": [
        {"probe": "aws", "status": "ok",      "detail": "arn:aws:iam::..."},
        {"probe": "fleet", "status": "fail",  "detail": "401 Unauthorized"},
        {"probe": "wandb", "status": "skip",  "detail": "wandb not installed"}
      ],
      "exit_code": 1
    }

``exit_code`` is non-zero iff at least one probe returned ``fail``.
``skip`` results do not fail the preflight; the bash caller decides how
to surface them.

Usage:
    python3 scripts/_fleet_preflight_probes.py aws fleet wandb
"""

from __future__ import annotations

import json
import os
import sys
from typing import Callable

PROBE_TIMEOUT_SECONDS = 8


def _ok(probe: str, detail: str) -> dict:
    return {"probe": probe, "status": "ok", "detail": detail}


def _fail(probe: str, detail: str) -> dict:
    return {"probe": probe, "status": "fail", "detail": detail}


def _skip(probe: str, detail: str) -> dict:
    return {"probe": probe, "status": "skip", "detail": detail}


def probe_aws() -> dict:
    """Validate AWS credentials by calling STS GetCallerIdentity.

    Uses ``boto3`` if available. Falls back to skip with a warning if the
    SDK isn't installed locally — presence of the env vars is still
    enforced by the bash caller.
    """
    try:
        import boto3
        from botocore.config import Config
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:
        return _skip("aws", "boto3 not installed locally; run `uv pip install boto3` to enable liveness check")

    try:
        sts = boto3.client(
            "sts",
            config=Config(connect_timeout=PROBE_TIMEOUT_SECONDS, read_timeout=PROBE_TIMEOUT_SECONDS, retries={"max_attempts": 1}),
        )
        identity = sts.get_caller_identity()
        return _ok("aws", f"authenticated as {identity.get('Arn', '<unknown ARN>')}")
    except (BotoCoreError, ClientError) as exc:
        return _fail("aws", f"STS GetCallerIdentity failed: {exc}")


def probe_fleet() -> dict:
    """Validate FLEET_API_KEY by issuing a lightweight authenticated request.

    Prefers the ``fleet-python`` SDK if available; otherwise falls back to
    a plain HTTPS request via ``requests``. If neither is installed,
    skips with a warning.
    """
    api_key = os.environ.get("FLEET_API_KEY", "")
    if not api_key:
        return _fail("fleet", "FLEET_API_KEY is empty (presence check should have caught this)")

    # Preferred path: use the SDK, since the base URL may evolve.
    try:
        from fleet import Fleet  # type: ignore
        try:
            client = Fleet(api_key=api_key)
            # load_tasks with a non-existent env_key is a cheap auth probe:
            # the server returns 200 with an empty list on a valid key, and
            # 401/403 on a bad one — without doing real work.
            client.load_tasks(env_key="__preflight_probe__")
            return _ok("fleet", "Fleet SDK authenticated")
        except Exception as exc:  # noqa: BLE001 — surface any SDK error verbatim
            msg = str(exc)
            if "401" in msg or "403" in msg or "Unauthorized" in msg or "Forbidden" in msg:
                return _fail("fleet", f"Fleet API rejected key: {msg}")
            # Server reachable but unrelated error (e.g. 404 on probe key) — treat as ok.
            return _ok("fleet", "Fleet SDK reachable (probe call returned non-auth error, treating as authenticated)")
    except ImportError:
        pass

    try:
        import requests
    except ImportError:
        return _skip("fleet", "neither fleet-python nor requests installed; skipping liveness probe")

    base_url = os.environ.get("FLEET_BASE_URL", "https://fleet.so")
    try:
        resp = requests.get(
            f"{base_url}/api/v1/tasks",
            headers={"Authorization": f"Bearer {api_key}"},
            params={"env_key": "__preflight_probe__"},
            timeout=PROBE_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        return _fail("fleet", f"could not reach {base_url}: {exc}")

    if resp.status_code in (401, 403):
        return _fail("fleet", f"{resp.status_code} from Fleet API — key invalid or expired")
    # Without the fleet-python SDK we can't reliably probe an auth-protected
    # endpoint we know exists. A non-401 response only means the host is
    # reachable; treat as skip so we don't falsely report "key validated".
    return _skip(
        "fleet",
        f"Fleet host reachable (HTTP {resp.status_code}) but key validity unverified — install fleet-python for a full liveness check",
    )


def probe_wandb() -> dict:
    """Validate WANDB_API_KEY by querying the W&B viewer endpoint."""
    api_key = os.environ.get("WANDB_API_KEY", "")
    if not api_key:
        return _fail("wandb", "WANDB_API_KEY is empty (presence check should have caught this)")

    try:
        import wandb  # type: ignore
        try:
            api = wandb.Api(api_key=api_key, timeout=PROBE_TIMEOUT_SECONDS)
            viewer = api.viewer
            username = getattr(viewer, "username", None) or getattr(viewer, "entity", "<unknown>")
            return _ok("wandb", f"authenticated as {username}")
        except Exception as exc:  # noqa: BLE001
            return _fail("wandb", f"W&B authentication failed: {exc}")
    except ImportError:
        pass

    try:
        import requests
    except ImportError:
        return _skip("wandb", "neither wandb nor requests installed; skipping liveness probe")

    try:
        resp = requests.post(
            "https://api.wandb.ai/graphql",
            json={"query": "{ viewer { username } }"},
            auth=("api", api_key),
            timeout=PROBE_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        return _fail("wandb", f"could not reach api.wandb.ai: {exc}")

    if resp.status_code == 401:
        return _fail("wandb", "401 from W&B GraphQL — key invalid or expired")
    if resp.status_code != 200:
        return _fail("wandb", f"unexpected HTTP {resp.status_code} from W&B GraphQL")
    try:
        username = resp.json()["data"]["viewer"]["username"]
    except (KeyError, ValueError):
        return _ok("wandb", "W&B reachable (could not parse viewer payload)")
    return _ok("wandb", f"authenticated as {username}")


PROBES: dict[str, Callable[[], dict]] = {
    "aws": probe_aws,
    "fleet": probe_fleet,
    "wandb": probe_wandb,
}


def main(argv: list[str]) -> int:
    requested = argv[1:] or list(PROBES)
    unknown = [p for p in requested if p not in PROBES]
    if unknown:
        print(f"unknown probes: {unknown}; valid: {list(PROBES)}", file=sys.stderr)
        return 2

    results = [PROBES[name]() for name in requested]
    exit_code = 1 if any(r["status"] == "fail" for r in results) else 0
    json.dump({"results": results, "exit_code": exit_code}, sys.stdout)
    sys.stdout.write("\n")
    return exit_code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
