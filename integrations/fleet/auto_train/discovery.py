"""Query Supabase for candidate Fleet datasets + resolve modalities.

In the Fleet UI these are 'Datasets'. In Supabase they live in `task_projects`
with a `project_key` column. We expose them as Dataset objects with
`dataset_key` to match the UI; Supabase queries use the literal column names.

Mirrors the query patterns in
fleet-research-scripts/training-data-pipeline/v6/export_s3.py.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from dataclasses import dataclass

import httpx

from .config import COMPUTER_USE_ENV_PREFIX, EXCLUDED_ENVS

logger = logging.getLogger(__name__)


def _supabase_url() -> str:
    return os.environ.get("SUPABASE_URL", "https://ehefoavidbttssbleuyv.supabase.co")


def _supabase_headers() -> dict:
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ["SUPABASE_KEY"]
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


@dataclass
class Dataset:
    """A Fleet dataset (Supabase row from `task_projects`)."""
    id: str                # task_projects.id (UUID)
    dataset_key: str       # task_projects.project_key (the stable string identifier)
    name: str
    status: str
    team_id: str | None = None
    created_at: str | None = None


def list_active_datasets(
    client: httpx.Client,
    team_id: str | None = None,
) -> list[Dataset]:
    """All active Fleet datasets, optionally filtered by team_id. Sorted newest-first."""
    out: list[Dataset] = []
    offset = 0
    while True:
        params = {
            "select": "id,project_key,name,status,team_id,created_at",
            "status": "eq.active",
            "offset": str(offset),
            "limit": "1000",
            "order": "created_at.desc",
        }
        if team_id:
            params["team_id"] = f"eq.{team_id}"
        r = client.get(
            f"{_supabase_url()}/rest/v1/task_projects",
            params=params,
            headers=_supabase_headers(),
            timeout=30,
        )
        r.raise_for_status()
        rows = r.json()
        for row in rows:
            key = row.get("project_key") or row.get("name")
            if not key:
                continue
            out.append(Dataset(
                id=row["id"],
                dataset_key=key,
                name=row.get("name", ""),
                status=row["status"],
                team_id=row.get("team_id"),
                created_at=row.get("created_at"),
            ))
        if len(rows) < 1000:
            break
        offset += 1000
    logger.info("Active datasets: %d (team_id=%s)", len(out), team_id or "all")
    return out


def get_dataset_task_ids(client: httpx.Client, dataset_id: str) -> list[str]:
    """Task ids linked to a dataset via eval_task_projects."""
    out: list[str] = []
    offset = 0
    while True:
        r = client.get(
            f"{_supabase_url()}/rest/v1/eval_task_projects",
            params={
                "select": "eval_task_id",
                "project_id": f"eq.{dataset_id}",
                "offset": str(offset),
                "limit": "1000",
            },
            headers=_supabase_headers(),
            timeout=30,
        )
        r.raise_for_status()
        rows = r.json()
        out.extend(row["eval_task_id"] for row in rows)
        if len(rows) < 1000:
            break
        offset += 1000
    return out


def fetch_task_records(client: httpx.Client, task_ids: list[str]) -> list[dict]:
    """Pull env_key, env_version, env_data_*, task_modality, current_version_id for given ids."""
    out: list[dict] = []
    for i in range(0, len(task_ids), 100):
        batch = task_ids[i : i + 100]
        r = client.get(
            f"{_supabase_url()}/rest/v1/eval_tasks",
            params={
                "select": "id,key,env_key,env_version,env_data_key,env_data_version,task_modality,current_version_id",
                "id": f"in.({','.join(batch)})",
            },
            headers=_supabase_headers(),
            timeout=30,
        )
        r.raise_for_status()
        out.extend(r.json())
    return out


def resolve_modality(raw_modality: str | None, env_key: str | None) -> str | None:
    """Resolve effective modality from (raw_modality, env_key).

    Rules:
      - fos-* env_key is deterministically computer_use (Fleet OS desktop apps).
        This overrides any DB label, since the env identity is ground truth.
        Inverse: 80% of fos-* tasks ARE labeled computer_use in DB; 18% are
        NULL; we infer the NULL case from env_key.
      - For non-fos envs, trust the DB label, with one override: 'computer_use'
        on a non-fos env actually means browser_use (v6 'DB-is-wrong' rule;
        sentry/jira/amazon/etc. labeled computer_use are really browser_use).
      - Returns None for excluded envs, missing fields, or unknown labels
        we can't infer.
    """
    if not env_key:
        return None
    if env_key in EXCLUDED_ENVS:
        return None

    is_fos = env_key.startswith(COMPUTER_USE_ENV_PREFIX)

    # fos-* envs are computer_use by env identity, regardless of label.
    if is_fos:
        return "computer_use"

    if not raw_modality:
        # Non-fos env with NULL modality: we can't reliably infer.
        return None

    m = raw_modality.strip().lower()
    if m == "tool_use":
        return "tool_use"
    if m == "browser_use":
        return "browser_use"
    if m == "computer_use":
        # Non-fos env labeled computer_use → really browser_use (v6 rule).
        return "browser_use"
    return None


def get_dataset_modalities(client: httpx.Client, dataset_id: str) -> dict[str, int]:
    """Return effective modality -> task count for a dataset (after fos-* override).

    Includes 'computer_use' if any real fos-* tasks exist (so the caller can
    detect and raise NotImplementedError).
    """
    task_ids = get_dataset_task_ids(client, dataset_id)
    if not task_ids:
        return {}
    records = fetch_task_records(client, task_ids)
    counts: Counter[str] = Counter()
    for rec in records:
        m = resolve_modality(rec.get("task_modality"), rec.get("env_key"))
        if m is None:
            continue
        counts[m] += 1
    return dict(counts)
