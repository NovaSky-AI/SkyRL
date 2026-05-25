"""Export project tasks → OpenEnv JSON → S3.

Mirrors fleet-research-scripts/training-data-pipeline/v6/export_s3.py but
scoped to a single project + modality. Reads directly from Supabase.
"""

from __future__ import annotations

import json
import logging
from typing import Iterable

import boto3
import httpx
from botocore.config import Config

from .config import (
    AWS_DEFAULT_REGION,
    OPENENV_FIELDS,
    S3_DATASET_BUCKET,
    S3_DATASET_PATH_TEMPLATE,
    SUPPORTED_MODALITIES,
)
from .discovery import (
    _supabase_headers,
    _supabase_url,
    fetch_task_records,
    get_project_task_ids,
    resolve_modality,
)

logger = logging.getLogger(__name__)


def _fetch_task_versions(
    client: httpx.Client, version_ids: Iterable[str]
) -> dict[str, dict]:
    """version_id -> {prompt, verifier_id, env_variables}"""
    out: dict[str, dict] = {}
    vids = list(version_ids)
    for i in range(0, len(vids), 100):
        batch = vids[i : i + 100]
        r = client.get(
            f"{_supabase_url()}/rest/v1/eval_task_versions",
            params={
                "select": "id,prompt,verifier_id,env_variables",
                "id": f"in.({','.join(batch)})",
            },
            headers=_supabase_headers(),
            timeout=30,
        )
        r.raise_for_status()
        for row in r.json():
            out[row["id"]] = row
    return out


def _fetch_verifier_code(
    client: httpx.Client, verifier_ids: Iterable[str]
) -> dict[str, str]:
    """verifier_id -> latest display_src."""
    out: dict[str, str] = {}
    vids = list(set(verifier_ids))
    for i in range(0, len(vids), 100):
        batch = vids[i : i + 100]
        r = client.get(
            f"{_supabase_url()}/rest/v1/verifier_versions",
            params={
                "select": "verifier_id,display_src,version",
                "verifier_id": f"in.({','.join(batch)})",
                "order": "version.desc",
            },
            headers=_supabase_headers(),
            timeout=30,
        )
        r.raise_for_status()
        seen: set[str] = set()
        for row in r.json():
            vid = row["verifier_id"]
            if vid in seen:
                continue
            seen.add(vid)
            src = row.get("display_src")
            if src:
                out[vid] = src
    return out


def build_openenv_tasks(
    client: httpx.Client, project_id: str, modality: str
) -> list[dict]:
    """Return list of task dicts with OPENENV_FIELDS for the given project + modality."""
    if modality not in SUPPORTED_MODALITIES:
        raise NotImplementedError(f"modality {modality!r} not supported")

    task_ids = get_project_task_ids(client, project_id)
    if not task_ids:
        return []
    records = fetch_task_records(client, task_ids)

    # Filter to target modality (after fos-* override)
    tasks_by_key: dict[str, dict] = {}
    version_to_key: dict[str, str] = {}
    for rec in records:
        env_key = rec.get("env_key")
        effective = resolve_modality(rec.get("task_modality"), env_key)
        if effective != modality:
            continue
        key = rec.get("key")
        if not key:
            continue
        tasks_by_key[key] = {
            "task_key": key,
            "env_key": env_key,
            "env_version": rec.get("env_version"),
            "data_key": rec.get("env_data_key"),
            "data_version": rec.get("env_data_version"),
            "task_modality": modality,
        }
        vid = rec.get("current_version_id")
        if vid:
            version_to_key[vid] = key

    if not tasks_by_key:
        return []

    # Pull prompt + verifier_id + env_variables
    versions = _fetch_task_versions(client, version_to_key.keys())
    verifier_to_keys: dict[str, list[str]] = {}
    for vid, vrow in versions.items():
        key = version_to_key.get(vid)
        if not key or key not in tasks_by_key:
            continue
        if vrow.get("prompt"):
            tasks_by_key[key]["prompt"] = vrow["prompt"]
        if vrow.get("env_variables"):
            tasks_by_key[key]["env_variables"] = vrow["env_variables"]
        verifier_id = vrow.get("verifier_id")
        if verifier_id:
            verifier_to_keys.setdefault(verifier_id, []).append(key)

    # Pull verifier source
    verifier_code = _fetch_verifier_code(client, verifier_to_keys.keys())
    for verifier_id, keys in verifier_to_keys.items():
        code = verifier_code.get(verifier_id)
        if not code:
            continue
        for k in keys:
            tasks_by_key[k]["verifier_code"] = code

    # Filter: must have prompt + verifier_code
    out: list[dict] = []
    dropped_no_prompt = 0
    dropped_no_verifier = 0
    for t in tasks_by_key.values():
        if not t.get("prompt"):
            dropped_no_prompt += 1
            continue
        if not t.get("verifier_code"):
            dropped_no_verifier += 1
            continue
        # Restrict to documented fields
        out.append({field: t.get(field) for field in OPENENV_FIELDS})

    if dropped_no_prompt or dropped_no_verifier:
        logger.warning(
            "Dropped %d tasks (missing prompt=%d, missing verifier=%d)",
            dropped_no_prompt + dropped_no_verifier,
            dropped_no_prompt,
            dropped_no_verifier,
        )
    return out


def _s3_client():
    return boto3.client(
        "s3",
        region_name=AWS_DEFAULT_REGION,
        config=Config(retries={"max_attempts": 3, "mode": "adaptive"}),
    )


def export_to_s3(
    tasks: list[dict],
    project_key: str,
    modality: str,
    bucket: str = S3_DATASET_BUCKET,
) -> str:
    """Upload tasks to s3://{bucket}/{project_key}/openenv/all_{modality}.json.

    Returns the S3 URI.
    """
    key = S3_DATASET_PATH_TEMPLATE.format(project_key=project_key, modality=modality)
    payload = json.dumps({"tasks": tasks}, ensure_ascii=False).encode("utf-8")
    s3 = _s3_client()
    s3.put_object(Bucket=bucket, Key=key, Body=payload, ContentType="application/json")
    uri = f"s3://{bucket}/{key}"
    logger.info("Uploaded %d tasks → %s (%d KB)", len(tasks), uri, len(payload) // 1024)
    return uri


def export_project(
    client: httpx.Client,
    project_id: str,
    project_key: str,
    modality: str,
    bucket: str = S3_DATASET_BUCKET,
) -> tuple[str, int]:
    """Full export flow: query tasks → build JSON → upload. Returns (s3_uri, task_count)."""
    tasks = build_openenv_tasks(client, project_id, modality)
    if not tasks:
        raise ValueError(
            f"No exportable tasks for {project_key}/{modality} "
            "(check prompt/verifier coverage)"
        )
    uri = export_to_s3(tasks, project_key, modality, bucket=bucket)
    return uri, len(tasks)
