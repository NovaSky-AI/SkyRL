"""S3-backed state for the auto-train trigger.

State file: s3://fleet-internal-datasets/.auto_train_state.json
{
  "seen_datasets": ["proj-a", "proj-b", ...],   # dataset_keys seen at least once
  "processed_pairs": [                            # (dataset_key, modality) triggered
    ["proj-a", "tool_use"],
    ["proj-b", "browser_use"]
  ],
  "seeded_at": "2026-05-21T..."                   # absence ⇒ never seeded
}

(In the Fleet UI these are called Datasets. In Supabase they live in
`task_projects` with a `project_key` column. We use the UI term in our
state file and user-facing messages.)

Two-level model:
  - seen_datasets: cheap dataset-level filter. Seeded with all current active
    datasets on first run, then incrementally extended.
  - processed_pairs: per-modality completion. Marked after a successful launch
    (or a NotImplementedError skip) so we don't retry forever.

A single GitHub Actions runner mutates this (concurrency group serializes
runs), so no locking needed.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Iterable

import boto3
from botocore.exceptions import ClientError

from .config import AWS_DEFAULT_REGION, S3_DATASET_BUCKET, S3_STATE_KEY

logger = logging.getLogger(__name__)


def _s3_client():
    from botocore.config import Config

    return boto3.client(
        "s3",
        region_name=AWS_DEFAULT_REGION,
        config=Config(retries={"max_attempts": 3, "mode": "adaptive"}),
    )


class ProcessedState:
    def __init__(self, bucket: str = S3_DATASET_BUCKET, key: str = S3_STATE_KEY):
        self.bucket = bucket
        self.key = key
        self._seen_datasets: set[str] = set()
        self._processed_pairs: set[tuple[str, str]] = set()
        self._seeded_at: str | None = None
        self._s3 = _s3_client()
        self._load()

    @property
    def is_seeded(self) -> bool:
        return self._seeded_at is not None

    def _load(self) -> None:
        try:
            resp = self._s3.get_object(Bucket=self.bucket, Key=self.key)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404"):
                logger.info("State file s3://%s/%s missing, first run", self.bucket, self.key)
                return
            raise
        data = json.loads(resp["Body"].read().decode("utf-8"))
        # Back-compat: accept old `seen_projects` key if present
        self._seen_datasets = set(data.get("seen_datasets") or data.get("seen_projects", []))
        self._processed_pairs = {tuple(p) for p in data.get("processed_pairs", [])}
        self._seeded_at = data.get("seeded_at")
        logger.info(
            "Loaded state: %d seen datasets, %d processed pairs, seeded_at=%s",
            len(self._seen_datasets),
            len(self._processed_pairs),
            self._seeded_at,
        )

    def _save(self) -> None:
        payload = {
            "seen_datasets": sorted(self._seen_datasets),
            "processed_pairs": sorted([list(p) for p in self._processed_pairs]),
            "seeded_at": self._seeded_at,
        }
        body = json.dumps(payload, indent=2).encode("utf-8")
        self._s3.put_object(
            Bucket=self.bucket, Key=self.key, Body=body, ContentType="application/json"
        )

    # per-dataset
    def is_seen(self, dataset_key: str) -> bool:
        return dataset_key in self._seen_datasets

    def mark_seen(self, dataset_key: str) -> None:
        if dataset_key not in self._seen_datasets:
            self._seen_datasets.add(dataset_key)
            self._save()

    # per-(dataset, modality)
    def is_processed(self, dataset_key: str, modality: str) -> bool:
        return (dataset_key, modality) in self._processed_pairs

    def mark_processed(self, dataset_key: str, modality: str) -> None:
        self._processed_pairs.add((dataset_key, modality))
        self._seen_datasets.add(dataset_key)
        self._save()

    def all_processed(self) -> set[tuple[str, str]]:
        return set(self._processed_pairs)

    def all_seen(self) -> set[str]:
        return set(self._seen_datasets)

    def seed(self, dataset_keys: Iterable[str]) -> None:
        """First-run seed: every current active dataset marked 'seen' so we
        don't retroactively launch training. Does NOT enumerate modalities,
        we only do that for genuinely new datasets."""
        if self.is_seeded:
            logger.warning("State already seeded at %s; ignoring", self._seeded_at)
            return
        self._seen_datasets.update(dataset_keys)
        self._seeded_at = datetime.now(timezone.utc).isoformat()
        self._save()
        logger.info("Seeded state with %d datasets", len(self._seen_datasets))
