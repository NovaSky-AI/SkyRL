"""Smoke test: provision sample envs via Fleet SDK before exporting / training.

Lightweight check that the envs in a dataset are healthy. Uses fleet.make()
to provision a small sample and verifies the call returns without error.
"""

from __future__ import annotations

import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .config import (
    EXCLUDED_ENVS,
    SMOKE_RETRIES,
    SMOKE_SAMPLE_ENVS,
    SMOKE_TTL_SECONDS,
    SUPPORTED_MODALITIES,
)

logger = logging.getLogger(__name__)


@dataclass
class SmokeResult:
    env_key: str
    env_version: Optional[str]
    success: bool
    error: Optional[str] = None
    elapsed_seconds: float = 0.0


@dataclass
class SmokeReport:
    dataset_key: str
    modality: str
    total_envs_in_dataset: int
    sampled_envs: int
    results: list[SmokeResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.results) > 0 and all(r.success for r in self.results)

    def summary(self) -> str:
        ok = sum(1 for r in self.results if r.success)
        return f"{ok}/{len(self.results)} envs ok (of {self.total_envs_in_dataset} envs in dataset)"

    def failures(self) -> list[SmokeResult]:
        return [r for r in self.results if not r.success]


def _sample_env_keys(
    tasks: list[dict],
    n: int = SMOKE_SAMPLE_ENVS,
    rng: random.Random | None = None,
) -> list[tuple[str, str | None]]:
    """Pick n distinct (env_key, env_version) pairs, biased toward most common."""
    rng = rng or random.Random()
    groups: dict[tuple[str, str | None], int] = defaultdict(int)
    for t in tasks:
        env_key = t.get("env_key")
        if not env_key or env_key in EXCLUDED_ENVS:
            continue
        groups[(env_key, t.get("env_version"))] += 1
    if not groups:
        return []
    # Sort by count desc and pick top n distinct env_keys (not just versions)
    sorted_pairs = sorted(groups.items(), key=lambda kv: kv[1], reverse=True)
    chosen: list[tuple[str, str | None]] = []
    seen_keys: set[str] = set()
    for (env_key, env_version), _count in sorted_pairs:
        if env_key in seen_keys:
            continue
        chosen.append((env_key, env_version))
        seen_keys.add(env_key)
        if len(chosen) >= n:
            break
    return chosen


def _provision_one(
    api_key: str,
    env_key: str,
    env_version: str | None,
    ttl_seconds: int = SMOKE_TTL_SECONDS,
    retries: int = SMOKE_RETRIES,
) -> SmokeResult:
    try:
        from fleet import Fleet
    except ImportError:
        return SmokeResult(
            env_key=env_key,
            env_version=env_version,
            success=False,
            error="fleet-python not installed (pip install 'fleet-python<=0.2.119')",
        )

    fleet = Fleet(api_key=api_key)
    # The SDK accepts "env_key" or "env_key:version" depending on use; v6 smoke
    # test passes env_key without version, which uses the latest. Match that.
    last_error = "no attempts"
    for attempt in range(1, retries + 1):
        start = time.time()
        try:
            env = fleet.make(env_key=env_key, ttl_seconds=ttl_seconds)
            elapsed = time.time() - start
            logger.info("smoke[%s] ok in %.1fs (attempt %d)", env_key, elapsed, attempt)
            # Try to close cleanly; ignore errors since the instance has a short TTL
            close = getattr(env, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            return SmokeResult(
                env_key=env_key,
                env_version=env_version,
                success=True,
                elapsed_seconds=elapsed,
            )
        except Exception as e:
            last_error = str(e)
            elapsed = time.time() - start
            logger.warning(
                "smoke[%s] attempt %d/%d failed in %.1fs: %s",
                env_key, attempt, retries, elapsed, e,
            )
    return SmokeResult(
        env_key=env_key,
        env_version=env_version,
        success=False,
        error=last_error,
    )


def run_smoke_test(
    tasks: list[dict],
    modality: str,
    dataset_key: str,
    api_key: str,
    sample_envs: int = SMOKE_SAMPLE_ENVS,
) -> SmokeReport:
    """Smoke test a sample of envs from a dataset.

    Raises NotImplementedError if modality is not supported (e.g., real
    fos-* computer_use).
    """
    if modality not in SUPPORTED_MODALITIES:
        raise NotImplementedError(
            f"Modality {modality!r} not supported (dataset={dataset_key}). "
            f"Supported: {SUPPORTED_MODALITIES}"
        )

    unique_envs = {t.get("env_key") for t in tasks if t.get("env_key")}
    report = SmokeReport(
        dataset_key=dataset_key,
        modality=modality,
        total_envs_in_dataset=len(unique_envs),
        sampled_envs=0,
    )

    sampled = _sample_env_keys(tasks, n=sample_envs)
    report.sampled_envs = len(sampled)
    if not sampled:
        logger.warning("No sampleable envs for %s/%s", dataset_key, modality)
        return report

    for env_key, env_version in sampled:
        result = _provision_one(api_key, env_key, env_version)
        report.results.append(result)

    logger.info("smoke %s/%s: %s", dataset_key, modality, report.summary())
    return report
