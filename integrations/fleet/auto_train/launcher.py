"""Launch training via scripts/fleet-launch.sh."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from .config import MODALITY_YAML_MAP, REQUIRED_LAUNCH_ENV_VARS, SUPPORTED_MODALITIES

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "scripts" / "fleet-launch.sh").exists():
            return parent
    raise FileNotFoundError("scripts/fleet-launch.sh not found above auto_train module")


def build_launch_command(
    project_key: str,
    modality: str,
    env_vars: dict[str, str],
    extra_sky_args: list[str] | None = None,
) -> list[str]:
    if modality not in SUPPORTED_MODALITIES:
        raise NotImplementedError(
            f"modality {modality!r} not supported (project={project_key})"
        )

    root = _repo_root()
    yaml_path = root / MODALITY_YAML_MAP[modality]
    launch_script = root / "scripts" / "fleet-launch.sh"

    cmd: list[str] = [
        "bash", str(launch_script), str(yaml_path),
        "--env", f"DATA_VERSION={project_key}",
        "--env", f"MODALITY={modality}",
    ]
    for k, v in env_vars.items():
        cmd.extend(["--env", f"{k}={v}"])
    cmd.extend(["--retry-until-up", "-y"])
    if extra_sky_args:
        cmd.extend(extra_sky_args)
    return cmd


def launch_training(
    project_key: str,
    modality: str,
    dry_run: bool = False,
    extra_sky_args: list[str] | None = None,
) -> bool:
    """Returns True if `sky launch` exited 0 (success)."""
    missing = [v for v in REQUIRED_LAUNCH_ENV_VARS if not os.environ.get(v)]
    if missing and not dry_run:
        logger.error("Missing required env vars: %s", missing)
        return False

    env_vars = {k: os.environ.get(k, "") for k in REQUIRED_LAUNCH_ENV_VARS}
    cmd = build_launch_command(project_key, modality, env_vars, extra_sky_args)

    if dry_run:
        # Redact credential values in dry-run output
        printable = []
        i = 0
        while i < len(cmd):
            if cmd[i] == "--env" and i + 1 < len(cmd):
                kv = cmd[i + 1]
                k, _, _v = kv.partition("=")
                if k in REQUIRED_LAUNCH_ENV_VARS:
                    printable.extend([cmd[i], f"{k}=<redacted>"])
                else:
                    printable.extend([cmd[i], cmd[i + 1]])
                i += 2
            else:
                printable.append(cmd[i])
                i += 1
        logger.info("[DRY RUN] %s", " ".join(printable))
        return True

    logger.info("Launching: %s/%s", project_key, modality)
    result = subprocess.run(cmd, cwd=str(_repo_root()))
    if result.returncode != 0:
        logger.error(
            "fleet-launch.sh failed (exit=%d) for %s/%s",
            result.returncode, project_key, modality,
        )
        return False
    logger.info("fleet-launch.sh succeeded for %s/%s", project_key, modality)
    return True
