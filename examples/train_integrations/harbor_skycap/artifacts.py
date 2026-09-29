"""Each step's skycap documents, uploaded to W&B as one version of a ``skycap-records`` artifact."""

import re
from pathlib import Path
from typing import Iterable, Optional

import wandb
from loguru import logger

ARTIFACT_TYPE = "skycap-records"


def artifact_name(run_id: str) -> str:
    return f"{ARTIFACT_TYPE}-{re.sub(r'[^a-zA-Z0-9_.-]', '-', run_id)}"


def log_step_records(
    record_dir: Path, trajectory_ids: Iterable[str], step: Optional[int], phase: str = "train"
) -> Optional[str]:
    """Upload the step's documents, without sidecars, aliased ``{phase}-step-N`` and ``latest``. Returns the artifact name."""
    run = wandb.run
    if run is None:
        return None
    documents = (record_dir / f"{trajectory_id}.json.zst" for trajectory_id in set(trajectory_ids))
    files = sorted(path for path in documents if path.exists())
    if not files:
        logger.warning(f"skycap artifact: no {phase} records in {record_dir} for step {step}")
        return None
    artifact = wandb.Artifact(
        name=artifact_name(run.id),
        type=ARTIFACT_TYPE,
        metadata={
            "global_step": step,
            "training_phase": phase,
            "num_trajectories": len(set(trajectory_ids)),
            "run_id": run.id,
            "contents": "documents",
        },
    )
    for path in files:
        artifact.add_file(str(path), name=path.name)
    aliases = ["latest"] if step is None else [f"{phase}-step-{step}", "latest"]
    run.log_artifact(artifact, aliases=aliases)
    logger.info(f"skycap artifact {artifact.name}:{aliases[0]}: {len(files)} files")
    return artifact.name
