"""Control-plane request models.

Payloads are intentionally shallow, as the PRD requires: optional
configuration can be added later without changing the lifecycle shape.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TrajectoryCreate(BaseModel):
    # Required. The SDK generates it before it sends this, so the id exists
    # before the trajectory does and a retry after a lost response names the
    # same trial. It is also the session key sent to the inference engine and
    # the name of this trajectory's files.
    trajectory_id: str
    project: str
    # Where this attempt sits: project / run / trajectory, with the task it
    # attempted and the step that produced it. All optional -- a one-off trial
    # belongs to no run. The run is created on first use, like the project.
    run_id: str | None = None
    task_id: str | None = None
    step: int | None = None
    # Bare string tags for filtering; free-form key/value for everything else.
    labels: list[str] = Field(default_factory=list)
    annotations: dict[str, Any] = Field(default_factory=dict)
    bodies: Literal["full", "sampled"] = "full"
    source_metadata: dict[str, Any] = Field(default_factory=dict)


class TrajectoryFinish(BaseModel):
    # Metadata is not sealed by finishing, so these merge like any other write.
    labels: list[str] | None = None
    annotations: dict[str, Any] | None = None
    command_result: str | None = None
    # What to render from the committed record and return in the reply. The
    # default is the whole trajectory; the other three are the training
    # projections of it. Rendering is not a job -- the record is already
    # compiled by the time this is read.
    format: str = "graph"
    options: dict[str, Any] = Field(default_factory=dict)


class MetadataUpdate(BaseModel):
    """Merge annotations, add and remove labels. Any subset may be given."""

    annotations: dict[str, Any] = Field(default_factory=dict)
    remove_annotations: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    remove_labels: list[str] = Field(default_factory=list)


class ExportCreate(BaseModel):
    format: str
    # Exactly one scope. A run is the one an RL harness wants: the rows this
    # training run produced.
    project: str | None = None
    run: str | None = None
    trajectory: str | None = None
    # Where to deliver a copy of the finished artifact: any object-store URI
    # the deployment can write (file://, s3://, gs://). There is no registry of
    options: dict[str, Any] = Field(default_factory=dict)
