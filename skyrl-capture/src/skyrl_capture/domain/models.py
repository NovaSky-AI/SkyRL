"""The edge: what a caller may name, what creation hands back, and the formats.

Everything here is about the boundary between a request and the domain. The
trajectory shapes themselves live in `domain/records.py`; what is left is the
validation a command runs before it decides anything, and the two small
documents that are neither an active trajectory nor a finished one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from skyrl_capture.domain.records import ExportError, TrajectoryError
from skyrl_capture.routes import DATA_PLANE_PREFIX


@dataclass(frozen=True)
class CreatedTrajectory:
    """What creation hands back: the trajectory, and the route to send it to.

    No credential. The route's trajectory id is correlation, not
    authorization, and returning a generated secret beside it said otherwise.
    """

    id: str
    base_url: str
    mode: str
    protocol: str
    status: str

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "base_url": self.base_url,
            "mode": self.mode,
            "protocol": self.protocol,
            "status": self.status,
        }


def route_base_url(public_url: str, trajectory_id: str, client_suffix: str) -> str:
    return f"{public_url.rstrip('/')}{DATA_PLANE_PREFIX}{trajectory_id}{client_suffix}"


# A caller-supplied id becomes the first segment of every route this
# trajectory serves, and the name of its files on disk, so it has to survive
# both a URL and a path untouched.
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_identifier(identifier: str) -> str:
    if not IDENTIFIER_PATTERN.match(identifier):
        raise TrajectoryError(
            f"trajectory id {identifier!r} must be 1-128 characters of letters, digits, "
            "'.', '_' or '-', starting with a letter or digit: it is the first path "
            "segment of this trajectory's route and the name of its record"
        )
    return identifier


# -- exports --------------------------------------------------------------------------
# One format per use case. None is a superset of another, and everything is a
# projection of ``graph``.
EXPORT_FORMATS = ("graph", "replay", "text_samples", "token_samples")


def normalize_export_format(value: str) -> str:
    """Accept both ``text-samples`` (CLI) and ``text_samples`` (API)."""
    candidate = value.strip().lower().replace("-", "_")
    if candidate not in EXPORT_FORMATS:
        raise ExportError(
            f"unknown export format {value!r}; expected one of {', '.join(EXPORT_FORMATS)}"
        )
    return candidate


def export_public(row: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "id": row["id"],
        "status": row["status"],
        "format": row["format"],
        "project": row["project"],
        "run": row["run_id"],
        "trajectory": row["trajectory_id"],
        "selected_trajectory_ids": list(row["selected_trajectory_ids"] or []),
        # Which flags produced this export. An export that trains on nothing
        # should be able to explain itself without a separate manifest.
        "options": row["options"] or {},
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
    if row["status"] == "ready":
        payload.update(
            {
                "output_uri": row["output_uri"],
                "byte_count": row["byte_count"],
                "record_count": row["record_count"],
                "checksum": row["checksum"],
            }
        )
    if row["status"] == "failed":
        payload["error"] = row["error"]
    return payload
