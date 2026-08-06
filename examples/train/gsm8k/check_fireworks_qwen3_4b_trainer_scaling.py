#!/usr/bin/env python3
"""Inspect Fireworks Qwen3-4B trainer scaling without creating resources.

This distinguishes two different requests:

* Changing ``accelerator_count`` or ``node_count`` changes the topology of one
  trainer replica. A validated training shape owns those values, so callers
  cannot override them.
* ``trainer_replica_count`` replicates the validated shape for data-parallel
  HSDP training. It is a supported run-level setting.

The script performs GET requests and local SDK validation only. It does not
provision or delete resources and does not incur accelerator charges.
"""

from __future__ import annotations

import argparse
import os
from typing import Any
from urllib.parse import urlencode

from fireworks.training.sdk import TrainerJobConfig, TrainerJobManager

DEFAULT_BASE_MODEL = "accounts/fireworks/models/qwen3-4b"
DEFAULT_SHAPE = "accounts/fireworks/trainingShapes/qwen3-4b-minimum"


def _shape_versions(
    manager: TrainerJobManager, base_model: str
) -> list[dict[str, Any]]:
    params = urlencode(
        {
            "filter": (f'snapshot.base_model="{base_model}" AND latest_validated=true'),
            "pageSize": 200,
        }
    )
    response = manager._get(
        f"/v1/accounts/-/trainingShapes/-/versions?{params}", timeout=30
    )
    if not response.is_success:
        raise RuntimeError(
            "Failed to list training shapes "
            f"(HTTP {response.status_code}): {response.text}"
        )
    body = response.json() or {}
    return (
        body.get("trainingShapeVersions") or body.get("training_shape_versions") or []
    )


def _validate_case(label: str, config: TrainerJobConfig) -> None:
    print(f"\n{label}")
    try:
        config.validate()
    except ValueError as exc:
        print(f"  REJECTED by SDK: {exc}")
        return
    payload = TrainerJobManager._build_trainer_create_payload(config)
    print("  ACCEPTED by SDK validation")
    print(f"  create payload: {payload}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--training-shape", default=DEFAULT_SHAPE)
    parser.add_argument("--trainer-replicas", type=int, default=2)
    args = parser.parse_args()

    if args.trainer_replicas <= 0:
        parser.error("--trainer-replicas must be positive")
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        parser.error("FIREWORKS_API_KEY is not set")

    manager = TrainerJobManager(api_key=api_key)
    profile = manager.resolve_training_profile(args.training_shape)
    shape_version = profile.training_shape_version

    print("Selected validated training shape")
    print(f"  shape: {args.training_shape}")
    print(f"  version: {shape_version}")
    print(f"  mode: {profile.trainer_mode}")
    print(f"  accelerator: {profile.accelerator_type}")
    print(f"  accelerators per replica: {profile.accelerator_count}")
    print(f"  nodes per replica: {profile.node_count}")
    print(f"  pipeline parallelism: {profile.pipeline_parallelism}")

    print(f"\nLatest validated shapes for {args.base_model}")
    matching_two_chip_shapes = []
    for version in _shape_versions(manager, args.base_model):
        snapshot = version.get("snapshot") or {}
        accelerators = int(snapshot.get("acceleratorCount") or 0)
        nodes = int(snapshot.get("nodeCount") or 0)
        print(
            f"  {version.get('name')}: mode={snapshot.get('trainerMode')}, "
            f"accelerator={snapshot.get('acceleratorType')}, "
            f"accelerators={accelerators}, nodes={nodes}"
        )
        if accelerators == 2 or nodes == 2:
            matching_two_chip_shapes.append(version.get("name"))
    if not matching_two_chip_shapes:
        print("  result: no validated two-accelerator or two-node shape is exposed")

    common = {
        "base_model": args.base_model,
        "lora_rank": 0,
        "training_shape_ref": shape_version,
    }
    _validate_case(
        "Attempt A: override one shaped replica to accelerator_count=2",
        TrainerJobConfig(**common, accelerator_count=2),
    )
    _validate_case(
        "Attempt B: override one shaped replica to node_count=2",
        TrainerJobConfig(**common, node_count=2),
    )
    _validate_case(
        f"Attempt C: request trainer_replica_count={args.trainer_replicas}",
        TrainerJobConfig(**common, trainer_replica_count=args.trainer_replicas),
    )

    total_accelerators = profile.accelerator_count * args.trainer_replicas
    total_nodes = profile.node_count * args.trainer_replicas
    print("\nEffective supported data-parallel topology")
    print(
        f"  {args.trainer_replicas} trainer replicas x "
        f"{profile.node_count} node/replica x "
        f"{profile.accelerator_count} accelerator/replica"
    )
    print(f"  total: {total_nodes} replica-nodes, {total_accelerators} accelerators")
    print(
        "  This replicates the complete 4B model; it is data parallelism, "
        "not a two-GPU model-parallel trainer shape."
    )


if __name__ == "__main__":
    main()
