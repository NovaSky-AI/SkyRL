"""CLI entry point: python -m integrations.fleet.auto_train ...

Subcommands:
  trigger    Discover new (project, modality) pairs → smoke → export → launch
  status     Print processed pairs from S3 state
  seed       Force-seed the S3 state with all current pairs (first-run helper)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import httpx

from .config import SUPPORTED_MODALITIES
from .discovery import (
    get_project_modalities,
    list_active_projects,
)
from .exporter import build_openenv_tasks, export_to_s3
from .launcher import launch_training
from .notify import (
    notify_launch,
    notify_launch_failure,
    notify_not_implemented,
    notify_smoke_failure,
)
from .smoke import run_smoke_test
from .state import ProcessedState

logger = logging.getLogger("auto_train")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_status(args: argparse.Namespace) -> int:
    state = ProcessedState()
    print(f"Seeded at: {state._seeded_at or '(not seeded yet)'}")
    print(f"Seen projects: {len(state.all_seen())}")
    pairs = sorted(state.all_processed())
    print(f"Processed pairs: {len(pairs)}")
    for project_key, modality in pairs:
        print(f"  {project_key:<40} {modality}")
    return 0


def cmd_seed(args: argparse.Namespace) -> int:
    state = ProcessedState()
    if state.is_seeded and not args.force:
        logger.error("State already seeded; pass --force to overwrite")
        return 1
    with httpx.Client() as client:
        projects = list_active_projects(client)
    keys = [p.project_key for p in projects]
    if args.force:
        state._seen_projects = set()
        state._processed_pairs = set()
        state._seeded_at = None
    state.seed(keys)
    print(f"Seeded {len(keys)} projects")
    return 0


def _process_one(
    client: httpx.Client,
    project,
    modality: str,
    state: ProcessedState,
    *,
    dry_run: bool,
    skip_smoke: bool,
    api_key: str,
) -> int:
    """Returns 0 on success, non-zero on failure (still marks processed if appropriate)."""
    project_key = project.project_key

    if modality == "computer_use":
        # Real fos-* computer_use; we don't have a training YAML
        modalities = get_project_modalities(client, project.id)
        notify_not_implemented(project_key, modality, modalities.get("computer_use", 0))
        state.mark_processed(project_key, modality)
        logger.warning("Skipping unsupported modality %s for %s", modality, project_key)
        return 0

    if modality not in SUPPORTED_MODALITIES:
        logger.warning("Unknown modality %s for %s; skipping", modality, project_key)
        return 0

    # Build tasks once — used by both smoke test and exporter
    logger.info("Building task list for %s/%s", project_key, modality)
    try:
        tasks = build_openenv_tasks(client, project.id, modality)
    except Exception as e:
        logger.exception("Failed to build tasks for %s/%s: %s", project_key, modality, e)
        notify_launch_failure(project_key, modality, f"task build error: {e}")
        return 1
    if not tasks:
        logger.warning("No exportable tasks for %s/%s; skipping", project_key, modality)
        state.mark_processed(project_key, modality)
        return 0

    # Smoke test
    if not skip_smoke:
        if not api_key:
            logger.error("FLEET_API_KEY not set; cannot run smoke test")
            return 2
        try:
            report = run_smoke_test(tasks, modality, project_key, api_key)
        except NotImplementedError as e:
            notify_not_implemented(project_key, modality, len(tasks))
            state.mark_processed(project_key, modality)
            logger.warning("%s", e)
            return 0
        if not report.passed:
            notify_smoke_failure(project_key, modality, report)
            logger.error("Smoke failed for %s/%s — NOT launching", project_key, modality)
            # Do NOT mark processed; retry on next tick
            return 2

    # Export
    if dry_run:
        logger.info(
            "[DRY RUN] Would upload %d tasks to s3 for %s/%s",
            len(tasks), project_key, modality,
        )
        s3_uri = f"s3://(dry-run)/{project_key}/openenv/all_{modality}.json"
    else:
        try:
            s3_uri = export_to_s3(tasks, project_key, modality)
        except Exception as e:
            logger.exception("Export failed: %s", e)
            notify_launch_failure(project_key, modality, f"S3 export error: {e}")
            return 1

    # Launch
    launched = launch_training(project_key, modality, dry_run=dry_run)
    if not launched:
        notify_launch_failure(project_key, modality, "fleet-launch.sh returned non-zero")
        return 1

    if not dry_run:
        state.mark_processed(project_key, modality)
        notify_launch(project_key, modality, len(tasks), s3_uri)
    return 0


def cmd_trigger(args: argparse.Namespace) -> int:
    state = ProcessedState()
    api_key = os.environ.get("FLEET_API_KEY", "")

    with httpx.Client() as client:
        projects = list_active_projects(client, team_id=args.team_id)

        # First-run seed: every current project marked 'seen' so we don't
        # retroactively trigger training for the 800+ existing projects.
        # --ignore-seed bypasses this for targeted testing.
        if not state.is_seeded and not args.ignore_seed:
            keys = [p.project_key for p in projects]
            if args.dry_run:
                logger.info("[DRY RUN] Would seed state with %d projects", len(keys))
                return 0
            state.seed(keys)
            logger.info("First-run seed complete; no training launched this tick")
            return 0

        # Per-tick: only enumerate modalities for projects we haven't seen
        # (unless --ignore-seed is set, in which case treat all as candidates).
        if args.ignore_seed:
            new_projects = list(projects)
            logger.info("--ignore-seed: treating all %d projects as candidates", len(new_projects))
        else:
            new_projects = [p for p in projects if not state.is_seen(p.project_key)]
            logger.info("Active projects: %d, new since last tick: %d",
                        len(projects), len(new_projects))

        if args.max_projects and len(new_projects) > args.max_projects:
            logger.info("Capping to first %d projects (newest-first)", args.max_projects)
            new_projects = new_projects[: args.max_projects]

        exit_code = 0
        for project in new_projects:
            modalities = get_project_modalities(client, project.id)
            if not modalities:
                logger.info("No supported tasks in %s; marking seen", project.project_key)
                if not args.dry_run:
                    state.mark_seen(project.project_key)
                continue
            project_ok = True
            for modality in modalities:
                if state.is_processed(project.project_key, modality) and not args.ignore_seed:
                    continue
                if args.modality and modality != args.modality:
                    continue
                rc = _process_one(
                    client, project, modality, state,
                    dry_run=args.dry_run,
                    skip_smoke=args.skip_smoke,
                    api_key=api_key,
                )
                if rc != 0:
                    project_ok = False
                    exit_code = rc
            # Only mark the project 'seen' if every modality landed in
            # processed_pairs (success or expected skip). Otherwise we leave
            # it unseen so the next tick retries the failed modalities.
            if project_ok and not args.dry_run:
                state.mark_seen(project.project_key)
        return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="auto_train")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_trig = sub.add_parser("trigger", help="Discover and process new pairs")
    p_trig.add_argument("--modality", choices=SUPPORTED_MODALITIES, default=None)
    p_trig.add_argument("--dry-run", action="store_true")
    p_trig.add_argument("--skip-smoke", action="store_true")
    p_trig.add_argument("--team-id", default=None,
                        help="Filter task_projects by team_id (e.g., Fleet Research)")
    p_trig.add_argument("--max-projects", type=int, default=None,
                        help="Cap the number of new projects processed this tick (newest-first)")
    p_trig.add_argument("--ignore-seed", action="store_true",
                        help="Treat all projects as candidates (bypass seen_projects and processed_pairs filters). For targeted testing.")
    p_trig.set_defaults(func=cmd_trigger)

    p_status = sub.add_parser("status", help="Show processed pairs")
    p_status.set_defaults(func=cmd_status)

    p_seed = sub.add_parser("seed", help="Force-seed state with all current pairs")
    p_seed.add_argument("--force", action="store_true",
                        help="Overwrite existing seed")
    p_seed.set_defaults(func=cmd_seed)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
