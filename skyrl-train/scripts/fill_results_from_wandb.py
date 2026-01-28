#!/usr/bin/env python3
"""
Fill results tables in project.md from WandB run data.

Usage:
    python fill_results_from_wandb.py --run-id <wandb_run_id> [--entity thefleet] [--project fleet-task-grpo]
    
Environment:
    WANDB_API_KEY: WandB API key

Example:
    WANDB_API_KEY=<key> python fill_results_from_wandb.py --run-id mk6nr5ij
"""

import argparse
import os
import wandb


def fetch_run_metrics(entity: str, project: str, run_id: str) -> dict:
    """Fetch metrics from a WandB run."""
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    metrics = {
        "summary": dict(run.summary),
        "config": dict(run.config),
        "name": run.name,
        "state": run.state,
    }

    # Get history for time series data
    history = run.history()
    metrics["history"] = history

    return metrics


def extract_env_metrics(metrics: dict) -> dict:
    """Extract per-environment metrics from WandB data."""
    summary = metrics["summary"]
    results = {}

    # Look for environment-specific metrics
    envs = [
        "outlook",
        "github",
        "booking",
        "reddit",
        "ticketmaster",
        "fira",
        "zillow",
        "hubspot",
        "google-maps",
        "dropbox",
    ]

    for env in envs:
        env_data = {}

        # Look for pass@k metrics
        for key, value in summary.items():
            if env in key.lower():
                env_data[key] = value

        # Look for avg_turns
        turns_key = f"environment/fleet_task/{env}/avg_turns"
        if turns_key in summary:
            env_data["avg_turns"] = summary[turns_key]

        # Look for reward/score
        for pattern in [f"{env}/reward", f"{env}/score", f"{env}/pass"]:
            for key, value in summary.items():
                if pattern in key.lower():
                    env_data[key] = value

        if env_data:
            results[env] = env_data

    return results


def print_results_table(metrics: dict):
    """Print results in a format suitable for markdown tables."""
    summary = metrics["summary"]

    print("\n=== Run Summary ===")
    print(f"Run Name: {metrics['name']}")
    print(f"State: {metrics['state']}")

    print("\n=== All Metrics ===")
    for key in sorted(summary.keys()):
        if not key.startswith("_"):
            print(f"{key}: {summary[key]}")

    print("\n=== Per-Environment Metrics ===")
    env_metrics = extract_env_metrics(metrics)
    for env, data in env_metrics.items():
        print(f"\n{env}:")
        for key, value in data.items():
            print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Fill results tables from WandB")
    parser.add_argument("--run-id", required=True, help="WandB run ID")
    parser.add_argument("--entity", default="thefleet", help="WandB entity")
    parser.add_argument("--project", default="fleet-task-grpo", help="WandB project")
    args = parser.parse_args()

    if not os.environ.get("WANDB_API_KEY"):
        print("Error: WANDB_API_KEY environment variable required")
        return 1

    print(f"Fetching metrics from {args.entity}/{args.project}/{args.run_id}...")
    metrics = fetch_run_metrics(args.entity, args.project, args.run_id)
    print_results_table(metrics)

    return 0


if __name__ == "__main__":
    exit(main())
