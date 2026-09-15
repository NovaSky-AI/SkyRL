"""Start a native SkyRL GLM service on an already-owned Ray cluster."""

import argparse
import json
import os
import sys
from pathlib import Path

CONFIG_DIR = Path(__file__).parent / "configs"


def parse_profile_ranks(value: str) -> list[int]:
    try:
        ranks = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("profile ranks must be comma-separated integers") from error
    if not ranks or any(rank < 0 for rank in ranks) or len(ranks) != len(set(ranks)):
        raise argparse.ArgumentTypeError("profile ranks must be unique nonnegative integers")
    return ranks


def validate_profile_ranks(ranks: list[int], world_size: int) -> None:
    if any(rank >= world_size for rank in ranks):
        raise ValueError(f"profile ranks {ranks} exceed trainer world size {world_size}")


def get_profiles() -> list[str]:
    return sorted(path.stem for path in CONFIG_DIR.glob("*.json") if path.stem != "common")


def build_config(profile: str, model_path: Path, state_dir: Path, profile_dir: Path) -> dict:
    overrides = json.loads((CONFIG_DIR / f"{profile}.json").read_text())
    parent = overrides.pop("extends", None)
    config = json.loads((CONFIG_DIR / f"{parent}.json").read_text()) if parent else {}
    for key, value in overrides.items():
        if isinstance(value, dict) and key in config:
            config[key].update(value)
        else:
            config[key] = value
    config["trainer.policy.model.path"] = str(model_path)
    config["generator.inference_engine.engine_init_kwargs"]["model"] = str(model_path)
    config["trainer.policy.model.lora.lora_sync_path"] = str(state_dir / "lora-sync")
    if not profile_dir.is_absolute():
        raise ValueError("profile-dir must be an absolute path on the policy nodes")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=get_profiles())
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--database-path", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--profile-dir",
        type=Path,
        required=True,
        help="Operator-owned trainer trace directory; the client controls capture sessions",
    )
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument(
        "--profile-ranks",
        type=parse_profile_ranks,
        default=[0],
        help="Comma-separated trainer ranks to capture (default: 0)",
    )
    args = parser.parse_args()
    if not all(path.is_absolute() for path in (args.model_path, args.state_dir, args.database_path)):
        parser.error("model-path, state-dir and database-path must be absolute")
    config = build_config(args.profile, args.model_path, args.state_dir, args.profile_dir)
    world_size = config["trainer.placement.policy_num_nodes"] * config["trainer.placement.policy_num_gpus_per_node"]
    validate_profile_ranks(args.profile_ranks, world_size)
    if args.print_config:
        print(json.dumps(config, indent=2))
        return
    if not (args.model_path / "config.json").is_file():
        parser.error("model-path must contain a downloaded Hugging Face snapshot")
    args.state_dir.mkdir(parents=True, exist_ok=True)
    args.database_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "skyrl.tinker.api",
        "--backend",
        "megatron",
        "--base-model",
        str(args.model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--torch-profiler",
        json.dumps(
            {
                "export_dir": str(args.profile_dir),
                "ranks": args.profile_ranks,
            }
        ),
        "--backend-config",
        json.dumps(config),
        "--database-url",
        f"sqlite:///{args.database_path}",
        "--checkpoints-base",
        str(args.state_dir / "checkpoints"),
        "--external-inference-lora-base",
        str(args.state_dir / "lora-models"),
        "--forwarding-inference-max-connections",
        "512",
        "--forwarding-inference-timeout-sec",
        "1800",
    ]
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
