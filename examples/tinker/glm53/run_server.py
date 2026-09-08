"""Start a native SkyRL GLM service on an already-owned Ray cluster."""

import argparse
import json
import os
from pathlib import Path
import sys


def build_config(profile: str, model_path: Path, state_dir: Path) -> dict:
    root = Path(__file__).parent
    config = json.loads((root / "common.json").read_text())
    overrides = json.loads((root / f"{profile}.json").read_text())
    for key, value in overrides.items():
        if isinstance(value, dict):
            config[key].update(value)
        else:
            config[key] = value
    config["trainer.policy.model.path"] = str(model_path)
    config["generator.inference_engine.engine_init_kwargs"]["model"] = str(model_path)
    config["trainer.policy.model.lora.lora_sync_path"] = str(state_dir / "lora-sync")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=["32k-2n", "256k-2n", "256k-3n"])
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--database-path", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    if not all(path.is_absolute() for path in (args.model_path, args.state_dir, args.database_path)):
        parser.error("model-path, state-dir and database-path must be absolute")
    config = build_config(args.profile, args.model_path, args.state_dir)
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
