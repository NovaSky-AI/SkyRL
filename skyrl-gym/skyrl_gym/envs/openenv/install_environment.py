#!/usr/bin/env python3
#
# Helper that pulls prebuilt OpenEnv environment Docker images and tags them locally.
#
# Examples:
#   uv run skyrl_gym/envs/openenv/install_environment.py echo-env
#   uv run skyrl_gym/envs/openenv/install_environment.py coding-env
#   uv run skyrl_gym/envs/openenv/install_environment.py atari-env
#   uv run skyrl_gym/envs/openenv/install_environment.py chat-env
#   docker kill $(docker ps -q)
         # pulls all images
#

import argparse
import sys
import subprocess

# Image mapping: from https://github.com/meta-pytorch/OpenEnv/pkgs/container/ 
ENV_IMAGES = {
    "base": "ghcr.io/meta-pytorch/openenv-base:sha-64d4b10",
    "atari-env": "ghcr.io/meta-pytorch/openenv-atari-env:sha-64d4b10",
    "coding-env": "ghcr.io/meta-pytorch/openenv-coding-env:sha-64d4b10",
    "chat-env": "ghcr.io/meta-pytorch/openenv-chat-env:sha-64d4b10",
    "echo-env": "ghcr.io/meta-pytorch/openenv-echo-env:sha-64d4b10",
}

def run_command(cmd, check=True):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=check)


def pull_image(image_url, local_tag):
    print(f"Pulling image from URL:{image_url}")
    run_command(["docker", "pull", image_url])
    print(f"Tagging as {local_tag}:latest")
    run_command(["docker", "tag", image_url, f"{local_tag}:latest"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull prebuilt OpenEnv Docker images.")
    parser.add_argument(
        "env_name",
        nargs="?", 
        help="Environment name (e.g., echo-env, coding-env, atari-env, chat-env). Leave blank to pull all.",
    )
    args = parser.parse_args()

    try:
        print("Pulling OpenEnv base image...")
        pull_image(ENV_IMAGES["base"], "openenv-base")

        if args.env_name:
            env_name = args.env_name
            if env_name not in ENV_IMAGES:
                print(f"Error: Unknown environment '{env_name}'.")
                print("Available environments:", ", ".join(ENV_IMAGES.keys()))
                sys.exit(1)

            print(f"Pulling {env_name} image...")
            pull_image(ENV_IMAGES[env_name], env_name)

        else:
            print("No environment specified. Pulling all environments...")
            for name, url in ENV_IMAGES.items():
                if name == "base":
                    continue  
                pull_image(url, name)

        print("All images pulled successfully.")

    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
