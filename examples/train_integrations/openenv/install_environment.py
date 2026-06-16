#!/usr/bin/env python3
#
# Helper that pulls prebuilt OpenEnv environment Docker images and tags them locally.
#
# Examples:
#   uv run skyrl_gym/envs/openenv/install_environment.py echo-env
#   uv run skyrl_gym/envs/openenv/install_environment.py coding-env
#   uv run skyrl_gym/envs/openenv/install_environment.py atari-env
#   uv run skyrl_gym/envs/openenv/install_environment.py openspiel-env
#   uv run skyrl_gym/envs/openenv/install_environment.py sumo-rl-env
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
    "echo-env": "ghcr.io/meta-pytorch/openenv-echo-env:sha-64d4b10",
    "openspiel-env": "ghcr.io/meta-pytorch/openenv-openspiel-base:sha-e622c7e",
    "sumo-rl-env": "ghcr.io/meta-pytorch/openenv-sumo-rl-env:sha-c25298c",
    "finrl-env": "",
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
        help="Environment name (e.g., echo-env, coding-env, atari-env, openspiel-env, sumo-rl-env). Leave blank to pull all.",
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

            if env_name == "finrl-env":
                # NOTE(shu): finrl-env is not available remotely; need to build from source
                print("Building finrl-env from source (no remote package available)...")

                repo_url = "https://github.com/meta-pytorch/OpenEnv.git"
                repo_dir = "OpenEnv"

                try:
                    # clone and build base
                    run_command(["git", "clone", "--depth", "1", repo_url, repo_dir])

                    # NOTE(shu): original docker file does not work; need to replace finrl==0.3.6 with 0.3.7
                    finrl_dockerfile = f"{repo_dir}/src/envs/finrl_env/server/Dockerfile"
                    run_command(["sed", "-i", "s/finrl==0.3.6/finrl==0.3.7/g", finrl_dockerfile])

                    base_dockerfile = f"{repo_dir}/src/core/containers/images/Dockerfile"
                    run_command(["docker", "build", "-t", "envtorch-base:latest", "-f", base_dockerfile, repo_dir])

                    finrl_dockerfile = f"{repo_dir}/src/envs/finrl_env/server/Dockerfile"
                    run_command(["docker", "build", "-t", "finrl-env:latest", "-f", finrl_dockerfile, repo_dir])

                    # Now start the docker with default sample data
                    run_command(["docker", "run", "-p", "8000:8000", "finrl-env:latest"])

                finally:
                    run_command(["rm", "-rf", repo_dir])
            else:
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
