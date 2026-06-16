#!/usr/bin/env python3
#
# Helper that installs the Verifiers environment and adds it to the uv project.
#
# Example:
#   uv run integrations/verifiers/install_environment.py will/wordle@0.1.4
#

import argparse
import sys
import subprocess


def verifiers_env_to_uv_args(spec: str):
    org, sep, rest = spec.partition("/")
    if not sep:
        rest = spec
    name, _, version = rest.partition("@")
    index_url = f"https://hub.primeintellect.ai/{org}/simple/"
    with_req = name if not version else f"{name}=={version}"
    return [with_req, "--index", index_url]


def main() -> None:
    parser = argparse.ArgumentParser(description="Install a Verifiers environment into the current uv project.")
    parser.add_argument(
        "env_id",
        help="Environment ID in the form 'org/name@version' (e.g., will/wordle@0.1.4)",
    )
    args = parser.parse_args()

    env_id = args.env_id
    uv_args = verifiers_env_to_uv_args(env_id)

    result = subprocess.run(["uv", "add", "--active", *uv_args])
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
