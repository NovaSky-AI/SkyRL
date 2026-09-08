# /// script
# requires-python = ">=3.12"
# dependencies = ["tomlkit==0.13.3", "packaging==25.0"]
# ///
"""Generate and apply the complete, frozen B300 image dependency profile."""

import argparse
import functools
import hashlib
import http.server
import json
import shutil
import subprocess
import tempfile
import threading
import urllib.request
from pathlib import Path

import tomlkit
from packaging.markers import Marker
from packaging.requirements import Requirement

from build_wheels import canonical_name, json_bytes, requirement_name, sha256

HERE = Path(__file__).resolve().parent
PROFILE = "b300-fa4"


def project_manifest(source, release):
    manifest = tomlkit.parse(source)
    build = release["build"]
    project = manifest["project"]
    project["requires-python"] = build["requires_python"]
    extras = project["optional-dependencies"]
    for requirements in extras.values():
        for item in requirements:
            requirement = Requirement(item)
            if requirement_name(item) == "torch" and "2.11.0" not in requirement.specifier:
                raise ValueError("The B300 profile requires a Torch 2.11 source manifest")
    found = 0
    for requirements in extras.values():
        for index, requirement in enumerate(requirements):
            if requirement_name(requirement) == "flash-attn":
                requirements[index] = f"flash-attn=={build['combined_version']}; sys_platform == 'linux'"
                found += 1
    if found != 3:
        raise ValueError(f"Expected FA2 pins in three extras, found {found}")
    if any(requirement_name(item) == "flash-attn-4" for items in extras.values() for item in items):
        raise ValueError("The source manifest must not enable FA4 globally")
    extras["megatron"].append(f"flash-attn-4[cu13]=={build['companion_version']}; sys_platform == 'linux'")
    uv = manifest["tool"]["uv"]
    uv["environments"] = ["sys_platform == 'linux' and platform_machine == 'x86_64'"]
    uv["required-environments"] = ["sys_platform == 'linux' and platform_machine == 'x86_64'"]
    target = {
        "sys_platform": "linux",
        "platform_system": "Linux",
        "platform_machine": "x86_64",
        "python_version": "3.12",
        "python_full_version": "3.12.0",
        "os_name": "posix",
        "implementation_name": "cpython",
        "platform_python_implementation": "CPython",
    }
    # uv fetches metadata for inactive source alternatives before resolving markers.
    for name, sources in list(uv["sources"].items()):
        if isinstance(sources, list):
            selected = [entry for entry in sources if "marker" not in entry or Marker(entry["marker"]).evaluate(target)]
            if selected:
                uv["sources"][name] = selected
            else:
                del uv["sources"][name]
    overrides = uv["override-dependencies"]
    for name, requirement in build["runtime_pins"].items():
        indices = [index for index, item in enumerate(overrides) if requirement_name(item) == name]
        value = requirement + "; sys_platform == 'linux'"
        if indices:
            for index in indices:
                overrides[index] = value
        else:
            overrides.append(value)
    for artifact in release["artifacts"]:
        name = canonical_name(artifact["filename"].split("-", 1)[0])
        uv["sources"][name] = tomlkit.inline_table()
        uv["sources"][name]["url"] = artifact["url"]
    return tomlkit.dumps(manifest)


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


def generate(root, artifacts):
    release = json.loads((artifacts / "release-manifest.json").read_text())
    if release["build"] != json.loads((HERE / "inputs.json").read_text()):
        raise ValueError("Artifacts were built from a different input manifest")
    for artifact in release["artifacts"]:
        if sha256(artifacts / artifact["filename"]) != artifact["sha256"]:
            raise ValueError(f"Release artifact hash mismatch: {artifact['filename']}")
    source = (root / "pyproject.toml").read_text()
    projected = project_manifest(source, release)
    handler = functools.partial(QuietHandler, directory=str(artifacts.resolve()))
    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory(prefix="skyrl-fa4-lock-") as directory:
                stage = Path(directory)
                for name in ("skyrl", "skyrl-gym"):
                    shutil.copytree(root / name, stage / name, ignore=shutil.ignore_patterns("__pycache__", ".venv"))
                shutil.copy(root / "README.md", stage / "README.md")
                local = projected
                substitutions = {}
                for artifact in release["artifacts"]:
                    url = f"http://127.0.0.1:{server.server_port}/{artifact['filename'].replace('+', '%2B')}"
                    substitutions[url] = artifact["url"]
                    local = local.replace(artifact["url"], url)
                (stage / "pyproject.toml").write_text(local)
                shutil.copy(root / "uv.lock", stage / "uv.lock")
                subprocess.run(["uv", "lock", "--python", "3.12", "--directory", str(stage)], check=True)
                lock = (stage / "uv.lock").read_text()
                for local_url, release_url in substitutions.items():
                    lock = lock.replace(local_url, release_url)
                    lock = lock.replace(local_url.replace("%2B", "+"), release_url)
                if "127.0.0.1" in lock:
                    raise ValueError("The generated lock still contains a staging URL")
        finally:
            server.shutdown()
            thread.join()
    (HERE / "pyproject.toml").write_text(projected)
    (HERE / "uv.lock").write_text(lock)
    (HERE / "release-manifest.json").write_bytes(json_bytes(release))
    (HERE / "profile.json").write_bytes(
        json_bytes(
            {
                "profile": PROFILE,
                "source_manifest_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "generated_manifest_sha256": sha256(HERE / "pyproject.toml"),
                "lock_sha256": sha256(HERE / "uv.lock"),
                "release_manifest_sha256": sha256(HERE / "release-manifest.json"),
            }
        )
    )


def check(root):
    profile = json.loads((HERE / "profile.json").read_text())
    for name, field in (
        ("pyproject.toml", "generated_manifest_sha256"),
        ("uv.lock", "lock_sha256"),
        ("release-manifest.json", "release_manifest_sha256"),
    ):
        if sha256(HERE / name) != profile[field]:
            raise ValueError(f"The B300 profile {name} changed; regenerate the profile")
    actual = sha256(root / "pyproject.toml")
    if actual not in (profile["source_manifest_sha256"], profile["generated_manifest_sha256"]):
        raise ValueError("SkyRL's source manifest changed; regenerate the B300 profile")
    if actual == profile["generated_manifest_sha256"] and sha256(root / "uv.lock") != profile["lock_sha256"]:
        raise ValueError("The applied B300 lock changed")
    release = json.loads((HERE / "release-manifest.json").read_text())
    if release["build"] != json.loads((HERE / "inputs.json").read_text()):
        raise ValueError("The wheel inputs changed; rebuild the pair and regenerate the profile")
    return profile


def apply(root):
    check(root)
    shutil.copyfile(HERE / "pyproject.toml", root / "pyproject.toml")
    shutil.copyfile(HERE / "uv.lock", root / "uv.lock")


def check_release():
    release = json.loads((HERE / "release-manifest.json").read_text())
    for artifact in release["artifacts"]:
        with urllib.request.urlopen(artifact["url"], timeout=60) as response:
            if hashlib.file_digest(response, "sha256").hexdigest() != artifact["sha256"]:
                raise ValueError(f"Published artifact hash mismatch: {artifact['filename']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "check", "apply", "check-release"))
    parser.add_argument("--root", type=Path, default=HERE.parents[2])
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    if args.command == "generate":
        if args.artifacts is None:
            parser.error("generate requires --artifacts")
        generate(args.root, args.artifacts)
    elif args.command == "check":
        print(json.dumps(check(args.root), indent=2))
    elif args.command == "apply":
        apply(args.root)
    else:
        check_release()


if __name__ == "__main__":
    main()
