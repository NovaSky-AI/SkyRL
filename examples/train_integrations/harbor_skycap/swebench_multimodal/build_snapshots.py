"""One Daytona snapshot per task image, so trials start from Daytona's own registry, not Docker Hub.

A trial started from `docker_image` makes Daytona pull it from Docker Hub on whichever runner the sandbox
lands. At 64 sandboxes at once that spreads over runners without the image cached, and Docker Hub
rate-limits the anonymous pulls ("failed to resolve source metadata ... failed to auth"): most trials of a
pass@8 run failed to build. A snapshot is pulled once, here, a few at a time with retries, and the run
starts every sandbox of a task from it (`snapshot_template_name`). A snapshot fixes the sandbox's
resources, so they are part of its name.

    python build_snapshots.py --tasks-dir ~/data/swebench_multimodal/tasks --memory-gb 16
    # then run_swebench_multimodal.sh with the same SANDBOX_* sizes uses them
"""

import argparse
import asyncio
import json
import tomllib
import urllib.request
from pathlib import Path

#: Daytona refuses ":latest" for a snapshot, so each image is pinned by its digest. A manifest HEAD doesn't
#: count against Docker Hub's pull limit.
MANIFEST_TYPES = ", ".join(
    [
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ]
)


def snapshot_name(prefix: str, task_name: str) -> str:
    """The snapshot a task's sandboxes start from; ``task_name`` is Harbor's short name, the instance id."""
    return f"{prefix}{task_name}"


def pinned(image: str) -> str:
    """``docker.io/<repo>:<tag>`` as ``docker.io/<repo>@<digest>``."""
    repo, _, tag = image.removeprefix("docker.io/").rpartition(":")
    scope = f"repository:{repo}:pull"
    url = f"https://auth.docker.io/token?service=registry.docker.io&scope={scope}"
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 - Docker Hub
        token = json.load(response)["token"]
    request = urllib.request.Request(
        f"https://registry-1.docker.io/v2/{repo}/manifests/{tag}",
        method="HEAD",
        headers={"Authorization": f"Bearer {token}", "Accept": MANIFEST_TYPES},
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - Docker Hub
        return f"docker.io/{repo}@{response.headers['Docker-Content-Digest']}"


def snapshot_prefix(cpus: int, memory_gb: int, disk_gb: int) -> str:
    return f"swebm-c{cpus}m{memory_gb}d{disk_gb}-"


async def build(daytona, name: str, image: str, resources, gate: asyncio.Semaphore, attempts: int) -> str:
    from daytona import CreateSnapshotParams

    try:
        existing = await daytona.snapshot.get(name)
        if "error" not in str(existing.state).lower():
            return f"{name}: exists ({existing.state})"
        await daytona.snapshot.delete(existing)
    except Exception:  # noqa: BLE001 - not found
        pass
    for attempt in range(1, attempts + 1):
        async with gate:
            try:
                await daytona.snapshot.create(
                    CreateSnapshotParams(name=name, image=image, resources=resources), timeout=0
                )
                return f"{name}: built"
            except Exception as error:  # noqa: BLE001 - Docker Hub rate limits are transient
                message = str(error)[:200]
                if "not allowed" in message:
                    return f"{name}: FAILED: {message}"
        try:
            await daytona.snapshot.delete(name)
        except Exception:  # noqa: BLE001
            pass
        if attempt < attempts:
            await asyncio.sleep(60 * attempt)
    return f"{name}: FAILED after {attempts} attempts: {message}"


async def main() -> None:
    from daytona import AsyncDaytona, Resources

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--cpus", type=int, default=1)
    parser.add_argument("--memory-gb", type=int, default=16)
    parser.add_argument("--disk-gb", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()

    prefix = snapshot_prefix(args.cpus, args.memory_gb, args.disk_gb)
    resources = Resources(cpu=args.cpus, memory=args.memory_gb, disk=args.disk_gb)
    tasks = []
    for task in sorted(args.tasks_dir.expanduser().iterdir()):
        config = task / "task.toml"
        if config.is_file():
            parsed = tomllib.loads(config.read_text())
            tasks.append((parsed["task"]["name"].split("/", 1)[1], parsed["environment"]["docker_image"]))
    gate = asyncio.Semaphore(args.concurrency)
    async with AsyncDaytona() as daytona:
        jobs = [
            build(daytona, snapshot_name(prefix, name), pinned(image), resources, gate, args.attempts)
            for name, image in tasks
        ]
        for done in asyncio.as_completed(jobs):
            print(await done, flush=True)
    print(f"snapshot_template_name={prefix}{{name}}")


if __name__ == "__main__":
    asyncio.run(main())
