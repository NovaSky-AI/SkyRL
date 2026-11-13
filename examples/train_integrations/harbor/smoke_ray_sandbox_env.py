"""Smoke test for RaySandboxEnvironment without training.

Starts one sandbox the way a Harbor trial would, then runs a command, a timeout,
two commands at once, and a file round trip.

    uv run --isolated --extra fsdp -m examples.train_integrations.harbor.smoke_ray_sandbox_env
"""

import asyncio
import tempfile
import time
from pathlib import Path

import ray
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from examples.train_integrations.harbor.ray_sandbox_env import RaySandboxEnvironment


async def main():
    tmp = Path(tempfile.mkdtemp())
    env_dir = tmp / "environment"
    env_dir.mkdir()
    env = RaySandboxEnvironment(
        environment_dir=env_dir,
        environment_name="smoke",
        session_id="smoke__0",
        trial_paths=TrialPaths(trial_dir=tmp / "trial"),
        task_env_config=EnvironmentConfig(docker_image="python:3.12-slim", cpus=1, memory_mb=512),
    )
    await env.start(force_build=False)
    try:
        r = await env.exec("python3 -c 'print(6 * 7)'")
        assert r.return_code == 0 and r.stdout.strip() == "42", r

        r = await env.exec("sleep 10", timeout_sec=1)
        assert r.return_code == 124, r

        # The actor runs calls concurrently, so two 2 s commands take ~2 s.
        start = time.monotonic()
        results = await asyncio.gather(env.exec("sleep 2"), env.exec("sleep 2"))
        assert all(r.return_code == 0 for r in results), results
        assert time.monotonic() - start < 3.5, time.monotonic() - start

        src = tmp / "src"
        (src / "sub").mkdir(parents=True)
        (src / "sub" / "a.txt").write_text("hello")
        await env.upload_dir(src, "/workspace/in")
        await env.exec("cp -r /workspace/in /workspace/out")
        await env.download_dir("/workspace/out", tmp / "out")
        assert (tmp / "out" / "sub" / "a.txt").read_text() == "hello"
        print("OK")
    finally:
        await env.stop(delete=True)


if __name__ == "__main__":
    ray.init()
    asyncio.run(main())
