"""Harbor environment backed by Ray sandboxes (``ray.experimental.sandbox``).

Each Harbor trial gets one ``Sandbox`` actor, which runs the task image in a
gVisor container on whichever Ray node has room. Select it from the Harbor
trial config::

    environment:
      import_path: examples.train_integrations.harbor.ray_sandbox_env:RaySandboxEnvironment

Requirements:
  * Ray >= 2.59.0 on every node (SkyRL's pin).
  * Nodes that host sandboxes need gVisor's ``runsc`` on PATH. See
    Dockerfile.ray-sandbox. Ray master additionally needs ``mkfs.erofs`` >= 1.7
    and ``slirp4netns``.
  * Tasks must name a prebuilt, publicly pullable image (``[environment].docker_image``
    or the ``image`` kwarg below). Ray sandboxes cannot build Dockerfiles.
"""

import io
import shlex
import tarfile
import uuid
from pathlib import Path
from typing import Any

import ray
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import (
    EnvironmentCapabilities,
    EnvironmentResourceCapabilities,
)
from harbor.environments.definition import effective_exec_cwd, parse_dockerfile_workdir
from harbor.models.task.config import NetworkMode
from ray.experimental import sandbox
from ray.experimental.sandbox import DOCKER_DEFAULT_CAPABILITIES, SandboxTimeoutError

# Same exit code GNU ``timeout`` uses, which is what Harbor agents expect.
_TIMEOUT_EXIT_CODE = 124


class RaySandboxEnvironment(BaseEnvironment):
    def __init__(
        self,
        *args,
        image: str | None = None,
        resources: dict[str, float] | None = None,
        ttl_seconds: int | None = 3600,
        shell: str | None = None,
        network: str | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            image: Overrides the task's ``docker_image`` (e.g. one prebuilt
                image shared by every CodeContests task).
            resources: Extra Ray resources for the sandbox actor, e.g.
                ``{"sandbox": 0.001}`` to pin sandboxes to nodes labelled with
                a custom ``sandbox`` resource (and keep them off GPU nodes).
            ttl_seconds: Wall-clock safety net in case ``stop`` never runs.
            shell: Shell for string commands (Ray's default is /bin/bash).
            network: Ray network mode for every trial, e.g. "none" to cut
                network even for tasks that allow it. Defaults to "none" for
                no-network tasks and "public" otherwise.
        """
        # Drop provider kwargs from other backends' configs (e.g. Daytona's
        # auto_stop_interval_mins) so the same YAML can be reused.
        kwargs.pop("auto_stop_interval_mins", None)
        kwargs.pop("sandbox_timeout_secs", None)
        # Set before super().__init__, which calls _validate_definition().
        self._image_override = image
        super().__init__(*args, **kwargs)
        self._resources = resources
        self._ttl_seconds = ttl_seconds
        self._shell = shell
        self._network = network
        self._dockerfile_workdir = parse_dockerfile_workdir(
            self.environment_dir / "Dockerfile"
        )
        self._sb = None

    @staticmethod
    def type() -> str:
        return "ray-sandbox"

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        # cpu/memory become both a Ray reservation and a cgroup limit.
        return EnvironmentResourceCapabilities(
            cpu_limit=True, cpu_request=True, memory_limit=True, memory_request=True
        )

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        # No egress allowlist: Harbor rejects allowlist tasks up front.
        return EnvironmentCapabilities(disable_internet=True)

    @property
    def _image(self) -> str | None:
        return self._image_override or self.task_env_config.docker_image

    def _validate_definition(self):
        if not self._image:
            raise FileNotFoundError(
                f"{self.environment_dir}: Ray sandboxes need a prebuilt image. Set "
                "[environment].docker_image in the task or pass `image` in "
                "environment.kwargs."
            )

    def _workdir(self) -> str | None:
        return self.task_env_config.workdir or self._dockerfile_workdir

    async def start(self, force_build: bool) -> None:
        cpus = self._effective_cpus
        memory_mb = self._effective_memory_mb
        self._sb = sandbox.create(
            image=self._image,
            cpu=float(cpus) if cpus else None,
            memory=f"{memory_mb}Mi" if memory_mb else None,
            env=self._persistent_env or None,
            workdir=self._workdir(),
            ttl_seconds=self._ttl_seconds,
            # Docker caps so apt/pip/chown behave as they do under Docker.
            capabilities=DOCKER_DEFAULT_CAPABILITIES,
            # Agents write all over the filesystem; writes go to a per-sandbox
            # copy-on-write overlay that is discarded on delete.
            readonly=False,
            # In Ray 2.59.0, "public" is runsc host networking: the sandbox shares
            # the node's network namespace, including 127.0.0.1 services such as
            # the raylet and dashboard. Only use it for trusted tasks or on a
            # dedicated sandbox cluster; pass network="none" otherwise.
            network=self._network or ("none" if self._network_disabled else "public"),
            resources=self._resources,
            # First pull of an image on a node includes download + EROFS build.
            timeout_seconds=self.task_env_config.build_timeout_sec,
        )
        # The actor's __init__ creates the container; any creation error
        # surfaces here.
        await self._sb.get_instance_id.remote()
        await self.ensure_dirs(self._mount_targets(writable_only=True))
        await self._upload_environment_dir_after_start()

    async def stop(self, delete: bool):
        if self._sb is None:
            return
        try:
            await self._sb.delete.remote()
        except Exception as e:
            self.logger.warning(f"Error deleting Ray sandbox: {e}")
        finally:
            # delete() only removes the container; free the actor's resources too.
            ray.kill(self._sb)
            self._sb = None

    def _require_sandbox(self):
        if self._sb is None:
            raise RuntimeError("Sandbox not found. Please start the environment first.")
        return self._sb

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        sb = self._require_sandbox()
        user = self._resolve_user(user)
        if user is not None and str(user) not in ("root", "0"):
            # Sandbox.exec has no `user` argument yet; drop privileges in-shell.
            command = f"su -s /bin/bash {shlex.quote(str(user))} -c {shlex.quote(command)}"
        try:
            result = await sb.exec.remote(
                command,
                timeout=timeout_sec,
                cwd=effective_exec_cwd(cwd, self.task_env_config.workdir, self._dockerfile_workdir),
                env=self._merge_env(env),
                shell=self._shell,
            )
        except SandboxTimeoutError as e:
            return ExecResult(stdout=None, stderr=str(e), return_code=_TIMEOUT_EXIT_CODE)
        return ExecResult(
            stdout=result.stdout, stderr=result.stderr, return_code=result.exit_code
        )

    # File transfer. Sandbox.upload_file/download_file use the *actor's* node
    # filesystem, so bytes are shipped through Ray instead.

    async def upload_file(self, source_path: Path | str, target_path: str):
        await self._require_sandbox().write_file.remote(
            target_path, Path(source_path).read_bytes()
        )

    async def upload_dir(self, source_dir: Path | str, target_dir: str):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(str(source_dir), arcname=".")
        remote_tar = f"/tmp/.hb-upload-{uuid.uuid4().hex}.tar.gz"
        await self._require_sandbox().write_file.remote(remote_tar, buf.getvalue())
        q = shlex.quote
        result = await self.exec(
            f"mkdir -p {q(target_dir)} && tar -xzf {remote_tar} -C {q(target_dir)}"
            f" && rm -f {remote_tar}",
            user="root",
        )
        if result.return_code != 0:
            raise RuntimeError(f"upload_dir to {target_dir} failed: {result.stderr}")

    async def download_file(self, source_path: str, target_path: Path | str):
        data = await self._require_sandbox().read_file.remote(source_path)
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        Path(target_path).write_bytes(data)

    async def download_dir(self, source_dir: str, target_dir: Path | str):
        remote_tar = f"/tmp/.hb-download-{uuid.uuid4().hex}.tar.gz"
        result = await self.exec(
            f"tar -czf {remote_tar} -C {shlex.quote(source_dir)} .", user="root"
        )
        if result.return_code != 0:
            raise RuntimeError(f"download_dir of {source_dir} failed: {result.stderr}")
        sb = self._require_sandbox()
        data = await sb.read_file.remote(remote_tar)
        await sb.exec.remote(f"rm -f {remote_tar}")
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            tar.extractall(target_dir, filter="data")
