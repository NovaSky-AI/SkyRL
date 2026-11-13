"""Harbor environment backed by Ray sandboxes (``ray.experimental.sandbox``).

Each Harbor trial gets one ``HarborSandbox`` actor (ray_sandbox_actor.py), which
runs the task image in a gVisor container on whichever Ray node has room. Select it from the Harbor
trial config::

    environment:
      import_path: examples.train_integrations.harbor.ray_sandbox_env:RaySandboxEnvironment

Requirements:
  * Ray nightly at the commit pinned in pyproject.toml, on every node.
  * Nodes that host sandboxes need ``runsc``, ``mkfs.erofs`` >= 1.7, and
    ``slirp4netns`` (network="public" only). Dockerfile.ray-sandbox has all three.
    For network="public" they also need /dev/net/tun; the actor creates it with
    passwordless sudo if it's missing.
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
from ray.experimental.sandbox import DOCKER_DEFAULT_CAPABILITIES, SandboxTimeoutError

from examples.train_integrations.harbor.ray_sandbox_actor import (
    SANDBOX_RUNTIME_ENV,
    HarborSandbox,
)

# Same exit code GNU ``timeout`` uses, which is what Harbor agents expect.
_TIMEOUT_EXIT_CODE = 124


class RaySandboxEnvironment(BaseEnvironment):
    def __init__(
        self,
        *args,
        image: str | None = None,
        resources: dict[str, float] | None = None,
        ttl_seconds: int | None = 86400,
        shell: str | None = None,
        network: str | None = None,
        max_concurrency: int = 8,
        enforce_limits: bool | None = None,
        cidr_allowlist: list[str] | None = None,
        **kwargs: Any,
    ):
        """
        Args:
            image: Overrides the task's ``docker_image`` (e.g. one prebuilt
                image shared by every CodeContests task).
            resources: Extra Ray resources for the sandbox actor, e.g.
                ``{"sandbox": 0.001}`` to pin sandboxes to nodes labelled with
                a custom ``sandbox`` resource (and keep them off GPU nodes).
            ttl_seconds: Wall-clock safety net in case ``stop`` never runs. It
                must outlast the agent and verifier timeouts, which reach 5400 s
                for SWE-bench tasks. The sandbox also stops when its actor dies.
            shell: Shell for string commands (Ray's default is /bin/bash).
            network: Ray network mode for every trial, e.g. "none" to cut
                network even for tasks that allow it. Defaults to "none" for
                no-network tasks and "public" otherwise.
            max_concurrency: Calls the sandbox actor runs at once, so a
                long command doesn't hold up the agent's other commands and
                file transfers.
            enforce_limits: Whether the task's cpu and memory also become the
                sandbox's cgroup limits. None (the default) enforces them on
                nodes that delegate a cgroup v2 subtree to Ray (set
                ``RAY_SANDBOX_CGROUP_PATH``) and only reserves them elsewhere;
                True fails a trial on a node without one.
            cidr_allowlist: Egress allowlist for "public" sandboxes. None
                (the default) is Ray's: the public internet only. To also
                reach a host inside the cluster, such as a model server an
                agent in the sandbox calls, list it with the internet, e.g.
                ``["internet", "10.0.12.34/32"]``.
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
        self._max_concurrency = max_concurrency
        self._enforce_limits = enforce_limits
        self._cidr_allowlist = cidr_allowlist
        self._dockerfile_workdir = parse_dockerfile_workdir(self.environment_dir / "Dockerfile")
        self._sb = None

    @staticmethod
    def type() -> str:
        return "ray-sandbox"

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        # cpu/memory become a Ray reservation, and a cgroup limit on nodes that
        # delegate one to Ray.
        return EnvironmentResourceCapabilities(cpu_limit=True, cpu_request=True, memory_limit=True, memory_request=True)

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
        cpus = float(self._effective_cpus or 1)
        memory_mb = self._effective_memory_mb
        storage_mb = self._effective_storage_mb
        actor_options = {
            "num_cpus": cpus,
            "runtime_env": SANDBOX_RUNTIME_ENV,
            "max_concurrency": self._max_concurrency,
        }
        if memory_mb:
            actor_options["memory"] = memory_mb * 1024 * 1024
        if self._resources:
            actor_options["resources"] = self._resources
        self._sb = HarborSandbox.options(**actor_options).remote(
            image=self._image,
            enforce_limits=self._enforce_limits,
            # The same cpu/memory also become the sandbox's cgroup limits.
            cpu=cpus,
            memory=f"{memory_mb}Mi" if memory_mb else 0,
            # The task's storage caps the writable rootfs (writes past it get
            # ENOSPC). Unlike cpu and memory, this needs nothing from the node.
            disk=f"{storage_mb}Mi" if storage_mb else 0,
            env=self._persistent_env or None,
            workdir=self._workdir(),
            ttl_seconds=self._ttl_seconds,
            # Docker caps so apt/pip/chown behave as they do under Docker.
            capabilities=DOCKER_DEFAULT_CAPABILITIES,
            # Agents write all over the filesystem; writes go to a per-sandbox
            # copy-on-write overlay that is discarded on delete.
            readonly=False,
            # "public" has its own network namespace with egress to the public
            # internet only: Ray leaves out internal addresses (the Ray GCS and
            # other nodes, as a best effort) and always blocks instance metadata
            # and the Kubernetes API. cidr_allowlist changes what it can reach.
            network=self._network or ("none" if self._network_disabled else "public"),
            cidr_allowlist=self._cidr_allowlist,
            # First pull of an image on a node includes download + EROFS build.
            timeout_seconds=self.task_env_config.build_timeout_sec,
        )
        # The actor's __init__ creates the container; any creation error
        # surfaces here.
        await self._sb.ready.remote()
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
        try:
            result = await sb.exec.remote(
                command,
                timeout=timeout_sec,
                cwd=effective_exec_cwd(cwd, self.task_env_config.workdir, self._dockerfile_workdir),
                env=self._merge_env(env),
                shell=self._shell,
                user=str(user) if user is not None else None,
            )
        except SandboxTimeoutError as e:
            return ExecResult(stdout=None, stderr=str(e), return_code=_TIMEOUT_EXIT_CODE)
        return ExecResult(stdout=result.stdout, stderr=result.stderr, return_code=result.exit_code)

    # File transfer: bytes are shipped through Ray, since the actor runs on
    # another node.

    async def upload_file(self, source_path: Path | str, target_path: str):
        await self._require_sandbox().write_file.remote(target_path, Path(source_path).read_bytes())

    async def upload_dir(self, source_dir: Path | str, target_dir: str):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(str(source_dir), arcname=".")
        remote_tar = f"/tmp/.hb-upload-{uuid.uuid4().hex}.tar.gz"
        await self._require_sandbox().write_file.remote(remote_tar, buf.getvalue())
        q = shlex.quote
        result = await self.exec(
            f"mkdir -p {q(target_dir)} && tar -xzf {remote_tar} -C {q(target_dir)} && rm -f {remote_tar}",
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
        result = await self.exec(f"tar -czf {remote_tar} -C {shlex.quote(source_dir)} .", user="root")
        if result.return_code != 0:
            raise RuntimeError(f"download_dir of {source_dir} failed: {result.stderr}")
        sb = self._require_sandbox()
        data = await sb.read_file.remote(remote_tar)
        await sb.exec.remote(f"rm -f {remote_tar}")
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            tar.extractall(target_dir, filter="data")
