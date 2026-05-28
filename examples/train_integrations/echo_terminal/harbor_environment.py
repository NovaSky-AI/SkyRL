import asyncio
import io
import json
import logging
import os
import shutil
import tarfile
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.docker.docker import DockerEnvironment
from harbor.environments.factory import EnvironmentFactory
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig
from harbor.models.task.task import Task
from harbor.models.trial.paths import TrialPaths
from harbor.verifier.verifier import Verifier

logger = logging.getLogger(__name__)


class EnvironmentStartTimeoutError(asyncio.TimeoutError):
    pass


class HarborEnvironment:
    def __init__(
        self,
        task_name: str,
        shared_task: Task,
        shared_env_name: str,
        shared_task_env_config: EnvironmentConfig,
        rollout_id: str,
        base_temp_dir: Path | None = None,
        keep_container: bool = False,
        environment_build_timeout_sec: float = 600.0,
        force_build: bool = False,
        use_prebuilt_image: bool = False,
    ) -> None:
        self.task_name = task_name
        self._shared_task = shared_task
        self._shared_env_name = shared_env_name
        self._shared_task_env_config = shared_task_env_config
        self._rollout_id = rollout_id
        self._base_temp_dir = base_temp_dir
        self._keep_container = keep_container
        self._environment_build_timeout_sec = environment_build_timeout_sec
        self._force_build = force_build
        self._use_prebuilt_image = use_prebuilt_image
        self._work_dir: Path | None = None
        self._trial_paths: TrialPaths | None = None
        self._environment: BaseEnvironment | None = None
        self._is_setup = False
        self._env_setup_sec: float | None = None

    async def setup(self) -> None:
        if self._is_setup:
            return
        dir_label = self.task_name.replace("/", "_").replace(" ", "_")[:40] or "tbench"
        dir_suffix = self._rollout_id or uuid.uuid4().hex[:8]
        if self._base_temp_dir:
            self._base_temp_dir.mkdir(parents=True, exist_ok=True)
            self._work_dir = self._base_temp_dir / f"{dir_label}_r{dir_suffix}"
        else:
            self._work_dir = Path(tempfile.gettempdir()) / f"tbench_{dir_label}_r{dir_suffix}"

        trial_dir = self._work_dir / "trial"
        self._trial_paths = TrialPaths(trial_dir)
        self._trial_paths.mkdir()

        env_name = self._shared_env_name
        session_id = f"{env_name[:32]}__{uuid.uuid4().hex[:7]}"

        if self._use_prebuilt_image:
            task_env_config = self._shared_task_env_config
            force_build = False
        elif self._force_build:
            task_env_config = self._shared_task_env_config
            force_build = True
        else:
            task_env_config = self._shared_task_env_config.model_copy()
            task_env_config.docker_image = f"hb__{env_name}"
            self._ensure_pull_policy_override(self._shared_task.paths.environment_dir)
            force_build = False

        self._environment = EnvironmentFactory.create_environment(
            type=EnvironmentType.DOCKER,
            environment_dir=self._shared_task.paths.environment_dir,
            environment_name=self._shared_env_name,
            session_id=session_id,
            trial_paths=self._trial_paths,
            task_env_config=task_env_config,
            keep_containers=self._keep_container,
            suppress_override_warnings=True,
        )
        t_env = time.monotonic()
        await self._start_environment_with_retry(force_build=force_build)
        self._env_setup_sec = time.monotonic() - t_env
        self._is_setup = True

    async def _start_environment_with_retry(self, force_build: bool = True) -> None:
        last_err: BaseException | None = None
        for attempt in range(1, 3):
            try:
                await asyncio.wait_for(
                    self._environment.start(force_build=force_build),
                    timeout=self._environment_build_timeout_sec,
                )
                if force_build:
                    await self._tag_built_image()
                return
            except asyncio.TimeoutError as exc:
                last_err = EnvironmentStartTimeoutError(
                    f"Environment start timed out after {self._environment_build_timeout_sec} seconds"
                )
            except RuntimeError as exc:
                last_err = exc
            if attempt < 2:
                await asyncio.sleep(1)
        raise last_err

    async def _tag_built_image(self) -> None:
        target_image = f"hb__{self._shared_env_name}"
        try:
            result = await self._environment._run_docker_compose_command(["images", "--format", "json"], check=False)
            images = json.loads(result.stdout or "[]")
            if not images:
                return
            source_image = f"{images[0]['Repository']}:{images[0]['Tag']}"
            proc = await asyncio.subprocess.create_subprocess_exec(
                "docker",
                "tag",
                source_image,
                target_image,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.warning("Failed to tag %s as %s: %s", source_image, target_image, stderr.decode())
        except Exception as exc:
            logger.warning("Could not query/tag container image: %s", exc)

    @staticmethod
    def _ensure_pull_policy_override(environment_dir: Path) -> None:
        compose_path = environment_dir / "docker-compose.yaml"
        if not compose_path.exists():
            compose_path.write_text("services:\n  main:\n    pull_policy: never\n")

    async def exec(self, command: str, timeout: float | None = None) -> ExecResult:
        if not self._is_setup:
            raise RuntimeError("Environment not set up. Call setup() first.")
        timeout_sec = int(timeout) if timeout is not None else None
        return await self._environment.exec(command, timeout_sec=timeout_sec)

    async def run_verifier(self, timeout: float | None = None) -> tuple[float, str | None]:
        if not self._is_setup:
            raise RuntimeError("Environment not set up. Call setup() first.")
        try:
            verifier = Verifier(task=self._shared_task, trial_paths=self._trial_paths, environment=self._environment)
            result = await asyncio.wait_for(verifier.verify(), timeout=timeout) if timeout is not None else await verifier.verify()
            reward = result.rewards.get("reward", 0.0) if result.rewards else 0.0
            return float(reward), None if result.rewards else "no_reward"
        except asyncio.TimeoutError:
            logger.warning("Verifier timed out after %ss", timeout)
            return 0.0, "verifier_timeout"
        except Exception as exc:
            logger.warning("Verifier failed: %s", exc)
            return 0.0, "verifier_error"

    async def cleanup(self) -> None:
        if self._environment is not None:
            try:
                if self._keep_container:
                    await self._environment.stop(delete=False)
                else:
                    await self._environment._run_docker_compose_command(["down", "--volumes", "--remove-orphans"])
            except Exception as exc:
                logger.warning("Environment stop failed: %s", exc)
            self._environment = None

        if self._work_dir and self._work_dir.exists() and not self._keep_container:
            shutil.rmtree(self._work_dir, ignore_errors=True)
            self._work_dir = None
        self._is_setup = False


class _SharedTaskImage:
    def __init__(
        self,
        task_binary: bytes,
        task_name: str = "",
        base_temp_dir: Path | None = None,
        memory_mb: int | None = None,
        cpus: int | None = None,
    ) -> None:
        self._task_binary = task_binary
        self._task_name = task_name
        self._base_temp_dir = base_temp_dir
        self._memory_mb = memory_mb
        self._cpus = cpus
        self._work_dir: Path | None = None
        self._task: Task | None = None
        self._env_name: str | None = None
        self._task_env_config: EnvironmentConfig | None = None
        self._is_setup = False

    @property
    def env_name(self) -> str | None:
        return self._env_name

    @property
    def has_prebuilt_image(self) -> bool:
        return self._is_setup and bool(self._task_env_config.docker_image)

    def setup(self) -> None:
        if self._is_setup:
            return
        dir_suffix = uuid.uuid4().hex[:8]
        dir_label = self._task_name.replace("/", "_").replace(" ", "_")[:40] or "tbench"
        if self._base_temp_dir:
            self._base_temp_dir.mkdir(parents=True, exist_ok=True)
            self._work_dir = self._base_temp_dir / f"shared_{dir_label}_{dir_suffix}"
        else:
            self._work_dir = Path(tempfile.gettempdir()) / f"tbench_shared_{dir_label}_{dir_suffix}"

        task_dir = self._work_dir / dir_label
        task_dir.mkdir(parents=True, exist_ok=True)
        _safe_extract_tar(self._task_binary, task_dir)
        self._task = Task(task_dir)
        self._task_env_config = self._task.config.environment
        if self._memory_mb is not None:
            self._task_env_config.memory_mb = self._memory_mb
        if self._cpus is not None:
            self._task_env_config.cpus = self._cpus
        self._env_name = self._task.name
        self._is_setup = True

    def create_environment(self, rollout_id: str, force_build: bool = False) -> HarborEnvironment:
        if not self._is_setup:
            raise RuntimeError("_SharedTaskImage not set up. Call setup() first.")
        return HarborEnvironment(
            task_name=self._task_name,
            shared_task=self._task,
            shared_env_name=self._env_name,
            shared_task_env_config=self._task_env_config,
            rollout_id=rollout_id,
            base_temp_dir=self._base_temp_dir,
            force_build=force_build,
            use_prebuilt_image=self.has_prebuilt_image,
        )

    async def build_image(self) -> None:
        if not self._is_setup:
            raise RuntimeError("_SharedTaskImage not set up. Call setup() first.")
        build_env = self.create_environment(rollout_id="build", force_build=True)
        try:
            await build_env.setup()
        finally:
            await build_env.cleanup()

    async def pull_image(self) -> None:
        if not self._is_setup:
            raise RuntimeError("_SharedTaskImage not set up. Call setup() first.")
        image = self._task_env_config.docker_image
        proc = await asyncio.subprocess.create_subprocess_exec(
            "docker",
            "pull",
            image,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"docker pull failed for {image!r}: {stderr.decode()}")

    async def cleanup(self) -> None:
        if self._env_name and not self.has_prebuilt_image:
            try:
                proc = await asyncio.subprocess.create_subprocess_exec(
                    "docker",
                    "rmi",
                    "-f",
                    f"hb__{self._env_name}",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.wait()
            except Exception:
                logger.debug("Failed to remove image hb__%s", self._env_name)
        if self._work_dir and self._work_dir.exists():
            shutil.rmtree(self._work_dir, ignore_errors=True)
            self._work_dir = None
        self._is_setup = False


class HarborEnvironmentProvider:
    def __init__(
        self,
        docker_memory_mb: int | None = None,
        docker_cpus: int | None = None,
        base_temp_dir: Path | None = None,
        max_concurrent_builds: int = 32,
        max_build_retries: int = 3,
    ) -> None:
        self._docker_memory_mb = docker_memory_mb
        self._docker_cpus = docker_cpus
        self._base_temp_dir = base_temp_dir
        self._max_concurrent_builds = max_concurrent_builds
        self._max_build_retries = max_build_retries
        self._shared_images: list[_SharedTaskImage] = []
        self._failed_prompts: set[int] = set()
        self._rollout_counter: dict[int, int] = {}

    async def prepare_batch(self, batch_environment_data: list[dict[str, Any]], num_generations: int) -> None:
        DockerEnvironment._image_build_locks.clear()
        self._shared_images = []
        self._failed_prompts = set()
        self._rollout_counter = {}

        for prompt_idx, env_data in enumerate(batch_environment_data):
            task_binary = env_data.get("task_binary")
            task_name = env_data.get("path", "unknown")
            if task_binary is None:
                raise ValueError(f"prompt {prompt_idx}: environment_data must contain 'task_binary'.")
            env_data["_prompt_idx"] = prompt_idx
            shared_image = _SharedTaskImage(
                task_binary=task_binary,
                task_name=task_name,
                base_temp_dir=self._base_temp_dir,
                memory_mb=self._docker_memory_mb,
                cpus=self._docker_cpus,
            )
            shared_image.setup()
            self._shared_images.append(shared_image)

        build_semaphore = asyncio.Semaphore(self._max_concurrent_builds)

        async def _with_retries(action, img: _SharedTaskImage) -> None:
            async with build_semaphore:
                last_err: BaseException | None = None
                for attempt in range(1, self._max_build_retries + 1):
                    try:
                        await action()
                        return
                    except Exception as exc:
                        last_err = exc
                        logger.warning(
                            "Image build/pull attempt %s/%s failed for %s: %s",
                            attempt,
                            self._max_build_retries,
                            img.env_name or "?",
                            exc,
                        )
                        if attempt < self._max_build_retries:
                            await asyncio.sleep(min(2**attempt, 10))
                raise last_err

        work = []
        for idx, img in enumerate(self._shared_images):
            action = img.pull_image if img.has_prebuilt_image else img.build_image
            work.append((idx, _with_retries(action, img)))
        results = await asyncio.gather(*[task for _, task in work], return_exceptions=True)
        for (prompt_idx, _), result in zip(work, results):
            if isinstance(result, BaseException):
                logger.error("Image build/pull failed for prompt %s: %s", prompt_idx, result)
                self._failed_prompts.add(prompt_idx)

    async def create(self, environment_data: dict[str, Any]) -> HarborEnvironment:
        prompt_idx = environment_data.get("_prompt_idx", 0)
        if prompt_idx in self._failed_prompts:
            raise RuntimeError(f"Docker image build failed for prompt {prompt_idx}.")
        if prompt_idx >= len(self._shared_images):
            raise RuntimeError(f"prompt_idx {prompt_idx} out of range.")
        rollout_num = self._rollout_counter.get(prompt_idx, 0)
        self._rollout_counter[prompt_idx] = rollout_num + 1
        return self._shared_images[prompt_idx].create_environment(rollout_id=f"{prompt_idx}_{rollout_num}")

    async def cleanup_batch(self) -> None:
        for shared_image in self._shared_images:
            try:
                await shared_image.cleanup()
            except Exception:
                logger.exception("[shared-image-cleanup] Failed")
        self._shared_images = []
        self._failed_prompts = set()
        self._rollout_counter = {}


def _sanitize_tar_member_name(name: str) -> str:
    p = PurePosixPath(name)
    parts = [part for part in p.parts if part not in ("..", ".", "")]
    while parts and parts[0] == "/":
        parts.pop(0)
    return str(PurePosixPath(*parts)) if parts else ""


def _is_within(base: Path, target: Path) -> bool:
    try:
        return os.path.commonpath([str(base.resolve()), str(target.resolve())]) == str(base.resolve())
    except Exception:
        return False


def _safe_extract_tar(archive_bytes: bytes, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO(archive_bytes)
    with tarfile.open(fileobj=buf, mode="r:*") as tf:
        for member in tf.getmembers():
            member_name = _sanitize_tar_member_name(member.name)
            if not member_name or member_name.endswith("/"):
                dir_path = member_name.rstrip("/")
                if dir_path:
                    (dest_dir / dir_path).mkdir(parents=True, exist_ok=True)
                continue
            if ".snapshot" in PurePosixPath(member_name).parts:
                continue
            target = (dest_dir / member_name).resolve()
            if not _is_within(dest_dir, target):
                raise RuntimeError(f"Unsafe path in archive: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if member.isfile():
                with tf.extractfile(member) as src:
                    if src is None:
                        continue
                    with open(target, "wb") as dst:
                        dst.write(src.read())
                if member.mode & 0o111:
                    target.chmod(target.stat().st_mode | 0o111)
            elif member.isdir():
                target.mkdir(parents=True, exist_ok=True)
