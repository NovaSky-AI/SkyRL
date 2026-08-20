"""Lifecycle for a dedicated Fireworks trainer and rollout deployment."""

from __future__ import annotations

import asyncio
import os
import re
import threading
import time
import uuid
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from skyrl.train.config import FireworksConfig


def _checkpoint_short_name(row: dict) -> str:
    return str(row.get("name", "")).rstrip("/").rsplit("/", 1)[-1]


def _checkpoint_create_timestamp(row: dict) -> float | None:
    value = row.get("createTime")
    if not isinstance(value, str) or not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


@dataclass(frozen=True)
class SamplerVersion:
    """Snapshot identity visible to rollout calls admitted after publication."""

    version: int
    snapshot_path: str


@dataclass(frozen=True)
class ResumableCheckpoint:
    """DCP checkpoint identity used for same-trainer or cross-job resume."""

    requested_name: str
    checkpoint_name: str
    checkpoint_resource: str | None
    checkpoint_type: str | None


@dataclass(frozen=True)
class PromotableCheckpoint:
    """Inference checkpoint persisted for later model promotion."""

    snapshot_path: str
    checkpoint_resource: str | None
    checkpoint_type: str | None


@dataclass(frozen=True)
class FireworksInferenceEndpoint:
    """Native OpenAI-compatible endpoint for the managed deployment."""

    api_base: str
    model: str


def _close_quietly(resource: Any) -> None:
    close = getattr(resource, "close", None)
    if close is None:
        return
    try:
        close()
    except Exception as exc:  # pragma: no cover - provider-specific teardown
        warnings.warn(
            f"Failed to close Fireworks resource {type(resource).__name__}: {exc}",
            RuntimeWarning,
        )


class FireworksRuntime:
    """Own one dedicated trainer, deployment, and stable sampling client.

    Fireworks hot-loads new snapshots into the deployment without replacing
    its URL or the local sampling client. ``_active_samples`` exists only so
    teardown does not close that client underneath an in-flight request; it
    does not represent multiple model clients or deployed versions.
    """

    def __init__(
        self,
        *,
        service: Any,
        training_client: Any,
        tokenizer: Any,
        config: FireworksConfig,
    ) -> None:
        self.service = service
        self.training_client = training_client
        self.tokenizer = tokenizer
        self.config = config
        self._state_lock = threading.Condition()
        self._provider_operation_lock = threading.RLock()
        self._publish_lock = asyncio.Lock()
        self._sampler: Any | None = None
        self._sampler_identity: SamplerVersion | None = None
        self._active_samples = 0
        self._next_version = 0
        self._closed = False

    @classmethod
    def connect(
        cls,
        *,
        config: FireworksConfig,
        tokenizer: Any,
        tokenizer_model: str,
        lora_rank: int,
        learning_rate: float,
        create_deployment: bool = True,
    ) -> FireworksRuntime:
        """Provision an SDK-managed trainer and optional rollout deployment."""

        api_key = os.environ.get("FIREWORKS_API_KEY")
        if not api_key:
            raise RuntimeError("FIREWORKS_API_KEY is required for Fireworks training")
        if not config.base_model or not config.training_shape_id:
            raise ValueError("Dedicated Fireworks training requires base_model and training_shape_id")
        if not config.trainer_job_id:
            raise ValueError("Dedicated Fireworks training requires a stable trainer_job_id")
        if create_deployment and not config.deployment_id:
            raise ValueError("A stable deployment_id is required when create_deployment=True")

        try:
            from fireworks.training.sdk import FiretitanServiceClient
        except ImportError as exc:
            raise ImportError(
                "The Fireworks backend requires fireworks-ai[training]; " "install SkyRL with --extra fireworks"
            ) from exc

        cleanup_deployment = (
            config.cleanup_deployment_on_close if create_deployment and config.cleanup_on_exit else None
        )
        service = FiretitanServiceClient.from_firetitan_config(
            api_key=api_key,
            base_url=config.base_url,
            base_model=config.base_model,
            tokenizer_model=tokenizer_model,
            lora_rank=lora_rank,
            training_shape_id=config.training_shape_id,
            trainer_job_id=config.trainer_job_id,
            deployment_id=config.deployment_id if create_deployment else None,
            create_deployment=create_deployment,
            max_context_length=config.max_seq_len,
            learning_rate=learning_rate,
            trainer_replica_count=config.trainer_replica_count,
            replica_count=config.replica_count,
            trainer_timeout_s=config.trainer_timeout_s,
            deployment_timeout_s=config.deployment_timeout_s,
            hotload_timeout_s=config.hotload_timeout_s,
            cleanup_trainer_on_close=config.cleanup_on_exit,
            cleanup_deployment_on_close=cleanup_deployment,
        )
        try:
            training_client = service.create_training_client(
                base_model=config.base_model,
                lora_rank=lora_rank,
            )
        except BaseException:
            # Include KeyboardInterrupt so a wall-clock supervisor can still
            # invoke SDK cleanup during a long provisioning wait.
            _close_quietly(service)
            raise
        return cls(
            service=service,
            training_client=training_client,
            tokenizer=tokenizer,
            config=config,
        )

    @property
    def trainer_job_id(self) -> str | None:
        return getattr(self.service, "trainer_job_id", None)

    @property
    def deployment_id(self) -> str | None:
        return getattr(self.service, "deployment_id", None)

    @property
    def inference_endpoint(self) -> FireworksInferenceEndpoint:
        """Return the native endpoint backing the managed rollout deployment."""

        handle = getattr(self.service, "_managed_handle", None)
        deployment = getattr(handle, "deployment", None)
        deployment_manager = getattr(handle, "deployment_manager", None)
        model = getattr(deployment, "inference_model", None)
        inference_url = getattr(deployment_manager, "inference_url", None)
        if not model or not inference_url:
            raise RuntimeError(
                "Fireworks SDK-managed deployment did not expose its native "
                "inference model and URL after provisioning"
            )
        return FireworksInferenceEndpoint(
            api_base=f"{str(inference_url).rstrip('/')}/inference/v1",
            model=str(model),
        )

    @property
    def weight_version(self) -> int:
        with self._state_lock:
            if self._sampler_identity is None:
                return -1
            return self._sampler_identity.version

    @property
    def snapshot_path(self) -> str | None:
        with self._state_lock:
            if self._sampler_identity is None:
                return None
            return self._sampler_identity.snapshot_path

    def _snapshot_name(self, version: int) -> str:
        # Dedicated checkpoint names are lowercase DNS labels.
        prefix = re.sub(r"[^a-z0-9-]+", "-", self.config.snapshot_prefix.lower()).strip("-") or "skyrl"
        suffix = f"-v{version:08d}-{uuid.uuid4().hex[:8]}"
        # Fireworks appends another ``-<8 hex>`` suffix. Keep our input at 54
        # characters so the provider-side name remains at most 63 characters.
        prefix = prefix[: 54 - len(suffix)].rstrip("-") or "skyrl"
        return f"{prefix}{suffix}"

    async def publish_sampler_weights(self) -> SamplerVersion:
        """Save current weights and hot-load them into the stable deployment."""

        async with self._publish_lock:
            with self._state_lock:
                if self._closed:
                    raise RuntimeError("Fireworks runtime is closed")
                version = self._next_version

            name = self._snapshot_name(version)

            def _save() -> str:
                future = self.training_client.save_weights_for_sampler(name)
                result = future.result(timeout=self.config.request_timeout_s)
                path = getattr(result, "path", None)
                if not path:
                    raise RuntimeError(f"Fireworks save_weights_for_sampler({name!r}) returned no path")
                return str(path)

            snapshot_path = await asyncio.to_thread(_save)
            await asyncio.to_thread(self.service.hotload_sampler_snapshot, snapshot_path)

            # The client is created once, after the first snapshot is ready.
            with self._state_lock:
                needs_sampler = self._sampler is None
            if needs_sampler:
                sampler = await asyncio.to_thread(
                    self.service.create_sampling_client,
                    tokenizer=self.tokenizer,
                )
                with self._state_lock:
                    self._sampler = sampler

            identity = SamplerVersion(version=version, snapshot_path=snapshot_path)
            with self._state_lock:
                self._sampler_identity = identity
                self._next_version = version + 1
            return identity

    def resolve_resumable_checkpoint(
        self,
        requested_name: str,
        *,
        save_started_at: float,
        appear_timeout_s: float = 90.0,
        poll_s: float = 3.0,
    ) -> ResumableCheckpoint:
        with self._provider_operation_lock:
            with self._state_lock:
                if self._closed:
                    raise RuntimeError("Fireworks runtime is closed")

            deadline = time.monotonic() + appear_timeout_s
            matches: list[dict] = []
            while time.monotonic() < deadline:
                try:
                    rows = self.service.list_checkpoints(self.trainer_job_id)
                except Exception:
                    time.sleep(poll_s)
                    continue
                matches = [
                    row
                    for row in rows
                    if str(row.get("checkpointType", "")).endswith(("TRAINING", "TRAINING_LORA"))
                    and (_checkpoint_create_timestamp(row) or 0) >= save_started_at - 10
                ]
                if matches:
                    break
                time.sleep(poll_s)
            if matches:
                checkpoint = max(matches, key=lambda row: _checkpoint_create_timestamp(row) or 0)
                return ResumableCheckpoint(
                    requested_name=requested_name,
                    checkpoint_name=_checkpoint_short_name(checkpoint),
                    checkpoint_resource=checkpoint["name"],
                    checkpoint_type=checkpoint.get("checkpointType"),
                )
            warnings.warn(
                f"Resumable Fireworks checkpoint {requested_name!r} was saved but did not "
                "surface on the control plane before the visibility timeout",
                RuntimeWarning,
            )
            return ResumableCheckpoint(
                requested_name=requested_name,
                checkpoint_name=requested_name,
                checkpoint_resource=None,
                checkpoint_type=None,
            )

    def save_promotable_checkpoint(
        self,
        checkpoint_name: str,
        *,
        appear_timeout_s: float = 90.0,
        poll_s: float = 3.0,
    ) -> PromotableCheckpoint:
        with self._provider_operation_lock:
            with self._state_lock:
                if self._closed:
                    raise RuntimeError("Fireworks runtime is closed")

            result = self.training_client.save_weights_for_sampler(
                checkpoint_name,
                checkpoint_type="base",
            ).result(timeout=self.config.request_timeout_s)
            snapshot_path = str(getattr(result, "path", "") or "")
            if not snapshot_path:
                raise RuntimeError(f"Fireworks save_weights_for_sampler({checkpoint_name!r}) returned no path")

            snapshot_id = snapshot_path.rstrip("/").rsplit("/", 1)[-1]
            deadline = time.monotonic() + appear_timeout_s
            matches: list[dict] = []
            while time.monotonic() < deadline:
                try:
                    rows = self.service.list_checkpoints(self.trainer_job_id)
                except Exception:
                    time.sleep(poll_s)
                    continue
                matches = [row for row in rows if row.get("promotable") and _checkpoint_short_name(row) == snapshot_id]
                if matches:
                    break
                time.sleep(poll_s)
            if len(matches) > 1:
                raise RuntimeError(
                    f"Expected at most one promotable Fireworks checkpoint for {checkpoint_name!r}, "
                    f"got {len(matches)}"
                )
            if not matches:
                warnings.warn(
                    f"Promotable Fireworks checkpoint {checkpoint_name!r} was saved but did not "
                    "surface on the control plane before the visibility timeout",
                    RuntimeWarning,
                )
                return PromotableCheckpoint(
                    snapshot_path=snapshot_path,
                    checkpoint_resource=None,
                    checkpoint_type=None,
                )
            checkpoint = matches[0]
            return PromotableCheckpoint(
                snapshot_path=snapshot_path,
                checkpoint_resource=checkpoint["name"],
                checkpoint_type=checkpoint.get("checkpointType"),
            )

    async def promote_checkpoint_resource(
        self,
        *,
        checkpoint: PromotableCheckpoint,
        output_model_id: str,
    ) -> dict[str, Any]:
        if not checkpoint.checkpoint_resource:
            raise ValueError("Promotable checkpoint has no control-plane resource")

        def _promote() -> dict:
            with self._provider_operation_lock:
                with self._state_lock:
                    if self._closed:
                        raise RuntimeError("Fireworks runtime is closed")
                return self.service.promote_checkpoint(
                    name=checkpoint.checkpoint_resource,
                    output_model_id=output_model_id,
                    base_model=self.config.base_model,
                )

        model = await asyncio.to_thread(_promote)
        return {
            "sampler_path": checkpoint.snapshot_path,
            "checkpoint_resource": checkpoint.checkpoint_resource,
            "checkpoint_type": checkpoint.checkpoint_type,
            "output_model_id": output_model_id,
            "model": model,
        }

    async def promote_final_model(
        self,
        *,
        checkpoint_name: str,
        output_model_id: str,
        appear_timeout_s: float = 90.0,
        poll_s: float = 3.0,
    ) -> dict[str, Any]:
        async with self._publish_lock:
            with self._state_lock:
                if self._closed:
                    raise RuntimeError("Fireworks runtime is closed")

            def _save() -> str:
                future = self.training_client.save_weights_for_sampler(
                    checkpoint_name,
                    checkpoint_type="base",
                )
                result = future.result(timeout=self.config.request_timeout_s)
                path = str(getattr(result, "path", "") or "")
                if not path:
                    raise RuntimeError(f"Fireworks save_weights_for_sampler({checkpoint_name!r}) returned no path")
                return path

            snapshot_path = await asyncio.to_thread(_save)
            snapshot_id = snapshot_path.rstrip("/").rsplit("/", 1)[-1]
            deadline = time.monotonic() + appear_timeout_s
            matches: list[dict] = []
            while time.monotonic() < deadline:
                try:
                    rows = await asyncio.to_thread(self.service.list_checkpoints, self.trainer_job_id)
                except Exception:
                    await asyncio.sleep(poll_s)
                    continue
                matches = [row for row in rows if row.get("promotable") and _checkpoint_short_name(row) == snapshot_id]
                if matches:
                    break
                await asyncio.sleep(poll_s)
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one promotable Fireworks checkpoint for {checkpoint_name!r}, got {len(matches)}"
                )

            checkpoint = matches[0]
            model = await asyncio.to_thread(
                self.service.promote_checkpoint,
                name=checkpoint["name"],
                output_model_id=output_model_id,
                base_model=self.config.base_model,
            )
            return {
                "sampler_path": snapshot_path,
                "checkpoint_resource": checkpoint["name"],
                "checkpoint_type": checkpoint.get("checkpointType"),
                "output_model_id": output_model_id,
                "model": model,
            }

    async def delete_trainer(self, *, timeout_s: float = 60.0, poll_s: float = 5.0) -> str:
        from fireworks.training.sdk import TrainerJobManager

        api_key = os.environ.get("FIREWORKS_API_KEY")
        if not api_key or not self.trainer_job_id:
            raise RuntimeError("Fireworks trainer deletion requires API credentials and a trainer ID")
        manager = TrainerJobManager(api_key=api_key, base_url=self.config.base_url)
        try:
            await asyncio.to_thread(manager.delete, self.trainer_job_id)
            deadline = time.monotonic() + timeout_s
            state = "unknown"
            while time.monotonic() < deadline:
                row = await asyncio.to_thread(manager.try_get, self.trainer_job_id)
                state = "absent" if row is None else str(row.get("state", "?"))
                if state in ("absent", "JOB_STATE_DELETED"):
                    return state
                await asyncio.sleep(poll_s)
            raise RuntimeError(
                f"Fireworks trainer {self.trainer_job_id!r} did not reach a deleted state; state={state}"
            )
        finally:
            await asyncio.to_thread(manager.close)

    @contextmanager
    def _use_sampler(self) -> Iterator[tuple[Any, SamplerVersion]]:
        """Keep the stable sampling client open for one active request."""

        with self._state_lock:
            if self._closed:
                raise RuntimeError("Fireworks runtime is closed")
            if self._sampler is None or self._sampler_identity is None:
                raise RuntimeError("Fireworks sampler weights have not been published yet")
            sampler = self._sampler
            identity = self._sampler_identity
            self._active_samples += 1
        try:
            yield sampler, identity
        finally:
            with self._state_lock:
                self._active_samples -= 1
                self._state_lock.notify_all()

    async def sample_async(
        self,
        *,
        prompt: Any,
        sampling_params: Any,
    ) -> tuple[Any, SamplerVersion]:
        """Sample from the deployment without consuming an executor thread.

        The active-call scope covers the whole stream, including a provider-side
        pause/resume during an asynchronous hot-load.
        """

        with self._use_sampler() as (sampler, identity):
            native_sample = getattr(sampler, "sample_async", None)
            if native_sample is None:
                raise RuntimeError("The dedicated Fireworks sampler must expose sample_async()")
            result = await asyncio.wait_for(
                native_sample(
                    prompt=prompt,
                    num_samples=1,
                    sampling_params=sampling_params,
                ),
                timeout=self.config.sampling_timeout_s,
            )
            return result, identity

    async def close(self) -> None:
        """Drain active calls, then close the sampler and service. Idempotent."""

        # Publication and teardown must not mutate the deployment concurrently.
        async with self._publish_lock:
            with self._state_lock:
                if self._closed:
                    return
                self._closed = True

            def _wait_for_samples() -> tuple[Any | None, int]:
                deadline = time.monotonic() + self.config.sampling_timeout_s
                with self._state_lock:
                    while self._active_samples > 0:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        self._state_lock.wait(timeout=remaining)
                    sampler = self._sampler
                    active_samples = self._active_samples
                    self._sampler = None
                    self._sampler_identity = None
                    return sampler, active_samples

            sampler, active_calls = await asyncio.to_thread(_wait_for_samples)
            if active_calls:
                warnings.warn(
                    f"Closing Fireworks sampler with {active_calls} call(s) still "
                    f"active after sampling_timeout_s={self.config.sampling_timeout_s}",
                    RuntimeWarning,
                )
            if sampler is not None:
                await asyncio.to_thread(_close_quietly, sampler)

            def _close_service() -> None:
                with self._provider_operation_lock:
                    _close_quietly(self.service)

            await asyncio.to_thread(_close_service)
