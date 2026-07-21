"""Shared lifecycle for Fireworks hosted training and sampling clients."""

from __future__ import annotations

import asyncio
import os
import re
import threading
import time
import uuid
import warnings
from dataclasses import dataclass
from typing import Any

from skyrl.train.config import FireworksConfig


@dataclass(frozen=True)
class SamplerVersion:
    """Identity of the sampler currently visible to new rollout calls."""

    version: int
    snapshot_path: str


@dataclass(frozen=True)
class FireworksInferenceEndpoint:
    """Native OpenAI-compatible endpoint for an SDK-managed deployment."""

    api_base: str
    model: str


@dataclass
class _SamplerLease:
    sampler: Any
    identity: SamplerVersion
    active: int = 0
    retired: bool = False


def _close_quietly(resource: Any) -> None:
    close = getattr(resource, "close", None)
    if close is not None:
        try:
            close()
        except (
            Exception
        ) as exc:  # pragma: no cover - provider/client-specific teardown failure
            warnings.warn(
                f"Failed to close Fireworks resource {type(resource).__name__}: {exc}",
                RuntimeWarning,
            )


class FireworksRuntime:
    """Own one Fireworks service, training client, and current sampler.

    Dedicated deployments use one stable sampling client while Fireworks
    hot-loads successive snapshots into the deployment. Serverless snapshot
    routes require a new client, so that mode uses an RCU pointer: the old
    sampler closes only after its active calls finish.
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
        self._publish_lock = asyncio.Lock()
        self._current: _SamplerLease | None = None
        self._retired: list[_SamplerLease] = []
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
    ) -> "FireworksRuntime":
        if config.infrastructure == "serverless":
            return cls.connect_serverless(
                config=config, tokenizer=tokenizer, lora_rank=lora_rank
            )
        if config.infrastructure == "dedicated":
            return cls.connect_dedicated(
                config=config,
                tokenizer=tokenizer,
                tokenizer_model=tokenizer_model,
                lora_rank=lora_rank,
                learning_rate=learning_rate,
            )
        raise ValueError(
            f"Unsupported Fireworks infrastructure: {config.infrastructure!r}"
        )

    @classmethod
    def connect_serverless(
        cls, *, config: FireworksConfig, tokenizer: Any, lora_rank: int
    ) -> "FireworksRuntime":
        """Open a paid Fireworks serverless session.

        Callers must run this only after the resolved paid-run plan has been
        shown to and confirmed by the operator.
        """

        api_key = os.environ.get("FIREWORKS_API_KEY")
        if not api_key:
            raise RuntimeError("FIREWORKS_API_KEY is required for Fireworks training")
        if not config.base_model:
            raise ValueError("Fireworks serverless training requires a base_model")

        try:
            from fireworks.training.sdk import FiretitanServiceClient
        except ImportError as exc:
            raise ImportError(
                "The Fireworks backend requires fireworks-ai[training]; install SkyRL with --extra fireworks"
            ) from exc

        serverless_base_url = f"{config.base_url.rstrip('/')}/training/v1/serverless"
        service = FiretitanServiceClient(api_key=api_key, base_url=serverless_base_url)
        try:
            training_client = service.create_lora_training_client(
                base_model=config.base_model,
                rank=lora_rank,
            )
        except Exception:
            _close_quietly(service)
            raise
        return cls(
            service=service,
            training_client=training_client,
            tokenizer=tokenizer,
            config=config,
        )

    @classmethod
    def connect_dedicated(
        cls,
        *,
        config: FireworksConfig,
        tokenizer: Any,
        tokenizer_model: str,
        lora_rank: int,
        learning_rate: float,
    ) -> "FireworksRuntime":
        """Provision an SDK-managed trainer and linked rollout deployment."""

        api_key = os.environ.get("FIREWORKS_API_KEY")
        if not api_key:
            raise RuntimeError("FIREWORKS_API_KEY is required for Fireworks training")
        if not config.base_model or not config.training_shape_id:
            raise ValueError(
                "Dedicated Fireworks training requires base_model and training_shape_id"
            )
        if not config.trainer_job_id or not config.deployment_id:
            raise ValueError(
                "Dedicated Fireworks training requires stable trainer_job_id and deployment_id"
            )

        try:
            from fireworks.training.sdk import FiretitanServiceClient
        except ImportError as exc:
            raise ImportError(
                "The Fireworks backend requires fireworks-ai[training]; install SkyRL with --extra fireworks"
            ) from exc

        cleanup_deployment = (
            config.cleanup_deployment_on_close if config.cleanup_on_exit else None
        )
        service = FiretitanServiceClient.from_firetitan_config(
            api_key=api_key,
            base_url=config.base_url,
            base_model=config.base_model,
            tokenizer_model=tokenizer_model,
            lora_rank=lora_rank,
            training_shape_id=config.training_shape_id,
            trainer_job_id=config.trainer_job_id,
            deployment_id=config.deployment_id,
            create_deployment=True,
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
        """Return the native endpoint backing a dedicated hosted sampler.

        Agent integrations such as Harbor need to issue several token-in/token-out
        ``/completions`` calls for one trajectory.  They should talk directly to
        the same managed deployment that the SDK sampler uses, rather than route
        those calls through a local SkyRL HTTP proxy.
        """

        if self.config.infrastructure != "dedicated":
            raise RuntimeError(
                "A stable native inference endpoint is currently available only "
                "for dedicated Fireworks deployments"
            )
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
            return -1 if self._current is None else self._current.identity.version

    @property
    def snapshot_path(self) -> str | None:
        with self._state_lock:
            return (
                None if self._current is None else self._current.identity.snapshot_path
            )

    def _snapshot_name(self, version: int) -> str:
        # Managed dedicated checkpoints are DNS labels: lowercase ASCII,
        # digits, and hyphens only. Use the stricter form for both modes so a
        # configuration can move between serverless and dedicated unchanged.
        prefix = (
            re.sub(r"[^a-z0-9-]+", "-", self.config.snapshot_prefix.lower()).strip("-")
            or "skyrl"
        )
        suffix = f"-v{version:08d}-{uuid.uuid4().hex[:8]}"
        # Dedicated checkpoint creation appends another ``-<8 hex>`` suffix.
        # Keep our input at 54 characters so the final provider-side name is a
        # valid DNS label of at most 63 characters. This conservative limit is
        # also harmless for serverless snapshots.
        prefix = prefix[: 54 - len(suffix)].rstrip("-") or "skyrl"
        return f"{prefix}{suffix}"

    async def publish_sampler_weights(self) -> SamplerVersion:
        """Save current weights and make them visible to hosted sampling."""

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
                    raise RuntimeError(
                        f"Fireworks save_weights_for_sampler({name!r}) returned no path"
                    )
                return str(path)

            snapshot_path = await asyncio.to_thread(_save)

            if self.config.infrastructure == "dedicated":
                # The managed deployment has a stable URL/model. Hot-loading
                # changes server-side weights; it does not require replacing
                # the local HTTP client (and its concurrency controller).
                await asyncio.to_thread(
                    self.service.hotload_sampler_snapshot, snapshot_path
                )
                with self._state_lock:
                    needs_sampler = self._current is None
                sampler = None
                if needs_sampler:
                    sampler = await asyncio.to_thread(
                        self.service.create_sampling_client,
                        tokenizer=self.tokenizer,
                    )

                identity = SamplerVersion(version=version, snapshot_path=snapshot_path)
                close_now: Any | None = None
                with self._state_lock:
                    if self._closed:
                        close_now = sampler
                    elif self._current is None:
                        if sampler is None:  # pragma: no cover - serialized invariant
                            raise AssertionError("Dedicated sampler was not created")
                        self._current = _SamplerLease(
                            sampler=sampler, identity=identity
                        )
                        self._next_version = version + 1
                    else:
                        # Active calls retain the identity captured at
                        # admission, while the stable deployment/client can
                        # span the provider-side ASYNC weight transition.
                        self._current.identity = identity
                        self._next_version = version + 1
                if close_now is not None:
                    await asyncio.to_thread(_close_quietly, close_now)
                    raise RuntimeError(
                        "Fireworks runtime closed while publishing sampler weights"
                    )
                return identity

            # Serverless sampling routes are snapshot-qualified, so install a
            # new client and retire the previous client with RCU semantics.
            sampler = await asyncio.to_thread(
                self.service.create_sampling_client,
                model_path=snapshot_path,
                tokenizer=self.tokenizer,
            )
            identity = SamplerVersion(version=version, snapshot_path=snapshot_path)
            new_lease = _SamplerLease(sampler=sampler, identity=identity)
            close_now: Any | None = None

            with self._state_lock:
                if self._closed:
                    close_now = sampler
                else:
                    old = self._current
                    self._current = new_lease
                    self._next_version = version + 1
                    if old is not None:
                        old.retired = True
                        if old.active == 0:
                            close_now = old.sampler
                        else:
                            self._retired.append(old)

            if close_now is not None:
                await asyncio.to_thread(_close_quietly, close_now)
            if self._closed:
                raise RuntimeError(
                    "Fireworks runtime closed while publishing sampler weights"
                )
            return identity

    def acquire_sampler(self) -> _SamplerLease:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("Fireworks runtime is closed")
            if self._current is None:
                raise RuntimeError(
                    "Fireworks sampler weights have not been published yet"
                )
            self._current.active += 1
            return self._current

    def release_sampler(self, lease: _SamplerLease) -> None:
        close_now: Any | None = None
        with self._state_lock:
            if lease.active <= 0:
                raise RuntimeError("Fireworks sampler lease released more than once")
            lease.active -= 1
            if lease.retired and lease.active == 0 and not self._closed:
                if lease in self._retired:
                    self._retired.remove(lease)
                close_now = lease.sampler
            self._state_lock.notify_all()
        if close_now is not None:
            _close_quietly(close_now)

    def sample(
        self, *, prompt: Any, sampling_params: Any
    ) -> tuple[Any, SamplerVersion]:
        """Blocking sample call intended to run through ``asyncio.to_thread``."""

        lease = self.acquire_sampler()
        identity = lease.identity
        try:
            future = lease.sampler.sample(
                prompt=prompt, num_samples=1, sampling_params=sampling_params
            )
            result = future.result(timeout=self.config.sampling_timeout_s)
            return result, identity
        finally:
            self.release_sampler(lease)

    async def sample_async(
        self,
        *,
        prompt: Any,
        sampling_params: Any,
    ) -> tuple[Any, SamplerVersion]:
        """Sample without consuming one executor thread per active HTTP stream.

        Fireworks' hosted sampler exposes a native async streaming method.  A
        lease remains active for the whole stream, including any provider-side
        pause/resume during an asynchronous hot-load.  The synchronous fallback
        keeps unit-test doubles and older compatible clients working.
        """

        lease = self.acquire_sampler()
        identity = lease.identity
        try:
            sampler = lease.sampler
            native_sample = getattr(sampler, "sample_async", None)
            if native_sample is not None:
                result = await asyncio.wait_for(
                    native_sample(
                        prompt=prompt,
                        num_samples=1,
                        sampling_params=sampling_params,
                    ),
                    timeout=self.config.sampling_timeout_s,
                )
            else:

                def _sample() -> Any:
                    future = sampler.sample(
                        prompt=prompt,
                        num_samples=1,
                        sampling_params=sampling_params,
                    )
                    return future.result(timeout=self.config.sampling_timeout_s)

                result = await asyncio.to_thread(_sample)
            return result, identity
        finally:
            self.release_sampler(lease)

    async def close(self) -> None:
        """Drain active calls, then close every sampler and service. Idempotent."""

        # Serialize teardown with snapshot publication. This also makes two
        # concurrent close() calls wait for the same completed teardown.
        async with self._publish_lock:
            with self._state_lock:
                if self._closed:
                    return
                self._closed = True

            def _wait_for_leases() -> tuple[list[_SamplerLease], int]:
                deadline = time.monotonic() + self.config.sampling_timeout_s
                with self._state_lock:
                    while True:
                        leases = (
                            [self._current] if self._current is not None else []
                        ) + list(self._retired)
                        if not any(lease.active > 0 for lease in leases):
                            break
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        self._state_lock.wait(timeout=remaining)
                    self._current = None
                    self._retired = []
                    return leases, sum(lease.active for lease in leases)

            # Cancelling an asyncio.to_thread call does not stop the underlying
            # HTTP request. Give those threads their sampling timeout to release
            # their leases before closing clients underneath them.
            leases, active_calls = await asyncio.to_thread(_wait_for_leases)
            if active_calls:
                warnings.warn(
                    f"Closing Fireworks samplers with {active_calls} call(s) still active after "
                    f"sampling_timeout_s={self.config.sampling_timeout_s}",
                    RuntimeWarning,
                )
            seen: set[int] = set()
            for lease in leases:
                sampler_id = id(lease.sampler)
                if sampler_id not in seen:
                    seen.add(sampler_id)
                    await asyncio.to_thread(_close_quietly, lease.sampler)
            await asyncio.to_thread(_close_quietly, self.service)
