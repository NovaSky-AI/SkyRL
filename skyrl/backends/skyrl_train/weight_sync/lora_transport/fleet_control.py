"""Fleet-wide transaction for a named native LoRA adapter replacement."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

ServerCall = Callable[[str], Awaitable[Any]]


class LoRATransportRetirementError(RuntimeError):
    """The new generation is active, but its predecessor is not retired everywhere."""


class LoRATransportFleetTransaction:
    """Stage every deployment before one paused activation transaction.

    A deployment is a vLLM server plus all of its tensor-parallel workers. The
    server-specific stage/activate operations are collective RPCs, while this
    class provides the missing data-parallel fleet barrier.
    """

    def __init__(self, server_urls: list[str]) -> None:
        if not server_urls:
            raise ValueError("lora_transport requires at least one inference server")
        self._server_urls = tuple(server_urls)

    async def replace(
        self,
        stage: ServerCall,
        pause: ServerCall,
        activate: ServerCall,
        rollback: ServerCall,
        commit: ServerCall,
        resume: ServerCall,
        prepare_activation: Callable[[], Awaitable[Any]] | None = None,
    ) -> Mapping[str, Any]:
        """Run a fleet-atomic replacement and restore every server on failure."""
        staged = await self._run_phase(stage)
        if isinstance(staged, BaseException):
            await self._rollback_preserving_primary(rollback, staged)
        if prepare_activation is not None:
            try:
                await prepare_activation()
            except BaseException as error:
                await self._rollback_preserving_primary(rollback, error)
        paused = await self._run_phase(pause)
        if isinstance(paused, BaseException):
            await self._recover_failed_pause(rollback, resume, paused)
        resume_after_update = False
        primary_error = None
        try:
            activated = await self._run_phase(activate)
            if isinstance(activated, BaseException):
                try:
                    await self._rollback_all(rollback)
                except BaseException as rollback_error:
                    activated.add_note(f"LoRA fleet rollback also failed: {rollback_error}")
                    raise activated from rollback_error
                resume_after_update = True
                raise activated
            resume_after_update = True
            committed = await self._run_phase(commit)
            if isinstance(committed, BaseException):
                # Old adapters are retained through activation. A commit failure
                # cannot expose a mixed generation, but it must stop a later
                # replacement until the retained old buffer is reconciled.
                raise LoRATransportRetirementError(
                    "LoRA activated everywhere but failed to retire an old adapter"
                ) from committed
            return activated
        except BaseException as error:
            primary_error = error
            raise
        finally:
            if resume_after_update:
                resumed = await self._run_phase(resume)
                if isinstance(resumed, BaseException):
                    if primary_error is not None:
                        primary_error.add_note(f"LoRA fleet resume also failed: {resumed}")
                    else:
                        raise resumed

    async def _run_phase(self, operation: ServerCall) -> Mapping[str, Any] | BaseException:
        results = await asyncio.gather(
            *(operation(server_url) for server_url in self._server_urls),
            return_exceptions=True,
        )
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            return errors[0]
        return dict(zip(self._server_urls, results, strict=True))

    async def _rollback_preserving_primary(
        self,
        rollback: ServerCall,
        primary: BaseException,
    ) -> None:
        """Raise the operation failure even when cleanup also fails."""
        try:
            await self._rollback_all(rollback)
        except BaseException as rollback_error:
            primary.add_note(f"LoRA fleet rollback also failed: {rollback_error}")
            raise primary from rollback_error
        raise primary

    async def _rollback_all(self, rollback: ServerCall) -> None:
        result = await self._run_phase(rollback)
        if isinstance(result, BaseException):
            raise RuntimeError("LoRA fleet rollback failed") from result

    async def _recover_failed_pause(
        self,
        rollback: ServerCall,
        resume: ServerCall,
        primary: BaseException,
    ) -> None:
        """Discard staging and resume servers after a partial fleet pause."""
        cleanup_error = None
        try:
            await self._rollback_all(rollback)
        except BaseException as error:
            cleanup_error = error
            primary.add_note(f"LoRA fleet rollback also failed: {error}")
        resumed = await self._run_phase(resume)
        if isinstance(resumed, BaseException):
            cleanup_error = resumed
            primary.add_note(f"LoRA fleet resume also failed: {resumed}")
        if cleanup_error is not None:
            raise primary from cleanup_error
        raise primary
