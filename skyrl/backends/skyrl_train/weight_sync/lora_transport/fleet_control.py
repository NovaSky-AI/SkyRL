"""Fleet-wide transaction for a named native LoRA adapter replacement."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

ServerCall = Callable[[str], Awaitable[Any]]


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
        pause: Callable[[], Awaitable[Any]],
        activate: ServerCall,
        rollback: ServerCall,
        commit: ServerCall,
        resume: Callable[[], Awaitable[Any]],
        prepare_activation: Callable[[], Awaitable[Any]] | None = None,
    ) -> Mapping[str, Any]:
        """Run a fleet-atomic replacement and restore every server on failure."""
        staged = await self._run_phase(stage)
        if isinstance(staged, BaseException):
            await self._rollback_preserving_primary(rollback, staged)
        try:
            if prepare_activation is not None:
                await prepare_activation()
            await pause()
        except BaseException as error:
            await self._rollback_preserving_primary(rollback, error)
        resume_after_update = False
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
                raise RuntimeError("LoRA activated everywhere but failed to retire an old adapter") from committed
            return activated
        finally:
            if resume_after_update:
                await resume()

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
        results = await asyncio.gather(
            *(rollback(server_url) for server_url in self._server_urls),
            return_exceptions=True,
        )
        errors = [result for result in results if isinstance(result, BaseException)]
        if errors:
            raise RuntimeError("LoRA fleet rollback failed") from errors[0]
