"""Atomic named-adapter replacement over vLLM's pause and collective RPC APIs."""

from typing import Any, Mapping, NamedTuple

from .contracts import LoRAUpdateRequest
from .control_protocol import (
    LORA_TRANSPORT_ACTIVATE_METHOD,
    LORA_TRANSPORT_DISCARD_METHOD,
    LORA_TRANSPORT_REMOVE_METHOD,
    LORA_TRANSPORT_RESTORE_METHOD,
)


class LoRATransportRollbackError(RuntimeError):
    """A generation could not be safely restored or discarded."""


class LoRATransportStagedGeneration(NamedTuple):
    request: LoRAUpdateRequest
    adapter_id: int
    previous_id: int | None
    prepared: bool = False


class LoRATransportServerLifecycle:
    """Own named active adapter ids while the server owns request admission."""

    def __init__(self) -> None:
        self._active_ids: dict[str, int] = {}
        self._staged: dict[str, LoRATransportStagedGeneration] = {}
        self._terminal: dict[str, tuple[LoRAUpdateRequest, int, str]] = {}
        self._unloaded: set[str] = set()

    def get_active_adapter_id(self, adapter_name: str) -> int | None:
        """Return the currently routable adapter id for one adapter name."""
        return self._active_ids.get(adapter_name)

    async def stage_transport(
        self,
        engine: Any,
        request: LoRAUpdateRequest,
        adapter_id: int,
        method: str,
        kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        """Stage through one trusted transport-specific worker method."""
        if not method:
            raise ValueError("LoRA staging requires a worker method")
        if adapter_id <= 0:
            raise ValueError(f"LoRA adapter ids must be positive, got {adapter_id}")
        if self._get_terminal_outcome(request, adapter_id) is not None:
            raise ValueError(
                f"LoRA generation {request.generation} has already completed"
            )
        self._validate_fixed_publication_contract(request)
        if request.adapter_name in self._unloaded:
            raise ValueError(f"LoRA adapter {request.adapter_name!r} has been unloaded")
        if request.adapter_name in self._staged:
            raise ValueError(
                f"LoRA adapter {request.adapter_name!r} already has a staged generation"
            )
        staged = LoRATransportStagedGeneration(
            request,
            adapter_id,
            self._active_ids.get(request.adapter_name),
        )
        self._staged[request.adapter_name] = staged
        stage_kwargs = {
            "request": request.to_json_dict(),
            "adapter_id": adapter_id,
            **dict(kwargs or {}),
        }
        try:
            await engine.collective_rpc(method, kwargs=stage_kwargs)
        except BaseException:
            try:
                await engine.collective_rpc(
                    LORA_TRANSPORT_DISCARD_METHOD,
                    kwargs={"adapter_id": adapter_id},
                )
            except BaseException as cleanup_error:
                raise LoRATransportRollbackError(
                    f"LoRA transport could not discard partially staged adapter id {adapter_id}"
                ) from cleanup_error
            del self._staged[request.adapter_name]
            raise
        self._staged[request.adapter_name] = staged._replace(prepared=True)

    async def activate(
        self, engine: Any, request: LoRAUpdateRequest, adapter_id: int
    ) -> None:
        """Activate a staged local generation while fleet admission is closed."""
        if not self._get_staged(request, adapter_id).prepared:
            raise ValueError(
                f"LoRA generation {request.generation} did not finish staging"
            )
        if self._active_ids.get(request.adapter_name) == adapter_id:
            return
        await engine.collective_rpc(
            LORA_TRANSPORT_ACTIVATE_METHOD,
            kwargs={"request": request.to_json_dict(), "adapter_id": adapter_id},
        )
        self._active_ids[request.adapter_name] = adapter_id

    async def commit(
        self, engine: Any, request: LoRAUpdateRequest, adapter_id: int
    ) -> bool:
        """Retire the prior buffer once and report whether the route needs finalizing."""
        outcome = self._get_terminal_outcome(request, adapter_id)
        if outcome == "committed":
            return False
        if outcome is not None:
            raise ValueError(
                f"LoRA generation {request.generation} was already rolled back"
            )
        previous_id = self._get_staged(request, adapter_id).previous_id
        if self._active_ids.get(request.adapter_name) != adapter_id:
            raise ValueError(
                f"LoRA generation {request.generation} has not been activated"
            )
        if previous_id is not None:
            await engine.collective_rpc(
                LORA_TRANSPORT_REMOVE_METHOD, kwargs={"adapter_id": previous_id}
            )
        del self._staged[request.adapter_name]
        self._terminal[request.adapter_name] = (request, adapter_id, "committed")
        return True

    async def abort(self, engine: Any, request: LoRAUpdateRequest) -> bool:
        """Cancel a generation even when its stage response never reached the caller."""
        terminal = self._terminal.get(request.adapter_name)
        if terminal is not None:
            outcome = self._get_terminal_outcome(request, terminal[1])
            if outcome == "rolled_back":
                return False
            if outcome is not None:
                raise ValueError(
                    f"LoRA generation {request.generation} was already committed"
                )
        staged = self._staged.get(request.adapter_name)
        if staged is not None:
            return await self.rollback(engine, request, staged[1])
        # No adapter id was allocated; prevent a delayed stage from reviving this generation.
        self._terminal[request.adapter_name] = (request, 0, "rolled_back")
        return False

    async def rollback(
        self, engine: Any, request: LoRAUpdateRequest, adapter_id: int
    ) -> bool:
        """Restore the prior route once without replaying destructive cleanup."""
        outcome = self._get_terminal_outcome(request, adapter_id)
        if outcome == "rolled_back":
            return False
        if outcome is not None:
            raise ValueError(
                f"LoRA generation {request.generation} was already committed"
            )
        previous_id = self._get_staged(request, adapter_id).previous_id
        if previous_id is not None:
            await engine.collective_rpc(
                LORA_TRANSPORT_RESTORE_METHOD, kwargs={"adapter_id": previous_id}
            )
            self._active_ids[request.adapter_name] = previous_id
        elif self._active_ids.get(request.adapter_name) == adapter_id:
            self._active_ids.pop(request.adapter_name, None)
        await engine.collective_rpc(
            LORA_TRANSPORT_DISCARD_METHOD, kwargs={"adapter_id": adapter_id}
        )
        self._staged.pop(request.adapter_name, None)
        self._terminal[request.adapter_name] = (request, adapter_id, "rolled_back")
        return True

    async def unload(self, engine: Any, adapter_name: str) -> None:
        """Release the active buffer while admission is closed, retaining a name tombstone."""
        if adapter_name in self._staged:
            raise ValueError(
                f"LoRA adapter {adapter_name!r} has an unfinished replacement"
            )
        self._unloaded.add(adapter_name)
        adapter_id = self._active_ids.get(adapter_name)
        if adapter_id is not None:
            await engine.collective_rpc(
                LORA_TRANSPORT_REMOVE_METHOD, kwargs={"adapter_id": adapter_id}
            )
        self._active_ids.pop(adapter_name, None)
        self._terminal.pop(adapter_name, None)

    def _get_terminal_outcome(
        self, request: LoRAUpdateRequest, adapter_id: int
    ) -> str | None:
        terminal = self._terminal.get(request.adapter_name)
        if terminal is None:
            return None
        previous_request, previous_id, outcome = terminal
        if request.generation < previous_request.generation:
            raise ValueError(
                f"LoRA generation {request.generation} is older than the last completed transaction"
            )
        if request.generation == previous_request.generation:
            if request != previous_request or adapter_id != previous_id:
                raise ValueError(
                    f"LoRA generation {request.generation} does not match its completed transaction"
                )
            return outcome
        return None

    def _validate_fixed_publication_contract(self, request: LoRAUpdateRequest) -> None:
        """Reject incompatible updates before dispatching a worker collective."""
        terminal = self._terminal.get(request.adapter_name)
        if terminal is None:
            return
        previous_request = terminal[0]
        if request.layout_digest != previous_request.layout_digest:
            raise ValueError("LoRA update changed the fixed source layout")
        if request.source_dtype != previous_request.source_dtype:
            raise ValueError("LoRA update changed the fixed source dtype")

    def _get_staged(
        self, request: LoRAUpdateRequest, adapter_id: int
    ) -> LoRATransportStagedGeneration:
        staged = self._staged.get(request.adapter_name)
        if staged is None or staged[:2] != (request, adapter_id):
            raise ValueError(
                f"LoRA adapter {request.adapter_name!r} generation {request.generation} is not staged"
            )
        return staged
