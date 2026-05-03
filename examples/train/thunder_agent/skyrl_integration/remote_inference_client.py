"""ThunderAgent-aware HTTP inference client wrappers."""

from __future__ import annotations

from typing import Any, Dict

from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
    PauseMode,
    RemoteInferenceClient,
    raise_for_status,
)


class ThunderAgentRemoteInferenceClient(RemoteInferenceClient):
    """RemoteInferenceClient with ThunderAgent control-plane hooks."""

    async def _call_proxy(self, endpoint: str, json: Dict[str, Any], method: str = "POST") -> Dict[str, Any]:
        session = await self._get_session()
        url = f"{self.proxy_url}{endpoint}"

        async with session.request(method, url, json=json) as resp:
            body = await resp.json() if resp.content_length else None
            raise_for_status(resp, body)
            return {"status": resp.status, "body": body}

    async def _end_weight_sync_safely(self) -> None:
        try:
            await self._call_proxy("/weight_sync/end", {})
        except Exception:
            # Best-effort cleanup if the backend-side resume path fails.
            pass

    async def pause(
        self,
        mode: PauseMode | str = PauseMode.KEEP,
        clear_cache: bool = False,
    ) -> Dict[str, Any]:
        await self._call_proxy("/weight_sync/begin", {})
        try:
            return await super().pause(mode=mode, clear_cache=clear_cache)
        except Exception:
            await self._end_weight_sync_safely()
            raise

    async def resume(self) -> Dict[str, Any]:
        try:
            return await super().resume()
        finally:
            await self._end_weight_sync_safely()

    async def release_program(self, program_id: str) -> Dict[str, Any]:
        return await self._call_proxy("/programs/release", {"program_id": program_id})
