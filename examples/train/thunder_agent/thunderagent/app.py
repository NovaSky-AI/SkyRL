"""ThunderAgent FastAPI application entry point."""

import logging
from typing import Any, Dict, Mapping, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .config import Config, get_config
from .scheduler import MultiBackendRouter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_program_id(payload: Dict[str, Any], headers: Optional[Mapping[str, str]] = None) -> str:
    """Extract program_id from the request payload.

    Checks payload["program_id"] first, then payload["extra_body"]["program_id"].
    Falls back to the X-Session-ID header so SkyRL session routing can act as
    the ThunderAgent program identifier when no explicit program_id is sent.
    """
    if "program_id" in payload:
        return str(payload["program_id"])
    extra_body = payload.get("extra_body", {})
    if isinstance(extra_body, dict) and "program_id" in extra_body:
        return str(extra_body["program_id"])
    if headers is not None:
        session_id = headers.get("X-Session-ID") or headers.get("x-session-id")
        if session_id:
            return str(session_id)
    return "default"


def register_routes(app: FastAPI, ta_router: MultiBackendRouter, config: Optional[Config] = None) -> None:
    """Register all ThunderAgent HTTP routes on the given FastAPI app.

    This function adds the standard ThunderAgent endpoints to any FastAPI app.
    Used by both the standalone ThunderAgent server and integration layers
    (e.g. SkyRL's ThunderAgentRouter).

    Args:
        app: FastAPI application to register routes on.
        ta_router: ThunderAgent MultiBackendRouter instance.
        config: Optional config override. If None, uses get_config().
    """

    def _get_config() -> Config:
        return config if config is not None else get_config()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """Handle chat completions request with program-aware scheduling."""
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON") from exc

        program_id = get_program_id(payload, request.headers)
        program_state = ta_router.get_or_create_program(program_id)

        if program_state.profile:
            program_state.profile.on_request_arrive()

        await ta_router.update_program_before_request(program_id, program_state, payload)

        if program_state.profile:
            program_state.profile.on_request_start()

        backend = ta_router.get_backend_for_program(program_id)

        async def on_usage(total_tokens: int, prompt_tokens: int, cached_tokens: int) -> None:
            ta_router.update_program_after_request(program_id, program_state, total_tokens, prompt_tokens)
            if program_state.profile:
                program_state.profile.on_request_end(prompt_tokens, cached_tokens)

        def on_token_progress(delta_tokens: int) -> None:
            ta_router.update_program_tokens_streaming(program_state, delta_tokens)

        return await ta_router.proxy_request(
            backend,
            payload,
            on_usage=on_usage,
            on_first_token=program_state.profile.on_first_token if program_state.profile else None,
            on_token=program_state.profile.on_token if program_state.profile else None,
            on_token_progress=on_token_progress,
        )

    @app.get("/programs")
    async def list_programs():
        """List all programs (includes profile data if profiling enabled)."""
        result = {}
        for pid, state in ta_router.programs.items():
            program_data = {
                "backend": state.backend_url,
                "context_len": state.context_len,
                "total_tokens": state.total_tokens,
                "step_count": state.step_count,
                "status": state.status.value,
                "state": state.state.value,
            }
            if state.profile:
                program_data["profile"] = state.profile.to_dict()
            result[pid] = program_data
        return JSONResponse(result)

    @app.post("/programs/release")
    async def release_program(request: Request):
        """Release a program."""
        try:
            payload = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON") from exc

        program_id = payload.get("program_id")
        if not program_id:
            raise HTTPException(status_code=400, detail="Missing program_id")
        program_id = str(program_id)

        released = await ta_router.release_program(program_id)
        logger.info(f"/programs/release program_id={program_id} released={released}")
        return JSONResponse({"program_id": program_id, "released": released})

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        cfg = _get_config()
        stats = ta_router.get_program_stats()
        return JSONResponse(
            {
                "status": "ok",
                "router_mode": cfg.router_mode,
                "scheduling_enabled": ta_router.scheduling_enabled,
                "backends": list(ta_router.backends.keys()),
                "programs_count": stats["total"],
                "reasoning_count": stats["reasoning"],
                "acting_count": stats["acting"],
                "paused_count": stats["paused"],
                "per_backend": stats["per_backend"],
                "profile_enabled": ta_router.profile_enabled,
            }
        )

    @app.get("/profiles")
    async def list_profiles():
        """List all program profiles (timing metrics)."""
        if not ta_router.profile_enabled:
            return JSONResponse({"error": "Profiling not enabled. Start with --profile flag."}, status_code=400)
        result = {}
        for pid, state in ta_router.programs.items():
            if state.profile:
                result[pid] = state.profile.to_dict()
        return JSONResponse(result)

    @app.get("/profiles/{program_id}")
    async def get_profile(program_id: str):
        """Get profile for a specific program."""
        if not ta_router.profile_enabled:
            return JSONResponse({"error": "Profiling not enabled. Start with --profile flag."}, status_code=400)
        state = ta_router.programs.get(program_id)
        if state is None or state.profile is None:
            raise HTTPException(status_code=404, detail=f"Profile not found for program: {program_id}")
        return JSONResponse(state.profile.to_dict())

    @app.get("/v1/models")
    async def list_models():
        """List available models (proxy to first backend)."""
        if not ta_router.backends:
            return JSONResponse({"object": "list", "data": []})
        backend_url = next(iter(ta_router.backends.keys()))
        return await ta_router.proxy_get(backend_url, "/v1/models")

    @app.get("/metrics")
    async def get_metrics():
        """Get vLLM metrics from all backends."""
        cfg = _get_config()
        paused_counts = ta_router.get_paused_counts_by_backend()
        return JSONResponse(
            {
                "metrics_enabled": cfg.metrics_enabled,
                "metrics_interval": cfg.metrics_interval if cfg.metrics_enabled else None,
                "backends": {
                    url: backend.to_dict(paused_program_count=paused_counts.get(url, 0))
                    for url, backend in ta_router.backends.items()
                },
            }
        )

    # -- Weight sync coordination --

    @app.post("/weight_sync/begin")
    async def weight_sync_begin():
        """Notify TA that weight sync is starting. Holds new requests."""
        await ta_router.begin_weight_sync()
        return JSONResponse({"status": "ok", "weight_sync_active": True})

    @app.post("/weight_sync/end")
    async def weight_sync_end():
        """Notify TA that weight sync has completed. Releases held requests."""
        await ta_router.end_weight_sync()
        return JSONResponse({"status": "ok", "weight_sync_active": False})


# =============================================================================
# Standalone Application
# =============================================================================


def _create_router() -> MultiBackendRouter:
    """Create router with current config."""
    config = get_config()
    return MultiBackendRouter(
        config.backends,
        profile_enabled=config.profile_enabled,
        scheduling_enabled=(config.router_mode == "tr"),
        scheduler_interval=config.scheduler_interval,
        backend_type=config.backend_type,
        acting_token_weight=config.acting_token_weight,
        use_acting_token_decay=config.use_acting_token_decay,
    )


router = _create_router()
app = FastAPI(title="ThunderAgent - Program State Tracking Proxy")


@app.on_event("startup")
async def startup_event():
    await router.start()


@app.on_event("shutdown")
async def shutdown_event():
    await router.stop()


register_routes(app, router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8300, log_level="info")
