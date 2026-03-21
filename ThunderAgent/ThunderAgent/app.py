"""ThunderAgent FastAPI application entry point."""
import logging
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .config import get_config
from .scheduler import MultiBackendRouter
from .program import ProgramStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# FastAPI Application
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


def get_program_id(payload: Dict[str, Any]) -> str:
    """Extract program_id from the request."""
    if "program_id" in payload:
        return str(payload["program_id"])
    extra_body = payload.get("extra_body", {})
    if isinstance(extra_body, dict) and "program_id" in extra_body:
        return str(extra_body["program_id"])
    return "default"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Handle chat completions request."""
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON") from exc
    # Get or create program state (new programs go to waiting queue)
    program_id = get_program_id(payload)
    program_state = router.get_or_create_program(program_id)

    # Profile: record request arrival BEFORE pause check (for accurate tool_call_time)
    if program_state.profile:
        program_state.profile.on_request_arrive()

    # Update state: check capacity, may pause and wait (max 20 min, then force resume)
    await router.update_program_before_request(program_id, program_state, payload)
    
    # Profile: record request start AFTER pause (captures pause_time)
    if program_state.profile:
        program_state.profile.on_request_start()

    # Resolve backend after any pause/resume to honor migrations.
    backend = router.get_backend_for_program(program_id)

    # Callback to update state after response
    async def on_usage(
        total_tokens: int,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        cached_tokens: int | None,
    ) -> None:
        router.update_program_after_request(
            program_id,
            program_state,
            total_tokens,
            prompt_tokens or 0,
        )
        # Profile: record request end with KV cache info
        if program_state.profile:
            program_state.profile.on_request_end(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
            )

    # Callback for streaming token progress updates (every 20 tokens)
    def on_token_progress(delta_tokens: int) -> None:
        router.update_program_tokens_streaming(program_state, delta_tokens)

    # Forward to vLLM (sticky unless rescheduled from the global paused pool)
    # Pass profile callbacks for token timing
    return await router.proxy_request(
        backend, payload,
        on_usage=on_usage,
        on_first_token=program_state.profile.on_first_token if program_state.profile else None,
        on_token=program_state.profile.on_token if program_state.profile else None,
        on_token_progress=on_token_progress,
    )


@app.get("/programs")
async def list_programs():
    """List all programs (includes profile data if profiling enabled)."""
    result = {}
    for pid, state in router.programs.items():
        program_data = {
            "backend": state.backend_url,
            "context_len": state.context_len,
            "total_tokens": state.total_tokens,
            "step_count": state.step_count,
            "status": state.status.value,
            "state": state.state.value,
        }
        # Include profile data if available
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

    released = await router.release_program(program_id)
    return JSONResponse({"program_id": program_id, "released": released})


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    config = get_config()
    stats = router.get_program_stats()
    return JSONResponse({
        "status": "ok",
        "router_mode": config.router_mode,
        "scheduling_enabled": router.scheduling_enabled,
        "backends": list(router.backends.keys()),
        "programs_count": stats["total"],
        "reasoning_count": stats["reasoning"],
        "acting_count": stats["acting"],
        "paused_count": stats["paused"],
        "per_backend": stats["per_backend"],
        "profile_enabled": router.profile_enabled,
    })


@app.get("/profiles")
async def list_profiles():
    """List all program profiles (timing metrics)."""
    if not router.profile_enabled:
        return JSONResponse({"error": "Profiling not enabled. Start with --profile flag."}, status_code=400)
    result = {}
    for pid, state in router.programs.items():
        if state.profile:
            result[pid] = state.profile.to_dict()
    return JSONResponse(result)


@app.get("/profiles/{program_id}")
async def get_profile(program_id: str):
    """Get profile for a specific program."""
    if not router.profile_enabled:
        return JSONResponse({"error": "Profiling not enabled. Start with --profile flag."}, status_code=400)
    state = router.programs.get(program_id)
    if state is None or state.profile is None:
        raise HTTPException(status_code=404, detail=f"Profile not found for program: {program_id}")
    return JSONResponse(state.profile.to_dict())


@app.get("/v1/models")
async def list_models():
    """List available models (proxy to first backend)."""
    # Forward to the first available backend
    if not router.backends:
        return JSONResponse({"object": "list", "data": []})
    
    backend_url = next(iter(router.backends.keys()))
    return await router.proxy_get(backend_url, "/models")


@app.get("/metrics")
async def get_metrics():
    """Get vLLM metrics from all backends."""
    config = get_config()
    paused_counts = router.get_paused_counts_by_backend()
    return JSONResponse({
        "metrics_enabled": config.metrics_enabled,
        "metrics_interval": config.metrics_interval if config.metrics_enabled else None,
        "backends": {
            url: backend.to_dict(paused_program_count=paused_counts.get(url, 0))
            for url, backend in router.backends.items()
        },
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8300, log_level="info")
