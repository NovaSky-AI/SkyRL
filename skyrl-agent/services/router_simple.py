import os
import json
import asyncio
import itertools
import hashlib
import time
import random
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Request, Response
from starlette.responses import StreamingResponse, JSONResponse
import httpx


"""
Simple Web Summary router (no keys). It only handles requests whose JSON
body has model == SUMMARY_MODEL, and forwards them to one of fixed
summary servers using either Rendezvous Hash (HRW) sticky routing (per
trajectory) or round-robin (when no sticky key), with the ability to
hard-disable failed upstreams (auto or manual) and skip them for all
subsequent requests.

  - Configure via SUMMARY_UPSTREAMS (e.g. http://localhost:8000/v1)

Standby (used when primaries are unavailable):
  - Configure via SUMMARY_STANDBY_UPSTREAMS

All other requests are rejected, so your LLM judge and unrelated traffic
stay untouched unless explicitly pointed here.

Usage
- Configure via env (optional):
  - SUMMARY_MODEL: defaults to "Qwen/Qwen3-32B"
  - SUMMARY_UPSTREAMS: comma-separated upstream bases (e.g. "http://host1:8000/v1,http://host2:8000/v1").
  - SUMMARY_STANDBY_UPSTREAMS: comma-separated standby bases.
  - ROUTER_ENFORCE_MODEL: "1" to enforce model match (default), "0" to forward all.
- Run: uvicorn router_simple:app --host 0.0.0.0 --port 8080
- Point only the Web Summary client to: http://localhost:8080/v1
  (set WEB_SUMMARY_API_BASE to this). Judge stays on your current base.
"""

app = FastAPI()
client = httpx.AsyncClient(timeout=None)

# Only handle Web Summary requests; identified by model in JSON body
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "Qwen/Qwen3-32B")
ENFORCE_MODEL = os.getenv("ROUTER_ENFORCE_MODEL", "1").strip().lower() in ("1", "true", "yes", "on")
DISABLE_STICKY = os.getenv("SUMMARY_DISABLE_STICKY", "1").strip().lower() in ("1", "true", "yes", "on")


# Fixed summary servers (provided by user). Stored without trailing /v1
def _normalize_base(u: str) -> str:
    b = (u or "").strip().rstrip("/")
    if b.endswith("/v1"):
        b = b[:-3]
    return b


def _parse_upstreams_env(env_name: str, default_bases: List[str]) -> List[str]:
    raw = os.getenv(env_name, "")
    if not raw.strip():
        return default_bases
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    bases = [_normalize_base(p) for p in parts]
    return bases or default_bases


_DEFAULT_SUMMARY_UPSTREAMS_BASE: List[str] = []

_SUMMARY_UPSTREAMS_BASE: List[str] = _parse_upstreams_env(
    "SUMMARY_UPSTREAMS",
    _DEFAULT_SUMMARY_UPSTREAMS_BASE,
)

if not _SUMMARY_UPSTREAMS_BASE:
    raise RuntimeError(
        "No SUMMARY_UPSTREAMS configured. Set the environment variable to a comma-separated list of upstream base URLs."
    )

# Standby upstreams (not part of sticky hashing; used only when needed)
_DEFAULT_STANDBY_UPSTREAMS_BASE: List[str] = []

_STANDBY_UPSTREAMS_BASE: List[str] = _parse_upstreams_env(
    "SUMMARY_STANDBY_UPSTREAMS",
    _DEFAULT_STANDBY_UPSTREAMS_BASE,
)

# Weighted bias settings: send a portion of traffic to a preferred upstream
_BIAS_TARGET_BASE = _normalize_base(os.getenv("SUMMARY_BIAS_TARGET", ""))
_BIAS_TARGET_BASE2 = _normalize_base(os.getenv("SUMMARY_BIAS_TARGET2", ""))
try:
    _BIAS_RATIO = float(os.getenv("SUMMARY_BIAS_RATIO", "0.0"))
except Exception:
    _BIAS_RATIO = 0.0
try:
    _BIAS_RATIO2 = float(os.getenv("SUMMARY_BIAS_RATIO2", "0.0"))
except Exception:
    _BIAS_RATIO2 = 0.0

_rr_iter = itertools.cycle(_SUMMARY_UPSTREAMS_BASE)
_rr_lock = asyncio.Lock()

# Ensure bias targets are part of upstreams even if env forgot them
if _BIAS_TARGET_BASE and _BIAS_TARGET_BASE not in _SUMMARY_UPSTREAMS_BASE:
    _SUMMARY_UPSTREAMS_BASE.append(_BIAS_TARGET_BASE)
if _BIAS_TARGET_BASE2 and _BIAS_TARGET_BASE2 not in _SUMMARY_UPSTREAMS_BASE:
    _SUMMARY_UPSTREAMS_BASE.append(_BIAS_TARGET_BASE2)

# Optional smooth weighted round-robin for non-sticky requests
USE_SWRR = os.getenv("SUMMARY_USE_SWRR", "1").strip().lower() in ("1", "true", "yes", "on")
_swrr_weights: Dict[str, int] = {}
_swrr_current: Dict[str, int] = {}
_swrr_known_set: set[str] = set()


def _rebuild_swrr(primaries: List[str]) -> None:
    global _swrr_weights, _swrr_current, _swrr_known_set
    actives = [p for p in primaries if p not in _disabled]
    _swrr_weights = {}
    _swrr_current = {}
    _swrr_known_set = set(actives)
    if not actives:
        return
    scale = 100  # percent basis
    total = int(round(scale))
    # compute weights for up to two preferred targets
    w1 = int(round(_BIAS_RATIO * scale)) if _BIAS_TARGET_BASE in actives else 0
    w2 = int(round(_BIAS_RATIO2 * scale)) if _BIAS_TARGET_BASE2 in actives else 0
    rem = max(0, total - w1 - w2)
    others = [p for p in actives if p not in (_BIAS_TARGET_BASE, _BIAS_TARGET_BASE2)]
    if others:
        base_w = rem // len(others)
        extra = rem - base_w * len(others)
        for i, p in enumerate(others):
            w = base_w + (1 if i < extra else 0)
            _swrr_weights[p] = max(0, w)
    if _BIAS_TARGET_BASE in actives:
        _swrr_weights[_BIAS_TARGET_BASE] = max(1, w1)
    if _BIAS_TARGET_BASE2 in actives:
        _swrr_weights[_BIAS_TARGET_BASE2] = max(1, w2)
    if not _swrr_weights:
        # fallback: equal weights if nothing assigned
        base_w = total // len(actives)
        extra = total - base_w * len(actives)
        for i, p in enumerate(actives):
            w = base_w + (1 if i < extra else 0)
            _swrr_weights[p] = max(1, w)
    for p in actives:
        _swrr_current[p] = 0


def _swrr_pick(primaries: List[str]) -> Optional[str]:
    actives = [p for p in primaries if p not in _disabled]
    if set(actives) != _swrr_known_set:
        _rebuild_swrr(actives)
    if not actives:
        return None
    if not _swrr_weights:
        _rebuild_swrr(actives)
    total_w = sum(_swrr_weights.get(p, 0) for p in actives)
    if total_w <= 0:
        return random.choice(actives)
    best_p = None
    best_val = None
    for p in actives:
        _swrr_current[p] = _swrr_current.get(p, 0) + _swrr_weights.get(p, 0)
        v = _swrr_current[p]
        if best_val is None or v > best_val:
            best_val = v
            best_p = p
    if best_p is None:
        return random.choice(actives)
    _swrr_current[best_p] = _swrr_current.get(best_p, 0) - total_w
    return best_p


async def _choose_summary_upstream() -> str:
    async with _rr_lock:
        return next(_rr_iter)


def _is_summary_request(body_obj) -> bool:
    return isinstance(body_obj, dict) and body_obj.get("model") == SUMMARY_MODEL


def _choose_sticky_upstream(key: str) -> str:
    # stable choice based on sha1(key)
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    idx = int.from_bytes(digest[:4], "big") % len(_SUMMARY_UPSTREAMS_BASE)
    return _SUMMARY_UPSTREAMS_BASE[idx]


# Hard-disable state and simple metrics
_disabled: Dict[str, Dict[str, Any]] = {}
_fail_counts: Dict[str, int] = {}
_metrics: Dict[str, Dict[str, Any]] = {}
FAIL_THRESHOLD = int(os.getenv("ROUTER_FAIL_THRESHOLD", "2"))
LOAD_BETA = float(os.getenv("ROUTER_LOAD_BETA", "0.5"))
_sticky_map: Dict[str, str] = {}
_in_flight: Dict[str, int] = {}


def _active_primaries() -> List[str]:
    return [b for b in _SUMMARY_UPSTREAMS_BASE if b not in _disabled]


def _next_active_rr() -> Optional[str]:
    # Get next round-robin base that is not disabled
    for _ in range(len(_SUMMARY_UPSTREAMS_BASE)):
        base = next(_rr_iter)
        if base not in _disabled:
            return base
    return None


def _any_primary_disabled() -> bool:
    return any(b in _disabled for b in _SUMMARY_UPSTREAMS_BASE)


def _active_set_for_sticky() -> List[str]:
    primaries = _active_primaries()
    # If any primary is disabled, include standby to share the load
    if len(primaries) < len(_SUMMARY_UPSTREAMS_BASE):
        for sb in _STANDBY_UPSTREAMS_BASE:
            if sb not in _disabled and sb not in primaries:
                primaries.append(sb)
    return primaries


def _hrw_order(sticky_key: str, servers: List[str], load_aware: bool) -> List[str]:
    scored = []
    for b in servers:
        h = hashlib.sha1()
        h.update(sticky_key.encode("utf-8"))
        h.update(b.encode("utf-8"))
        score = int.from_bytes(h.digest()[:8], "big")
        if load_aware:
            load = float(_in_flight.get(b, 0))
            denom = 1.0 + LOAD_BETA * load
            adj = score / denom if denom > 0 else float(score)
            scored.append((adj, b))
        else:
            scored.append((float(score), b))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [b for _, b in scored]


def _ordered_candidates(sticky_key: Optional[str]) -> List[str]:
    primaries = _active_primaries()
    ordered: List[str] = []
    if primaries:
        if sticky_key:
            # Rendezvous Hash (HRW) over active primaries
            ordered = _hrw_order(sticky_key, primaries, load_aware=False)
        else:
            if USE_SWRR:
                pick = _swrr_pick(primaries)
                if pick is None:
                    ordered = primaries
                else:
                    rest = [b for b in primaries if b != pick]
                    ordered = [pick] + rest
            else:
                # Weighted bias for non-sticky requests (randomized, up to two preferred targets)
                bias1 = _BIAS_TARGET_BASE in primaries
                bias2 = _BIAS_TARGET_BASE2 in primaries
                others = [b for b in primaries if b not in (_BIAS_TARGET_BASE, _BIAS_TARGET_BASE2)]
                p = random.random()
                if bias1 and p < _BIAS_RATIO:
                    ordered = [_BIAS_TARGET_BASE] + ([b for b in primaries if b != _BIAS_TARGET_BASE])
                elif bias2 and p < (_BIAS_RATIO + _BIAS_RATIO2):
                    ordered = [_BIAS_TARGET_BASE2] + ([b for b in primaries if b != _BIAS_TARGET_BASE2])
                else:
                    # Keep a simple rotation start to avoid starving any server
                    start = _next_active_rr()
                    if start and start in others:
                        sidx = others.index(start)
                        ordered = others[sidx:] + others[:sidx]
                    else:
                        ordered = others if others else primaries
                    # Put bias targets at the end as fallbacks if present
                    if bias1 and _BIAS_TARGET_BASE not in ordered:
                        ordered += [_BIAS_TARGET_BASE]
                    if bias2 and _BIAS_TARGET_BASE2 not in ordered:
                        ordered += [_BIAS_TARGET_BASE2]
    # Append standby (also skip disabled if ever set via admin)
    # Standby will be fully included for sticky path via _active_set_for_sticky when any primary is disabled
    ordered += [b for b in _STANDBY_UPSTREAMS_BASE if b not in _disabled]
    return ordered


def _mark_ok(base: str, status: int) -> None:
    _fail_counts[base] = 0
    m = _metrics.setdefault(base, {"ok": 0, "err": 0, "last_status": None, "last_error": None, "disabled": False})
    m["ok"] += 1
    m["last_status"] = status
    m["last_error"] = None
    m["disabled"] = base in _disabled


def _mark_err(base: str, error: str) -> None:
    cnt = _fail_counts.get(base, 0) + 1
    _fail_counts[base] = cnt
    m = _metrics.setdefault(base, {"ok": 0, "err": 0, "last_status": None, "last_error": None, "disabled": False})
    m["err"] += 1
    m["last_error"] = error
    # Hard-disable once threshold reached
    if cnt >= FAIL_THRESHOLD:
        _disabled[base] = {
            "disabled_at": time.time(),
            "reason": f"auto_fail_threshold_{FAIL_THRESHOLD}",
            "last_error": error,
        }
        m["disabled"] = True


@app.get("/health")
async def health():
    return {
        "ok": True,
        "service": "router_simple",
        "expected_model": SUMMARY_MODEL,
        "enforce_model": ENFORCE_MODEL,
        "upstreams": _SUMMARY_UPSTREAMS_BASE,
        "standby": _STANDBY_UPSTREAMS_BASE,
        "disabled": list(_disabled.keys()),
    }


@app.get("/metrics")
async def metrics():
    # include in_flight in metrics snapshot
    snapshot: Dict[str, Dict[str, Any]] = {}
    for b in set(_SUMMARY_UPSTREAMS_BASE + _STANDBY_UPSTREAMS_BASE):
        m = _metrics.setdefault(
            b, {"ok": 0, "err": 0, "last_status": None, "last_error": None, "disabled": (b in _disabled)}
        )
        m["disabled"] = b in _disabled
        m["in_flight"] = int(_in_flight.get(b, 0))
        snapshot[b] = dict(m)
    snapshot["sticky_map_size"] = {"count": len(_sticky_map)}
    return snapshot


@app.get("/admin/status")
async def admin_status():
    return {
        "active": _active_primaries(),
        "standby": [b for b in _STANDBY_UPSTREAMS_BASE if b not in _disabled],
        "disabled": _disabled,
        "fail_counts": _fail_counts,
        "sticky_disabled": DISABLE_STICKY,
    }


@app.post("/admin/disable")
@app.get("/admin/disable")
async def admin_disable(base: str):
    base = _normalize_base(base)
    _disabled[base] = {"disabled_at": time.time(), "reason": "manual"}
    m = _metrics.setdefault(base, {"ok": 0, "err": 0, "last_status": None, "last_error": None, "disabled": False})
    m["disabled"] = True
    return {"ok": True, "disabled": list(_disabled.keys())}


@app.post("/admin/enable")
@app.get("/admin/enable")
async def admin_enable(base: str):
    base = _normalize_base(base)
    _disabled.pop(base, None)
    _fail_counts[base] = 0
    m = _metrics.setdefault(base, {"ok": 0, "err": 0, "last_status": None, "last_error": None, "disabled": False})
    m["disabled"] = False
    return {"ok": True, "disabled": list(_disabled.keys())}


@app.post("/admin/clear_sticky")
@app.get("/admin/clear_sticky")
async def admin_clear_sticky():
    _sticky_map.clear()
    return {"ok": True, "sticky_map_size": len(_sticky_map)}


@app.api_route("/v1/{rest:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy(rest: str, request: Request):
    raw = await request.body()
    body_obj = None
    if raw:
        try:
            body_obj = json.loads(raw.decode("utf-8"))
        except Exception:
            body_obj = None

    # Determine sticky key from header or body
    sticky_key = None
    try:
        sticky_key = request.headers.get("x-trajectory-id") or None
    except Exception:
        sticky_key = None
    if sticky_key is None and isinstance(body_obj, dict):
        meta = body_obj.get("metadata") or {}
        if isinstance(meta, dict):
            sticky_key = meta.get("trajectory_id") or meta.get("traj_id")
        if sticky_key is None:
            sticky_key = body_obj.get("trajectory_id") or body_obj.get("traj_id")
        if sticky_key is None:
            # allow OpenAI chat.completions 'user' as sticky key
            u = body_obj.get("user")
            if isinstance(u, str) and u:
                sticky_key = u

    # Determine candidate list based on sticky and active set
    candidates: List[str] = []
    if (not ENFORCE_MODEL) or _is_summary_request(body_obj) or (request.method == "GET" and rest.startswith("models")):
        if sticky_key and (not DISABLE_STICKY):
            actives = _active_set_for_sticky()
            mapped = _sticky_map.get(sticky_key)
            if mapped in actives:
                # stick to existing mapping; use HRW order for failover on remaining actives
                failover_order = [b for b in _hrw_order(sticky_key, actives, load_aware=False) if b != mapped]
                candidates = [mapped] + failover_order
            else:
                # Weighted bias for sticky mapping creation
                bias1 = _BIAS_TARGET_BASE in actives
                bias2 = _BIAS_TARGET_BASE2 in actives
                others = [b for b in actives if b not in (_BIAS_TARGET_BASE, _BIAS_TARGET_BASE2)]
                p = random.random()
                if bias1 and p < _BIAS_RATIO:
                    _sticky_map[sticky_key] = _BIAS_TARGET_BASE
                    failover_order = [
                        b for b in _hrw_order(sticky_key, actives, load_aware=True) if b != _BIAS_TARGET_BASE
                    ]
                    candidates = [_BIAS_TARGET_BASE] + failover_order
                elif bias2 and p < (_BIAS_RATIO + _BIAS_RATIO2):
                    _sticky_map[sticky_key] = _BIAS_TARGET_BASE2
                    failover_order = [
                        b for b in _hrw_order(sticky_key, actives, load_aware=True) if b != _BIAS_TARGET_BASE2
                    ]
                    candidates = [_BIAS_TARGET_BASE2] + failover_order
                else:
                    pool = others if (others) else actives
                    ordered = _hrw_order(sticky_key, pool, load_aware=True)
                    if ordered:
                        _sticky_map[sticky_key] = ordered[0]
                        failover_order = [
                            b for b in _hrw_order(sticky_key, actives, load_aware=True) if b != ordered[0]
                        ]
                        candidates = [ordered[0]] + failover_order
                    else:
                        candidates = []
        else:
            candidates = _ordered_candidates(None)
    if not candidates:
        return JSONResponse(
            {
                "error": "router_simple: request not handled (not the Web Summary model)",
                "expected_model": SUMMARY_MODEL,
            },
            status_code=400,
        )

    forward_path = rest

    headers = dict(request.headers)
    headers.pop("host", None)  # let httpx set Host for upstream
    params = dict(request.query_params)

    want_stream = (isinstance(body_obj, dict) and body_obj.get("stream") is True) or (
        "text/event-stream" in request.headers.get("accept", "")
    )

    FAILOVER_STATUS = {408, 429, 500, 502, 503, 504}
    tried: List[str] = []
    last_error: Optional[str] = None

    for idx, up in enumerate(candidates):
        if up in _disabled:
            continue
        tried.append(up)
        url = up.rstrip("/") + "/v1/" + forward_path.lstrip("/")
        try:
            timeout = httpx.Timeout(connect=2.0, read=None, write=10.0, pool=None)
            async with client.stream(
                request.method, url, params=params, headers=headers, content=raw, timeout=timeout
            ) as resp:
                if resp.status_code in FAILOVER_STATUS:
                    _mark_err(up, error=f"status_{resp.status_code}")
                    # Hard disable on threshold, then continue to next
                    continue
                ctype = resp.headers.get("content-type", "")
                if want_stream or ctype.startswith("text/event-stream"):
                    # Track in-flight for streaming until generator completes
                    _in_flight[up] = _in_flight.get(up, 0) + 1

                    async def gen():
                        try:
                            async for chunk in resp.aiter_bytes():
                                yield chunk
                        finally:
                            _in_flight[up] = max(0, _in_flight.get(up, 1) - 1)

                    _mark_ok(up, resp.status_code)
                    return StreamingResponse(
                        gen(),
                        status_code=resp.status_code,
                        media_type=ctype,
                        headers={
                            "X-Upstream-Server": up,
                            "X-Failover-Attempts": str(idx),
                            "X-Tried-Upstreams": ",".join(tried),
                        },
                    )
                # Non-streaming: count in-flight during body read
                _in_flight[up] = _in_flight.get(up, 0) + 1
                try:
                    content = await resp.aread()
                finally:
                    _in_flight[up] = max(0, _in_flight.get(up, 1) - 1)
                passthru = {
                    k: v
                    for k, v in resp.headers.items()
                    if k.lower() in ["content-type", "cache-control", "x-request-id"]
                }
                # annotate which upstream handled the request
                passthru["X-Upstream-Server"] = up
                passthru["X-Failover-Attempts"] = str(idx)
                passthru["X-Tried-Upstreams"] = ",".join(tried)
                _mark_ok(up, resp.status_code)
                return Response(content=content, status_code=resp.status_code, headers=passthru)
        except httpx.HTTPError as e:
            last_error = str(e)
            _mark_err(up, error=last_error)
            continue

    # All attempts failed or all candidates disabled
    detail = {
        "error": "All upstreams failed or disabled",
        "tried": tried,
        "last_error": last_error,
        "disabled": list(_disabled.keys()),
    }
    return JSONResponse(detail, status_code=502)
