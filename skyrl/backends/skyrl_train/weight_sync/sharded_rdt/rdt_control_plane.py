"""Synchronous control-plane client for the sharded-RDT trainer engine.

The vendored trainer engine (``sharded_rdt_trainer.py``) drives the inference
side through the *synchronous* ``VLLMWeightSyncClient`` protocol — a serial
``init -> start -> update -> finish`` handshake issued once per weight sync.
SkyRL's ``RemoteInferenceClient`` exposes those same routes as **coroutines** on
the worker's event loop, so the previous glue bounced every call back onto that
loop with ``run_coroutine_threadsafe`` — coupling the (thread-run) engine to a
specific loop instance and requiring a documented "never call from the loop
thread or you deadlock" invariant.

None of these four calls needs the async client's connection pooling or its
generation concurrency, so this client talks to the same ``/collective_rpc``
endpoints over **blocking HTTP**. The engine then runs entirely sync-to-sync in
its worker thread with zero event-loop involvement, and the deadlock class
disappears structurally rather than by convention.

What we deliberately keep from ``RemoteInferenceClient`` so sidestepping it costs
nothing (see ``_post`` / ``_fanout``):

* **Per-call fresh connections** (``Connection: close``). A full training step
  elapses between syncs, so any pooled keep-alive connection is stale by the next
  call; reusing it races the server's ``timeout_keep_alive`` (uvicorn 5s) and
  yields ECONNRESET. The async client dodges this with ``keepalive_timeout=2``;
  for once-per-step control calls it is simpler and strictly safer to not keep
  connections at all.
* **Concurrent fan-out** across servers. This is REQUIRED, not merely faster:
  the consumers pull over NIXL in lockstep and the producer only frees a served
  group once every consumer bound to it has pulled, so issuing ``update_weights``
  server-by-server would stall the producer's gather loop and deadlock. Mirrors
  ``RemoteInferenceClient._call_all_servers``.
* **Body-aware error messages** (``_error_message``), matching the client's
  ``raise_for_status`` — surface the response body's error detail, not just the
  HTTP reason phrase.
* **No timeout** (bake + NIXL pull are long) and **no retry** (retrying a
  half-done bake/pull would be wrong; ``Connection: close`` already removes the
  only transient race the async control path guarded against).

``requests`` is used rather than adding a new dependency: the sharded_rdt path
already hard-requires Ray, and Ray depends on ``requests``, so it is always
importable wherever this client runs. The import is local so non-RDT / non-Ray
code paths never pay for it.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

from skyrl.backends.skyrl_train.inference_servers.rdt_control_protocol import (
    COLLECTIVE_RPC_ENDPOINT,
    RDT_FINISH_METHOD,
    RDT_START_METHOD,
    RDT_UPDATE_METHOD,
    build_rdt_init_payloads,
)

logger = logging.getLogger(__name__)


class SyncRdtControlPlaneClient:
    """Blocking ``VLLMWeightSyncClient`` over the inference servers' ``/collective_rpc``.

    Implements the four synchronous control-plane calls the vendored trainer
    engine makes (``sharded_rdt_base.VLLMWeightSyncClient``). Constructed by
    ``RdtWeightSyncSender`` from the ``RemoteInferenceClient``'s ``server_urls``
    and ``data_parallel_size`` — the async session and event loop are NOT used.
    Safe to call from any thread; the engine calls it from the worker thread
    ``RdtWeightSyncSender`` runs ``send_weights`` on.
    """

    def __init__(self, server_urls: Sequence[str], data_parallel_size: int) -> None:
        import requests  # local: Ray (an RDT hard-dep) provides it; keep it off non-RDT paths.

        self._urls = list(server_urls)
        self._dp = int(data_parallel_size)
        if not self._urls:
            raise ValueError("SyncRdtControlPlaneClient requires at least one server_url.")

        self._session = requests.Session()
        # Per-call fresh connections — see module docstring (avoids the stale
        # keep-alive / ECONNRESET race across the long idle gap between syncs).
        self._session.headers["Connection"] = "close"
        # One worker per server so a fan-out call issues every POST concurrently.
        self._pool = ThreadPoolExecutor(max_workers=len(self._urls), thread_name_prefix="rdt-ctrl")

    # ---- VLLMWeightSyncClient protocol ----

    def init_weight_transfer_engine(self, init_info: Dict[str, Any]) -> None:
        self._fanout(build_rdt_init_payloads(init_info, self._urls, self._dp))

    def start_weight_update(self) -> None:
        self._fanout_uniform(RDT_START_METHOD, {"is_checkpoint_format": True})

    def update_weights(self, update_info: Dict[str, Any]) -> None:
        self._fanout_uniform(RDT_UPDATE_METHOD, {"update_info": update_info})

    def finish_weight_update(self) -> None:
        self._fanout_uniform(RDT_FINISH_METHOD, None)

    def close(self) -> None:
        """Release the HTTP session + fan-out pool. Idempotent."""
        self._pool.shutdown(wait=True)
        self._session.close()

    # ---- internals ----

    def _fanout_uniform(self, method: str, kwargs: Optional[Dict[str, Any]]) -> None:
        payload: Dict[str, Any] = {"method": method}
        if kwargs is not None:
            payload["kwargs"] = kwargs
        self._fanout([(url, payload) for url in self._urls])

    def _fanout(self, url_payloads: List[Tuple[str, Dict[str, Any]]]) -> None:
        """POST to every server concurrently; raise the first failure after all
        return. Concurrency is required for correctness, not speed (see module
        docstring): serial ``update_weights`` deadlocks the producer's
        ref-counted group free. We drain ALL futures before raising so a failure
        on one server never leaves POSTs in flight against the others."""
        futures = [self._pool.submit(self._post, url, payload) for url, payload in url_payloads]
        first_exc: Optional[BaseException] = None
        for fut in futures:
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                first_exc = first_exc or exc
        if first_exc is not None:
            raise first_exc

    def _post(self, url: str, payload: Dict[str, Any]) -> None:
        # No timeout (bake + NIXL pull are long) and no retry (retrying a
        # half-done stateful call would be wrong; Connection: close already
        # removes the stale-keepalive race the async path guarded against).
        resp = self._session.post(f"{url}{COLLECTIVE_RPC_ENDPOINT}", json=payload, timeout=None)
        if resp.status_code >= 400:
            raise RuntimeError(_error_message(url, payload, resp))


def _error_message(url: str, payload: Dict[str, Any], resp: Any) -> str:
    """Mirror ``RemoteInferenceClient.raise_for_status``: surface the response
    body's error detail (``{"error": {"message": ...}}``) rather than the bare
    HTTP reason phrase, which is usually unhelpful."""
    method = payload.get("method")
    detail = resp.reason
    try:
        body = resp.json()
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                detail = err.get("message", detail)
    except Exception:  # noqa: BLE001
        detail = (resp.text or resp.reason)[:1000]
    return f"RDT control-plane call {method!r} to {url} failed [{resp.status_code}]: {detail}"
