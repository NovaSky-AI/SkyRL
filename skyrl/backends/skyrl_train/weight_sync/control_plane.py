"""Blocking control-plane client for the trainer-side weight transfer engines.

vLLM's ``VLLMWeightSyncClient`` is a synchronous four-method protocol
(``init_weight_transfer_engine`` / ``start_weight_update`` / ``update_weights`` /
``finish_weight_update``), and the engine that drives it runs off the worker's
event loop. So this client uses blocking HTTP against the same routes rather
than SkyRL's async ``RemoteInferenceClient``, keeping the whole engine
sync-to-sync with no event-loop involvement.

Four properties are load-bearing:

* **Per-call fresh connections** (``Connection: close``). A full training step
  elapses between syncs, so a pooled keep-alive connection is stale by the next
  call and races the server's ``timeout_keep_alive`` (uvicorn 5s) into
  ECONNRESET.
* **Concurrent fan-out** across servers. Required for correctness on the RDT
  path, not just speed: consumers pull in lockstep and a producer frees a served
  group only once every bound consumer has pulled, so a serial
  ``update_weights`` stalls the producer's gather loop and deadlocks.
* **Body-aware error messages** (:func:`_error_message`) — surface the response
  body's error detail, not the bare HTTP reason phrase.
* **No timeout** (an RDT bake + NIXL pull are long) and **no retry** (retrying a
  half-done stateful call would be wrong).

``requests`` rather than a new dependency: Ray provides it, and Ray is a hard
requirement of every path that reaches here. The import is local so nothing else
pays for it.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# vLLM dev-mode RLHF routes (entrypoints/serve/dev/rlhf/api_router.py), plus
# /fetch_weights and /reset_prefix_cache, which SkyRL adds in vllm_server_actor.
INIT_ENGINE_ENDPOINT = "/init_weight_transfer_engine"
START_UPDATE_ENDPOINT = "/start_weight_update"
UPDATE_WEIGHTS_ENDPOINT = "/update_weights"
FINISH_UPDATE_ENDPOINT = "/finish_weight_update"
FETCH_WEIGHTS_ENDPOINT = "/fetch_weights"
RESET_PREFIX_CACHE_ENDPOINT = "/reset_prefix_cache"
PAUSE_ENDPOINT = "/pause"
RESUME_ENDPOINT = "/resume"


class SkyrlWeightSyncClient:
    """``VLLMWeightSyncClient`` over the inference servers' native HTTP routes.

    Args:
        server_urls: every inference server, ordered
            ``[engine0_dp0, engine0_dp1, ..., engine1_dp0, ...]`` — the order the
            per-server init rewrite (``init_payload_fn``) depends on.
        data_parallel_size: DP replicas per deployment. Servers are grouped into
            deployments of this many.
        init_payload_fn: rewrites the engine's single worker-side init dict into
            one payload per server, for the backends that need per-server fields:
            NCCL a cumulative ``rank_offset``, sharded RDT a deployment ordinal.
            See :func:`nccl_init_payloads` / :func:`rdt_init_payloads`. ``None``
            sends the same dict to every server.
    """

    def __init__(
        self,
        server_urls: Sequence[str],
        data_parallel_size: int = 1,
        *,
        init_payload_fn: Optional[Callable[[Dict[str, Any], Sequence[str], int], List[Dict[str, Any]]]] = None,
    ) -> None:
        import requests  # local: Ray (a hard dep of every path reaching here) provides it.

        self._urls = list(server_urls)
        self._dp = max(1, int(data_parallel_size))
        self._init_payload_fn = init_payload_fn
        if not self._urls:
            raise ValueError("SkyrlWeightSyncClient requires at least one server_url.")

        self._session = requests.Session()
        # See module docstring: fresh connection per call.
        self._session.headers["Connection"] = "close"
        # One worker per server so a fan-out call issues every POST concurrently.
        self._pool = ThreadPoolExecutor(max_workers=len(self._urls), thread_name_prefix="weight-sync-ctrl")

    # ---- VLLMWeightSyncClient protocol (the four methods vLLM's engines call) ----

    def init_weight_transfer_engine(self, init_info: Dict[str, Any]) -> None:
        if self._init_payload_fn is None:
            per_server = [dict(init_info) for _ in self._urls]
        else:
            per_server = self._init_payload_fn(init_info, self._urls, self._dp)
        if len(per_server) != len(self._urls):
            raise ValueError(
                f"init_payload_fn returned {len(per_server)} payloads for {len(self._urls)} servers; "
                "it must return exactly one per server, in server order."
            )
        self._fanout([(url, INIT_ENGINE_ENDPOINT, {"init_info": info}) for url, info in zip(self._urls, per_server)])

    def start_weight_update(self) -> None:
        self._fanout_uniform(START_UPDATE_ENDPOINT, None)

    def update_weights(self, update_info: Dict[str, Any]) -> None:
        self._fanout_uniform(UPDATE_WEIGHTS_ENDPOINT, {"update_info": _json_safe(update_info)})

    def finish_weight_update(self, weight_version: Optional[str] = None) -> None:
        # Omit the key entirely when unset so the route's default applies.
        body = {"weight_version": weight_version} if weight_version is not None else None
        self._fanout_uniform(FINISH_UPDATE_ENDPOINT, body)

    # ---- extras: the checkpoint-delta lifecycle ----
    #
    # Not the transport contract, and here anyway, because they have to be driven
    # from inside `send_weights`: `fetch_weights` needs the `target_version` the
    # publish produces mid-send, and must run before the pause. `send_weights` is
    # sync and runs off the event loop, so `RemoteInferenceClient` -- where these
    # routes otherwise live -- is reachable only via `run_coroutine_threadsafe`,
    # the loop coupling this client exists to avoid (see module docstring).
    # Splitting them into a second sync object would duplicate the session and
    # pool to narrow one type.
    #
    # The cost: an engine calling these needs a SkyRL client, not any object
    # satisfying VLLMWeightSyncClient. Only the delta engine does.

    def fetch_weights(self, target_version: int, sync_dir: Optional[str] = None, uri: Optional[str] = None) -> None:
        body: Dict[str, Any] = {"target_version": int(target_version)}
        if sync_dir is not None:
            body["sync_dir"] = sync_dir
        if uri is not None:
            body["uri"] = uri
        self._fanout_uniform(FETCH_WEIGHTS_ENDPOINT, body)

    def reset_prefix_cache(self, reset_running_requests: bool = True) -> None:
        self._fanout_uniform(RESET_PREFIX_CACHE_ENDPOINT, {"reset_running_requests": reset_running_requests})

    def pause_generation(self, clear_cache: bool = False) -> None:
        # /pause takes query params, not a body (mirrors RemoteInferenceClient.pause).
        self._fanout(
            [(url, f"{PAUSE_ENDPOINT}?mode=keep&clear_cache={str(clear_cache).lower()}", None) for url in self._urls]
        )

    def resume_generation(self) -> None:
        self._fanout_uniform(RESUME_ENDPOINT, None)

    def close(self) -> None:
        """Release the HTTP session + fan-out pool. Idempotent."""
        self._pool.shutdown(wait=True)
        self._session.close()

    # ---- internals ----

    def _fanout_uniform(self, endpoint: str, body: Optional[Dict[str, Any]]) -> None:
        self._fanout([(url, endpoint, body) for url in self._urls])

    def _fanout(self, calls: Sequence[Tuple[str, str, Optional[Dict[str, Any]]]]) -> None:
        """POST to every server concurrently; raise the first failure after all
        return.

        Every future is drained before raising, so a failure on one server never
        leaves POSTs in flight against the others."""
        futures = [self._pool.submit(self._post, url, endpoint, body) for url, endpoint, body in calls]
        first_exc: Optional[BaseException] = None
        for fut in futures:
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                first_exc = first_exc or exc
        if first_exc is not None:
            raise first_exc

    def _post(self, url: str, endpoint: str, body: Optional[Dict[str, Any]]) -> None:
        # No timeout, no retry -- see module docstring.
        resp = self._session.post(f"{url}{endpoint}", json=body, timeout=None)
        if resp.status_code >= 400:
            raise RuntimeError(_error_message(url, endpoint, resp))


def _json_safe(update_info: Dict[str, Any]) -> Dict[str, Any]:
    """Make an update_info dict JSON-serializable for HTTP transport.

    ``IPCTrainerWeightTransferEngine`` emits raw CUDA IPC handles -- tuples
    containing storage types and handle bytes -- because a Ray transport carries
    them natively. Over HTTP they must be pickled into ``ipc_handles_pickled``,
    which the worker auto-deserializes when
    ``VLLM_ALLOW_INSECURE_SERIALIZATION=1`` (set by ``vllm_server_actor``).

    Other backends carry only JSON-native metadata and pass through unchanged.
    """
    handles = update_info.get("ipc_handles")
    if handles is None:
        return update_info

    import base64
    import pickle

    out = {k: v for k, v in update_info.items() if k != "ipc_handles"}
    out["ipc_handles_pickled"] = base64.b64encode(pickle.dumps(handles)).decode("utf-8")
    return out


def nccl_init_payloads(
    init_info: Dict[str, Any],
    server_urls: Sequence[str],
    data_parallel_size: int,
) -> List[Dict[str, Any]]:
    """Per-server NCCL init payloads with cumulative ``rank_offset``.

    ``worker_init_process_group`` derives a worker's rank as
    ``dp_rank * world_size_per_dp + rank_within_dp + rank_offset``, and each
    deployment's indices restart at 0, so every deployment's offset must advance
    past the previous one's workers or their ranges collide. The engine builds
    one init info with ``rank_offset=1``, which is only correct for a single
    deployment.

    DP servers within one deployment share an offset -- vLLM's
    ``data_parallel_index`` already separates them, so advancing per DP server
    would double-count. The offset advances only at deployment boundaries.

    A wrong offset mis-maps ranks and hangs in the NCCL rendezvous rather than
    failing, so ``world_size`` is cross-checked against what the offsets imply.
    """
    dp = max(1, int(data_parallel_size))
    num_servers = len(server_urls)
    base_offset = int(init_info.get("rank_offset", 1))
    world_size = int(init_info["world_size"])
    num_deployments = max(1, num_servers // dp)
    # world_size counts the trainer sender (rank 0) plus every inference worker,
    # so the per-deployment worker count follows from it.
    workers_total = world_size - base_offset
    if workers_total <= 0 or workers_total % num_deployments != 0:
        raise ValueError(
            f"NCCL weight sync world_size={world_size} with rank_offset={base_offset} implies "
            f"{workers_total} inference workers, which does not divide across {num_deployments} "
            f"deployment(s) ({num_servers} servers / dp={dp})."
        )
    world_size_per_deployment = workers_total // num_deployments

    payloads: List[Dict[str, Any]] = []
    offset = base_offset
    for i in range(num_servers):
        payloads.append({**init_info, "rank_offset": offset})
        if (i + 1) % dp == 0:
            offset += world_size_per_deployment
    return payloads


def rdt_init_payloads(
    init_info: Dict[str, Any],
    server_urls: Sequence[str],
    data_parallel_size: int,
) -> List[Dict[str, Any]]:
    """Per-server sharded-RDT init payloads, stamped with the deployment ordinal.

    Each deployment has its own self-contained parallel config, so its internal
    worker index restarts at 0 and would collide under the M:N block assignment.
    Stamping each server with its ordinal (``server_index // data_parallel_size``)
    as ``replica_rank``, plus the deployment count as ``num_replicas``, lets the
    engine offset its consumers into a globally distinct range.

    The ordinal divides by ``data_parallel_size`` because the DP servers of one
    deployment share a parallel config, so they must share one ``replica_rank``.
    """
    dp = max(1, int(data_parallel_size))
    num_replicas = max(1, len(server_urls) // dp)
    return [{**init_info, "replica_rank": i // dp, "num_replicas": num_replicas} for i in range(len(server_urls))]


def _error_message(url: str, endpoint: str, resp: Any) -> str:
    """Surface the response body's error detail (``{"error": {"message": ...}}``
    or FastAPI's ``{"detail": ...}``) rather than the bare HTTP reason phrase."""
    detail = resp.reason
    try:
        body = resp.json()
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                detail = err.get("message", detail)
            elif isinstance(err, str):
                detail = err
            elif body.get("detail") is not None:
                detail = body["detail"]
    except Exception:  # noqa: BLE001
        detail = (resp.text or resp.reason)[:1000]
    return f"Weight-sync control-plane call {endpoint} to {url} failed [{resp.status_code}]: {detail}"
