"""
RemoteInferenceClient - Serializable HTTP client for inference.

This is a lightweight, fully serializable HTTP client that wraps the inference
server HTTP API. It is the concrete ``InferenceEngineInterface`` implementation
used for HTTP-based inference servers.

Architecture:
-------------
This client is responsible for BOTH data plane and control plane operations:

1. Data Plane (routed through proxy_url):
   - generate, chat_completion, completion, tokenize, detokenize, render
   - Uses proxy_url which points to a router (VLLMRouter or external)
   - Router handles load balancing and session-aware routing

2. Control Plane (fan-out to all server_urls):
   - pause, resume, sleep, wake_up, reset_prefix_cache
   - init_weight_transfer, update_weights_skyrl
   - Fans out directly to all backend servers (bypassing router)
   - This allows using external routers that only handle data plane

The router (proxy_url) is expected to be a data-plane-only router. Control plane
operations are always fanned out to all backends by this client directly.

Key features:
- Serializable: Can be pickled and passed between processes
- Two URL types:
  - proxy_url: Single URL for data plane operations (routed requests)
  - server_urls: List of backend URLs for control plane operations (fan-out)
- Fault tolerance (opt-in, ``fault_tolerance.enabled``): ``server_urls`` is the
  PROVISIONED list and is never mutated — position in it is engine identity for the
  whole run. ``active_server_urls`` is the subset that has not been found dead, and
  is what the control plane and weight sync talk to. Detection is reactive: a caught
  data-plane failure triggers one ``/health`` probe of the fleet, non-responders are
  dropped from the router, and generation retries land on the survivors.
- Lazy world_size fetching from /get_server_info
- Keep-mode pause: in-flight requests are frozen by the vLLM scheduler and
  resume where they left off after /resume. No client-side retry needed.

Usage:
    client = RemoteInferenceClient(
        proxy_url="http://router:8080",  # Data plane (router)
        server_urls=["http://backend1:8000", "http://backend2:8000"],  # Control plane
        data_parallel_size=1,
    )

Design notes:
- Talks directly to the router via HTTP, no Ray actor wrapping.
- The router handles session-aware routing; this client handles control plane fan-out.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Required,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

import aiohttp

from skyrl.backends.skyrl_train.inference_servers.base import (
    InferenceEngineInput,
    InferenceEngineInterface,
    InferenceEngineOutput,
    MMPlaceholderRangeInfo,
    MultiModalFeatures,
)
from skyrl.env_vars import (
    SKYRL_GENERATE_CONCURRENCY_PER_ENGINE,
    SKYRL_HTTP_CONNECTION_LIMIT,
)

_DATA_PLANE_RETRIES = 30

# Router worker-management routes (see `_router_post_worker`). Confirmed present in
# vllm-router 0.1.14.post1; the backend URL travels as the `url` query parameter.
_ROUTER_ADD_WORKER = "/add_worker"
_ROUTER_REMOVE_WORKER = "/remove_worker"
_ROUTER_LIST_WORKERS = "/list_workers"
_ROUTER_ADMIN_TIMEOUT_S = 10.0
"""Bound for router membership calls. These run on the failure path, where the
router may itself be struggling; an unbounded admin call would defeat the point."""

# Router status codes that mean "the backend behind me is gone or unreachable", as
# opposed to a client error we should surface. Retried (and reconciled) under FT.
_ROUTER_BACKEND_FAILURE_STATUSES = frozenset({502, 503, 504})

SKYRL_LORA_ADAPTER_NAME = "skyrl-lora"
"""Default LoRA adapter name used for single-LoRA training inside SkyRL."""

_TINKER_SAMPLE_TO_VLLM_PARAM_MAP = {
    "temperature": "temperature",
    "max_tokens": "max_tokens",
    "seed": "seed",
    "top_k": "top_k",
    "top_p": "top_p",
    "stop_strings": "stop",
    "stop_tokens": "stop_token_ids",
}

if TYPE_CHECKING:
    from skyrl.backends.skyrl_train.weight_sync.transfer_strategy import (
        WeightSyncInitInfo,
    )
    from skyrl.train.config.config import InferenceFaultToleranceConfig


logger = logging.getLogger(__name__)


class EngineFleetError(RuntimeError):
    """Too few inference engines are alive to continue.

    Raised by ``_reconcile_fleet`` when a probe would take the active set below
    ``fault_tolerance.min_live_engines``. Part 1 cannot bring engines back, so
    there is nothing to wait for — failing here is strictly better than training
    against a fleet that cannot serve.
    """


def _extract_session_id_and_body(
    request_payload: Dict[str, Any],
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Extract session_id and a clean body from an OpenAI-style request payload.

    Returns (session_id, body) where body is a shallow copy without the session_id key.
    """
    body = request_payload.get("json", {})
    session_id = body.get("session_id")
    clean_body = {k: v for k, v in body.items() if k != "session_id"}
    return session_id, clean_body


class PauseMode(Enum):
    """
    Pause mode for inference servers.

    Maps to the ``mode`` query parameter on vLLM's ``/pause`` endpoint.

    Modes:
        ABORT: Abort in-flight requests immediately. Clients receive partial
            tokens with ``finish_reason="abort"`` and must retry.
        KEEP: Freeze in-flight requests in the scheduler. They resume
            exactly where they left off when ``/resume`` is called.
            No retry needed. KV cache is preserved.
        WAIT: Wait for in-flight requests to complete before pausing.
            New requests are blocked. No retry needed.
    """

    ABORT = "abort"
    KEEP = "keep"
    WAIT = "wait"


class SampleRequestBody(TypedDict, total=False):
    """Tinker-style sample request body, mirroring tinker SamplingClient.sample"""

    prompt: Required[Dict[str, Any]]
    num_samples: int
    sampling_params: Dict[str, Any]
    session_id: str
    include_prompt_logprobs: bool
    prompt_logprobs: bool
    topk_prompt_logprobs: int


class SampleRequestPayload(TypedDict):
    """Wrapper for sample request (matches the {"json": ...} convention)."""

    json: SampleRequestBody


class SampleResponse(TypedDict):
    """Return value of RemoteInferenceClient.sample(), mirrors tinker SampleResponse"""

    type: Literal["sample"]
    sequences: List[Dict[str, Any]]
    prompt_logprobs: Optional[List[Optional[float]]]
    topk_prompt_logprobs: Optional[List[Optional[List[Tuple[int, float]]]]]


@dataclass
class RemoteInferenceClient(InferenceEngineInterface):
    """
    Serializable HTTP client for inference. The concrete InferenceEngineInterface.

    This class maintains two URL types:
    - proxy_url: Single URL for data plane operations (routed requests)
    - server_urls: List of backend URLs for control plane operations (fan-out)

    The router (proxy_url) is expected to be a data-plane-only router (like
    VLLMRouter or an external router). Control plane operations
    are always fanned out to all backends directly by this client.

    Usage:
        client = RemoteInferenceClient(
            proxy_url="http://router:8080",  # Data plane (router)
            server_urls=["http://backend1:8000", "http://backend2:8000"],  # Control plane
            data_parallel_size=1, # data parallel size for deployments
        )
    """

    proxy_url: str
    """Data plane URL (single endpoint - router or direct server)."""

    server_urls: List[str]
    """Control plane URLs (list of backend servers for fan-out)."""

    data_parallel_size: int
    """Data parallel size. Used to compute total inference world size correctly:
    server_urls contains num_engines * data_parallel_size entries, but vLLM already
    reports the full DP world size per server, so we divide by num_deployments."""

    server_slots: Optional[List[int]] = None
    """Stable deployment ordinal per entry of ``server_urls`` (see ``ServerInfo.slot``).

    ``None`` (the default) derives them positionally as ``index // data_parallel_size``,
    which is exactly what a static fleet means and what every pre-fault-tolerance
    caller got implicitly. Passing them explicitly is what lets a URL *change* --
    a restarted engine comes back on a re-reserved port -- without renumbering the
    consumer-id blocks every surviving engine already owns.

    Normalized to a concrete list in ``__post_init__``, so readers never handle
    ``None``.
    """

    model_name: str = "default"
    """The model identifier accepted by the inference server for the base model.

    This is usually the model path, but may be ``served_model_name`` when vLLM
    is started with an alias. It is never a LoRA adapter name. LoRA adapters are
    addressed by the names callers register them under via
    ``load_lora_adapter(name, path)``, and per-call routing is done by
    passing that name as ``model`` on the data-plane methods.

    Used internally only by ``tokenize``/``detokenize``, which are LoRA-
    agnostic but still require a ``model`` field per the OpenAI schema.
    """

    enable_return_routed_experts: bool = False
    """Whether to return routed expert indices (R3 / rollout router replay)."""

    uses_lora_weight_sync: bool = False
    """True when the trainer syncs LoRA adapters (rather than full/merged weights). When True,
    `sleep()` is forced to level=1: level=2 discards the base model from VRAM with no CPU backup,
    and LoRA-only broadcasts cannot repopulate it. Must be kept in sync with the same gate vLLM
    uses for `enable_lora` (see `_uses_lora_weight_sync` in inference_servers/utils.py)."""

    tokenizer: Optional[Any] = None
    """Optional HF tokenizer for local tokenize/detokenize (avoids HTTP round-trips)."""

    fault_tolerance: Optional["InferenceFaultToleranceConfig"] = None
    """Engine fault-tolerance policy (``generator.inference_engine.fault_tolerance``).
    ``None`` or ``enabled=False`` leaves every FT path inert."""

    # Private fields excluded from repr for cleaner output
    _session: Optional[aiohttp.ClientSession] = field(default=None, repr=False)
    _world_size: Optional[Tuple[int, int]] = field(default=None, repr=False)
    _gen_sem: Optional[asyncio.Semaphore] = field(default=None, repr=False)
    _detok_sem: Optional[asyncio.Semaphore] = field(default=None, repr=False)
    _sem_loop: Optional[asyncio.AbstractEventLoop] = field(default=None, repr=False)
    # Monotonic counter of weight syncs (see `increment_weight_version`); source of the prefix-cache salt.
    _weight_version: int = field(default=0, repr=False)
    # [FT] Backends a health probe found dead. Driver-side; written only by
    # `_reconcile_fleet`. `server_urls` itself is NEVER mutated -- it is the
    # provisioned list, and position in it IS engine identity for the whole run
    # (`replica_rank = index // data_parallel_size`). Slots are never compacted.
    _disabled_server_urls: set = field(default_factory=set, repr=False)
    # Bumps on every membership change. Callers capture it before a request and
    # pass it back on failure, so a burst of failures under one membership state
    # triggers exactly one probe.
    _membership_generation: int = field(default=0, repr=False)
    # The single in-flight probe (see `_reconcile_fleet`), and the loop it belongs
    # to. Both are dropped on pickle.
    _reconcile_task: Optional[Any] = field(default=None, repr=False)
    _reconcile_loop: Optional[asyncio.AbstractEventLoop] = field(default=None, repr=False)
    # [FT Part 2] The driver's EngineSupervisor, when engine restarts are enabled.
    # Carried here because this client is the one object already threaded to every
    # driver-side consumer (trainer, dispatch, generator) -- the alternative was a new
    # constructor argument on each. Driver-only: dropped on pickle like the session and
    # the probe task, so no worker ever sees a ServerGroup or a placement group.
    _engine_supervisor: Optional[Any] = field(default=None, repr=False)

    @property
    def weight_version(self) -> int:
        """Number of weight syncs to the engines so far (0 before the first sync); the policy version."""
        return self._weight_version

    def increment_weight_version(self) -> None:
        """Advance the weight version. Called once per completed weight sync to the engines."""
        self._weight_version += 1

    # ---------------------------
    # Fleet membership (fault tolerance)
    # ---------------------------

    @property
    def _ft_enabled(self) -> bool:
        return bool(self.fault_tolerance is not None and self.fault_tolerance.enabled)

    @property
    def active_server_urls(self) -> List[str]:
        """Provisioned URLs minus the ones a health probe found dead.

        In provisioned order, and never compacted: a URL leaving this list does not
        renumber the ones after it. Use this for anything that talks to backends now
        (control plane, weight sync); use ``server_urls`` for anything that encodes
        identity or geometry (``replica_rank``, ``num_consumers``).
        """
        if not self._disabled_server_urls:
            return list(self.server_urls)
        return [u for u in self.server_urls if u not in self._disabled_server_urls]

    @property
    def disabled_server_urls(self) -> List[str]:
        """Provisioned URLs currently marked dead, in provisioned order."""
        return [u for u in self.server_urls if u in self._disabled_server_urls]

    @property
    def num_provisioned_replicas(self) -> int:
        """Distinct deployments provisioned for this run.

        Counted from ``server_slots``, which is never compacted, so this stays at
        the provisioned value however many engines are currently dead. RDT sizes
        each consumer's id offset from it, so deriving it from a live URL count
        instead would silently re-map survivors onto each other's slices.
        """
        return max(1, len(set(self.server_slots or [0])))

    @property
    def url_slots(self) -> List[Tuple[str, int]]:
        """``(url, slot)`` for every provisioned server, in provisioned order."""
        return list(zip(self.server_urls, self.server_slots or []))

    @property
    def live_url_slots(self) -> List[Tuple[str, int]]:
        """``(url, slot)`` for the servers not currently marked dead.

        The pair, rather than a bare URL list, is what the RDT sender needs: it maps
        a server to the consumer ids it owns, and after a restart the URL is new
        while the slot -- and therefore the consumer block -- is unchanged. A sender
        that looked the URL up positionally would reject the restarted engine as
        "outside the provisioned set".
        """
        if not self._disabled_server_urls:
            return self.url_slots
        return [(u, s) for u, s in self.url_slots if u not in self._disabled_server_urls]

    def slot_of(self, url: str) -> int:
        """The slot a provisioned URL currently occupies.

        Raises:
            KeyError: the URL is not in the provisioned set.
        """
        try:
            return (self.server_slots or [])[self.server_urls.index(url)]
        except ValueError as e:
            raise KeyError(f"{url} is not a provisioned server URL ({self.server_urls}).") from e

    @property
    def engine_supervisor(self) -> Optional[Any]:
        """The driver's ``EngineSupervisor``, or ``None`` (disabled, or a worker copy)."""
        return self._engine_supervisor

    def attach_engine_supervisor(self, supervisor: Optional[Any]) -> None:
        """Install the supervisor. Called once, on the driver, at fleet construction."""
        self._engine_supervisor = supervisor

    def mark_dead(self, url: str) -> None:
        """Record that a provisioned backend is unreachable. Local bookkeeping only.

        For a caller that already knows — the engine supervisor, whose own ``/health``
        probe just concluded it — so that the conclusion does not have to be
        rediscovered by ``_reconcile_fleet``'s probe. Unknown or already-dead URLs are
        no-ops, which is what keeps it safe to call once per reconcile.
        """
        if url not in self.server_urls or url in self._disabled_server_urls:
            return
        self._disabled_server_urls.add(url)
        self._membership_generation += 1
        logger.info(
            f"[ft] marked {url} dead. Active {len(self.active_server_urls)}/{len(self.server_urls)} "
            f"(membership generation {self._membership_generation})"
        )

    def mark_alive(self, url: str) -> None:
        """Undo a ``mark_dead``. Local bookkeeping only.

        For a backend the reactive probe wrongly condemned: that probe fires once, with a
        few-seconds bound, at the moment every surviving engine is saturated, so an engine
        that is merely slow is indistinguishable from a dead one. When a later, calmer
        probe disagrees, the eviction has to be reversible or a transient stall costs an
        engine for the rest of the run.

        The caller is responsible for the ROUTER half -- and specifically for not
        re-admitting the engine until it has taken part in a weight sync, since while it
        was disabled it was excluded from the sync fan-out and may be holding stale
        weights (the supervisor returns it via PENDING_SYNC, never straight to LIVE).
        """
        if url not in self._disabled_server_urls:
            return
        self._disabled_server_urls.discard(url)
        self._membership_generation += 1
        logger.info(
            f"[ft] marked {url} alive again. Active {len(self.active_server_urls)}/"
            f"{len(self.server_urls)} (membership generation {self._membership_generation})"
        )

    def replace_server_url(self, slot: int, new_url: str) -> None:
        """Point ``slot`` at a restarted engine's new URL. Local bookkeeping only.

        Called by the engine supervisor once a replacement is healthy. The slot keeps
        its position in ``server_urls`` -- so ``num_provisioned_replicas`` and every
        surviving engine's consumer block are untouched -- and the old URL is dropped
        from the dead set, since the thing it named no longer exists.

        The new engine is left OUT of the router by the supervisor until it has taken
        part in a weight sync; it is deliberately *in* ``active_server_urls`` from
        here, because that is the set the control plane and the weight sync fan out
        over, and taking part in that sync is precisely what it is waiting to do.

        Raises:
            KeyError: no such slot.
            ValueError: ``new_url`` already names a different provisioned slot.
        """
        slots = self.server_slots or []
        try:
            idx = slots.index(slot)
        except ValueError as e:
            raise KeyError(f"slot {slot} is not provisioned (slots={slots}).") from e
        if new_url in self.server_urls and self.server_urls.index(new_url) != idx:
            raise ValueError(
                f"{new_url} is already registered for slot {slots[self.server_urls.index(new_url)]}; "
                f"refusing to give slot {slot} a URL another slot answers on."
            )
        old_url = self.server_urls[idx]
        self.server_urls[idx] = new_url
        self._disabled_server_urls.discard(old_url)
        self._disabled_server_urls.discard(new_url)
        self._membership_generation += 1
        logger.info(
            f"[ft] slot {slot} moved {old_url} -> {new_url}. "
            f"Active {len(self.active_server_urls)}/{len(self.server_urls)} "
            f"(membership generation {self._membership_generation})"
        )

    @property
    def membership_generation(self) -> int:
        """Monotonic counter of membership changes; 0 until the first engine dies."""
        return self._membership_generation

    def sync_membership(self, live_url_slots: Optional[Sequence[Tuple[str, int]]]) -> None:
        """Adopt someone else's membership view. Local bookkeeping only, no I/O.

        For copies of this client that cannot reconcile for themselves: a worker's
        copy was pickled at ``init_weight_sync_state``, so its ``_disabled_server_urls``
        is frozen at "nothing has died". Without this, the rank-0
        ``reset_prefix_cache`` fan-out inside a weight sync would discover the dead
        engine on its own — re-probing the fleet and re-issuing router removals from
        the worker, in the middle of the sync window, for a conclusion the driver
        already reached.

        Keyed on ``(url, slot)`` rather than on URLs alone because a restarted engine
        comes back on a re-reserved port. A pickled copy still holds the URL the dead
        engine answered on, so a URL-only view would look like an unprovisioned
        server and be rejected; the slot says "this is the same engine, at a new
        address", and the copy adopts it.

        ``None`` means the whole provisioned fleet is live.

        Raises:
            ValueError: a slot outside the provisioned set, which would mean the
                fleet geometry moved rather than one engine relocating.
        """
        if live_url_slots is None:
            live = set(self.server_urls)
        else:
            provisioned = set(self.server_slots or [])
            unknown = sorted({s for _, s in live_url_slots} - provisioned)
            if unknown:
                raise ValueError(
                    f"sync_membership got slots outside the provisioned set: {unknown}. "
                    f"Provisioned: {sorted(provisioned)}"
                )
            for url, slot in live_url_slots:
                if self.server_urls[(self.server_slots or []).index(slot)] != url:
                    # A restart moved this slot; adopt the new address. This also
                    # clears the old URL from the dead set (see replace_server_url).
                    self.replace_server_url(slot, url)
            live = {u for u, _ in live_url_slots}
        disabled = {u for u in self.server_urls if u not in live}
        if disabled != self._disabled_server_urls:
            self._disabled_server_urls = disabled
            self._membership_generation += 1

    async def _probe_health(self, url: str) -> bool:
        """``GET /health`` against a backend. True iff it is alive.

        This one function is the ONLY place the fleet decides whether an engine is dead,
        and every caller -- the reactive path and the engine supervisor -- goes through
        it. That is deliberate: the previous design had the reactive path condemn on a
        single short probe while the supervisor required several, and the translation
        layer between those two standards is where a healthy engine got killed.

        Accuracy comes from distinguishing the two failure kinds rather than retrying
        blindly:

        * **Definitive** -- connection refused, or an answer that is not 200. Nothing is
          listening, or something is listening and says it is unwell. A dead engine is
          therefore still detected in MILLISECONDS, with no added latency, which is what
          the retry path must not spoil.
        * **Ambiguous** -- a timeout. This is exactly what a healthy but SATURATED engine
          looks like: the reactive probe fires right after a data-plane failure, when
          every survivor is serving hundreds of concurrent requests, and a few-second
          bound is easy to miss. Observed twice on consecutive GPU runs: one engine was
          killed by the chaos script and a second, healthy one was condemned with it.

        So only the ambiguous case is retried, with the bound doubling each attempt so a
        transient load spike is outlived rather than mistaken for death.
        ``health_failure_threshold`` is the attempt budget -- the same "N misses before I
        believe it" intent it always had, moved into the probe where the evidence is,
        instead of being counted by a caller after the fact.
        """
        assert self.fault_tolerance is not None
        attempts = max(1, int(getattr(self.fault_tolerance, "health_failure_threshold", 3)))
        budget = float(self.fault_tolerance.health_probe_timeout_s)
        session = await self._get_session()

        for attempt in range(1, attempts + 1):
            try:
                async with session.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=budget)) as resp:
                    await resp.read()  # drain so the keep-alive connection is reusable
                    if resp.status == 200:
                        return True
                    # It answered, so it is reachable and self-reporting unwell (e.g. the
                    # engine core died behind a live HTTP server). No amount of waiting
                    # changes that.
                    logger.info(f"[ft] health probe for {url} answered HTTP {resp.status}: treating as dead")
                    return False
            except asyncio.CancelledError:
                raise
            except (asyncio.TimeoutError, TimeoutError):
                if attempt == attempts:
                    logger.warning(
                        f"[ft] health probe for {url} timed out {attempts}x "
                        f"(last budget {budget:.0f}s): treating as dead"
                    )
                    return False
                logger.info(
                    f"[ft] health probe for {url} timed out after {budget:.0f}s "
                    f"(attempt {attempt}/{attempts}); it may just be saturated -- retrying with a "
                    "longer bound before condemning it"
                )
                budget *= 2
                # A brief pause so a load spike can pass rather than being re-measured
                # immediately.
                await asyncio.sleep(0.5)
            except Exception as e:  # noqa: BLE001
                # Refused / DNS / reset: nothing is listening. Definitive.
                logger.debug(f"[ft] health probe for {url} failed definitively: {type(e).__name__}: {e}")
                return False
        return False

    async def _probe_and_disable(self) -> None:
        """Probe every active backend once; disable and de-route the non-responders.

        This is the whole detector. There is no interval, no background task and no
        per-URL failure counter carried across time — it runs only because a request
        already failed. See ``_reconcile_fleet`` for why it is a probe rather than a
        reading of the failure itself.
        """
        assert self.fault_tolerance is not None
        candidates = self.active_server_urls
        if not candidates:
            return
        alive = await asyncio.gather(*[self._probe_health(u) for u in candidates])
        dead = [u for u, ok in zip(candidates, alive) if not ok]
        if not dead:
            logger.info(
                "[ft] fleet probe found no dead engines (%d active); the failure was transient",
                len(candidates),
            )
            return

        self._disabled_server_urls.update(dead)
        self._membership_generation += 1
        remaining = len(candidates) - len(dead)
        logger.warning(
            "[ft] inference engines died: %s. Active %d -> %d (membership generation %d).",
            dead,
            len(candidates),
            remaining,
            self._membership_generation,
        )

        floor = self.fault_tolerance.min_live_engines
        if remaining < floor:
            # Nothing can bring them back in Part 1, so waiting would only delay the
            # same outcome. Skip the router calls: the run is over.
            raise EngineFleetError(
                f"only {remaining} inference engine(s) still alive, below "
                f"fault_tolerance.min_live_engines={floor}. Dead: {dead}"
            )

        # Best-effort de-route. The router's own health checker is the backstop, so a
        # failure here costs latency (requests keep hashing onto a dead backend until
        # it ejects), not correctness — and the URL is already locally disabled.
        await asyncio.gather(*[self._router_remove_worker(u) for u in dead], return_exceptions=True)

    async def _reconcile_fleet(self, seen_generation: Optional[int] = None) -> int:
        """Turn "something in the fleet is broken" into "these URLs are broken".

        Triggered by a caught data-plane failure, never by a timer. A probe rather
        than a reading of the failure itself because the failure arrives from the
        *router* (a 502/503/504), which does not reliably name the backend that died.

        Debounced two ways so a batch of 512 trajectories failing at once costs one
        probe, not 512:

        * callers pass the ``membership_generation`` they failed under; if it has
          already moved, someone else's probe covered them and this returns at once;
        * otherwise they all await one shared probe task.

        Returns the membership generation after reconciling, for the caller to carry
        into its next attempt.
        """
        if not self._ft_enabled:
            return self._membership_generation
        if seen_generation is not None and seen_generation != self._membership_generation:
            return self._membership_generation

        loop = asyncio.get_running_loop()
        # No await between the check and the assignment, so concurrent callers on this
        # loop cannot both create a task.
        task = self._reconcile_task
        if task is None or task.done() or self._reconcile_loop is not loop:
            task = loop.create_task(self._probe_and_disable())
            self._reconcile_task = task
            self._reconcile_loop = loop
        # Shielded: one caller giving up (cancelled trajectory) must not cancel the
        # probe every other caller is waiting on.
        await asyncio.shield(task)
        return self._membership_generation

    # ---------------------------
    # Router worker management
    # ---------------------------

    async def _router_worker_call(self, endpoint: str, url: str) -> Tuple[int, str]:
        """POST a worker-management route on the router. Returns (status, body text).

        The backend URL travels as the ``url`` query parameter and the response body
        is plain text, not JSON — verified against vllm-router 0.1.14.post1, which
        answers ``400 missing field `url``` to any other spelling.
        """
        session = await self._get_session()
        async with session.post(
            f"{self.proxy_url}{endpoint}",
            params={"url": url},
            timeout=aiohttp.ClientTimeout(total=_ROUTER_ADMIN_TIMEOUT_S),
        ) as resp:
            return resp.status, (await resp.text())

    async def _router_remove_worker(self, url: str) -> bool:
        """Drop a backend from the router's ring. Best-effort; never raises.

        Idempotent server-side: the router answers 200 for a URL it does not have.
        ``consistent_hash`` rehashes onto the remaining ring on removal, so sessions
        land on survivors with no policy change.
        """
        try:
            status, body = await self._router_worker_call(_ROUTER_REMOVE_WORKER, url)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[ft] router remove_worker({url}) failed: {type(e).__name__}: {e}")
            return False
        if status != 200:
            logger.warning(f"[ft] router remove_worker({url}) returned HTTP {status}: {body[:200]}")
            return False
        logger.info(f"[ft] removed {url} from the router")
        return True

    async def _router_add_worker(self, url: str) -> bool:
        """Add a backend to the router's ring. Best-effort; never raises.

        Unused in Part 1 (nothing is restarted, so nothing rejoins) — it exists so
        the membership surface is symmetric and testable. Unlike removal the router
        is NOT idempotent here: re-adding a live worker is a 400 ``already exists``,
        which we treat as success since the post-condition holds either way.
        """
        try:
            status, body = await self._router_worker_call(_ROUTER_ADD_WORKER, url)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[ft] router add_worker({url}) failed: {type(e).__name__}: {e}")
            return False
        if status == 400 and "already exists" in body:
            return True
        if status != 200:
            logger.warning(f"[ft] router add_worker({url}) returned HTTP {status}: {body[:200]}")
            return False
        logger.info(f"[ft] added {url} to the router")
        return True

    async def router_list_workers(self) -> List[str]:
        """Backends the router currently routes to (``GET /list_workers``).

        The router's view, which is not necessarily ours: it lags on its own health
        interval, and a worker we removed locally but failed to de-route is still
        here. Used by tests and diagnostics, never as an input to routing decisions.
        """
        session = await self._get_session()
        async with session.get(
            f"{self.proxy_url}{_ROUTER_LIST_WORKERS}",
            timeout=aiohttp.ClientTimeout(total=_ROUTER_ADMIN_TIMEOUT_S),
        ) as resp:
            body = await resp.json(content_type=None)
            raise_for_status(resp, body if isinstance(body, dict) else None)
        return list(body.get("urls", []) if isinstance(body, dict) else body)

    def __post_init__(self):
        if self.data_parallel_size <= 0:
            raise ValueError(f"Expected `data_parallel_size` >0, got {self.data_parallel_size}")

        if len(self.server_urls) % self.data_parallel_size != 0:
            raise ValueError(
                f"Expected number of servers to be divisible by data parallel size, got {self.server_urls} and {self.data_parallel_size}"
            )

        if self.server_slots is None:
            self.server_slots = [i // self.data_parallel_size for i in range(len(self.server_urls))]
        elif len(self.server_slots) != len(self.server_urls):
            raise ValueError(
                f"Expected one slot per server URL, got {len(self.server_slots)} slots "
                f"for {len(self.server_urls)} URLs."
            )

    def get_endpoint_url(self) -> str:
        """Data-plane endpoint base URL (the router/proxy that load-balances requests)."""
        return self.proxy_url

    # ---------------------------
    # Session Management
    # ---------------------------

    def _get_semaphores(self) -> Tuple[Optional[asyncio.Semaphore], Optional[asyncio.Semaphore]]:
        """Get or create the shared generate/detokenize semaphores for this client.

        Semaphores are event-loop-bound (Python 3.10+). If the running loop has
        changed since they were created, recreate them.

        All concurrent generate() calls on the same client instance share these
        semaphores, capping total in-flight requests at
        SKYRL_GENERATE_CONCURRENCY_PER_ENGINE × num_engines.
        """
        current_loop = asyncio.get_running_loop()
        if self._sem_loop is not current_loop:
            if SKYRL_GENERATE_CONCURRENCY_PER_ENGINE > 0:
                concurrency = SKYRL_GENERATE_CONCURRENCY_PER_ENGINE * len(self.server_urls)
                logger.info(f"Capping concurrency for generation to a maximum of {concurrency} requests")
                self._gen_sem = asyncio.Semaphore(concurrency)
                self._detok_sem = asyncio.Semaphore(concurrency)
            else:
                self._gen_sem = None
                self._detok_sem = None
            self._sem_loop = current_loop
        return self._gen_sem, self._detok_sem

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create the aiohttp session."""
        # Re-use the existing session object if it is not closed.
        # Note that we also create a new session object if the event loop has changed, since
        # aiohttp.ClientSession is tied to the event loop.
        current_loop = asyncio.get_running_loop()
        if self._session is not None and not self._session.closed and self._session.loop != current_loop:
            # Event loop changed - the old session is unusable (bound to a dead loop).
            self._session = None
        if self._session is None or self._session.closed:
            # keepalive_timeout must be shorter than the server's timeout_keep_alive
            # (uvicorn default: 5s). Otherwise aiohttp reuses connections the server
            # has already closed, causing ECONNRESET under high concurrency.
            connector = aiohttp.TCPConnector(
                limit=SKYRL_HTTP_CONNECTION_LIMIT,
                keepalive_timeout=2,
            )
            self._session = aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=None))
        return self._session

    async def _post(self, url: str, json: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Any:
        """POST with retry + backoff on transient connection errors.

        Between generate bursts the pool's keep-alive connections go stale
        (server closes them after ``timeout_keep_alive``).  An immediate
        retry would grab another stale connection from the same pool, so we
        sleep briefly to let the connector detect and purge dead sockets
        before the next attempt.

        Two of the retryable classes are new and apply **unconditionally**, because
        they were always transport failures that a retry is the right answer to:
        ``ClientPayloadError`` (a response truncated mid-body) and
        ``asyncio.TimeoutError`` (which also covers ``ServerTimeoutError``).
        ``ClientConnectorError`` is already a ``ClientOSError`` subclass.

        Under fault tolerance, two more behaviours switch on:

        * a router 502/503/504 is retried. This is the actual signature of a dead
          backend — the router answers, the engine behind it does not — and without
          it ``raise_for_status`` raises straight out and kills the trajectory.
        * a per-request total timeout (the shared session has none), so a *wedged*
          engine — one that accepts the connection and never answers — becomes a
          bounded failure instead of an eternal hang.

        On the first retryable failure the fleet is reconciled before backing off,
        so the retry is issued after the dead backend has left the router's ring
        rather than hashing straight back onto it.
        """
        session = await self._get_session()
        post_kwargs: Dict[str, Any] = {}
        if self._ft_enabled and self.fault_tolerance.request_timeout_s is not None:
            post_kwargs["timeout"] = aiohttp.ClientTimeout(total=self.fault_tolerance.request_timeout_s)
        last_exc: Optional[Exception] = None
        reconciled = False
        generation = self._membership_generation
        for attempt in range(_DATA_PLANE_RETRIES):
            # INVARIANT: exactly one place below decides "retryable", and exactly one
            # tail reconciles + backs off. Do not add an early `continue` here.
            try:
                async with session.post(url, json=json, headers=headers, **post_kwargs) as resp:
                    try:
                        body = await resp.json(content_type=None)
                    except Exception as e:
                        if 400 <= resp.status < 500:
                            # Non-JSON client error (e.g. plain text 422 from vllm-router).
                            # Raise immediately — client errors won't succeed on retry.
                            text = await resp.text()
                            raise aiohttp.ClientResponseError(
                                resp.request_info,
                                resp.history,
                                status=resp.status,
                                message=text or resp.reason,
                                headers=resp.headers,
                            )
                        # Non-JSON 5xx: retryable, as it always was. Under FT it is
                        # re-raised carrying the STATUS so the classifier below sees a
                        # dead backend, instead of a JSONDecodeError that names the
                        # parser rather than the cause.
                        #
                        # This branch used to `continue` straight back to the top of
                        # the loop, skipping the reconcile below -- and since the
                        # router answers PLAIN TEXT 5xx when its backend is gone, that
                        # made the dead-engine signature the one path through _post
                        # that never triggered detection. Measured: 30 retries, no
                        # probe, 33s lost, and a JSONDecodeError at the end. Every
                        # retryable outcome now funnels through the single tail below.
                        if self._ft_enabled:
                            text = await resp.text()
                            raise aiohttp.ClientResponseError(
                                resp.request_info,
                                resp.history,
                                status=resp.status,
                                message=text or resp.reason,
                                headers=resp.headers,
                            ) from e
                        last_exc = e
                    else:
                        raise_for_status(resp, body)
                        return body
            except aiohttp.ClientResponseError as e:
                if not (self._ft_enabled and e.status in _ROUTER_BACKEND_FAILURE_STATUSES):
                    raise
                last_exc = e
            except (
                aiohttp.ServerDisconnectedError,
                aiohttp.ClientOSError,
                aiohttp.ClientPayloadError,
                asyncio.TimeoutError,
            ) as e:
                last_exc = e
            logger.debug(f"POST retry {attempt + 1}/{_DATA_PLANE_RETRIES} for {url=}: {last_exc}")
            if not reconciled:
                reconciled = True
                generation = await self._reconcile_fleet(seen_generation=generation)
            await asyncio.sleep(1)
        raise last_exc  # type: ignore[misc]

    # ---------------------------
    # Data Plane
    # ---------------------------

    def _resolve_model(self, model: Optional[str], method_name: str) -> str:
        """Pick the target model name for a data-plane call.

        - If ``model`` is non-empty, use it as-is.
        - Otherwise, when LoRA is in use (``uses_lora_weight_sync=True``) raise
          ``ValueError`` — the caller must name the adapter explicitly because
          falling back to the base model would silently bypass LoRA.
        - Otherwise return ``self.model_name`` (the base model the server was
          started with).
        """
        if model:
            return model
        if self.uses_lora_weight_sync:
            raise ValueError(
                f"RemoteInferenceClient.{method_name}: `model` is required when LoRA "
                f"is enabled (uses_lora_weight_sync=True). Pass the LoRA adapter name "
                f"explicitly so the request doesn't silently target the base model."
            )
        return self.model_name

    async def generate(
        self,
        input_batch: InferenceEngineInput,
        model: Optional[str] = None,
    ) -> InferenceEngineOutput:
        """
        Generate completions via /v1/completions.

        This is the interface for token-in-token-out workflows. Input will have
        token ids, and the output is token ids as well.

        Each prompt is sent as a separate request to allow the router to route
        based on session_id. All requests are made in parallel.

        With keep-mode pause, in-flight requests are frozen and resume
        transparently after /resume -- no client-side retry needed.

        Args:
            input_batch: Contains prompt_token_ids, sampling_params, and optional session_ids.
            model: Optional model identifier — the base model name or a loaded
                LoRA adapter name. When omitted, defaults to ``self.model_name``
                if LoRA is not in use; raises ``ValueError`` if it is.

        Returns:
            InferenceEngineOutput with responses, response_ids, and stop_reasons.
        """
        model = self._resolve_model(model, "generate")

        prompt_token_ids = input_batch.get("prompt_token_ids")
        if prompt_token_ids is None:
            raise ValueError("RemoteInferenceClient only accepts `prompt_token_ids`, not `prompts`.")

        sampling_params = input_batch.get("sampling_params") or {}
        if sampling_params.get("n", 1) > 1:
            raise ValueError("n > 1 is not supported. Use `config.generator.n_samples_per_prompt` instead.")

        session_ids = input_batch.get("session_ids")
        mm_features = input_batch.get("mm_features")
        cache_salt = input_batch.get("cache_salt")
        get_logprobs = sampling_params.get("logprobs") is not None

        # Two semaphores decouple the generate and detokenize stages:
        #   gen_sem:   limits concurrent in-flight generate requests so we don't
        #              overwhelm the router/vLLM scheduler.  Released as soon as
        #              generation finishes, so the GPU slot is freed immediately.
        #   detok_sem: limits concurrent detokenize calls independently.  Uses the
        #              same concurrency limit so detokenize never starves generate.
        # Semaphores are shared across all concurrent generate() calls on this client
        # instance, so total in-flight requests are capped at
        # SKYRL_GENERATE_CONCURRENCY_PER_ENGINE × num_engines regardless of how many
        # callers invoke generate() simultaneously.
        # TODO (sumanthrh) (RemoteInferenceClient data-plane-deprecation): We should move this outside of the client to a runner abstraction that will also parallelize client requests across processes.
        gen_sem, detok_sem = self._get_semaphores()
        batch_size = len(prompt_token_ids)

        async def _throttled_generate(idx: int) -> Dict[str, Any]:
            if gen_sem is None:
                return await self._generate_single(
                    prompt_token_ids=prompt_token_ids[idx],
                    sampling_params=sampling_params,
                    session_id=session_ids[idx] if session_ids and idx < len(session_ids) else None,
                    mm_features=mm_features[idx] if mm_features and idx < len(mm_features) else None,
                    model=model,
                    cache_salt=cache_salt,
                )
            async with gen_sem:
                return await self._generate_single(
                    prompt_token_ids=prompt_token_ids[idx],
                    sampling_params=sampling_params,
                    session_id=session_ids[idx] if session_ids and idx < len(session_ids) else None,
                    mm_features=mm_features[idx] if mm_features and idx < len(mm_features) else None,
                    model=model,
                    cache_salt=cache_salt,
                )

        async def _throttled_detokenize(token_ids: List[int]) -> str:
            if detok_sem is None:
                return (await self.detokenize([token_ids]))[0]
            async with detok_sem:
                return (await self.detokenize([token_ids]))[0]

        raw_results = await asyncio.gather(*[_throttled_generate(idx) for idx in range(batch_size)])
        responses = await asyncio.gather(*[_throttled_detokenize(r["response_ids"]) for r in raw_results])

        rollout_expert_indices = [r.get("routed_experts") for r in raw_results]
        has_routed_experts = any(x is not None for x in rollout_expert_indices)

        return InferenceEngineOutput(
            responses=responses,
            stop_reasons=[r["stop_reason"] for r in raw_results],
            response_ids=[r["response_ids"] for r in raw_results],
            response_logprobs=[r["response_logprobs"] for r in raw_results] if get_logprobs else None,
            rollout_expert_indices=rollout_expert_indices if has_routed_experts else None,
        )

    async def _generate_single(
        self,
        prompt_token_ids: List[int],
        sampling_params: Dict[str, Any],
        session_id: Optional[Any],
        model: str,
        mm_features: Optional[MultiModalFeatures] = None,
        cache_salt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate completion for a single prompt.

        A keep-mode pause needs no retry: in-flight requests are frozen by the vLLM
        scheduler and resume where they left off after /resume.

        An engine DEATH is a different matter, and ``_post`` below does retry it --
        reissuing this same request, with this same ``X-Session-ID``, through the
        router. That preserves the trajectory (earlier turns and env state survive;
        only this turn's generation is redone), and it is why the fleet reconcile
        happens before the first backoff: routing is ``consistent_hash`` on the
        session id, so a retry against a ring that still contains the dead worker
        would hash straight back onto it. Only if ``_post`` exhausts its attempts
        does the whole trajectory get re-run (``_agent_loop_with_retry``).

        Returns:
            Dict with keys: stop_reason, response_ids, response_logprobs
        """
        url = (
            f"{self.proxy_url}/skyrl/v1/generate"
            if self.enable_return_routed_experts
            else f"{self.proxy_url}/inference/v1/generate"
        )

        payload: dict[str, Any] = {
            "sampling_params": sampling_params,
            "model": model,
            "token_ids": prompt_token_ids,
        }
        if mm_features:
            payload["features"] = mm_features
        # `cache_salt` is a top-level request field (forwarded to vLLM's TokensPrompt), not a sampling
        # param.
        if cache_salt is not None:
            payload["cache_salt"] = cache_salt

        headers = {"Content-Type": "application/json"}
        if session_id:
            headers["X-Session-ID"] = str(session_id)

        response = await self._post(url, json=payload, headers=headers)

        choice = response["choices"][0]
        token_ids = choice["token_ids"]
        stop_reason = choice["finish_reason"]

        response_logprobs: Optional[List[float]] = None
        logprobs = choice.get("logprobs")
        if logprobs is not None:
            logprobs_content = logprobs.get("content", [])
            if logprobs_content:
                response_logprobs = [logprob_info["logprob"] for logprob_info in logprobs_content]

        routed_experts = choice.get("routed_experts")

        return {
            "stop_reason": stop_reason,
            "response_ids": token_ids,
            "response_logprobs": response_logprobs,
            "routed_experts": routed_experts,
        }

    async def _render_for_sample(
        self,
        prompt: Dict[str, Any],
        session_id: Optional[str],
        model: str,
    ) -> Tuple[List[int], Optional[MultiModalFeatures]]:
        """Build token_ids and optional multi-modal features from a Tinker prompt.

        For text-only prompts this simply flattens chunk tokens (no HTTP call).
        When image chunks are present, calls /v1/chat/completions/render to
        process images, then splices the resulting placeholder tokens into the
        pre-tokenized text stream and adjusts placeholder offsets.

        Returns:
            (token_ids, features) where features is None for text-only prompts.
        """
        chunks = prompt.get("chunks", [])

        # No images → flatten text tokens directly.
        image_chunks = [c for c in chunks if c.get("type") in ("image", "image_asset_pointer")]
        if not image_chunks:
            token_ids = [tok for c in chunks for tok in c.get("tokens", [])]
            return token_ids, None

        # Build OpenAI chat template with only image_urls
        content_parts: List[Dict[str, Any]] = []
        for c in image_chunks:
            if c["type"] == "image":
                # model_dump() on Base64Bytes produces bytes with the b64 string.
                raw = c["data"]
                b64_str = raw.decode("ascii") if isinstance(raw, bytes) else raw
                url = f"data:image/{c.get('format', 'jpeg')};base64,{b64_str}"
            else:  # image_asset_pointer
                url = c["location"]
            content_parts.append({"type": "image_url", "image_url": {"url": url}})

        render_payload: Dict[str, Any] = {
            "json": {
                "model": model,
                "messages": [{"role": "user", "content": content_parts}],
            }
        }
        if session_id:
            render_payload["json"]["session_id"] = session_id

        render_resp = await self.render_chat_completion(render_payload)

        # Extract per-image placeholder token slices from the render output.
        features = render_resp.get("features") or {}
        render_token_ids = render_resp.get("token_ids", [])
        render_placeholders = features.get("mm_placeholders", {}).get("image", [])

        placeholder_token_slices: List[List[int]] = []
        for ph in render_placeholders:
            offset, length = ph["offset"], ph["length"]
            placeholder_token_slices.append(render_token_ids[offset : offset + length])

        if len(placeholder_token_slices) != len(image_chunks):
            raise ValueError(
                f"Expected {len(image_chunks)} placeholder token slices, got {len(placeholder_token_slices)}"
            )

        # Splice: walk chunks in order, substituting image placeholder tokens.
        final_token_ids: List[int] = []
        new_placeholders: List[MMPlaceholderRangeInfo] = []
        img_idx = 0

        for c in chunks:
            ctype = c.get("type", "encoded_text")
            if ctype == "encoded_text":
                final_token_ids.extend(c.get("tokens", []))
            elif ctype in ("image", "image_asset_pointer"):
                ph_tokens = placeholder_token_slices[img_idx]
                new_placeholders.append({"offset": len(final_token_ids), "length": len(ph_tokens)})
                final_token_ids.extend(ph_tokens)
                img_idx += 1

        # No need to decode, vllm handles decoding
        adjusted_features: MultiModalFeatures = {
            "mm_hashes": features.get("mm_hashes", {}),
            "mm_placeholders": {"image": new_placeholders},
            "kwargs_data": features.get("kwargs_data"),
        }

        return final_token_ids, adjusted_features

    async def sample(
        self,
        request_payload: SampleRequestPayload,
    ) -> SampleResponse:
        """
        Sample completions via /inference/v1/generate (Tinker API).

        Maps Tinker-style sample requests to the vLLM generate endpoint.
        Uses self._post() for automatic retry + backoff on transient errors.

        Args:
            request_payload: SampleRequestPayload with {"json": <request-body>}.
                Expected keys in json: prompt, num_samples, sampling_params,
                session_id, include_prompt_logprobs (bool), topk_prompt_logprobs (int).
                ``model`` is optional and resolved via ``_resolve_model``.

        Returns:
            SampleResponse with type="sample", sequences list, prompt_logprobs, and topk_prompt_logprobs.
        """
        session_id, body = _extract_session_id_and_body(request_payload)
        model = self._resolve_model(body.get("model"), "sample")
        body["model"] = model

        prompt = body.get("prompt", {})
        num_samples = body.get("num_samples", 1)
        tinker_params = body.get("sampling_params", {})

        # Note: Tinker SampleRequest uses "prompt_logprobs" (bool), while
        # SamplingClient.sample() uses "include_prompt_logprobs".
        include_prompt_logprobs = body.get("include_prompt_logprobs", body.get("prompt_logprobs", False))
        topk_prompt_logprobs_k = body.get("topk_prompt_logprobs", 0)

        # vLLM prompt logprob mapping
        prompt_logprobs_sp = None
        if include_prompt_logprobs:
            prompt_logprobs_sp = topk_prompt_logprobs_k if topk_prompt_logprobs_k > 0 else 0

        # Render prompt: flatten text tokens and, if images are present,
        # call the render endpoint to get placeholder tokens + features.
        token_ids, mm_features = await self._render_for_sample(prompt, session_id, model=model)

        # Map Tinker SamplingParams → vLLM format
        sampling_params: Dict[str, Any] = {
            "n": num_samples,
            "logprobs": 0,
            "output_kind": 2,
            "prompt_logprobs": prompt_logprobs_sp,
        }

        for tinker_key, vllm_key in _TINKER_SAMPLE_TO_VLLM_PARAM_MAP.items():
            val = tinker_params.get(tinker_key)
            if val is not None:
                sampling_params[vllm_key] = val

        payload: Dict[str, Any] = {
            "sampling_params": sampling_params,
            "model": model,
            "token_ids": token_ids,
        }
        if mm_features is not None:
            payload["features"] = mm_features

        headers = {"Content-Type": "application/json"}
        if session_id:
            headers["X-Session-ID"] = str(session_id)

        url = f"{self.proxy_url}/inference/v1/generate"
        gen_sem, _ = self._get_semaphores()
        if gen_sem is None:
            response = await self._post(url, json=payload, headers=headers)
        else:
            async with gen_sem:
                response = await self._post(url, json=payload, headers=headers)

        # vLLM returns: list[dict[str(token_id) → {"logprob": float, ...}] | None]
        result_prompt_logprobs: Optional[List[Optional[float]]] = None
        result_topk_prompt_logprobs: Optional[List[Optional[List[Tuple[int, float]]]]] = None

        raw_prompt_logprobs = response.get("prompt_logprobs")
        if raw_prompt_logprobs is not None and include_prompt_logprobs:
            result_prompt_logprobs = [
                (pos_dict.get(str(tid)) or {}).get("logprob") if pos_dict is not None else None
                for tid, pos_dict in zip(token_ids, raw_prompt_logprobs)
            ]
            if topk_prompt_logprobs_k > 0:
                # vLLM returns k or k+1 logprobs per position (the extra entry is the
                # prompt token when it falls outside the top-k). Tinker always returns
                # exactly top-k, so we sort and truncate below.
                result_topk_prompt_logprobs = [
                    (
                        sorted(
                            [(int(tid), entry["logprob"]) for tid, entry in pos_dict.items()],
                            key=lambda x: x[1],
                            reverse=True,
                        )[:topk_prompt_logprobs_k]
                        if pos_dict is not None
                        else None
                    )
                    for _, pos_dict in zip(token_ids, raw_prompt_logprobs)
                ]

        # Transform response choices → sequences
        sequences = []
        for choice in response.get("choices", []):
            seq_logprobs: Optional[List[float]] = None
            logprobs_data = choice.get("logprobs")
            if logprobs_data is not None:
                logprobs_content = logprobs_data.get("content", [])
                if logprobs_content:
                    seq_logprobs = [lp["logprob"] for lp in logprobs_content]

            sequences.append(
                {
                    "tokens": choice["token_ids"],
                    "logprobs": seq_logprobs,
                    "stop_reason": choice.get("finish_reason"),
                }
            )

        return {
            "type": "sample",
            "sequences": sequences,
            "prompt_logprobs": result_prompt_logprobs,
            "topk_prompt_logprobs": result_topk_prompt_logprobs,
        }

    async def chat_completion(
        self,
        request_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Chat completion via /v1/chat/completions.

        Args:
            request_payload: Dict with {"json": <request-body>, "headers": <headers-dict>}.
                The request body must be an OpenAI-compatible chat completion
                request. ``model`` is optional and resolved via
                ``_resolve_model``; if omitted the body is mutated to inject the
                resolved value before forwarding to vLLM. ``session_id`` can be
                included in the body for consistent routing.

        Returns:
            OpenAI-compatible chat completion response.
        """
        session_id, body = _extract_session_id_and_body(request_payload)
        body["model"] = self._resolve_model(body.get("model"), "chat_completion")

        headers = {"Content-Type": "application/json"}
        if session_id:
            headers["X-Session-ID"] = str(session_id)

        url = f"{self.proxy_url}/v1/chat/completions"
        gen_sem, _ = self._get_semaphores()
        if gen_sem is None:
            return await self._post(url, json=body, headers=headers)
        else:
            async with gen_sem:
                return await self._post(url, json=body, headers=headers)

    async def render_chat_completion(
        self,
        request_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Render a chat completion (apply chat template + tokenize) via /v1/chat/completions/render.

        Args:
            request_payload: Dict with {"json": <request-body>}.
                The request body should be OpenAI-compatible chat completion
                request. ``model`` is optional and resolved via
                ``_resolve_model``. session_id can be included in json for
                consistent routing.

        Returns:
            Rendered chat completion response (template-applied prompt and token IDs).
        """
        session_id, body = _extract_session_id_and_body(request_payload)
        body["model"] = self._resolve_model(body.get("model"), "render_chat_completion")

        headers = {"Content-Type": "application/json"}
        if session_id:
            headers["X-Session-ID"] = str(session_id)

        url = f"{self.proxy_url}/v1/chat/completions/render"
        gen_sem, _ = self._get_semaphores()
        if gen_sem is None:
            return await self._post(url, json=body, headers=headers)
        else:
            async with gen_sem:
                return await self._post(url, json=body, headers=headers)

    async def completion(
        self,
        request_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Completion via /v1/completions.

        Args:
            request_payload: Dict with {"json": <request-body>, "headers": <headers-dict>}.
                The request body should be OpenAI-compatible completion
                request. ``model`` is optional and resolved via
                ``_resolve_model``. session_id can be included in json for
                consistent routing.

        Returns:
            OpenAI-compatible completion response.
        """
        session_id, body = _extract_session_id_and_body(request_payload)
        body["model"] = self._resolve_model(body.get("model"), "completion")

        headers = {"Content-Type": "application/json"}
        if session_id:
            headers["X-Session-ID"] = str(session_id)

        url = f"{self.proxy_url}/v1/completions"
        gen_sem, _ = self._get_semaphores()
        if gen_sem is None:
            return await self._post(url, json=body, headers=headers)
        else:
            async with gen_sem:
                return await self._post(url, json=body, headers=headers)

    async def tokenize(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
    ) -> List[List[int]]:
        """
        Tokenize texts.

        Uses the local tokenizer if available, otherwise falls back to HTTP /tokenize.

        Args:
            texts: List of texts to tokenize.
            add_special_tokens: Whether to add special tokens.

        Returns:
            List of token ID lists.
        """
        if self.tokenizer is not None:
            return self.tokenizer(texts, add_special_tokens=add_special_tokens)["input_ids"]

        url = f"{self.proxy_url}/tokenize"

        # vLLM /tokenize expects individual requests, batch them
        results = []
        for text in texts:
            payload = {
                "model": self.model_name,
                "prompt": text,
                "add_special_tokens": add_special_tokens,
            }
            result = await self._post(url, json=payload)
            results.append(result.get("tokens", []))

        return results

    async def detokenize(
        self,
        token_ids: List[List[int]],
    ) -> List[str]:
        """
        Detokenize token IDs.

        Uses the local tokenizer if available, otherwise falls back to HTTP /detokenize.

        Args:
            token_ids: List of token ID lists.

        Returns:
            List of decoded texts.
        """
        if self.tokenizer is not None:
            return self.tokenizer.batch_decode(token_ids)

        url = f"{self.proxy_url}/detokenize"

        # vLLM /detokenize expects individual requests, batch them
        results = []
        for ids in token_ids:
            payload = {
                "model": self.model_name,
                "tokens": ids,
            }
            result = await self._post(url, json=payload)
            results.append(result.get("prompt", ""))

        return results

    async def finish_session(self, session_id: str) -> None:
        """Notify the router that a session (trajectory) is complete.

        Best-effort data-plane call to the router's ``/finish_session`` endpoint.
        Session-aware routing policies (e.g. ``sticky_least_loaded``)
        use this to release the replica capacity held by the session so that new
        trajectories are balanced onto less-busy engines.

        Failures are logged but never raised: this runs in trajectory cleanup
        paths (``finally`` blocks, cancellation handlers) and must not mask the
        original outcome. Routers/policies that don't track sessions treat this
        as a no-op, and unknown session ids are ignored server-side.
        """
        if not session_id:
            return
        url = f"{self.proxy_url}/finish_session"
        try:
            session = await self._get_session()
            # Bound this best-effort cleanup call: the shared session has no
            # timeout (total=None), so an unresponsive router would otherwise
            # hang the trajectory's finally block forever and wedge the loop.
            async with session.post(
                url,
                params={"session_id": str(session_id)},
                timeout=aiohttp.ClientTimeout(total=10.0),
            ) as resp:
                # Drain the body so the keep-alive connection can be reused.
                await resp.read()
                if resp.status >= 400:
                    logger.warning(f"finish_session for session_id={session_id!r} returned HTTP {resp.status}")
        except asyncio.TimeoutError:
            logger.warning(f"finish_session for session_id={session_id!r} timed out after 10s (router unresponsive)")
        except Exception as e:
            logger.warning(f"finish_session for session_id={session_id!r} failed: {e}")

    # ---------------------------
    # Control Plane (fan-out to all server_urls)
    # ---------------------------

    async def _call_server(
        self,
        server_url: str,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        method: str = "POST",
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Call endpoint on a single server.

        Args:
            server_url: Base URL of the server.
            endpoint: Endpoint path (e.g., "/pause").
            json: JSON payload to send as request body.
            method: HTTP method (default: POST).
            params: URL query parameters (e.g., for FastAPI Query() params).

        Returns:
            Tuple of (server_url, {"status": <int>, "body": <response>}).
        """
        session = await self._get_session()
        url = f"{server_url}{endpoint}"
        async with session.request(method, url, json=json, params=params) as resp:
            body = await resp.json() if resp.content_length else None
            raise_for_status(resp, body)
            return server_url, {"status": resp.status, "body": body}

    async def _call_all_servers(
        self,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        method: str = "POST",
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Call endpoint on all server_urls concurrently.

        Args:
            endpoint: Endpoint path (e.g., "/pause").
            json: JSON payload to send as request body.
            method: HTTP method (default: POST).
            params: URL query parameters (e.g., for FastAPI Query() params).

        Returns:
            Dict mapping server_url to response. Under fault tolerance this may be a
            SUBSET of the active set — see below.
        """
        urls = self.active_server_urls
        if not self._ft_enabled:
            results = await asyncio.gather(*[self._call_server(url, endpoint, json, method, params) for url in urls])
            return {url: resp for url, resp in results}

        if not urls:
            # Reachable only after a reconcile that already breached the floor (it
            # disables the URLs before raising). Returning {} here would read as
            # "fan-out succeeded" to every caller.
            raise EngineFleetError(f"{endpoint}: every inference server is marked dead ({self.server_urls}).")

        # [FT] Degrade instead of abort. A bare gather aborts the whole fan-out on the
        # first failure, so one dead engine breaks pause/resume/reset_prefix_cache
        # fleet-wide — and pause/resume bracket every non-colocated weight sync, which
        # means a dead engine breaks the sync path before RDT is even involved.
        outcomes = await asyncio.gather(
            *[self._call_server(url, endpoint, json, method, params) for url in urls],
            return_exceptions=True,
        )
        ok: Dict[str, Any] = {}
        failed: List[Tuple[str, BaseException]] = []
        for url, outcome in zip(urls, outcomes):
            if isinstance(outcome, BaseException):
                if isinstance(outcome, asyncio.CancelledError):
                    raise outcome
                failed.append((url, outcome))
            else:
                ok[outcome[0]] = outcome[1]
        if failed:
            logger.warning(
                "[ft] %s failed on %d/%d active servers: %s",
                endpoint,
                len(failed),
                len(urls),
                [(u, f"{type(e).__name__}: {e}") for u, e in failed],
            )
            # Reconcile first even when nothing answered: the probe's verdict ("the
            # whole fleet is gone") is a better error than one server's transport
            # exception, and it is the one that stops the run.
            await self._reconcile_fleet()
            if not ok:
                # Returning {} would read as "fan-out succeeded, no servers" to every
                # caller; surface the first failure instead.
                raise failed[0][1]
        return ok

    async def pause(self, mode: Union[PauseMode, str] = PauseMode.KEEP, clear_cache: bool = False) -> Dict[str, Any]:
        """
        Pause generation on all backends.

        Args:
            mode: Pause mode determining how in-flight requests are handled.
                Can be a PauseMode enum or string ("abort", "keep", "wait").
                - KEEP / "keep": Freeze in-flight requests in the scheduler.
                    They resume where they left off on /resume. KV cache is
                    preserved. No retry needed. (default)
                - ABORT / "abort": Abort in-flight requests immediately. Clients
                    receive partial tokens and must retry with accumulated context.
                - WAIT / "wait": Wait for in-flight requests to complete before
                    pausing. New requests are blocked. No retry needed.
            clear_cache: Whether to clear the KV cache on pause. Defaults to False.

        Returns:
            Dict mapping server_url to response.
        """
        if isinstance(mode, str):
            mode = PauseMode(mode.lower())

        params: Dict[str, Any] = {"mode": mode.value, "clear_cache": str(clear_cache).lower()}

        return await self._call_all_servers("/pause", params=params)

    async def resume(self) -> Dict[str, Any]:
        """Resume generation on all backends."""
        return await self._call_all_servers("/resume")

    async def pause_generation(self, clear_cache: bool = False) -> Dict[str, Any]:
        """Pause using keep mode."""
        return await self.pause(mode=PauseMode.KEEP, clear_cache=clear_cache)

    async def resume_generation(self) -> Dict[str, Any]:
        """Resume after pause."""
        return await self.resume()

    async def sleep(self, level: int = 2, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Put all backends to sleep (offload weights to CPU).

        Args:
            level: Sleep level (1 or 2). Level 2 offloads more aggressively.
            tags: Optional list of tags to sleep specific resources.
                Common tags: ["weights"], ["kv_cache"], or None for all.

        Returns:
            Dict mapping server_url to response.
        """
        # Mirror BaseVLLMInferenceEngine.sleep: when the trainer syncs LoRA adapters
        # only, force level=1 so the base model survives via CPU backup. level=2
        # discards weights with no source to restore from on wake_up(["weights"]).
        if self.uses_lora_weight_sync and level != 1:
            logger.info(
                "Forcing sleep level=1 (uses_lora_weight_sync=True); requested level=%d would discard the base model.",
                level,
            )
            level = 1
        params: Dict[str, Any] = {"level": str(level)}
        if tags:
            params["tags"] = tags
        return await self._call_all_servers("/sleep", params=params)

    async def wake_up(self, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Wake up all backends (load weights back to GPU).

        Args:
            tags: Optional list of tags to wake up specific resources.
                Common tags: ["weights"], ["kv_cache"], or None for all.
        """
        params = {"tags": tags} if tags else {}
        return await self._call_all_servers("/wake_up", params=params)

    async def reset_prefix_cache(
        self,
        reset_running_requests: bool = False,
    ) -> Dict[str, Any]:
        """
        Reset KV cache on all backends.

        Args:
            reset_running_requests: Whether to reset running requests.

        Returns:
            Dict mapping server_url to response.
        """
        return await self._call_all_servers("/reset_prefix_cache", {"reset_running_requests": reset_running_requests})

    # ---------------------------
    # Weight Sync (control plane - fan-out)
    # ---------------------------

    async def init_weight_update_communicator(
        self,
        init_info: "WeightSyncInitInfo",
    ) -> Dict[str, Any]:
        """
        Initialize weight sync via vLLM native /init_weight_transfer_engine.

        Fetches per-server world sizes, expands init_info into per-server
        payloads (with correct NCCL rank offsets), and fans out to all servers.

        Args:
            init_info: A WeightSyncInitInfo (e.g. BroadcastInitInfo) that supports
                for_servers() and to_api_payload().

        Returns:
            Dict mapping server_url to response.
        """
        _, world_size_per_server = await self.get_world_size()
        num_servers = len(self.server_urls)
        server_infos = init_info.for_servers(world_size_per_server, num_servers, dp_size=self.data_parallel_size)
        payloads = [{"init_info": x.to_api_payload()} for x in server_infos]
        results = await asyncio.gather(
            *[
                self._call_server(url, "/init_weight_transfer_engine", payload)
                for url, payload in zip(self.server_urls, payloads)
            ]
        )
        return {url: resp for url, resp in results}

    async def update_named_weights(
        self,
        update_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Update model weights via vLLM native /update_weights. Used for full parameter fine-tuning.

        For LoRA weight sync, use load_lora_adapter() instead.

        Args:
            update_info: Dict with keys expected by vLLM (names, dtype_names, shapes, packed, etc.)

        Returns:
            Dict mapping server_url to response.
        """
        return await self._call_all_servers(
            "/update_weights",
            {"update_info": update_info},
        )

    # TODO: Once https://github.com/vllm-project/vllm/pull/39212 lands, switch
    # these three methods from /collective_rpc to the native vLLM endpoints
    # (/start_weight_update, /update_weights, /finish_weight_update) and remove
    # the NewInferenceWorkerWrap worker extension.

    async def start_weight_update(
        self,
        is_checkpoint_format: bool = True,
    ) -> Dict[str, Any]:
        """
        Start a new chunked weight update via /collective_rpc.

        Calls the NewInferenceWorkerWrap.skyrl_start_weight_update method on all
        workers. For checkpoint-format weights this initializes layerwise
        reload. Must be called before any update_weights_ipc calls.

        Args:
            is_checkpoint_format: True if weights are in checkpoint format
                (need layerwise processing), False for kernel format.

        Returns:
            Dict mapping server_url to response.
        """
        return await self._call_all_servers(
            "/collective_rpc",
            {
                "method": "skyrl_start_weight_update",
                "kwargs": {"is_checkpoint_format": is_checkpoint_format},
            },
        )

    async def update_weights_ipc(
        self,
        update_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Send a single weight chunk via /collective_rpc.

        Calls NewInferenceWorkerWrap.update_weights_ipc on all workers.
        Can be called multiple times between skyrl_start_weight_update and
        skyrl_finish_weight_update.

        Args:
            update_info: Dict with backend-specific update info (names,
                dtype_names, shapes, ipc_handles_pickled or packed flag).

        Returns:
            Dict mapping server_url to response.
        """
        return await self._call_all_servers(
            "/collective_rpc",
            {
                "method": "update_weights_ipc",
                "kwargs": {"update_info": update_info},
            },
        )

    async def update_weights_nccl(
        self,
        update_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Send batched weight update via /collective_rpc to the broadcast receiver.

        Calls NewInferenceWorkerWrap.update_weights_nccl on all workers,
        which routes weight_transfer_engine.receive_weights through the
        set_current_vllm_config wrap. Used by the broadcast (NCCL) sender as
        a temporary substitute for vLLM's native /update_weights endpoint
        until the upstream patch (vllm-project/vllm weight-sync-fix) lands.

        Args:
            update_info: Dict with backend-specific update info (names,
                dtype_names, shapes, packed flag, etc.) — same shape vLLM's
                native /update_weights expects.

        Returns:
            Dict mapping server_url to response.
        """
        return await self._call_all_servers(
            "/collective_rpc",
            {
                "method": "update_weights_nccl",
                "kwargs": {"update_info": update_info},
            },
        )

    async def finish_weight_update(self) -> Dict[str, Any]:
        """
        Finish the current chunked weight update via /collective_rpc.

        Calls NewInferenceWorkerWrap.skyrl_finish_weight_update on all workers.
        For checkpoint-format weights, runs layerwise postprocessing.

        Returns:
            Dict mapping server_url to response.
        """
        return await self._call_all_servers(
            "/collective_rpc",
            {"method": "skyrl_finish_weight_update"},
        )

    async def init_weight_transfer_engine_rdt(
        self,
        init_info: Dict[str, Any],
        targets: Optional[Sequence[Tuple[str, int]]] = None,
    ) -> Dict[str, Any]:
        """
        Initialize the sharded_rdt weight-transfer engine on all workers.

        Routed via /collective_rpc to NewInferenceWorkerWrap.init_weight_transfer_engine_rdt
        (rather than the native /init_weight_transfer_engine endpoint) so the
        worker can inject the model/device and run the bake under
        set_current_vllm_config — neither of which the native GPUWorker path does.

        Each independent inference DEPLOYMENT has its own self-contained parallel
        config, so the vLLM ``ShardedRDTWeightTransferEngine`` can't tell
        deployments apart on its own — every deployment's internal worker index
        restarts at 0 and would collide under the M:N block assignment. So (unlike
        the fully-identical broadcast NCCL init) we fan out a PER-SERVER payload
        here, stamping each server with its DEPLOYMENT ordinal as ``replica_rank``
        (with ``num_replicas`` = the deployment count). The engine offsets its
        consumers into a globally distinct range from those two fields; everything
        else in the payload is shared.

        ``server_urls`` holds ``num_deployments * data_parallel_size`` entries —
        the ``data_parallel_size`` servers of one deployment share a parallel
        config in which vLLM already assigns each a distinct
        ``data_parallel_index``, so they must share ONE ``replica_rank`` (their
        deployment's) and let ``_global_worker_index`` separate them. Hence the
        replica ordinal is ``server_index // data_parallel_size``, not the raw
        server index — otherwise a DP deployment would double-count.

        Args:
            init_info: asdict(ShardedRDTWeightTransferInitInfo) — trainer actor
                name(s)/namespace, produce method name, M:N + ring knobs, and the
                group-major names/dtype_names/shapes/group_lens the bake plans over.
                ``num_consumers`` must already be the whole-fleet total.
            targets: ``(url, slot)`` pairs to init instead of the whole fleet, for
                re-initializing a RESTARTED engine. Its slot pins the
                ``replica_rank``, so the replacement bakes a plan for the same
                consumer ids the dead engine owned — which is what makes it a
                rejoin rather than a second deployment 0. ``None`` inits the whole
                provisioned fleet (the once-per-run path).

        Returns:
            Dict mapping server_url to response.
        """
        # Shared with the sync trainer-side control plane so both stamp identical
        # per-server payloads (single source of truth for the replica_rank fan-out).
        from skyrl.backends.skyrl_train.inference_servers.rdt_control_protocol import (
            build_rdt_init_payloads,
        )

        if targets is None:
            # PROVISIONED, deliberately -- not `active_server_urls`. Position in this
            # list is what `replica_rank = index // data_parallel_size` means, so
            # handing it a degraded list would silently re-map every surviving
            # consumer. Whole-fleet init runs once, before anything can have died, so
            # the two are equal here anyway.
            urls: Sequence[str] = self.server_urls
            slots: Optional[Sequence[int]] = self.server_slots
        else:
            urls = [u for u, _ in targets]
            slots = [s for _, s in targets]
        # `num_replicas` always comes from the provisioned slot set, never from
        # `len(urls)`: with `targets` naming one restarted engine, the derivation
        # would collapse to 1 and that engine would bake a plan for consumer ids
        # belonging to deployment 0.
        payloads = build_rdt_init_payloads(
            init_info,
            urls,
            self.data_parallel_size,
            slots=slots,
            num_replicas=self.num_provisioned_replicas,
        )
        results = await asyncio.gather(
            *[self._call_server(url, "/collective_rpc", payload) for url, payload in payloads]
        )
        return {url: resp for url, resp in results}

    async def update_weights_rdt(
        self,
        update_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Receive a full set of weights via the sharded_rdt engine.

        Calls NewInferenceWorkerWrap.update_weights_rdt on all workers, which
        pulls each worker's consumed slices from the trainer actors over NIXL.
        Called ONCE between start_weight_update and finish_weight_update: each
        worker blocks inside it until it has pulled every gather group, so the
        call overlaps the trainer's gather/publish loop.

        Args:
            update_info: asdict(ShardedRDTWeightTransferUpdateInfo), which is
                empty — the pull plan is built once at init from the group-major
                init metadata and nothing arrives per sync.

        Returns:
            Dict mapping server_url to response.
        """
        return await self._call_all_servers(
            "/collective_rpc",
            {
                "method": "update_weights_rdt",
                "kwargs": {"update_info": update_info},
            },
        )

    async def load_lora_adapter(
        self,
        lora_name: str,
        lora_path: str,
    ) -> Dict[str, Any]:
        """
        Load (or reload) a LoRA adapter on all backend servers via the SkyRL
        custom /skyrl/v1/load_lora_adapter endpoint.

        After loading, generation/chat/completion requests can target this LoRA
        by passing ``model=lora_name``.

        TODO(aaron): switch back to vLLM's /v1/load_lora_adapter once the
        upstream fix in https://github.com/vllm-project/vllm/pull/41482 lands
        in a vLLM release we depend on.

        The custom endpoint (defined in vllm_server_actor.py) wraps add_lora
        with load_inplace=True (so the engine reloads the freshly-written
        safetensors) and then resets the cached LoRARequest's load_inplace=False
        (so subsequent generates don't reload from disk on every step). This
        avoids two vLLM 0.19.0 bugs that surface under colocate_all + tp=1 +
        num_engines>=2 — see vllm_server_actor.py:_skyrl_load_lora_adapter for
        the detailed explanation.

        Args:
            lora_name: Name to register the adapter under on each server.
            lora_path: Path to the LoRA adapter on disk (must be accessible from servers).

        Returns:
            Dict mapping server_url to response.
        """
        session = await self._get_session()

        async def _load_on_server(server_url: str):
            url = f"{server_url}/skyrl/v1/load_lora_adapter"
            payload = {"lora_name": lora_name, "lora_path": lora_path}
            async with session.post(url, json=payload) as resp:
                if resp.status >= 400:
                    body = await resp.json()
                    raise_for_status(resp, body)
                return server_url, {"status": resp.status, "body": await resp.text()}

        results = await asyncio.gather(*[_load_on_server(url) for url in self.active_server_urls])

        logger.info(f"Loaded LoRA adapter '{lora_name}' from {lora_path}")

        return {url: resp for url, resp in results}

    async def unload_lora_adapter(self, lora_name: str) -> Dict[str, Any]:
        """
        Unload a previously-loaded LoRA adapter on all backend servers via /v1/unload_lora_adapter.

        After unloading, ``lora_name`` is no longer accepted as a ``model``
        target on any server. The underlying CPU/GPU LRU entries on vLLM age
        out naturally as new adapters are loaded.

        Args:
            lora_name: Name of the adapter to unload.

        Returns:
            Dict mapping server_url to response.
        """
        payload = {"lora_name": lora_name}

        # Mirror load_lora_adapter: vLLM returns plain text on success and JSON
        # ErrorResponse (e.g. 404) on failure.
        session = await self._get_session()

        async def _unload_on_server(server_url: str):
            url = f"{server_url}/v1/unload_lora_adapter"
            async with session.post(url, json=payload) as resp:
                if resp.status >= 400:
                    body = await resp.json()
                    raise_for_status(resp, body)
                return server_url, {"status": resp.status, "body": await resp.text()}

        results = await asyncio.gather(*[_unload_on_server(url) for url in self.active_server_urls])

        logger.info(f"Unloaded LoRA adapter '{lora_name}'")

        return {url: resp for url, resp in results}

    # ---------------------------
    # Info
    # ---------------------------

    async def get_world_size(self) -> Tuple[int, int]:
        """
        Get total and per-server world size across all inference workers.

        Fetches from vLLM's /get_world_size endpoint on each server.
        All servers are expected to have the same world size.
        Result is cached after first call.

        When data_parallel_size > 1, server_urls contains num_engines * dp_size entries.
        vLLM reports the full DP * TP world size per server, which already
        covers all DP ranks in one deployment. To avoid double-counting,
        total_world_size = per_server_ws * num_deployments (not num_servers).

        Returns:
            Tuple of (total_world_size, world_size_per_server).
        """
        if self._world_size is not None:
            return self._world_size

        results = await self._call_all_servers("/get_world_size", {}, method="GET")

        # PROVISIONED, deliberately. The RDT geometry (`num_consumers`) is frozen at
        # provision and must not shrink when an engine dies -- deriving it from a
        # degraded list would re-map every surviving consumer. That also means this
        # must be resolved before any failure, which it is: the first call happens
        # during weight-sync init. A dead engine here is a real error.
        per_server = []
        for server_url in self.server_urls:
            resp = results.get(server_url)
            if resp is None:
                raise RuntimeError(f"No response for server {server_url}")
            body = resp.get("body", {})
            world_size = body.get("world_size")
            if world_size is None:
                raise RuntimeError(f"Missing world_size in response from {server_url}")
            per_server.append(world_size)

        assert all(
            ws == per_server[0] for ws in per_server
        ), f"All servers must have the same world_size, got {per_server}"

        # Each server is one DP rank. vLLM reports world_size = dp_size * tp_size * pp_size,
        # which is the worker count across ALL DP ranks in one deployment.
        # num_deployments = num_servers / dp_size (each deployment has dp_size servers).
        # Total unique workers = per_server_ws * num_deployments.
        num_deployments = len(self.server_urls) // self.data_parallel_size
        self._world_size = (per_server[0] * num_deployments, per_server[0])
        return self._world_size

    # ---------------------------
    # Lifecycle
    # ---------------------------

    async def teardown(self) -> None:
        """Close HTTP session."""
        task = self._reconcile_task
        self._reconcile_task = None
        self._reconcile_loop = None
        if task is not None and not task.done():
            # Shielded against its awaiters' cancellation, so it has to be cancelled here.
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> "RemoteInferenceClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.teardown()

    # ---------------------------
    # Serialization
    # ---------------------------

    def __getstate__(self) -> dict:
        """Exclude non-serializable fields from pickle.

        ``_disabled_server_urls`` DOES travel, but only as a snapshot: the driver is
        the single writer, and worker copies were pickled at ``init_weight_sync_state``
        — before anything could have died — so a worker's view never updates. That is
        why the live set is passed explicitly into the weight-sync call rather than
        read off the worker's client (see ``RdtWeightSyncSender.send``). A worker-side
        fan-out that hits a since-dead engine degrades through ``_call_all_servers``.
        """
        state = self.__dict__.copy()
        state["_session"] = None
        state["_gen_sem"] = None
        state["_detok_sem"] = None
        state["_sem_loop"] = None
        state["_reconcile_task"] = None
        state["_reconcile_loop"] = None
        # Holds ServerGroups (Ray actor handles, placement groups) and is meaningful
        # only on the driver, which is the only writer of fleet membership.
        state["_engine_supervisor"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """Restore state after unpickling."""
        self.__dict__.update(state)
        self._session = None
        self._gen_sem = None
        self._detok_sem = None
        self._sem_loop = None
        self._reconcile_task = None
        self._reconcile_loop = None
        self._engine_supervisor = None
        # Older pickles (and hand-built states in tests) predate the FT fields.
        self._disabled_server_urls = set(state.get("_disabled_server_urls", ()))
        self._membership_generation = int(state.get("_membership_generation", 0))

    async def aclose(self):
        if self._session is not None:
            try:
                await self._session.close()
            except Exception as e:
                logger.warning(f"Encountered exception {e} while closing client session")
                pass
            self._session = None


def raise_for_status(resp: aiohttp.ClientResponse, body: Optional[Any] = None) -> None:
    """Modified version of resp.raise_for_status() that reads the body for the error message.

    Raises aiohttp.ClientResponseError with the error message from the body if there is an error

    The standard `raise_for_status()` only uses the HTTP reason phrase (e.g. "Bad Request"), which is often unhelpful. APIs typically put more descriptive error details in the response body. This function bridges that gap by surfacing the body's error message in the exception.
    """
    if resp.status >= 400 and body is not None:
        error_detail = body.get("error", {})
        detail_msg = error_detail.get("message", resp.reason) if isinstance(error_detail, dict) else resp.reason
        raise aiohttp.ClientResponseError(
            resp.request_info,
            resp.history,
            status=resp.status,
            message=detail_msg,
            headers=resp.headers,
        )
    resp.raise_for_status()
