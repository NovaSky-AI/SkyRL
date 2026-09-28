"""Python SDK.

The design goal is that the workload does not change. A caller creates a
trajectory, applies its environment, runs the existing client or command
unmodified, and finishes:

    with capture(project="terminal-bench", labels=["task-17"]) as trajectory:
        run_client()              # existing application, no tracing changes

``capture()`` creates the trajectory, applies its environment, finishes it --
with an ``error`` outcome if the block raised -- and releases the HTTP client
it made. Doing it by hand means doing all four:

    trajectory = create_trajectory(project="terminal-bench")
    try:
        with trajectory.environment():
            run_client()
    finally:
        trajectory.finish(labels=["task-17"])
        trajectory.close()        # or pass your own CaptureClient and keep it

``environment()`` sets the provider environment variables the upstream's client
protocol uses and restores the previous values on exit. No per-request headers
and no client patching are involved: the route identity is the correlation key.

There is no control credential and nothing to configure over the API: the
capture process was started against one inference server, and creating a
trajectory is the only setup call there is.
"""

from __future__ import annotations

import contextlib
import os
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx
import orjson

from skyrl_capture import upstream
from skyrl_capture.compression import decompress
from skyrl_capture.ids import trajectory_id as new_trajectory_id

# zstd frame header. `compression.compress` writes these; anything else on this
# path is a plain artifact.
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"

DEFAULT_ENDPOINT = "http://127.0.0.1:8080"

# Finishing is the one call whose failure loses a trial's whole record, so it
# is the one call this SDK retries. Three attempts, short exponential backoff
# with jitter, and only for failures that say nothing about the request:
# a connection that did not complete, a timeout, or a gateway telling us to
# come back. A 409 is a decision and is never retried.
FINISH_ATTEMPTS = 3
FINISH_BACKOFF = 0.25
RETRYABLE_STATUSES = (502, 503, 504)

class CaptureError(Exception):
    pass


@dataclass
class Trajectory:
    """One live trajectory, and the route to send its inference at."""

    id: str
    base_url: str
    mode: str
    protocol: str
    project: str
    run_id: str | None
    task_id: str | None
    step: int | None
    client: CaptureClient = field(repr=False)
    finished: bool = False

    def __post_init__(self) -> None:
        # Not a constructor argument and not a field: only `create_trajectory`
        # knows whether it made the client, and it says so on the way out. A
        # caller-supplied client is never closed here -- a loop that makes one
        # client and many trajectories would lose it on the first `close()`.
        self._owns_client = False

    def env(self, *, api_key: str | None = None) -> dict[str, str]:
        """The environment variables that point a client at this trajectory.

        Capture authenticates nothing on the way in, so the key variable
        carries a placeholder rather than a credential -- most provider SDKs
        refuse to construct a client without one. ``api_key`` passes a
        deployment's own ingress credential through instead, for a capture
        process that sits behind something which checks it.
        """
        # Which variables an unchanged client reads is the upstream's
        # business. An unregistered protocol raises rather than defaulting:
        # injecting OpenAI's variables for an Anthropic target would leave the
        # workload pointing at nothing.
        protocol = upstream.get(self.protocol)
        environment = (
            protocol.client_environment(self.base_url, api_key)
            if api_key is not None
            else protocol.client_environment(self.base_url)
        )
        # Useful for a wrapped process that wants to report which trial it is.
        environment["SKYRL_CAPTURE_TRAJECTORY_ID"] = self.id
        return environment

    @contextlib.contextmanager
    def environment(self) -> Iterator[Trajectory]:
        """Apply :meth:`env` for the duration of the block."""
        previous: dict[str, str | None] = {}
        for name, value in self.env().items():
            previous[name] = os.environ.get(name)
            os.environ[name] = value
        try:
            yield self
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    # -- convenience -------------------------------------------------------
    def finish(
        self,
        *,
        labels: list[str] | None = None,
        annotations: dict[str, Any] | None = None,
        command_result: str | None = None,
        format: str = "graph",
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Finish this trajectory, and get its record back rendered.

        ``annotations`` is how a reward reaches the export: it belongs to the
        whole tree, so every branch of a branched trace carries it. Finishing
        and scoring in one call is what an RL harness wants in its ``finally``.

        The reply carries the trajectory rendered in ``format`` -- ``graph`` by
        default, or one of the three training projections -- read from the
        record this call committed. There is no second call to make and no job
        to poll for one trajectory's rows.

        Retried up to three times on a failure that says nothing about the
        request. The body is identical each time, and the server decides by the
        hash of it: a retry that reaches a committed record gets that record,
        and one that contradicts it gets a `409`.
        """
        payload: dict[str, Any] = {"format": format}
        if labels is not None:
            payload["labels"] = labels
        if annotations is not None:
            payload["annotations"] = annotations
        if command_result is not None:
            payload["command_result"] = command_result
        if options:
            payload["options"] = options
        result = self.client._post_with_retry(f"/v1/trajectories/{self.id}/finish", payload)
        self.finished = True
        return result

    def annotate(self, **values: Any) -> dict[str, Any]:
        """Merge free-form annotations onto the trajectory.

        Annotations belong to the whole tree, so a branched run needs no choice
        of which node to attach to.
        """
        return self.client.annotate_trajectory(self.id, annotations=values)

    def export(
        self,
        format: str = "token-samples",
        *,
        timeout: float = 300.0,
        **options: Any,
    ) -> list[dict[str, Any]]:
        """This trajectory's training rows, one per root-to-leaf branch.

        A compaction is a branch, so one trajectory can yield several rows. A
        sampled node reachable from more than one branch is trainable in
        exactly one of them.

        This drives the same export the CLI and ``/v1/exports`` use, scoped to
        one trajectory, and returns the parsed rows rather than an artifact.
        """
        job = self.client.create_export(format=format, trajectory=self.id, options=options)
        ready = self.client.wait_for_export(job["id"], timeout=timeout)
        if ready["status"] != "ready":
            raise CaptureError(f"export {job['id']} finished as {ready['status']}")
        payload = self.client.download_export(job["id"])
        # Artifacts are stored compressed -- right for the object store, and
        # not what a caller asking for rows wants. The CLI decodes on the way
        # out (`encode_artifact`); this path did not, so it handed zstd frames
        # to the JSON parser and every call failed with "str is not valid
        # UTF-8".
        #
        # Decided by the magic number rather than by catching the failure: an
        # uncompressed artifact must still work, and a *corrupt* one must say so
        # rather than reaching the JSON parser and reporting the wrong problem.
        # That matters most on the signed-URL path, where the bytes come
        # straight from the object store and a truncated download is real.
        if payload.startswith(ZSTD_MAGIC):
            payload = decompress(payload)
        return [orjson.loads(line) for line in payload.splitlines() if line.strip()]

    def tag(self, *labels: str, remove: list[str] | None = None) -> dict[str, Any]:
        return self.client.annotate_trajectory(
            self.id, labels=list(labels), remove_labels=remove or []
        )

    def close(self) -> None:
        """Release the HTTP client, if this trajectory created one.

        A trial per rollout means a connection pool per rollout otherwise, and
        they accumulate for as long as the training loop runs. Passing your own
        `CaptureClient` to `create_trajectory` is the other way to manage this:
        then the pool is shared and closing it is yours to do.
        """
        if self._owns_client:
            self.client.close()
            self._owns_client = False

class CaptureClient:
    """Thin synchronous control-plane client."""

    def __init__(self, endpoint: str | None = None, *, timeout: float = 30.0) -> None:
        self.endpoint = (endpoint or os.environ.get("CAPTURE_ENDPOINT") or DEFAULT_ENDPOINT).rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    # -- transport ---------------------------------------------------------
    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if extra:
            headers.update(extra)
        return headers

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        """Send one request, turning transport failures into ``CaptureError``.

        An unreachable control plane is the most common way to use this tool
        wrong, so it has to read as a message rather than as a stack trace from
        inside the HTTP client.
        """
        return self._handle(
            self._send(
                method,
                f"{self.endpoint}{path}",
                target=f"the control plane at {self.endpoint}",
                hint=" Is `skyrl-capture serve` running, and is --endpoint right?",
                **kwargs,
            )
        )

    def _send(self, method: str, url: str, *, target: str, hint: str = "", **kwargs: Any) -> httpx.Response:
        """The one place an httpx transport error becomes a `CaptureError`.

        Every request this client makes goes through here, including `/healthz`
        and the export download. The download especially: its URL may be a
        signed one at the object store rather than a control-plane path, so it
        is the call most likely to fail in a way a caller has to catch -- and
        it was the one leaking `httpx.ConnectError` past a documented contract
        that says everything here raises `CaptureError`.
        """
        try:
            return self._client.request(method, url, **kwargs)
        except httpx.TimeoutException as error:
            raise CaptureError(f"timed out talking to {target} ({error})") from error
        except httpx.RequestError as error:
            raise CaptureError(f"cannot reach {target}: {error}.{hint}") from error

    def _handle(self, response: httpx.Response) -> Any:
        if response.status_code >= 400:
            detail: Any
            try:
                body = response.json()
                detail = body.get("detail") or body
            except Exception:
                detail = response.text
            raise CaptureError(f"{response.status_code} {detail}")
        if not response.content:
            return None
        return response.json()

    # Private: a caller reaches an operation by name below, not by assembling
    # a path. A public `get(path)` makes every route this SDK does not support
    # look supported, and makes removing one a breaking change.
    def _get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        clean = {key: value for key, value in (params or {}).items() if value is not None}
        return self._request("GET", path, headers=self._headers(), params=clean)

    def _post(self, path: str, payload: Any = None, *, headers: dict[str, str] | None = None) -> Any:
        return self._request("POST", path, json=payload, headers=self._headers(headers))

    def _post_with_retry(self, path: str, payload: Any) -> Any:
        """POST, retrying the failures that say nothing about the request.

        Exactly `FINISH_ATTEMPTS` attempts in the worst case. The delay is
        exponential with jitter, because a capture replica that was restarted
        is about to be hit by every trajectory that was open on it at once.
        """
        url = f"{self.endpoint}{path}"
        for attempt in range(1, FINISH_ATTEMPTS + 1):
            last: Exception | None = None
            try:
                response = self._client.request(
                    "POST", url, json=payload, headers=self._headers()
                )
            except (httpx.TimeoutException, httpx.RequestError) as error:
                last = CaptureError(f"cannot reach the control plane at {self.endpoint}: {error}")
            else:
                if response.status_code not in RETRYABLE_STATUSES:
                    return self._handle(response)
                last = CaptureError(f"{response.status_code} {response.text[:200]}")
            if attempt == FINISH_ATTEMPTS:
                raise last
            time.sleep(FINISH_BACKOFF * (2 ** (attempt - 1)) * (0.5 + random.random()))
        raise CaptureError("unreachable")  # pragma: no cover

    def _patch(self, path: str, payload: Any = None) -> Any:
        return self._request("PATCH", path, json=payload, headers=self._headers())

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> CaptureClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- resources ---------------------------------------------------------
    def health(self) -> dict[str, Any]:
        return self._get("/healthz")

    def list_trajectories(self, **params: Any) -> dict[str, Any]:
        """Filterable by `project`, `run_id`, `task_id`, `step` and `status`."""
        return self._get("/v1/trajectories", params=params)

    def annotate_trajectory(
        self,
        trajectory: str,
        *,
        annotations: dict[str, Any] | None = None,
        remove_annotations: list[str] | None = None,
        labels: list[str] | None = None,
        remove_labels: list[str] | None = None,
    ) -> dict[str, Any]:
        """Merge labels and annotations onto a trajectory, by id.

        What `skyrl-capture annotate` runs, and what `Trajectory.annotate` and
        `Trajectory.tag` are the one-trajectory shorthands for. Rewards and
        labels usually arrive after the trial, when the `Trajectory` object is
        long gone and an id is all there is.
        """
        return self._patch(
            f"/v1/trajectories/{trajectory}/metadata",
            {
                "annotations": annotations or {},
                "remove_annotations": list(remove_annotations or []),
                "labels": list(labels or []),
                "remove_labels": list(remove_labels or []),
            },
        )

    def create_export(self, **fields: Any) -> dict[str, Any]:
        return self._post("/v1/exports", fields)

    def get_export(self, identifier: str) -> dict[str, Any]:
        return self._get(f"/v1/exports/{identifier}")

    def download_export(self, identifier: str) -> bytes:
        job = self.get_export(identifier)
        if job["status"] != "ready":
            raise CaptureError(f"export {identifier} is {job['status']}")
        url = job.get("download_url") or f"{self.endpoint}/v1/exports/{identifier}/download"
        response = self._send(
            "GET",
            url,
            target=f"the export artifact at {url}",
            headers=self._headers(),
            follow_redirects=True,
        )
        if response.status_code >= 400:
            raise CaptureError(f"download failed: {response.status_code} {response.text[:200]}")
        return response.content

    def wait_for_export(self, identifier: str, *, timeout: float = 300.0, interval: float = 0.5) -> dict[str, Any]:
        import time

        deadline = time.monotonic() + timeout
        while True:
            job = self.get_export(identifier)
            if job["status"] in ("ready", "failed"):
                return job
            if time.monotonic() >= deadline:
                raise CaptureError(f"export {identifier} did not finish within {timeout}s")
            time.sleep(interval)


# -- module-level lifecycle -------------------------------------------------
def create_trajectory(
    *,
    project: str,
    run_id: str | None = None,
    task_id: str | None = None,
    step: int | None = None,
    labels: list[str] | None = None,
    annotations: dict[str, Any] | None = None,
    bodies: str = "full",
    source_metadata: dict[str, Any] | None = None,
    trajectory_id: str | None = None,
    endpoint: str | None = None,
    client: CaptureClient | None = None,
) -> Trajectory:
    """Create one trajectory, and get the route to send its inference at.

    ``trajectory_id`` names the trajectory. When it is not given the SDK
    generates one *before* it sends the request, so a create whose response was
    lost can be repeated with the same id and the same body and is idempotent.
    The id is also the session key sent to the inference engine, so naming it
    is how a caller aligns capture's session with its own.

    ``run_id`` places this attempt in a run, creating the run if it is new;
    ``task_id`` and ``step`` say which task it attempted and which training
    step produced it. Both are filters on a listing -- `task_id` is one
    as, and they are indexed columns rather than annotations because every
    question about a run groups by them.

    What the upstream is, and how capture authenticates to it, is process
    configuration rather than anything a trajectory carries. What varies per
    inference goes in the inference request, where the provider already reads
    it. A trajectory carries capture metadata -- labels, annotations, the run
    it belongs to -- and not hidden upstream settings.
    """
    owned = client or CaptureClient(endpoint)
    # The id exists before the trajectory does. That is what makes a create
    # whose response was lost safe to repeat: the same id and the same body is
    # the same trajectory, and the server says so by the hash it persisted.
    identifier = trajectory_id or new_trajectory_id()
    payload: dict[str, Any] = {
        "trajectory_id": identifier,
        "project": project,
        "labels": labels or [],
        "annotations": annotations or {},
        "bodies": bodies,
        "source_metadata": source_metadata or {},
    }
    for name, value in (("run_id", run_id), ("task_id", task_id), ("step", step)):
        if value is not None:
            payload[name] = value
    mine = client is None
    try:
        created = owned._post("/v1/trajectories", payload)
    except BaseException:
        # The client exists only because this call needed it, and this call is
        # not going to return a `Trajectory` that could close it later.
        if mine:
            owned.close()
        raise
    trajectory = Trajectory(
        id=created["id"],
        base_url=created["base_url"],
        mode=created["mode"],
        protocol=created["protocol"],
        project=project,
        run_id=run_id,
        task_id=task_id,
        step=step,
        client=owned,
    )
    trajectory._owns_client = mine
    return trajectory



@contextlib.contextmanager
def capture(
    *,
    project: str,
    labels: list[str] | None = None,
    endpoint: str | None = None,
    **kwargs: Any,
) -> Iterator[Trajectory]:
    """Create, apply the environment, and finish -- in one context manager.

    On an exception the trajectory is finished with an ``error`` outcome, so an
    abandoned trial does not have to wait for the expiry sweep.
    """
    trajectory = create_trajectory(project=project, endpoint=endpoint, **kwargs)
    failed = False
    try:
        with trajectory.environment():
            yield trajectory
    except BaseException:
        failed = True
        raise
    finally:
        try:
            trajectory.finish(
                labels=labels,
                command_result="error" if failed else "success",
            )
        except Exception:
            # Finishing is best-effort in teardown; the expiry sweep is the backstop.
            pass
        # Whether or not finishing worked. A client this call created is this
        # call's to release, and leaking one per `with capture(...)` is how a
        # training loop runs out of sockets.
        trajectory.close()
