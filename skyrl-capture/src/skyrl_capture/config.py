"""Process configuration.

All configuration is environment-driven, so one image runs everywhere. Every
field has a working local default; nothing here requires a cloud account.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None or not raw.strip() else float(raw)




@dataclass(frozen=True)
class RecordConfig:
    """How journals are written. Where they go is ``Config.record_dir``."""

    # fsync policy: "always" | "interval" | "never". **"always" is the
    # default, and it is what the product contract rests on**: TITO promises
    # that a cleanly closed response has its exact exchange on disk, and that
    # promise is an fsync. The other two exist to measure what it costs, and
    # they weaken it -- a crash then loses whatever the last window held.
    fsync: str = field(default_factory=lambda: os.environ.get("RECORD_FSYNC", "always"))
    fsync_interval: float = field(default_factory=lambda: _env_float("RECORD_FSYNC_INTERVAL", 1.0))
    # zstd for records above a few kilobytes. A token turn compresses by an
    # order of magnitude; a lifecycle record is not worth the frame.
    compress: bool = field(default_factory=lambda: _env_bool("RECORD_COMPRESS", True))
    # Journal appends queued at once before capture is out of room. Text
    # refuses the work and records a gap; TITO waits for capacity before it
    # closes a response. Neither blocks a forwarded request.
    commit_capacity: int = field(default_factory=lambda: _env_int("RECORD_COMMIT_CAPACITY", 1024))


@dataclass(frozen=True)
class ProxyConfig:
    """Data-plane behaviour."""

    host: str = field(default_factory=lambda: os.environ.get("PROXY_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("PROXY_PORT", 8080))
    # Public base for the trajectory URLs handed back to callers.
    public_url: str = field(default_factory=lambda: os.environ.get("PUBLIC_URL", "http://127.0.0.1:8080"))
    max_request_bytes: int = field(default_factory=lambda: _env_int("PROXY_MAX_REQUEST_BYTES", 64 * 1024 * 1024))
    upstream_connect_timeout: float = field(default_factory=lambda: _env_float("UPSTREAM_CONNECT_TIMEOUT", 10.0))
    upstream_read_timeout: float = field(default_factory=lambda: _env_float("UPSTREAM_READ_TIMEOUT", 3600.0))
    # How many *idle* keep-alive connections to hold per upstream origin. Not
    # a concurrency limit: capping requests in flight would make capture queue
    # inference behind itself, and the upstream decides its own concurrency.
    upstream_idle_connections: int = field(
        default_factory=lambda: _env_int("UPSTREAM_IDLE_CONNECTIONS", 2048)
    )
    # Tokens held across cached token traces before the least recently used
    # ones are dropped. A dropped trace is rebuilt from the graph on its next
    # turn, so this trades memory for an occasional slow turn. 0 disables the
    # bound, which is what the tests want and no deployment does.
    token_trace_budget: int = field(
        default_factory=lambda: _env_int("TOKEN_TRACE_BUDGET_TOKENS", 20_000_000)
    )
    # Extra CA bundle for upstream TLS. Needed for a self-hosted inference
    # service behind a private CA, which is a common deployment.
    upstream_ca_bundle: str | None = field(
        default_factory=lambda: os.environ.get("UPSTREAM_CA_BUNDLE") or None
    )
    # What fraction of bodies a `bodies="sampled"` trajectory keeps. Whether
    # to sample at all is per-trajectory and decided at creation: a process-wide
    # setting could only disagree with what a caller asked for.
    payload_sample_rate: float = field(default_factory=lambda: _env_float("PAYLOAD_SAMPLE_RATE", 1.0))
    # Disables the capture path entirely. Used by the benchmark baseline.
    capture_enabled: bool = field(default_factory=lambda: _env_bool("CAPTURE_ENABLED", True))
    # Record per-chunk streaming timings. Summary timings are always recorded.
    capture_stream_chunks: bool = field(default_factory=lambda: _env_bool("CAPTURE_STREAM_CHUNKS", True))
    max_stream_chunk_records: int = field(default_factory=lambda: _env_int("MAX_STREAM_CHUNK_RECORDS", 4096))


class StartupError(RuntimeError):
    """A misconfiguration the operator can fix by relaunching.

    Raised during startup and reported as its message alone: a traceback for
    "this upstream needs a tokenizer" points at the import that raised, which is
    not where the mistake is.
    """


# -- the one upstream, in one of two shapes ------------------------------------------
#
# There is no target registry and no target CRUD. One capture process serves one
# upstream, chosen when it starts, because that is what an ephemeral in-cluster
# capture job actually needs: a trainer brings capture up beside its own
# inference server and tears both down together. Changing the upstream, or the
# mode, means relaunching.
#
# Two dataclasses rather than one with fields that matter only sometimes: a
# text upstream has no tokenizer and no context window, and a token upstream
# cannot work without a tokenizer. Which one a process holds *is* its mode, so
# there is one authoritative answer and nothing infers it per request.
#
# The credential is read from the environment and never stored, so there is
# nothing to encrypt at rest and nothing an API can leak.


@dataclass(frozen=True)
class TextUpstream:
    """An OpenAI- or Anthropic-compatible server, proxied as text."""

    #: A registered text server name: ``openai``, ``anthropic``, or one a
    #: module contributes.
    type: str = "openai"
    url: str = "http://127.0.0.1:8000/v1"
    model: str | None = None
    #: Presented to the upstream on every forwarded call. Never captured, never
    #: returned by the API, never written to the record.
    api_key: str | None = None

    mode = "text"
    #: What the token path reads and text has none of. Present so the two
    #: shapes answer the same questions.
    tokenizer = None
    max_model_len = None

    @property
    def protocol(self) -> Any:
        """The one adapter that owns everything provider-specific here.

        Resolved at startup and passed to the proxy and the commit path, so
        nothing looks a provider up per request.
        """
        from skyrl_capture import upstream

        return upstream.get(self.type)

    @property
    def client_suffix(self) -> str:
        return str(self.protocol.client_suffix)

    #: Which protocol a *client* speaks to this route. The same as the
    #: upstream's, in text mode.
    @property
    def client_protocol(self) -> str:
        return self.type

    @property
    def config(self) -> dict[str, Any]:
        return {}

    def provenance(self) -> dict[str, Any]:
        """Non-secret process-wide provenance, for the record's manifest.

        What a reader needs to interpret the record later. Never the key.
        """
        return {
            "mode": self.mode,
            "type": self.type,
            "protocol": self.client_protocol,
            "url": self.url,
            "model": self.model,
        }

    def validate(self) -> None:
        """Fail at startup rather than on the first request."""
        from skyrl_capture import upstream

        if not self.url:
            raise ValueError("UPSTREAM_URL is required")
        try:
            upstream.get(self.type)
        except upstream.UnknownProtocol as error:
            raise ValueError(str(error)) from error


@dataclass(frozen=True)
class TitoUpstream:
    """A token-in/token-out engine, rendered by this process.

    The tokenizer is required and is loaded once at startup: everything exact
    about this mode -- the prompt ids sent, the sampled ids stored, the token
    ranges attributed to messages -- is that renderer's output.
    """

    url: str = "http://127.0.0.1:8000/generate"
    #: A registered token server name: ``tokens``, ``vllm``, or one a module
    #: contributes.
    type: str = "tokens"
    tokenizer: str = ""
    model: str | None = None
    api_key: str | None = None
    max_model_len: int | None = None

    mode = "tokens"
    #: A token route speaks OpenAI chat completions to the workload whatever
    #: the engine underneath looks like, so both of these are fixed.
    client_suffix = "/v1"
    client_protocol = "openai"

    @property
    def engine(self) -> Any:
        """The wire this engine speaks. Resolved at startup, like a protocol."""
        from skyrl_capture.tito import upstream as tito

        return tito.get(self.type)

    @property
    def config(self) -> dict[str, Any]:
        """Extra knobs the token path reads off the upstream definition."""
        return {"max_model_len": self.max_model_len} if self.max_model_len else {}

    def provenance(self) -> dict[str, Any]:
        """Non-secret process-wide provenance, for the record's manifest.

        The tokenizer's identity is in here because reading these token ids
        later means knowing which renderer produced them. Never the key.
        """
        return {
            "mode": self.mode,
            "type": self.type,
            "protocol": self.client_protocol,
            "url": self.url,
            "model": self.model,
            "tokenizer": self.tokenizer,
            "config": self.config,
        }

    def validate(self) -> None:
        from skyrl_capture.tito import upstream as tito

        if not self.url:
            raise ValueError("UPSTREAM_URL is required")
        try:
            tito.get(self.type)
        except tito.UnknownTitoProtocol as error:
            raise ValueError(str(error)) from error
        if not self.tokenizer:
            raise ValueError(
                f"a {self.type} upstream requires a tokenizer "
                "(--tokenizer, or UPSTREAM_TOKENIZER)"
            )


Upstream = TextUpstream | TitoUpstream


def upstream_from_env(mode: str | None = None) -> Upstream:
    """The upstream this process was started with, and therefore its mode.

    The mode is stated, not inferred: ``CAPTURE_MODE`` picks which of the two
    shapes is built, and ``UPSTREAM_TYPE`` is recorded as a *name* and nothing
    more. Reading the environment deliberately consults no registry -- plugin
    modules are imported by `build_runtime`, after configuration exists, so a
    name resolved here would be resolved too early and a contributed type
    would be classified as whatever was registered before it loaded. The name
    is resolved once, in the mode's own registry, by `validate()`.
    """
    mode = mode or os.environ.get("CAPTURE_MODE", "text")
    if mode not in ("text", "tokens"):
        raise StartupError(f"CAPTURE_MODE must be 'text' or 'tokens', got {mode!r}")

    type_name = os.environ.get("UPSTREAM_TYPE", "openai" if mode == "text" else "tokens")
    api_key = os.environ.get("UPSTREAM_API_KEY") or None
    model = os.environ.get("UPSTREAM_MODEL") or None
    if mode == "tokens":
        return TitoUpstream(
            type=type_name,
            url=os.environ.get("UPSTREAM_URL", "http://127.0.0.1:8000/generate"),
            tokenizer=os.environ.get("UPSTREAM_TOKENIZER") or "",
            model=model,
            api_key=api_key,
            max_model_len=_env_int("UPSTREAM_MAX_MODEL_LEN", 0) or None,
        )
    return TextUpstream(
        type=type_name,
        url=os.environ.get("UPSTREAM_URL", "http://127.0.0.1:8000/v1"),
        model=model,
        api_key=api_key,
    )


@dataclass(frozen=True)
class Config:
    # Scratch: where exports go when no record directory says otherwise.
    # Nothing here is a result -- delete it whenever. Results go to
    # ``record_dir``.
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("CAPTURE_DATA_DIR", "./capture-data")))
    # Where the record goes. **Required**: persistence is not a mode, it is
    # what capture is. One journal per unfinished trajectory and one compiled
    # record per finished one, read by the exporters and the viewer with no
    # database and no server. See ``persistence/``.
    #
    # It may be a local directory or a shared volume. Several capture
    # processes may write one directory, under the external invariant that
    # routing gives each trajectory exactly one writer.
    record_dir: Path | None = field(
        default_factory=lambda: Path(os.environ["CAPTURE_RECORD_DIR"])
        if os.environ.get("CAPTURE_RECORD_DIR")
        else None
    )
    # Whether this replica also serves the read API and bulk exports. One
    # replica per record directory should, and the rest should not: there is
    # one indexer per record root.
    viewer: bool = field(default_factory=lambda: _env_bool("CAPTURE_VIEWER", True))
    # The one inference server this process captures. Startup configuration,
    # not a stored record: see ``TextUpstream`` and ``TitoUpstream``.
    # The one inference server this process captures, and -- by which shape it
    # is -- the one mode this process runs in.
    upstream: Upstream = field(default_factory=upstream_from_env)
    # How long `finish` waits for turns already in flight on the trajectory,
    # and then for their commits, before compiling without them. A turn that
    # outlives it lands late and says so on its exchange.
    finish_grace_seconds: float = field(default_factory=lambda: _env_float("FINISH_GRACE_SECONDS", 5.0))
    #: How often to finish trajectories whose caller asked and went away. A
    #: caller that abandons a call gets a retryable `503` from `finish`, and an
    #: RL generator retries on a *fresh* trajectory -- so the retry it was
    #: invited to make never arrives for the old one. Zero turns the sweep off.
    finish_sweep_seconds: float = field(
        default_factory=lambda: _env_float("FINISH_SWEEP_SECONDS", 60.0)
    )
    # Modules to import before anything resolves a target type, so an upstream
    # kind defined outside this package is registered.
    upstream_modules: tuple[str, ...] = ()
    capture_header_allowlist: tuple[str, ...] = (
        "content-type",
        "accept",
        "accept-encoding",
        "user-agent",
        "x-stainless-lang",
        "x-stainless-package-version",
        "x-stainless-runtime",
        "x-stainless-retry-count",
        "openai-organization",
        "openai-processing-ms",
        "openai-version",
        "anthropic-version",
        "anthropic-beta",
        "x-request-id",
        "request-id",
        "x-ratelimit-limit-requests",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-tokens",
        "retry-after",
    )
    record: RecordConfig = field(default_factory=RecordConfig)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)

    def require_record_dir(self) -> Path:
        """The record directory, or a `StartupError` naming how to set it.

        Refused at startup rather than at the first request: a capture process
        with nowhere to write is a process whose whole output is lost, and the
        run that discovers it has already been paid for.
        """
        if self.record_dir is None:
            raise StartupError(
                "capture needs a record directory: pass --record-dir, or set "
                "CAPTURE_RECORD_DIR. Every trajectory is persisted as it runs, so "
                "there is no mode in which one is optional."
            )
        return self.record_dir

    @property
    def export_dir(self) -> Path:
        """Where an export's artifacts go: beside the record they came from.

            <record-dir>/manifest.json      format and provenance
            <record-dir>/active/            a journal per unfinished trajectory
            <record-dir>/committed/         a record per finished one
            <record-dir>/exports/jobs/      what was asked for
            <record-dir>/exports/artifacts/ what it produced

        A record directory holds a run and everything derived from it, so
        copying the directory copies both.
        """
        return self.require_record_dir() / "exports"

    def with_overrides(self, **kwargs: object) -> Config:
        return replace(self, **kwargs)  # type: ignore[arg-type]


def load_config() -> Config:
    """The configuration this process was started with, from the environment."""
    return Config()
