"""Scrape vLLM engine metrics from Ray's per-node metrics agents.

When ``generator.inference_engine.enable_ray_prometheus_stats=true``, the vLLM
engines record their metrics through ``ray.util.metrics`` (via vLLM's
``RayPrometheusStatLogger``), and Ray's metrics agent on each node exposes them
in Prometheus text format.  This module scrapes those endpoints once per
training step and reduces a small fixed subset to scalars suitable for wandb.

Counters are summed across replicas; gauges are averaged.  Rates and average
latencies are derived from deltas vs. the previous sample.
"""

import asyncio
import math
import re
import statistics
import time
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

import httpx
import ray
from loguru import logger

from skyrl.backends.skyrl_train.inference_servers.common import format_http_url
from skyrl.train.utils.vllm_run_statistics import RunStatistics
from skyrl.train.utils.vllm_window_statistics import latency_metrics

# vLLM metric base names after RayPrometheusStatLogger sanitization (`:` -> `_`)
# AND the `ray_` prefix that Ray's metrics agent adds to every custom metric.
# Counters are exported by Ray in both legacy (no suffix) and proper (`_total`)
# forms; we use the proper form to avoid double-counting if both are summed.
# Histograms expose `_sum`/`_count`/`_bucket` samples.
_GAUGE_NUM_RUNNING = "ray_vllm_num_requests_running"
_GAUGE_NUM_WAITING = "ray_vllm_num_requests_waiting"
_GAUGE_KV_CACHE_USAGE = "ray_vllm_kv_cache_usage_perc"
_COUNTER_PREFIX_QUERIES = "ray_vllm_prefix_cache_queries_total"
_COUNTER_PREFIX_HITS = "ray_vllm_prefix_cache_hits_total"
_COUNTER_PROMPT_TOKENS = "ray_vllm_prompt_tokens_total"
_COUNTER_GENERATION_TOKENS = "ray_vllm_generation_tokens_total"
_COUNTER_PREEMPTIONS = "ray_vllm_num_preemptions_total"
_COUNTER_EXTERNAL_PREFIX_QUERIES = "ray_vllm_external_prefix_cache_queries_total"
_COUNTER_EXTERNAL_PREFIX_HITS = "ray_vllm_external_prefix_cache_hits_total"
_COUNTER_KV_OFFLOAD_STORE_BYTES = "ray_vllm_kv_offload_store_bytes_total"
_COUNTER_KV_OFFLOAD_LOAD_BYTES = "ray_vllm_kv_offload_load_bytes_total"
_HIST_TTFT_SUM = "ray_vllm_time_to_first_token_seconds_sum"
_HIST_TTFT_COUNT = "ray_vllm_time_to_first_token_seconds_count"
_HIST_TTFT_BUCKET = "ray_vllm_time_to_first_token_seconds_bucket"
_HIST_REQUEST_TPOT_SUM = "ray_vllm_request_time_per_output_token_seconds_sum"
_HIST_REQUEST_TPOT_COUNT = "ray_vllm_request_time_per_output_token_seconds_count"
_HIST_REQUEST_TPOT_BUCKET = "ray_vllm_request_time_per_output_token_seconds_bucket"
# Speculative-decoding (MTP draft) counters. The per-position counter additionally carries a
# `position` label ("0".."k-1"); it is summed per-position in `sum_by_position` rather than through
# `_SUM_METRICS` (which would collapse the label and lose the per-depth breakdown).
_COUNTER_SPEC_DRAFTS = "ray_vllm_spec_decode_num_drafts_total"
_COUNTER_SPEC_DRAFT_TOKENS = "ray_vllm_spec_decode_num_draft_tokens_total"
_COUNTER_SPEC_ACCEPTED_TOKENS = "ray_vllm_spec_decode_num_accepted_tokens_total"
_COUNTER_SPEC_ACCEPTED_PER_POS = "ray_vllm_spec_decode_num_accepted_tokens_per_pos_total"

_SUM_METRICS = (
    _GAUGE_NUM_RUNNING,
    _GAUGE_NUM_WAITING,
    _COUNTER_PREFIX_QUERIES,
    _COUNTER_PREFIX_HITS,
    _COUNTER_PROMPT_TOKENS,
    _COUNTER_GENERATION_TOKENS,
    _HIST_TTFT_SUM,
    _HIST_TTFT_COUNT,
    _COUNTER_SPEC_DRAFTS,
    _COUNTER_SPEC_DRAFT_TOKENS,
    _COUNTER_SPEC_ACCEPTED_TOKENS,
    _COUNTER_PREEMPTIONS,
    _COUNTER_EXTERNAL_PREFIX_QUERIES,
    _COUNTER_EXTERNAL_PREFIX_HITS,
    _COUNTER_KV_OFFLOAD_STORE_BYTES,
    _COUNTER_KV_OFFLOAD_LOAD_BYTES,
    _HIST_REQUEST_TPOT_SUM,
    _HIST_REQUEST_TPOT_COUNT,
)
_MEAN_METRICS = (_GAUGE_KV_CACHE_USAGE,)

ParsedSamples = Dict[Tuple[str, FrozenSet[Tuple[str, str]]], float]


# `metric_name{label="v",...} 12.34` — value may also be `+Inf`/`-Inf`/`NaN`.
# Optional trailing timestamp (ignored) per the Prometheus text format.
_METRIC_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)" r"(?:\{(?P<labels>[^}]*)\})?" r"\s+(?P<value>[^\s]+)" r"(?:\s+\d+)?\s*$"
)
_LABEL_RE = re.compile(r'(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)="(?P<val>(?:\\.|[^"\\])*)"')


def _coerce_value(raw: str) -> Optional[float]:
    if raw == "+Inf":
        return float("inf")
    if raw == "-Inf":
        return float("-inf")
    if raw == "NaN":
        return float("nan")
    try:
        return float(raw)
    except ValueError:
        return None


def parse_metrics_text(text: str) -> ParsedSamples:
    """Parse a Prometheus text payload into ``{(sample_name, labels): value}``.

    Sample names retain their exported suffix (``_total``, ``_sum``,
    ``_count``, ``_bucket``).  Labels are a frozenset of ``(key, value)`` pairs
    so the dict is hashable and label-permutation independent.

    Comment lines (``# HELP``/``# TYPE``) and blank lines are ignored.
    """
    out: ParsedSamples = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _METRIC_LINE_RE.match(line)
        if not m:
            continue
        value = _coerce_value(m.group("value"))
        if value is None:
            continue
        labels_str = m.group("labels") or ""
        labels = frozenset(
            (lm.group("key"), lm.group("val").replace('\\"', '"').replace("\\\\", "\\"))
            for lm in _LABEL_RE.finditer(labels_str)
        )
        out[(m.group("name"), labels)] = value
    return out


def aggregate(parsed: ParsedSamples, names: Iterable[str], how: str) -> Dict[str, float]:
    """Reduce per-(name, labels) values to one scalar per name.

    ``how`` is ``"sum"`` or ``"mean"``.  Names absent from ``parsed`` are
    omitted from the result rather than reported as 0 — that lets the caller
    distinguish "metric not seen yet" from "metric is zero".
    """
    result: Dict[str, float] = {}
    for name in names:
        vals = [v for (n, _labels), v in parsed.items() if n == name]
        if not vals:
            continue
        if how == "sum":
            result[name] = sum(vals)
        elif how == "mean":
            result[name] = sum(vals) / len(vals)
        else:
            raise ValueError(f"unknown aggregation: {how}")
    return result


def sum_by_position(parsed: ParsedSamples, name: str) -> Dict[str, float]:
    """Sum a per-position-labelled counter into ``{name::position: value}`` keys.

    The vLLM ``..._per_pos`` spec-decode counter carries a ``position`` label ("0".."k-1"), so
    plain ``aggregate`` would collapse all positions into one sum. Grouping by that label keeps the
    per-depth breakdown so ``_derive`` can emit one acceptance rate per draft position. Samples
    across replicas (different ReplicaId labels) add for the same position.
    """
    out: Dict[str, float] = {}
    for (sample_name, labels), value in parsed.items():
        if sample_name != name:
            continue
        pos = next((val for key, val in labels if key == "position"), None)
        if pos is None:
            continue
        key = f"{name}::{pos}"
        out[key] = out.get(key, 0.0) + value
    return out


def discover_ray_metrics_urls() -> List[str]:
    """Return ``http://<ip>:<port>/metrics`` for every alive Ray node."""
    urls: List[str] = []
    for node in ray.nodes():
        if not node.get("Alive", False):
            continue
        ip = node.get("NodeManagerAddress")
        port = node.get("MetricsExportPort")
        if not ip or not port:
            continue
        urls.append(format_http_url(ip, port, "/metrics"))
    return urls


class VLLMMetricsScraper:
    """Per-step snapshot of selected vLLM metrics from Ray's metrics agents.

    Two ways to derive a window:

    * ``sample()`` reports deltas vs. the previous call against a wall-clock (or
      caller-supplied generation) interval — used by the fully-async trainer.
    * ``start(label)`` / ``pause()`` / ``resume()`` / ``stop()`` measure an
      explicit window and own its timing. The scraper accumulates only the
      un-paused time between ``start`` and ``stop``, so the caller never has to
      thread a generation-time value back in or mark boundaries by ordering.
      ``stop()`` returns the metrics nested under ``{label}/`` (e.g.
      ``vllm/train/generation_throughput_tok_s``). The sync trainer uses this to
      report the train rollout under ``vllm/train/*`` and the eval rollout under
      ``vllm/eval/*`` instead of blending them.

    The two paths keep independent state, so a process may use either (the sync
    trainer uses windows; the fully-async trainer uses ``sample()``).
    """

    def __init__(
        self,
        urls: Optional[List[str]] = None,
        request_timeout_s: float = 2.0,
        worker_ids: Optional[Iterable[str]] = None,
    ):
        self._urls = urls if urls is not None else discover_ray_metrics_urls()
        self._timeout = request_timeout_s
        self._worker_ids = None if worker_ids is None else frozenset(worker_ids)
        self._role_scrapers: Dict[str, VLLMMetricsScraper] = {}
        self.run_statistics = RunStatistics()
        self._prev_aggregated: Optional[Dict[str, float]] = None
        self._prev_timestamp: Optional[float] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._warned_empty = False
        self._snapshot_complete = True
        self._engine_snapshot = {}
        self._previous_engines = {}
        self._window_engines = {}
        # Explicit-window state (start/pause/resume/stop). ``_label is None``
        # means no window is open.
        self._label: Optional[str] = None
        self._window_prev: Optional[Dict[str, float]] = None
        self._window_time_s: float = 0.0
        self._active_since: Optional[float] = None  # start of the current un-paused span
        self._paused: bool = False
        if not self._urls:
            logger.warning(
                "VLLMMetricsScraper: ray.nodes() returned no metrics endpoints; "
                "engine metrics will not appear in wandb."
            )

    def set_worker_ids(self, worker_ids: Iterable[str]) -> None:
        """Restrict snapshots to a fixed set of inference frontend workers."""
        self._worker_ids = frozenset(worker_ids)
        self._role_scrapers = {}

    @property
    def has_worker_roles(self) -> bool:
        """Whether both PD roles have known, separate worker membership."""
        return bool(self._role_scrapers)

    def set_worker_roles(self, worker_ids_by_role: Dict[str, Iterable[str]]) -> None:
        """Collect prefill and decode independently using their launched workers."""
        groups = {role: frozenset(ids) for role, ids in worker_ids_by_role.items()}
        if set(groups) != {"prefill", "decode"} or not all(groups.values()):
            raise ValueError("PD metrics require nonempty prefill and decode worker groups")
        if not groups["prefill"].isdisjoint(groups["decode"]):
            raise ValueError("Prefill and decode metric workers must be disjoint")
        self._worker_ids = groups["prefill"] | groups["decode"]
        self._role_scrapers = {
            role: VLLMMetricsScraper(urls=self._urls, request_timeout_s=self._timeout, worker_ids=ids)
            for role, ids in groups.items()
        }

    async def set_external_servers(self, server_urls: Iterable[str], enable_pd: bool) -> None:
        """Resolve frontend worker identities from external SkyRL servers."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            responses = await asyncio.gather(
                *(client.get(url.rstrip("/") + "/get_metrics_worker_info") for url in server_urls)
            )
        workers = []
        groups = {"prefill": [], "decode": []}
        for response in responses:
            response.raise_for_status()
            info = response.json()
            worker = info.get("worker_id")
            if not isinstance(worker, str) or not worker:
                raise ValueError("External server did not identify its Ray metrics worker")
            workers.append(worker)
            role = info.get("role")
            if role is not None:
                groups[role].append(worker)
        if enable_pd or any(groups.values()):
            if sum(len(ids) for ids in groups.values()) != len(workers):
                raise ValueError("External PD servers must all report a prefill or decode role")
            self.set_worker_roles(groups)
        else:
            self.set_worker_ids(workers)

    def _role_metrics(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        """Keep the existing metrics under separate prefill and decode scopes."""
        out = {}
        for role, metrics in zip(self._role_scrapers, results):
            for key, value in metrics.items():
                prefix, name = key.rsplit("/", 1)
                out[f"{prefix}/{role}/{name}"] = value
        return out

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        await asyncio.gather(*(scraper.aclose() for scraper in self._role_scrapers.values()))
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _fetch_one(self, client: httpx.AsyncClient, url: str) -> str:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.text
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.debug(f"VLLMMetricsScraper: failed to scrape {url}: {e}")
            return ""

    async def _fetch_all(self) -> ParsedSamples:
        client = await self._get_client()
        texts = await asyncio.gather(*(self._fetch_one(client, u) for u in self._urls))
        self._snapshot_complete = all(bool(text) for text in texts)
        merged: ParsedSamples = {}
        for text in texts:
            if not text:
                continue
            for key, value in parse_metrics_text(text).items():
                # Same (name, labels) tuple should not appear on two nodes for
                # vLLM metrics (ReplicaId is unique), so last-wins is safe.
                merged[key] = value
        return merged

    async def _read_snapshot(self) -> Optional[Dict[str, float]]:
        """Scrape every agent and reduce to one cumulative value per metric.

        Returns ``None`` when no endpoints are configured.
        """
        if not self._urls:
            return None

        parsed = await self._fetch_all()
        if not self._snapshot_complete:
            return None
        if not parsed and not self._warned_empty:
            logger.warning(
                "VLLMMetricsScraper: scraped Ray metrics agents but found no "
                "samples; check that engines were started with "
                "enable_ray_prometheus_stats=true."
            )
            self._warned_empty = True

        if self._worker_ids is not None:
            parsed = {key: value for key, value in parsed.items() if dict(key[1]).get("WorkerId") in self._worker_ids}
        if not parsed:
            self._engine_snapshot = {}
            return None
        engine_counters = {}
        for (name, labels), value in parsed.items():
            label_dict = dict(labels)
            worker = label_dict.get("WorkerId", label_dict.get("ReplicaId"))
            if worker is None:
                continue
            engine = (worker, label_dict.get("engine", "0"))
            if name == _GAUGE_NUM_RUNNING:
                engine_counters.setdefault(engine, {})
            elif name in (_COUNTER_PROMPT_TOKENS, _COUNTER_GENERATION_TOKENS):
                counters = engine_counters.setdefault(engine, {})
                counters[name] = counters.get(name, 0.0) + value
        self._engine_snapshot = engine_counters
        if self._worker_ids is not None and {engine[0] for engine in engine_counters} != self._worker_ids:
            return None
        buckets = {}
        schemas = {}
        for (name, labels), value in parsed.items():
            if name not in {_HIST_TTFT_BUCKET, _HIST_REQUEST_TPOT_BUCKET}:
                continue
            label_dict = dict(labels)
            bound = label_dict.get("le")
            if bound is None:
                continue
            owner = frozenset((k, v) for k, v in labels if k != "le")
            schemas.setdefault(name, {}).setdefault(owner, set()).add(bound)
            key = f"{name}::{bound}"
            buckets[key] = buckets.get(key, 0.0) + value
        for name, owners in schemas.items():
            if len({frozenset(bounds) for bounds in owners.values()}) != 1:
                buckets = {key: value for key, value in buckets.items() if not key.startswith(name + "::")}
        sums = aggregate(parsed, _SUM_METRICS, how="sum")
        means = aggregate(parsed, _MEAN_METRICS, how="mean")
        per_pos = sum_by_position(parsed, _COUNTER_SPEC_ACCEPTED_PER_POS)
        # Ray counters skip zero increments. A live engine gauge confirms the exporter exists.
        if any(name == _GAUGE_NUM_RUNNING for name, _ in parsed):
            sums.setdefault(_COUNTER_PREEMPTIONS, 0.0)
            sums.setdefault(_COUNTER_PROMPT_TOKENS, 0.0)
            sums.setdefault(_COUNTER_GENERATION_TOKENS, 0.0)
        for hits, queries in (
            (_COUNTER_PREFIX_HITS, _COUNTER_PREFIX_QUERIES),
            (_COUNTER_EXTERNAL_PREFIX_HITS, _COUNTER_EXTERNAL_PREFIX_QUERIES),
        ):
            if queries in sums:
                sums.setdefault(hits, 0.0)
        if _COUNTER_KV_OFFLOAD_STORE_BYTES in sums:
            sums.setdefault(_COUNTER_KV_OFFLOAD_LOAD_BYTES, 0.0)
        return {**sums, **means, **per_pos, **buckets}

    async def finalize(self) -> Dict[str, float]:
        """Attempt one final collection and close the HTTP client."""
        try:
            if self._role_scrapers:
                return self._role_metrics(
                    await asyncio.gather(*(scraper.finalize() for scraper in self._role_scrapers.values()))
                )
            if self._label is not None:
                await self.stop()
            else:
                await self.sample()
            return self.run_statistics.summary()
        finally:
            await self.aclose()

    async def sample(self, generation_time_s: Optional[float] = None) -> Dict[str, float]:
        """Return ``vllm/...`` scalars for the current step (empty if unavailable).

        ``generation_time_s`` is the throughput denominator (engine generation
        time since the previous call); ``None`` falls back to the wall-clock
        interval (fully-async overlap).
        """
        if self._role_scrapers:
            return self._role_metrics(
                await asyncio.gather(*(scraper.sample(generation_time_s) for scraper in self._role_scrapers.values()))
            )
        snapshot = await self._read_snapshot()
        if snapshot is None:
            self.run_statistics.incomplete.add("combined")
            self._prev_aggregated = None
            self._prev_timestamp = None
            self._previous_engines = {}
            return {}

        now = time.monotonic()
        if self._prev_aggregated is not None and self._prev_timestamp is not None:
            dt = max(now - self._prev_timestamp, 1e-9)
            window = generation_time_s if (generation_time_s is not None and generation_time_s > 0) else dt
            out = self._window_metrics(self._prev_aggregated, snapshot, window, "vllm/")
            self.run_statistics.add("combined", self._prev_aggregated, snapshot, window)
        else:
            out = self._window_metrics(None, snapshot, None, "vllm/")  # gauges only

        if self._prev_aggregated is not None:
            out.update(engine_imbalance(self._previous_engines, self._engine_snapshot, "vllm/"))
        self._previous_engines = self._engine_snapshot
        self._prev_aggregated = snapshot
        self._prev_timestamp = now
        return out

    async def start(self, label: str) -> None:
        """Open a metrics window labelled ``label`` (e.g. ``"vllm/train"``).

        Snapshots the counters and starts the active-time clock. The window is
        un-paused, so time accumulates immediately; call :meth:`pause` right
        after if the work between ``start`` and the first generation should be
        excluded from the throughput denominator.
        """
        if self._label is not None:
            raise ValueError(f"`start({label!r})` called while window {self._label!r} is still open")
        if self._role_scrapers:
            await asyncio.gather(*(scraper.start(label) for scraper in self._role_scrapers.values()))
            # Both roles begin timing after their baselines are ready.
            started = time.monotonic()
            for scraper in self._role_scrapers.values():
                scraper._active_since = started
            self._label = label
            return
        self._window_prev = await self._read_snapshot()
        self._window_engines = self._engine_snapshot
        self._label = label
        self._window_time_s = 0.0
        self._active_since = time.monotonic()
        self._paused = False

    def pause(self) -> None:
        """Stop accumulating active time until the next :meth:`resume`."""
        if self._role_scrapers:
            for scraper in self._role_scrapers.values():
                scraper.pause()
            return
        if self._label is None:
            raise ValueError("`pause` called without an open window")
        if self._paused:
            raise ValueError("`pause` called without `resume`")
        self._window_time_s += time.monotonic() - self._active_since
        self._active_since = None
        self._paused = True

    def resume(self) -> None:
        """Resume accumulating active time after a :meth:`pause`."""
        if self._role_scrapers:
            for scraper in self._role_scrapers.values():
                scraper.resume()
            return
        if self._label is None:
            raise ValueError("`resume` called without an open window")
        if not self._paused:
            raise ValueError("`resume` called without `pause`")
        self._active_since = time.monotonic()
        self._paused = False

    async def stop(self) -> Dict[str, float]:
        """Close the window and return its metrics nested under ``{label}/``.

        The throughput denominator is the active (un-paused) time accumulated
        between :meth:`start` and now. Returns ``{}`` when no endpoints are
        configured.
        """
        if self._label is None:
            raise ValueError("`stop` called without an open window")
        if self._role_scrapers:
            result = self._role_metrics(await asyncio.gather(*(s.stop() for s in self._role_scrapers.values())))
            self._label = None
            return result
        new_snapshot = await self._read_snapshot()
        if not self._paused:
            self._window_time_s += time.monotonic() - self._active_since
        label, prev, window = self._label, self._window_prev, self._window_time_s
        self._label = None
        self._window_prev = None
        self._active_since = None
        self._paused = False
        self.run_statistics.add(label.removeprefix("vllm/"), prev, new_snapshot, window)
        if new_snapshot is None:
            return {}
        result = self._window_metrics(prev, new_snapshot, window, f"{label}/")
        if prev is not None:
            result.update(engine_imbalance(self._window_engines, self._engine_snapshot, f"{label}/"))
        return result

    @classmethod
    def _window_metrics(
        cls,
        prev: Optional[Dict[str, float]],
        cur: Optional[Dict[str, float]],
        throughput_window_s: Optional[float],
        prefix: str,
    ) -> Dict[str, float]:
        """Gauges from ``cur`` plus derived rates over ``cur - prev``."""
        if cur is None:
            return {}
        out: Dict[str, float] = {}
        if _GAUGE_NUM_RUNNING in cur:
            out[f"{prefix}num_requests_running"] = cur[_GAUGE_NUM_RUNNING]
        if _GAUGE_NUM_WAITING in cur:
            out[f"{prefix}num_requests_waiting"] = cur[_GAUGE_NUM_WAITING]
        if _GAUGE_KV_CACHE_USAGE in cur:
            out[f"{prefix}kv_cache_usage_perc"] = cur[_GAUGE_KV_CACHE_USAGE]
        if prev is not None:
            out.update(cls._derive(cur, prev, throughput_window_s, prefix))
        return out

    @staticmethod
    def _derive(
        cur: Dict[str, float], prev: Dict[str, float], throughput_window_s: Optional[float], prefix: str
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}

        def delta(name: str) -> Optional[float]:
            if name not in cur or name not in prev:
                return None
            d = cur[name] - prev[name]
            # Counter resets (engine restart) shouldn't crash; just skip.
            return d if d >= 0 else None

        # Throughput needs a positive window; the rate/latency metrics don't.
        has_window = throughput_window_s is not None and throughput_window_s > 0

        gen_d = delta(_COUNTER_GENERATION_TOKENS)
        if gen_d is not None and has_window:
            out[f"{prefix}generation_throughput_tok_s"] = gen_d / throughput_window_s

        prompt_d = delta(_COUNTER_PROMPT_TOKENS)
        if prompt_d is not None and has_window:
            out[f"{prefix}prompt_throughput_tok_s"] = prompt_d / throughput_window_s

        q_d = delta(_COUNTER_PREFIX_QUERIES)
        h_d = delta(_COUNTER_PREFIX_HITS)
        if q_d is not None and h_d is not None and q_d > 0:
            out[f"{prefix}prefix_cache_hit_rate"] = h_d / q_d

        out.update(latency_metrics(prev, cur, prefix))
        preemptions = delta(_COUNTER_PREEMPTIONS)
        if preemptions is not None:
            out[prefix + "num_preemptions"] = preemptions
            if gen_d is not None and gen_d > 0:
                out[prefix + "preemptions_per_million_tokens"] = preemptions * 1e6 / gen_d
        external_q = delta(_COUNTER_EXTERNAL_PREFIX_QUERIES)
        external_h = delta(_COUNTER_EXTERNAL_PREFIX_HITS)
        if external_q is not None and external_q > 0 and external_h is not None:
            out[prefix + "external_prefix_cache_hit_rate"] = external_h / external_q
        load_bytes = delta(_COUNTER_KV_OFFLOAD_LOAD_BYTES)
        if load_bytes is not None and has_window:
            out[f"{prefix}kv_offload_load_throughput_bytes_s"] = load_bytes / throughput_window_s

        # Speculative-decoding (MTP draft) acceptance. Counters, so pure deltas over the window --
        # no throughput denominator needed. Keys mirror the legacy metrics: raw draft/accept counts,
        # an overall acceptance rate (accepted / drafted tokens), and one rate per draft position
        # (accepted-at-position / draft rounds), 1-based to match `..._pos_k`.
        drafts_d = delta(_COUNTER_SPEC_DRAFTS)
        drafted_d = delta(_COUNTER_SPEC_DRAFT_TOKENS)
        accepted_d = delta(_COUNTER_SPEC_ACCEPTED_TOKENS)
        if drafted_d is not None:
            out[f"{prefix}draft_num_draft_tokens"] = drafted_d
        if accepted_d is not None:
            out[f"{prefix}draft_num_accepted_tokens"] = accepted_d
        if drafted_d is not None and accepted_d is not None and drafted_d > 0:
            out[f"{prefix}draft_acceptance_rate"] = accepted_d / drafted_d
        if drafts_d is not None and accepted_d is not None and drafts_d > 0:
            # Mean acceptance length (vLLM's definition): 1 always-emitted target token per draft
            # round + the accepted draft tokens per round. This is the average number of tokens
            # produced per target forward pass -- i.e. the spec-decode speedup factor (1.0 == no gain).
            out[f"{prefix}draft_mean_acceptance_length"] = 1 + accepted_d / drafts_d
        if drafts_d is not None and drafts_d > 0:
            for name in cur:
                if not name.startswith(f"{_COUNTER_SPEC_ACCEPTED_PER_POS}::"):
                    continue
                pos_d = delta(name)
                if pos_d is None:
                    continue
                pos = int(name.rsplit("::", 1)[1])
                out[f"{prefix}draft_acceptance_rate_pos_{pos + 1}"] = pos_d / drafts_d

        return out


def engine_imbalance(previous, current, prefix):
    """Return cross-engine CV of token deltas over one common interval."""
    result = {}
    for counter, public in (
        (_COUNTER_PROMPT_TOKENS, "prompt_throughput_cv"),
        (_COUNTER_GENERATION_TOKENS, "generation_throughput_cv"),
    ):
        deltas = []
        for engine, counters in current.items():
            if engine not in previous:
                continue
            # A present engine with no positive-only token counter is idle.
            delta = counters.get(counter, 0.0) - previous[engine].get(counter, 0.0)
            if math.isfinite(delta) and delta >= 0:
                deltas.append(delta)
        if deltas:
            result[prefix + public + "_num_engines"] = len(deltas)
        if len(deltas) >= 2 and statistics.mean(deltas) > 0:
            result[prefix + public] = statistics.pstdev(deltas) / statistics.mean(deltas)
    return result
