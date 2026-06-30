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
import re
import time
from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple

import httpx
import ray
from loguru import logger

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
_HIST_TTFT_SUM = "ray_vllm_time_to_first_token_seconds_sum"
_HIST_TTFT_COUNT = "ray_vllm_time_to_first_token_seconds_count"
_HIST_ITL_SUM = "ray_vllm_inter_token_latency_seconds_sum"
_HIST_ITL_COUNT = "ray_vllm_inter_token_latency_seconds_count"

_SUM_METRICS = (
    _GAUGE_NUM_RUNNING,
    _GAUGE_NUM_WAITING,
    _COUNTER_PREFIX_QUERIES,
    _COUNTER_PREFIX_HITS,
    _COUNTER_PROMPT_TOKENS,
    _COUNTER_GENERATION_TOKENS,
    _HIST_TTFT_SUM,
    _HIST_TTFT_COUNT,
    _HIST_ITL_SUM,
    _HIST_ITL_COUNT,
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
        urls.append(f"http://{ip}:{port}/metrics")
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
    ):
        self._urls = urls if urls is not None else discover_ray_metrics_urls()
        self._timeout = request_timeout_s
        self._prev_aggregated: Optional[Dict[str, float]] = None
        self._prev_timestamp: Optional[float] = None
        self._client: Optional[httpx.AsyncClient] = None
        self._warned_empty = False
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

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
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
        if not parsed and not self._warned_empty:
            logger.warning(
                "VLLMMetricsScraper: scraped Ray metrics agents but found no "
                "samples; check that engines were started with "
                "enable_ray_prometheus_stats=true."
            )
            self._warned_empty = True

        sums = aggregate(parsed, _SUM_METRICS, how="sum")
        means = aggregate(parsed, _MEAN_METRICS, how="mean")
        return {**sums, **means}

    async def sample(self, generation_time_s: Optional[float] = None) -> Dict[str, float]:
        """Return ``vllm/...`` scalars for the current step (empty if unavailable).

        ``generation_time_s`` is the throughput denominator (engine generation
        time since the previous call); ``None`` falls back to the wall-clock
        interval (fully-async overlap).
        """
        snapshot = await self._read_snapshot()
        if snapshot is None:
            return {}

        now = time.monotonic()
        if self._prev_aggregated is not None and self._prev_timestamp is not None:
            dt = max(now - self._prev_timestamp, 1e-9)
            window = generation_time_s if (generation_time_s is not None and generation_time_s > 0) else dt
            out = self._window_metrics(self._prev_aggregated, snapshot, window, "vllm/")
        else:
            out = self._window_metrics(None, snapshot, None, "vllm/")  # gauges only

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
        self._window_prev = await self._read_snapshot()
        self._label = label
        self._window_time_s = 0.0
        self._active_since = time.monotonic()
        self._paused = False

    def pause(self) -> None:
        """Stop accumulating active time until the next :meth:`resume`."""
        if self._label is None:
            raise ValueError("`pause` called without an open window")
        if self._paused:
            raise ValueError("`pause` called without `resume`")
        self._window_time_s += time.monotonic() - self._active_since
        self._active_since = None
        self._paused = True

    def resume(self) -> None:
        """Resume accumulating active time after a :meth:`pause`."""
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
        new_snapshot = await self._read_snapshot()
        if not self._paused:
            self._window_time_s += time.monotonic() - self._active_since
        label, prev, window = self._label, self._window_prev, self._window_time_s
        self._label = None
        self._window_prev = None
        self._active_since = None
        self._paused = False
        if new_snapshot is None:
            return {}
        return self._window_metrics(prev, new_snapshot, window, f"{label}/")

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

        ttft_sum_d = delta(_HIST_TTFT_SUM)
        ttft_count_d = delta(_HIST_TTFT_COUNT)
        if ttft_sum_d is not None and ttft_count_d is not None and ttft_count_d > 0:
            out[f"{prefix}ttft_seconds_avg"] = ttft_sum_d / ttft_count_d

        itl_sum_d = delta(_HIST_ITL_SUM)
        itl_count_d = delta(_HIST_ITL_COUNT)
        if itl_sum_d is not None and itl_count_d is not None and itl_count_d > 0:
            out[f"{prefix}tpot_seconds_avg"] = itl_sum_d / itl_count_d

        return out
