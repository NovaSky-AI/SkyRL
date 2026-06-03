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

    The first ``sample()`` call establishes a baseline; rate-style metrics
    (throughput, hit rate, average latency) are reported starting from the
    second call.
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

    async def sample(self) -> Dict[str, float]:
        """Return a dict of ``vllm/...`` scalars for the current step.

        Empty if no agents are reachable or if no vLLM samples are present
        yet (e.g. before any inference has run).
        """
        if not self._urls:
            return {}

        parsed = await self._fetch_all()
        if not parsed and not self._warned_empty:
            logger.warning(
                "VLLMMetricsScraper: scraped Ray metrics agents but found no "
                "samples; check that engines were started with "
                "enable_ray_prometheus_stats=true."
            )
            self._warned_empty = True

        now = time.monotonic()

        sums = aggregate(parsed, _SUM_METRICS, how="sum")
        means = aggregate(parsed, _MEAN_METRICS, how="mean")
        snapshot = {**sums, **means}

        out: Dict[str, float] = {}
        # Instantaneous gauges expose directly.
        if _GAUGE_NUM_RUNNING in snapshot:
            out["vllm/num_requests_running"] = snapshot[_GAUGE_NUM_RUNNING]
        if _GAUGE_NUM_WAITING in snapshot:
            out["vllm/num_requests_waiting"] = snapshot[_GAUGE_NUM_WAITING]
        if _GAUGE_KV_CACHE_USAGE in snapshot:
            out["vllm/kv_cache_usage_perc"] = snapshot[_GAUGE_KV_CACHE_USAGE]

        # Derived metrics need a previous snapshot to take deltas.
        if self._prev_aggregated is not None and self._prev_timestamp is not None:
            dt = max(now - self._prev_timestamp, 1e-9)
            out.update(self._derive(snapshot, self._prev_aggregated, dt))

        self._prev_aggregated = snapshot
        self._prev_timestamp = now
        return out

    @staticmethod
    def _derive(cur: Dict[str, float], prev: Dict[str, float], dt: float) -> Dict[str, float]:
        out: Dict[str, float] = {}

        def delta(name: str) -> Optional[float]:
            if name not in cur or name not in prev:
                return None
            d = cur[name] - prev[name]
            # Counter resets (engine restart) shouldn't crash; just skip.
            return d if d >= 0 else None

        gen_d = delta(_COUNTER_GENERATION_TOKENS)
        if gen_d is not None:
            out["vllm/generation_throughput_tok_s"] = gen_d / dt

        prompt_d = delta(_COUNTER_PROMPT_TOKENS)
        if prompt_d is not None:
            out["vllm/prompt_throughput_tok_s"] = prompt_d / dt

        q_d = delta(_COUNTER_PREFIX_QUERIES)
        h_d = delta(_COUNTER_PREFIX_HITS)
        if q_d is not None and h_d is not None and q_d > 0:
            out["vllm/prefix_cache_hit_rate"] = h_d / q_d

        ttft_sum_d = delta(_HIST_TTFT_SUM)
        ttft_count_d = delta(_HIST_TTFT_COUNT)
        if ttft_sum_d is not None and ttft_count_d is not None and ttft_count_d > 0:
            out["vllm/ttft_seconds_avg"] = ttft_sum_d / ttft_count_d

        itl_sum_d = delta(_HIST_ITL_SUM)
        itl_count_d = delta(_HIST_ITL_COUNT)
        if itl_sum_d is not None and itl_count_d is not None and itl_count_d > 0:
            out["vllm/tpot_seconds_avg"] = itl_sum_d / itl_count_d

        return out
