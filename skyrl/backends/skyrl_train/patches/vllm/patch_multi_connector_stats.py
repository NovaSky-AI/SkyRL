"""Runtime patch: don't crash on KV-connector stats without Prometheus metrics.

vLLM 0.23's ``MultiKVConnectorPromMetrics.observe`` asserts that every child
connector reporting transfer stats is registered with a Prometheus metrics
class. Some connectors (e.g. ``MooncakeConnector``) produce ``KVConnectorStats``
but expose no prom-metrics class, so with a ``MultiConnector`` stack like
MooncakeConnector + MooncakeStoreConnector the assert fires inside
``AsyncLLM.output_handler`` on the first stats flush — killing the output
handler and hanging every in-flight request.

Patch: skip children without registered prom metrics (their stats still appear
in the periodic "KV Transfer metrics" log lines; only the Prometheus export is
unavailable for them).

TODO: drop once fixed upstream.
"""

from loguru import logger

_PATCHED = False


def apply_multi_connector_stats_patch() -> None:
    """Install the tolerant ``MultiKVConnectorPromMetrics.observe`` (idempotent)."""
    global _PATCHED
    if _PATCHED:
        return

    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiKVConnectorPromMetrics,
    )

    def observe(self, transfer_stats_data, engine_idx: int = 0):
        for connector_id, stats_data in transfer_stats_data.items():
            prom = self._prom_metrics.get(connector_id)
            if prom is None:
                # Child connector has stats but no prom-metrics class registered
                # (e.g. MooncakeConnector). Skip instead of asserting.
                continue
            prom.observe(stats_data["data"], engine_idx)

    MultiKVConnectorPromMetrics.observe = observe
    _PATCHED = True
    logger.info("SkyRL: installed tolerant MultiKVConnectorPromMetrics.observe patch")
