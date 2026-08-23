"""Wire protocol for the sharded-RDT control plane (dependency-light).

The RDT control plane is a serial ``init -> start -> update -> finish`` handshake
routed through the SkyRL worker extension's ``/collective_rpc`` (so the bake runs
under ``set_current_vllm_config``), not vLLM's native weight-transfer endpoints.

Two clients speak it: the async ``RemoteInferenceClient`` (over aiohttp) and the
trainer-side ``SyncRdtControlPlaneClient`` (over blocking HTTP; see
``weight_sync/sharded_rdt/rdt_control_plane.py``). This module is the single source of truth
for the method names and the per-server ``replica_rank`` fan-out so the two can
never drift. It is intentionally stdlib-only — no ray / torch / vllm — so either
client can import it without pulling backend deps.
"""

from typing import Any, Dict, List, Sequence, Tuple

COLLECTIVE_RPC_ENDPOINT = "/collective_rpc"

# Worker-extension methods each /collective_rpc call dispatches to.
RDT_INIT_METHOD = "init_weight_transfer_engine_rdt"
RDT_START_METHOD = "skyrl_start_weight_update"
RDT_UPDATE_METHOD = "update_weights_rdt"
RDT_FINISH_METHOD = "skyrl_finish_weight_update"


def build_rdt_init_payloads(
    init_info: Dict[str, Any],
    server_urls: Sequence[str],
    data_parallel_size: int,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Per-server ``/collective_rpc`` payloads for the RDT engine bake + init.

    Each independent inference *deployment* has its own self-contained parallel
    config, so the vLLM engine can't tell deployments apart — every deployment's
    internal worker index restarts at 0 and would collide under the M:N block
    assignment. So we stamp each server with its deployment ordinal
    (``server_index // data_parallel_size``) as ``replica_rank`` and the
    deployment count as ``num_replicas``; the engine offsets its consumers into a
    globally distinct range from those two fields. Every other field is shared.

    The replica ordinal divides by ``data_parallel_size`` because the DP servers
    of one deployment share a parallel config (vLLM's ``data_parallel_index``
    already separates them), so they must share ONE ``replica_rank`` or a DP
    deployment would double-count.
    """
    dp = max(1, data_parallel_size)
    num_replicas = max(1, len(server_urls) // dp)
    return [
        (
            url,
            {
                "method": RDT_INIT_METHOD,
                "kwargs": {
                    "init_info": {
                        **init_info,
                        "replica_rank": i // dp,
                        "num_replicas": num_replicas,
                    },
                },
            },
        )
        for i, url in enumerate(server_urls)
    ]
