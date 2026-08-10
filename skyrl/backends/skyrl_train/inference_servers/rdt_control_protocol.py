"""Wire protocol for the sharded-RDT control plane (dependency-light).

The RDT control plane is a serial ``init -> start -> update -> finish`` handshake
routed through the SkyRL worker extension's ``/collective_rpc`` (so the bake runs
under ``set_current_vllm_config``), not vLLM's native weight-transfer endpoints.

Two clients speak it: the async ``RemoteInferenceClient`` (over aiohttp) and the
trainer-side ``SyncRdtControlPlaneClient`` (over blocking HTTP; see
``weight_sync/rdt_control_plane.py``). This module is the single source of truth
for the method names and the per-server ``replica_rank`` fan-out so the two can
never drift. It is intentionally stdlib-only — no ray / torch / vllm — so either
client can import it without pulling backend deps.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

COLLECTIVE_RPC_ENDPOINT = "/collective_rpc"

# Worker-extension methods each /collective_rpc call dispatches to.
RDT_INIT_METHOD = "init_weight_transfer_engine_rdt"
RDT_START_METHOD = "skyrl_start_weight_update"
RDT_UPDATE_METHOD = "update_weights_rdt"
RDT_FINISH_METHOD = "skyrl_finish_weight_update"
RDT_ABORT_METHOD = "skyrl_abort_weight_update"
"""Abandon a half-finished update so the SURVIVING engines can take a retry.
Without it the retry's start call fails with "already active" (Part 2, §5.5)."""


class WeightSyncAborted(RuntimeError):
    """A weight sync was abandoned because an inference engine died inside it.

    Raised on the trainer ranks and caught on the DRIVER, which is the only place a
    retry can be issued from: the gather is a collective, so re-entering it means
    re-dispatching to every rank, not retrying inside one.

    Defined in this stdlib-only module on purpose. It crosses a Ray boundary, so the
    class has to be importable on the driver and on every worker without dragging in
    torch or vllm -- and it has to carry its payload in ``args`` so Ray's pickling of
    the exception preserves it.

    ``newly_dead_slots`` is what the driver acts on: the slots whose engines failed
    this sync, so the retry's live set can exclude them without waiting for a probe.
    """

    def __init__(self, message: str, newly_dead_slots: Sequence[int] = ()) -> None:
        # Through args, so the round trip through Ray's exception pickling keeps it.
        super().__init__(message, tuple(int(s) for s in newly_dead_slots))

    @property
    def message(self) -> str:
        return self.args[0] if self.args else ""

    @property
    def newly_dead_slots(self) -> Tuple[int, ...]:
        return self.args[1] if len(self.args) > 1 else ()


def build_rdt_init_payloads(
    init_info: Dict[str, Any],
    server_urls: Sequence[str],
    data_parallel_size: int,
    *,
    slots: Optional[Sequence[int]] = None,
    num_replicas: Optional[int] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Per-server ``/collective_rpc`` payloads for the RDT engine bake + init.

    Each independent inference *deployment* has its own self-contained parallel
    config, so the vLLM engine can't tell deployments apart — every deployment's
    internal worker index restarts at 0 and would collide under the M:N block
    assignment. So we stamp each server with its deployment ordinal as
    ``replica_rank`` and the deployment count as ``num_replicas``; the engine
    offsets its consumers into a globally distinct range from those two fields.
    Every other field is shared.

    The replica ordinal divides by ``data_parallel_size`` because the DP servers
    of one deployment share a parallel config (vLLM's ``data_parallel_index``
    already separates them), so they must share ONE ``replica_rank`` or a DP
    deployment would double-count.

    Args:
        slots: Explicit ``replica_rank`` per URL, defeating the positional
            derivation. Required whenever ``server_urls`` is not the whole
            provisioned fleet in provisioned order — re-initializing ONE restarted
            engine, say, where position would compute 0 for whatever slot it
            actually holds. ``None`` keeps the positional rule
            (``index // data_parallel_size``), which is what every static fleet
            wants and is byte-identical to the pre-fault-tolerance behaviour.
        num_replicas: The PROVISIONED deployment count. Also required with a
            partial URL list: derived from ``len(server_urls)`` it would shrink,
            and since the engine sizes its consumer-id offset from it, a shrunk
            value silently re-maps every surviving consumer onto another's slice.
            Defaults to the derivation when ``None``.

    Raises:
        ValueError: ``slots`` has a different length than ``server_urls``, or
            ``num_replicas`` cannot cover the largest slot given — both mean the
            caller's picture of the fleet is inconsistent, and the failure mode if
            it went through is silent slice corruption rather than an error.
    """
    dp = max(1, data_parallel_size)
    if slots is None:
        ranks = [i // dp for i in range(len(server_urls))]
    else:
        ranks = list(slots)
        if len(ranks) != len(server_urls):
            raise ValueError(f"build_rdt_init_payloads got {len(ranks)} slots for {len(server_urls)} server_urls.")
    replicas = max(1, len(server_urls) // dp) if num_replicas is None else int(num_replicas)
    if ranks and max(ranks) >= replicas:
        raise ValueError(
            f"build_rdt_init_payloads got replica_rank {max(ranks)} with num_replicas={replicas}: "
            "every rank must be addressable within the provisioned deployment count, or the "
            "engine's consumer-id offset would collide with another deployment's block."
        )
    return [
        (
            url,
            {
                "method": RDT_INIT_METHOD,
                "kwargs": {
                    "init_info": {
                        **init_info,
                        "replica_rank": rank,
                        "num_replicas": replicas,
                    },
                },
            },
        )
        for url, rank in zip(server_urls, ranks)
    ]
