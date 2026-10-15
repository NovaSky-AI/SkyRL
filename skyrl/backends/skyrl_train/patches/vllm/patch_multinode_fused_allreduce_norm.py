"""Skip FlashInfer's fused all-reduce + RMSNorm when the TP group spans nodes.

vLLM 0.30's DeepSeek-V3.2 model code (which serves GLM-5 / GLM-5.3, ``glm_moe_dsa``) calls
``models.common.ops.fused_allreduce_rms_norm`` in every decoder layer whenever TP > 1. That asks
``get_fi_ar_workspace`` for a FlashInfer all-reduce workspace, which on a multi-node TP group
auto-selects the ``mnnvl`` backend. Without multi-node NVLink (e.g. B200 nodes over RoCE) its
CUDA fd exchange times out after 30 s, and the failure is not cached, so every forward call
retries: a TP16 engine across two nodes never finishes its profile run. (Single-node TP falls
back to the ``trtllm`` backend and is unaffected.)

When the tensor-parallel group is larger than one node's share of ranks, report the FlashInfer
path as unavailable so ``fused_allreduce_rms_norm`` takes its own fallback: a NCCL all-reduce
followed by RMSNorm. No-op for single-node TP.
"""

from loguru import logger

_PATCHED = False


def apply_multinode_fused_allreduce_norm_patch() -> bool:
    """Install the patch once per process. Returns True if installed."""
    global _PATCHED
    if _PATCHED:
        return False
    try:
        from vllm.distributed import parallel_state
        from vllm.models.common.ops import fused_allreduce_rms_norm as fused_op
    except ImportError:
        return False
    original = getattr(fused_op, "_can_use_flashinfer", None)
    if original is None:
        return False

    spans_nodes: list[bool] = []  # computed once, after distributed init

    def _tp_group_spans_nodes() -> bool:
        if not spans_nodes:
            nodes = parallel_state.get_node_count()
            world_size = parallel_state.get_world_group().world_size
            tp_size = parallel_state.get_tensor_model_parallel_world_size()
            spans_nodes.append(nodes > 1 and tp_size > world_size // nodes)
        return spans_nodes[0]

    def _can_use_flashinfer(hidden_states, tp_size):
        if _tp_group_spans_nodes():
            return False, 0
        return original(hidden_states, tp_size)

    fused_op._can_use_flashinfer = _can_use_flashinfer
    _PATCHED = True
    logger.info("Installed multi-node fused all-reduce + RMSNorm patch")
    return True
