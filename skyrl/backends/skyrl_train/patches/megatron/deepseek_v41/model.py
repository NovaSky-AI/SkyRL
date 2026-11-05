"""DeepSeek-V4.1 changes to megatron-core's mHC ``HybridModel`` (all keyed on ``config.dsv4_version == "v4.1"``).

Parity reference: ``Block.forward`` / ``Transformer.forward`` in the checkpoint's ``inference/model.py``.

- Delayed pre-mix: a sublayer's mHC mappings give its own post / comb, but its ``pre`` (the weights that
  collapse the streams into the sublayer input) goes to the *next* sublayer. The first sublayer collapses
  with a one-hot on stream 0. V4 and megatron-core use each sublayer's own ``pre``.
- No ``hc_head``: the final collapse uses the last sublayer's ``pre`` (V4 has a learned output contraction).
  The stack's ``hc_head_*`` parameters are removed.
- Engram: ``engram_layer_ids`` add an n-gram lookup to the mHC streams before the block's attention.

Per-forward state (``attention.forward_state()``: the attention's ``CSA2State``, the pending ``pre``, engram hash
ids) is reset when ``HybridModel.forward`` starts.
"""

import functools
import os

import torch

from skyrl.backends.skyrl_train.patches.megatron.deepseek_v41.attention import (
    forward_state,
    is_v41,
    logical_layer_id,
)


# --- context parallelism (contiguous layout inside the stack, SkyRL's zigzag at its boundary) ---
# SkyRL packs one sequence per microbatch and shards it zigzag over 2 * cp chunks (rank r: chunks r and 2cp-1-r).
# Every V4.1 layer runs in the contiguous layout (rank r: chunks 2r, 2r+1): megatron-core converts the hidden
# states once at the stack's entry and exit, and the per-token side inputs below are converted here.


def cp_contiguous_group(config):
    """The CP group when V4.1 runs context-parallel in the contiguous layout, else None."""
    if not is_v41(config) or getattr(config, "attention_cp_layout", "zigzag") != "contiguous":
        return None
    import megatron.core.parallel_state as mpu

    group = mpu.get_context_parallel_group(check_initialized=False)
    return group if group is not None and group.size() > 1 else None


def _zigzag_order(cp: int):
    """Global chunk id of each rank-major zigzag chunk: [0, 2cp-1, 1, 2cp-2, ...]."""
    return [c for r in range(cp) for c in (r, 2 * cp - 1 - r)]


@torch.no_grad()
def zigzag_full(t: torch.Tensor, group, dim: int = 0) -> torch.Tensor:
    """All-gather this rank's zigzag rows of one sequence and return the whole sequence in order (no grad)."""
    cp = group.size()
    x = t.movedim(dim, 0).contiguous()
    parts = [torch.empty_like(x) for _ in range(cp)]
    torch.distributed.all_gather(parts, x, group=group)
    half = x.size(0) // 2
    chunks = [None] * (2 * cp)
    for r, part in enumerate(parts):
        chunks[r], chunks[2 * cp - 1 - r] = part[:half], part[half:]
    return torch.cat(chunks, dim=0).movedim(0, dim)


def contiguous_rows(full: torch.Tensor, group, dim: int = 0) -> torch.Tensor:
    """This rank's contiguous slice of a whole-sequence tensor."""
    n = full.size(dim) // group.size()
    return full.narrow(dim, group.rank() * n, n)


def contiguous_to_zigzag(t: torch.Tensor, group) -> torch.Tensor:
    """Differentiable: this rank's contiguous rows -> its zigzag rows (all-gather; backward reduce-scatter)."""
    from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

    cp = group.size()
    full = gather_from_sequence_parallel_region(t, tensor_parallel_output_grad=True, group=group)
    c = full.size(0) // (2 * cp)
    r = group.rank()
    return torch.cat([full[r * c : (r + 1) * c], full[(2 * cp - 1 - r) * c : (2 * cp - r) * c]], dim=0)

def _cp_batch(packed_seq_params, group):
    """megatron-core's ``ContextParallelBatch`` for one SkyRL packed microbatch: the zigzag (data) and contiguous
    (every V4.1 layer) views share SkyRL's global packed metadata, and the THD plan routes rows between them."""
    from megatron.core.context_parallel.layout import (
        _build_thd_zigzag_metadata,
        build_thd_cp_layout_plan,
    )
    from megatron.core.context_parallel.utils import ContextParallelBatch

    cu = packed_seq_params.cu_seqlens_q_padded
    cu = packed_seq_params.cu_seqlens_q if cu is None else cu
    meta = _build_thd_zigzag_metadata(cu, cu, group.size(), 1)
    total = int(cu[-1])
    assert int(meta.cu_seqlens_padded[-1]) == total, "SkyRL's CP padding must match megatron-core's zigzag padding"
    plan = build_thd_cp_layout_plan(meta.rank_order_indices, total, group)
    return ContextParallelBatch(
        boundary_layout="zigzag",
        batches_by_layout={"zigzag": {}, "contiguous": {}},
        packed_seq_params_by_layout={"zigzag": packed_seq_params, "contiguous": packed_seq_params},
        thd_plan=plan,
    )


# Debug (SKYRL_DSV41_DUMP_DIR): the first 256 rows of each hybrid layer's input and output for data-parallel rank 0's
# first microbatch, per CP rank, to diff a CP run against CP=1 layer by layer.
_DUMP_DIR = os.environ.get("SKYRL_DSV41_DUMP_DIR", "")
_DUMPED: set = set()


def _dump_layer(layer, hidden_in, out):
    import megatron.core.parallel_state as mpu

    if mpu.get_data_parallel_rank() != 0:
        return
    n = getattr(layer, "layer_number", -1)
    if n in _DUMPED:
        return
    _DUMPED.add(n)
    o = out[0] if isinstance(out, tuple) else out
    cp_rank = mpu.get_context_parallel_rank()
    rows = 256
    torch.save(
        {"in": hidden_in[:rows].detach().float().cpu(), "out": o[:rows].detach().float().cpu(),
         "out_rownorm": o.detach().float().flatten(1).norm(dim=1).cpu(), "cp_rank": cp_rank,
         "cp_size": mpu.get_context_parallel_world_size()},
        os.path.join(_DUMP_DIR, f"layer{n:03d}_cp{cp_rank}.pt"),
    )


_APPLIED = False
# The V4.1 stack's config (set when it is built), for module-level hooks that get no module.
hybrid_block_config: list = [None]


@functools.lru_cache(maxsize=2)
def get_hasher(engram_spec):
    from skyrl.backends.skyrl_train.patches.megatron.deepseek_v41.engram import (
        NgramHasher,
    )

    return NgramHasher(engram_spec)


def _engram(config):
    """Engram settings (upstream: NVIDIA/Megatron-LM#8225), kept apart from the CSA2 attention roles."""
    return getattr(config, "dsv41_engram", None)


def apply_deepseek_v41_patches() -> bool:
    """Install the V4.1 hooks on megatron-core's hybrid model. Idempotent; inert for other models."""
    global _APPLIED
    if _APPLIED:
        return True
    if os.environ.get("SKYRL_DSV41_FAULTHANDLER") == "1":
        # Debug aid for collective hangs and async CUDA faults (py-spy needs ptrace): `kill -USR1 <worker pid>`,
        # or the NCCL watchdog's SIGABRT, writes every thread's Python stack to the worker's stderr.
        import faulthandler
        import signal

        faulthandler.enable(all_threads=True)
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    try:
        from megatron.core.models.hybrid import hybrid_block, hybrid_model
        from megatron.core.models.hybrid.layers import hybrid_hyper_connection
        from megatron.core.transformer import hyper_connection
    except ImportError:
        return False

    # --- per-forward state: reset, hash the input ids once for every engram layer ---
    model_forward = hybrid_model.HybridModel.forward

    @functools.wraps(model_forward)
    def forward(self, input_ids, *args, **kwargs):
        if is_v41(self.config):
            state = forward_state()
            state.reset()
            engram = _engram(self.config)
            cp_group = cp_contiguous_group(self.config)
            if engram is not None and input_ids is not None:
                hasher = get_hasher(engram)
                if cp_group is None:
                    # [b, s, L, cols] -> [s, b, L, cols] to match the [s, b, ...] activations.
                    state.engram_hashes = hasher(input_ids).transpose(0, 1).contiguous()
                else:
                    # n-grams reach back across chunk boundaries: hash the whole sequence, keep this rank's
                    # contiguous rows (the layout the stack runs in).
                    full = hasher(zigzag_full(input_ids, cp_group, dim=1))
                    state.engram_hashes = contiguous_rows(full, cp_group, dim=1).transpose(0, 1).contiguous()
            if cp_group is not None:
                # Router replay targets were sharded zigzag with the tokens; the routers see contiguous rows.
                from megatron.core.transformer.moe.router_replay import RouterReplay

                for router in RouterReplay.global_router_replay_instances:
                    target = router.target_topk_idx
                    if target is not None and not getattr(target, "_skyrl_cp_contiguous", False):
                        target = contiguous_rows(zigzag_full(target, cp_group), cp_group)
                        target._skyrl_cp_contiguous = True
                        router.target_topk_idx = target
                packed = kwargs.get("packed_seq_params")
                if packed is not None and kwargs.get("cp_batch") is None:
                    kwargs["cp_batch"] = _cp_batch(packed, cp_group)
        return model_forward(self, input_ids, *args, **kwargs)

    hybrid_model.HybridModel.forward = forward

    # --- delayed pre-mix ---
    compute_mappings = hyper_connection.HyperConnectionModule.compute_mappings

    @functools.wraps(compute_mappings)
    def delayed_compute_mappings(self, x):
        h_pre, h_post, h_res = compute_mappings(self, x)
        if not is_v41(self.config):
            return h_pre, h_post, h_res
        state = forward_state()
        used = getattr(state, "pending_pre", None)
        if used is None:
            used = torch.zeros_like(h_pre)
            used[..., 0] = 1
        state.pending_pre = h_pre
        return used, h_post, h_res

    hyper_connection.HyperConnectionModule.compute_mappings = delayed_compute_mappings

    # --- final collapse with the last sublayer's pre instead of hc_head ---
    contract = hybrid_block.learned_output_contract

    @functools.wraps(contract)
    def output_contract(hidden_states, head_fn, base, scale, n, eps):
        if head_fn is not None:
            return contract(hidden_states, head_fn, base, scale, n, eps)
        pre = forward_state().pending_pre
        cp_group = cp_contiguous_group(hybrid_block_config[0]) if hybrid_block_config else None
        if cp_group is not None:
            # The stack has converted the hidden states back to the zigzag boundary layout; pre is still contiguous.
            pre = contiguous_to_zigzag(pre, cp_group)
        pre = pre.float()
        streams = hidden_states.float().view(*hidden_states.shape[:-1], n, -1)
        return torch.sum(pre.unsqueeze(-1) * streams, dim=-2).to(hidden_states.dtype)

    hybrid_block.learned_output_contract = output_contract

    stack_init = hybrid_block.HybridStack.__init__

    @functools.wraps(stack_init)
    def init_stack(self, config, *args, **kwargs):
        if is_v41(config) and getattr(config, "attention_cp_layout", "zigzag") == "contiguous":
            # Every layer contiguous (linear_cp_layout too, so the mHC / engram side state never changes layout
            # between sublayers) while the data stays in SkyRL's zigzag shards: convert only at the boundary.
            assert getattr(config, "linear_cp_layout", "zigzag") == "contiguous", "set linear_cp_layout=contiguous"
            kwargs.setdefault("boundary_layout", "zigzag")
            hybrid_block_config[0] = config
        stack_init(self, config, *args, **kwargs)
        if is_v41(config) and hasattr(self, "hc_head_fn"):
            self.hc_head_fn = None
            self.hc_head_base = None
            self.hc_head_scale = None

    hybrid_block.HybridStack.__init__ = init_stack

    # --- engram before the attention of engram layers ---
    layer_cls = hybrid_hyper_connection.HyperConnectionHybridLayer
    layer_init = layer_cls.__init__

    @functools.wraps(layer_init)
    def init_layer(self, config, layer, *args, **kwargs):
        layer_init(self, config, layer, *args, **kwargs)
        self.engram = None
        engram = _engram(config) if is_v41(config) else None
        if engram is None or self.layer_number % 2 == 0:
            return
        layer_id = logical_layer_id(self.layer_number)
        if layer_id in engram.layer_ids:
            from skyrl.backends.skyrl_train.patches.megatron.deepseek_v41.engram import (
                DSv41Engram,
            )

            self.engram = DSv41Engram(config, layer_id, engram, group_size=config.dsv41_engram_group_size)
            self.engram_hash_index = engram.layer_ids.index(layer_id)

    layer_cls.__init__ = init_layer

    layer_forward = layer_cls.forward

    @functools.wraps(layer_forward)
    def forward_layer(self, hidden_states, *args, **kwargs):
        if getattr(self, "engram", None) is not None:
            hashes = forward_state().engram_hashes
            assert hashes is not None, "DeepSeek-V4.1 engram needs the model's input_ids"
            hidden_states = self.engram(hidden_states, hashes[:, :, self.engram_hash_index])
        compressed = getattr(self, "_skyrl_frozen_store", None) is not None
        if getattr(self.config, "dsv41_stateful_recompute", False) and self.training and torch.is_grad_enabled():
            from skyrl.backends.skyrl_train.patches.megatron.deepseek_v41.recompute import (
                stateful_layer_forward,
            )

            run_layer = layer_forward
            if compressed:
                from skyrl.backends.skyrl_train.patches.megatron.deepseek_v41.frozen_weights import (
                    materialize,
                )

                def run_layer(layer, *a, **k):
                    # Both the forward and its replay in backward decompress this layer's weights first.
                    materialize(layer)
                    return layer_forward(layer, *a, **k)

            return stateful_layer_forward(run_layer, self, hidden_states, args, kwargs, forward_state)
        if compressed:
            from skyrl.backends.skyrl_train.patches.megatron.deepseek_v41.frozen_weights import (
                materialize,
            )

            materialize(self, grad_forward=torch.is_grad_enabled())
        out = layer_forward(self, hidden_states, *args, **kwargs)
        if _DUMP_DIR:
            _dump_layer(self, hidden_states, out)
        return out

    layer_cls.forward = forward_layer

    _APPLIED = True
    return True
