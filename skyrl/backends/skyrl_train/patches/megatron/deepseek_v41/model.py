"""DeepSeek-V4.1 changes to megatron-core's mHC ``HybridModel`` (all keyed on ``config.dsv41_spec``).

Parity reference: ``Block.forward`` / ``Transformer.forward`` in the checkpoint's ``inference/model.py``.

- Delayed pre-mix: a sublayer's mHC mappings give its own post / comb, but its ``pre`` (the weights that
  collapse the streams into the sublayer input) goes to the *next* sublayer. The first sublayer collapses
  with a one-hot on stream 0. V4 and megatron-core use each sublayer's own ``pre``.
- No ``hc_head``: the final collapse uses the last sublayer's ``pre`` (V4 has a learned output contraction).
  The stack's ``hc_head_*`` parameters are removed.
- Engram: ``engram_layer_ids`` add an n-gram lookup to the mHC streams before the block's attention.

Per-forward state (the pending ``pre``, engram hash ids, the shared compressed KV / top-k in
``attention.shared_state()``) is reset when ``HybridModel.forward`` starts.
"""

import functools

import torch

from skyrl.backends.skyrl_train.patches.megatron.deepseek_v41.attention import (
    logical_layer_id,
    shared_state,
)

_APPLIED = False


@functools.lru_cache(maxsize=2)
def get_hasher(engram_spec):
    from skyrl.backends.skyrl_train.patches.megatron.deepseek_v41.engram import (
        NgramHasher,
    )

    return NgramHasher(engram_spec)


def _spec(config):
    return getattr(config, "dsv41_spec", None)


def apply_deepseek_v41_patches() -> bool:
    """Install the V4.1 hooks on megatron-core's hybrid model. Idempotent; inert for other models."""
    global _APPLIED
    if _APPLIED:
        return True
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
        spec = _spec(self.config)
        if spec is not None:
            state = shared_state()
            state.reset()
            if spec.get("engram") is not None and input_ids is not None:
                hasher = get_hasher(spec["engram"])
                # [b, s, L, cols] -> [s, b, L, cols] to match the [s, b, ...] activations.
                state.engram_hashes = hasher(input_ids).transpose(0, 1).contiguous()
        return model_forward(self, input_ids, *args, **kwargs)

    hybrid_model.HybridModel.forward = forward

    # --- delayed pre-mix ---
    compute_mappings = hyper_connection.HyperConnectionModule.compute_mappings

    @functools.wraps(compute_mappings)
    def delayed_compute_mappings(self, x):
        h_pre, h_post, h_res = compute_mappings(self, x)
        if _spec(self.config) is None:
            return h_pre, h_post, h_res
        state = shared_state()
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
        pre = shared_state().pending_pre.float()
        streams = hidden_states.float().view(*hidden_states.shape[:-1], n, -1)
        return torch.sum(pre.unsqueeze(-1) * streams, dim=-2).to(hidden_states.dtype)

    hybrid_block.learned_output_contract = output_contract

    stack_init = hybrid_block.HybridStack.__init__

    @functools.wraps(stack_init)
    def init_stack(self, config, *args, **kwargs):
        stack_init(self, config, *args, **kwargs)
        if _spec(config) is not None and hasattr(self, "hc_head_fn"):
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
        spec = _spec(config)
        if spec is None or spec.get("engram") is None or self.layer_number % 2 == 0:
            return
        layer_id = logical_layer_id(self.layer_number)
        if layer_id in spec["engram"].layer_ids:
            from skyrl.backends.skyrl_train.patches.megatron.deepseek_v41.engram import (
                DSv41Engram,
            )

            self.engram = DSv41Engram(config, layer_id, spec["engram"], group_size=spec["engram_group_size"])
            self.engram_hash_index = spec["engram"].layer_ids.index(layer_id)

    layer_cls.__init__ = init_layer

    layer_forward = layer_cls.forward

    @functools.wraps(layer_forward)
    def forward_layer(self, hidden_states, *args, **kwargs):
        if getattr(self, "engram", None) is not None:
            hashes = shared_state().engram_hashes
            assert hashes is not None, "DeepSeek-V4.1 engram needs the model's input_ids"
            hidden_states = self.engram(hidden_states, hashes[:, :, self.engram_hash_index])
        return layer_forward(self, hidden_states, *args, **kwargs)

    layer_cls.forward = forward_layer

    _APPLIED = True
    return True
