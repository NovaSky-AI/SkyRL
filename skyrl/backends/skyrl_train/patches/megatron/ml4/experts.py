"""ML4 routed experts with the checkpoint's NVFP4 bytes resident, and the rollout-matching QAT.

:class:`ML4NVFP4GroupedMLP` replaces Megatron's ``TEGroupedMLP`` for the ModelOpt NVFP4 experts.
Each rank keeps only its local experts' packed E2M1 codes, E4M3 block scales and FP32
``weight_scale_2`` and decodes them just in time (the FSDP trainer's fused Triton decode), so
no BF16 copy of the experts exists. The math is the FSDP trainer's (``ml4_train/qat.py``), which
mirrors vLLM's ``TrtLlmNvFp4LoRAExperts`` with ``moe.activation=nvfp4_per_token``:

* QAT on (``ml4_nvfp4_act_qat``): FC1 input per-token NVFP4 over the full row; FC1 out =
  acc * s * ws2 in FP32; + gate/up LoRA delta (from the unquantized input) in FP32; SwiGLU in
  FP32 rounded to BF16; FC2 input per-token NVFP4 per rollout-TP shard of the intermediate dim;
  FC2 output rounded to BF16; contribution = w * base + BF16(w * down-LoRA delta) in FP32.
  The FP32 contributions are summed per token by Megatron's combine (ascending expert order)
  and rounded to BF16 once in :class:`ML4MoELayer`, before the shared expert is added.
* QAT off: the FSDP trainer's per-expert plain path: BF16 linear with the decoded weight
  times ws2, LoRA added in the activation dtype, SwiGLU in the activation dtype; contribution
  = FP32(y) * w.

The FP8_PB_WO linears (attention q/o, indexer wq_b, dense MLP, shared expert) get the FSDP
trainer's FP8 activation fake-quant as a forward pre-hook on the base linear
(``install_fp8_act_qat``), so a LoRA adapter on the same linear sees the unquantized input.
Every quantizer is the serving stack's own kernel (FlashInfer NVFP4, vLLM FP8 group quant)
with a straight-through backward, as in ``ml4_train/qat.py``.

LoRA: ``linear_fc1`` / ``linear_fc2`` subclass the TE grouped linears only so that
Megatron-Bridge's LoRA recognizes them and wraps them with its own (shared-outer) grouped
expert adapters; they hold no TE weights. The MLP calls the base GEMM and the adapter
separately to keep the FSDP trainer's (vLLM's) composition order.
"""

from collections import Counter

import torch
import torch.nn.functional as F
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelGroupedLinear,
    TERowParallelGroupedLinear,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_layer import MoELayer
from torch import nn

from skyrl.backends.skyrl_train.patches.megatron.ml4.ml4_train import qat

# Runtime call counters (proofs that each simulation runs where it should).
CALLS: Counter = Counter()


def reset_calls() -> None:
    CALLS.clear()


# --------------------------------------------------------------------------------------------
# NVFP4 grouped base linear (no TE weights)
# --------------------------------------------------------------------------------------------


class _NVFP4GroupedBase:
    """Mixin: packed NVFP4 weights of the local experts, decoded just in time.

    Tensors (all frozen; names carry ``nvfp4`` so the bridge leaves them alone and
    :meth:`~.bridge.ML4Bridge.load_weights_hf_to_megatron` fills them):
    ``nvfp4_ws2`` FP32 [E] (registered first: the LoRA adapter takes its dtype from the
    first parameter), ``nvfp4.packed`` uint8 [E, N, K/2], ``nvfp4.scale`` uint8 [E, N, K/16].
    """

    def _ml4_init(self, num_gemms, in_features, out_features, config, pg_collection=None):
        nn.Module.__init__(self)
        if in_features % qat.NVFP4_BLOCK:
            raise ValueError(f"in_features {in_features} not a multiple of 16")
        self.config = config
        self.num_gemms = num_gemms
        self.in_features = in_features
        self.out_features = out_features
        self.is_expert = True
        self.explicit_expert_comm = True
        self.parallel_mode = None
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.expt_tp if pg_collection is not None else None
        self.tp_size = 1
        self.nvfp4_ws2 = nn.Parameter(torch.ones(num_gemms, dtype=torch.float32), requires_grad=False)
        self.nvfp4 = nn.Module()
        u8 = dict(dtype=torch.uint8)
        self.nvfp4.packed = nn.Parameter(
            torch.zeros(num_gemms, out_features, in_features // 2, **u8), requires_grad=False
        )
        self.nvfp4.scale = nn.Parameter(
            torch.zeros(num_gemms, out_features, in_features // qat.NVFP4_BLOCK, **u8), requires_grad=False
        )

    def _apply(self, fn, recurse=True):
        # keep weight_scale_2 FP32 under Float16Module's .bfloat16() (as qat.NVFP4Linear does)
        saved = self.nvfp4_ws2.data.clone()  # values, not the Parameter (its .data is replaced)
        super()._apply(fn, recurse)
        if self.nvfp4_ws2.dtype != torch.float32:
            self.nvfp4_ws2.data = saved.to(device=self.nvfp4_ws2.device)
        return self

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        # Frozen checkpoint weights: never saved, reloaded from the model checkpoint on resume.
        return {}

    def _load_from_state_dict(self, *args, **kwargs):
        # The TE grouped-linear override remaps TE's own weight layout, which this module does not have.
        return nn.Module._load_from_state_dict(self, *args, **kwargs)

    def qat_gemm(self, x, counts, shards, xq, chunk):
        """QAT on, FP32 output [M, N]: ``qat._GroupedNVFP4Matmul`` over chunks of local experts
        (experts without rows skipped). Rows of x sorted by local expert (``counts``)."""
        outs, start = [], 0
        cnt_all = torch.tensor(counts, device=x.device)
        for c0 in range(0, self.num_gemms, chunk):
            sel = [e for e in range(c0, min(c0 + chunk, self.num_gemms)) if counts[e]]
            if not sel:
                continue
            cnt = [counts[e] for e in sel]
            m = sum(cnt)
            rows = slice(start, start + m)
            start += m
            idx = torch.tensor(sel, device=x.device)
            ws2_rows = self.nvfp4_ws2[idx].repeat_interleave(cnt_all[idx], output_size=m)
            outs.append(
                qat._GroupedNVFP4Matmul.apply(
                    x[rows],
                    self.nvfp4.packed[idx],
                    self.nvfp4.scale[idx],
                    ws2_rows,
                    cnt,
                    shards,
                    (xq[0][rows], xq[1][rows]),
                )
            )
        if not outs:
            return x.new_zeros((0, self.out_features), dtype=torch.float32)
        return torch.cat(outs)

    def plain_gemm(self, x, counts):
        """QAT off: per expert ``(F.linear(x, W_dec) * ws2)`` in x.dtype (``qat._NVFP4Matmul``)."""
        outs = []
        for e, xe in enumerate(x.split(counts)):
            outs.append(qat._NVFP4Matmul.apply(xe, self.nvfp4.packed[e], self.nvfp4.scale[e], self.nvfp4_ws2[e], None))
        return torch.cat(outs) if outs else x.new_zeros((0, self.out_features))

    def forward(self, x, m_splits):
        counts = m_splits.tolist() if torch.is_tensor(m_splits) else list(m_splits)
        return self.plain_gemm(x, counts), None


class ML4NVFP4ColumnGroupedLinear(_NVFP4GroupedBase, TEColumnParallelGroupedLinear):
    def __init__(self, num_gemms, in_features, out_features, config, pg_collection=None):
        self._ml4_init(num_gemms, in_features, out_features, config, pg_collection)


class ML4NVFP4RowGroupedLinear(_NVFP4GroupedBase, TERowParallelGroupedLinear):
    def __init__(self, num_gemms, in_features, out_features, config, pg_collection=None):
        self._ml4_init(num_gemms, in_features, out_features, config, pg_collection)


def _split_lora_wrapper(mod):
    """(base, adapter or None) of a Bridge ``LoRALinear``-wrapped (or plain) linear."""
    base = getattr(mod, "to_wrap", mod)
    adapter = getattr(mod, "adapter", None) if base is not mod else None
    if adapter is not None and not getattr(mod, "_adapter_enabled", True):
        adapter = None
    return base, adapter


# --------------------------------------------------------------------------------------------
# Experts
# --------------------------------------------------------------------------------------------


class ML4NVFP4GroupedMLP(MegatronModule):
    """Megatron experts interface (``forward(hidden, tokens_per_expert, probs)``) for NVFP4 experts."""

    def __init__(self, num_local_experts, config, submodules=None, pg_collection=None, **kwargs):
        super().__init__(config=config)
        from megatron.core import parallel_state

        if parallel_state.get_expert_tensor_parallel_world_size() != 1:
            raise NotImplementedError("ML4 NVFP4 experts: expert tensor parallelism is not supported")
        h, inter = config.hidden_size, config.moe_ffn_hidden_size
        self.num_local_experts = num_local_experts
        self.inter = inter
        self.linear_fc1 = ML4NVFP4ColumnGroupedLinear(num_local_experts, h, 2 * inter, config, pg_collection)
        self.linear_fc2 = ML4NVFP4RowGroupedLinear(num_local_experts, inter, h, config, pg_collection)
        self.act_qat = bool(config.ml4_nvfp4_act_qat)
        self.fc2_shards = int(config.ml4_rollout_moe_tp)
        self.chunk = int(config.ml4_moe_grouped_chunk)
        if self.act_qat and (inter % self.fc2_shards or (inter // self.fc2_shards) % qat.NVFP4_BLOCK):
            raise ValueError(f"moe_ffn_hidden_size={inter} not splittable into {self.fc2_shards} 16-blocks")

    def forward(self, x, tokens_per_expert, permuted_probs, **kwargs):
        counts = tokens_per_expert.tolist() if torch.is_tensor(tokens_per_expert) else list(tokens_per_expert)
        w = permuted_probs.float().reshape(-1)
        fc1, ad1 = _split_lora_wrapper(self.linear_fc1)
        fc2, ad2 = _split_lora_wrapper(self.linear_fc2)
        I = self.inter  # noqa: E741
        if x.shape[0] == 0:
            # x and the probs still take part in the graph: their backward runs the EP all-to-alls of the
            # token dispatcher, which every EP rank must issue even when it got no tokens.
            out = x.new_zeros((0, x.shape[-1]), dtype=torch.float32) + (x.float().sum() + w.sum()) * 0
            # every LoRA parameter takes part in the graph (zero gradient), as in the FSDP trainer
            for ad in (ad1, ad2):
                if ad is not None and torch.is_grad_enabled():
                    out = out + sum(p.narrow(0, 0, 0).sum().float() for p in ad.parameters() if p.requires_grad)
            return out, None
        if self.act_qat:
            CALLS["nvfp4_fc1_input_quant"] += 1
            xq = qat.nvfp4_per_token_quant(x, 1)
            h = fc1.qat_gemm(x, counts, 1, xq, self.chunk)
            if ad1 is not None:
                h = h + ad1(x, counts).float()
            a = (F.silu(h[:, :I]) * h[:, I:]).to(x.dtype)
            CALLS["nvfp4_fc2_input_quant"] += 1
            y = fc2.qat_gemm(a, counts, self.fc2_shards, qat.nvfp4_per_token_quant(a, self.fc2_shards), self.chunk)
            contrib = y.to(x.dtype).float() * w[:, None]
            if ad2 is not None:
                contrib = contrib + (ad2(a, counts).float() * w[:, None]).to(x.dtype).float()
        else:
            CALLS["nvfp4_plain"] += 1
            h = fc1.plain_gemm(x, counts)
            if ad1 is not None:
                h = h + ad1(x, counts).to(h.dtype)
            a = F.silu(h[:, :I]) * h[:, I:]
            y = fc2.plain_gemm(a, counts)
            if ad2 is not None:
                y = y + ad2(a, counts).to(y.dtype)
            contrib = y.float() * w[:, None]
        return contrib, None


class ML4MoELayer(MoELayer):
    """MoE layer whose routed experts return FP32 weighted contributions: the per-token FP32 sum
    (Megatron's combine) is rounded to the activation dtype once, then the shared expert is
    added in that dtype (vLLM / FSDP-trainer order)."""

    def postprocess(self, output, shared_expert_output):
        output = self.token_dispatcher.combine_postprocess(output)
        if shared_expert_output is not None:
            output = output.to(shared_expert_output.dtype) + shared_expert_output
        else:
            output = output.to(self.config.params_dtype)
        return output


# --------------------------------------------------------------------------------------------
# FP8 activation QAT on the FP8_PB_WO linears
# --------------------------------------------------------------------------------------------


def _fp8_pre_hook(name):
    def hook(module, args, kwargs=None):
        x = args[0]
        CALLS[f"fp8_act:{name}"] += 1
        xq = qat.ste(x, qat.fp8_block_act_quant(x))
        return (xq,) + tuple(args[1:])

    return hook


_FSDP_TO_MEGATRON = {
    "self_attn.q_proj": "self_attention.linear_q_proj",
    "self_attn.o_proj": "self_attention.linear_proj",
    "self_attn.kv_a_proj_with_mqa": "self_attention.linear_kv_down_proj",
    "self_attn.kv_b_proj": "self_attention.linear_kv_up_proj",
    "self_attn.indexer.wq_b": "self_attention.indexer.wq_b",
    "self_attn.indexer.wk_weights_proj": "self_attention.indexer.wk_weights_proj",
    "mlp.gate_proj": "mlp.linear_fc1",
    "mlp.up_proj": "mlp.linear_fc1",
    "mlp.down_proj": "mlp.linear_fc2",
    "mlp.shared_experts.gate_proj": "mlp.shared_experts.linear_fc1",
    "mlp.shared_experts.up_proj": "mlp.shared_experts.linear_fc1",
    "mlp.shared_experts.down_proj": "mlp.shared_experts.linear_fc2",
}


def fp8_act_qat_targets(config) -> dict:
    """{(global layer idx, Megatron module path within the layer)} -> FSDP names, for the
    checkpoint's FP8_PB_WO layers (``quantization_config.quantized_layers``, mapped with the FSDP
    loader's name map). A fused Megatron linear (gate+up) must have all its parts FP8_PB_WO."""
    from skyrl.backends.skyrl_train.patches.megatron.ml4.ml4_train.loading import (
        map_name,
    )

    out: dict = {}
    for key, spec in (config.ml4_quantized_layers or {}).items():
        if spec["quant_algo"] != "FP8_PB_WO":
            continue
        name = map_name(key)[0].removeprefix("language_model.model.")
        layer, rest = name.split(".", 2)[1], name.split(".", 2)[2]
        if rest not in _FSDP_TO_MEGATRON:
            raise ValueError(f"FP8_PB_WO layer {key} -> {name}: no Megatron module mapping")
        out.setdefault((int(layer), _FSDP_TO_MEGATRON[rest]), []).append(rest)
    for (layer, path), parts in out.items():
        if path.endswith("linear_fc1") and len(set(parts)) != 2:
            raise ValueError(f"layer {layer} {path}: FP8_PB_WO on {parts}, need both gate and up")
    return out


def install_fp8_act_qat(model: nn.Module, config) -> int:
    """Forward pre-hooks (input FP8 fake quant, STE) on the base linears of the checkpoint's
    FP8_PB_WO layers. Must run before LoRA wraps them, so the hooks sit on the base linear and
    the adapter sees the unquantized input. Returns the number of hooked modules."""
    targets = fp8_act_qat_targets(config)
    layers = {
        m.layer_number - 1: m for m in model.modules() if hasattr(m, "self_attention") and hasattr(m, "layer_number")
    }
    n = 0
    for (li, path), parts in targets.items():
        if li not in layers:
            continue  # another pipeline stage
        mod = layers[li].get_submodule(path)
        mod.register_forward_pre_hook(_fp8_pre_hook(path))
        n += 1
    return n
