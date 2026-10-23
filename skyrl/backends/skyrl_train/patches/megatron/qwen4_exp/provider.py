"""Model provider for Qwen3.8-Flash-Next (``qwen4_exp``)."""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union

import torch
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.core import parallel_state
from megatron.core.transformer.spec_utils import ModuleSpec

from skyrl.backends.skyrl_train.patches.megatron.qwen4_exp.layer_specs import (
    Qwen4ExpTransformerLayer,
    build_qwen4_exp_layer_spec,
)


@dataclass
class Qwen4ExpModelProvider(GPTModelProvider):
    """Megatron configuration and provider for the Qwen4Exp language model.

    On top of the Qwen3.5 hybrid GDN / gated-attention MoE configuration (set by the bridge),
    the ``qwen4_*`` fields describe the gated hyper-connections, the PLE n-gram embedding and
    the QSA indexer. Supported parallelism: TP=1, CP=1, PP=1, any EP dividing the 16 n-gram heads
    (the PLE table is sharded over the expert-parallel group).
    """

    transformer_layer_spec: Union[ModuleSpec, Callable] = build_qwen4_exp_layer_spec

    qwen4_hc_count: int = 4
    qwen4_hc_lowrank: int = 320
    qwen4_ple_layer_ids: List[int] = field(default_factory=list)
    qwen4_ple_embed_dim: int = 2560
    qwen4_ple_conv_kernel_size: int = 4
    qwen4_ngram_size: int = 3
    qwen4_heads_per_ngram: int = 8
    qwen4_eos_token_id: int = 248044
    qwen4_indexer_n_heads: int = 4
    qwen4_indexer_kv_heads: int = 1
    qwen4_indexer_head_dim: int = 128
    qwen4_indexer_budget: int = 2048
    qwen4_indexer_compress_ratio: int = 4
    # Where the PLE n-gram table is read from, and the HF prefix of the PLE layer
    # (``model.language_model.layers.<i>.ple.``); set by the bridge.
    qwen4_hf_path: Optional[str] = None
    qwen4_ple_hf_prefix: Optional[str] = None

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        if self.tensor_model_parallel_size != 1 or self.context_parallel_size != 1:
            raise NotImplementedError(
                "Qwen4Exp on Megatron supports tensor_model_parallel_size=1 and context_parallel_size=1 "
                "(PLE needs each packed sequence whole on one rank); scale with expert parallelism."
            )
        if self.pipeline_model_parallel_size != 1:
            raise NotImplementedError("Qwen4Exp on Megatron does not support pipeline parallelism yet.")
        model = super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)

        ples = [m.ple for m in model.modules() if isinstance(m, Qwen4ExpTransformerLayer) and m.ple is not None]
        group = parallel_state.get_expert_model_parallel_group() if torch.distributed.is_initialized() else None
        for ple in ples:
            ple.load_table(group)

        def _stash_tokens(module, args, kwargs):
            input_ids = args[0] if args else kwargs.get("input_ids")
            packed = kwargs.get("packed_seq_params")
            cu = None
            if packed is not None:
                cu = packed.cu_seqlens_q_padded if packed.cu_seqlens_q_padded is not None else packed.cu_seqlens_q
            for ple in ples:
                ple.set_tokens(input_ids, cu)

        model.register_forward_pre_hook(_stash_tokens, with_kwargs=True)
        return model
