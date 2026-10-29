"""LoRA targets for ML4 and the adapter-key contract with vLLM (LORA_NOTES.md).

Trainer module names are vLLM's LoRA module names for the served Pixtral
checkpoint, so ``get_peft_model_state_dict`` output is loaded by vLLM as is:
no renaming, splitting or prefix rewrite at export.

Targets (all ``nn.Linear``):

* attention: ``self_attn.q_proj``, ``self_attn.kv_a_proj_with_mqa``,
  ``self_attn.o_proj``;
* MLP of dense layers, the shared expert and each routed expert:
  ``gate_proj``, ``up_proj``, ``down_proj``. vLLM packs gate/up as
  ``gate_up_proj -> [gate_proj, up_proj]`` (two slices, each with its own A)
  and maps experts as ``experts.{e}.gate_proj|up_proj|down_proj``.

Excluded:

* ``kv_b_proj``: vLLM MLA absorbs it into W_UK/W_UV; a LoRA would apply in
  MHA prefill but not in decode;
* the indexer (``wq_b``, ``wk_weights_proj``): discrete top-k, no LM-loss
  gradient, and vLLM has no packed mapping for ``wk_weights_proj``;
* the router ``mlp.gate``.
"""

import re

ATTN_TARGETS = ("q_proj", "kv_a_proj_with_mqa", "o_proj")
MLP_TARGETS = ("gate_proj", "up_proj", "down_proj")

_MODULE = (
    r"self_attn\.(q_proj|kv_a_proj_with_mqa|o_proj)"
    r"|mlp\.(shared_experts\.|experts\.\d+\.)?(gate_proj|up_proj|down_proj)"
)

# PEFT ``target_modules`` (full-matched against module names).
LORA_TARGET_REGEX = rf"language_model\.model\.layers\.\d+\.({_MODULE})"

# Every key of an exported adapter state dict must match this.
ADAPTER_KEY_REGEX = (
    rf"^base_model\.model\.(?P<module>language_model\.model\.layers\.\d+\.({_MODULE}))"
    r"\.lora_(A|B)\.weight$"
)
_ADAPTER_KEY = re.compile(ADAPTER_KEY_REGEX)


def adapter_module_names(adapter_state_dict: dict) -> set[str]:
    """Module names (= vLLM module names) of an adapter; raises on any key
    outside the contract."""
    out = set()
    for k in adapter_state_dict:
        m = _ADAPTER_KEY.match(k)
        if m is None:
            raise ValueError(f"adapter key outside the vLLM contract: {k}")
        out.add(m.group("module"))
    return out
