# Per-model specs for serialized FP8 weight sync

Serialized FP8 weight sync needs these model-specific answers, grouped in a
`ModelFp8Spec` (`base.py`) and resolved once per checkpoint with
`resolve_fp8_spec(hf_config)`:

| Field | Question it answers |
| --- | --- |
| `matches(hf_config)` | Does this spec support the checkpoint layout? |
| `should_quantize(name, shape, wire_format)` | Should this exported HF weight be FP8 on the wire? (Linear weights yes; embeddings, norms, conv, router gates no.) |
| `base_exclude_list(hf_config, wire_format)` | Which modules does the engine always build unquantized, before any user exclusions, to match the checkpoint-format stream? |
| `fp8_modules(hf_config, wire_format)` | Which tensors does the spec sync as FP8 on this wire? One entry per tensor, named as Megatron-Bridge exports it minus a trailing `.weight`, e.g. `model.layers.3.self_attn.q_proj`, Qwen3.5's stacked `model.language_model.layers.3.mlp.experts.gate_up_proj`, or GLM-5's per-expert `model.layers.3.mlp.experts.17.gate_proj`. User exclusions match against this list. |
| `stacked_experts(name)` (optional) | Does this exported tensor stack a projection of every routed expert in a layer, and which projections, in what order? `None` (the default) for ordinary tensors. |

Everything else — blockwise casting, wire naming, the vLLM quantization
config, the receiver's fused-MoE loading — is generic and lives outside this
package. Which vLLM FusedMoE parameter each stacked projection loads into is
fixed by vLLM, so the receiver's table (`batched_moe_wire_targets()`) comes
from one constant in `base.py`, not from the specs.

## User exclusions

`generator.inference_engine.fp8_weight_sync_exclude_modules` keeps modules in
the model dtype on top of the spec. `resolve_user_provided_exclude_list(spec,
hf_config, wire_format, patterns)` expands the globs against `fp8_modules`; both
consumers call it on the same config, so they see the same modules:

- **engine init**: `engine_exclude_list` appends the user's modules to
  `base_exclude_list` (a layer's excluded routed experts are listed as
  `<experts>.0.{gate,up,down}_proj`, the names vLLM reads to decide the
  precision of all of them);
- **weight sync**: `SerializedFp8Config.user_provided_exclude_list` makes the
  sender pass those tensors through unquantized.

A pattern that matches no FP8 tensor is rejected before the engine starts, and
so is excluding only some of a layer's routed experts: vLLM runs them all at
one precision and does not reject the split itself. Excluding only part of a
module vLLM fuses (e.g. `q_proj` without `k_proj` and `v_proj`) is left to
vLLM, which rejects it when the engine starts.

## Adding a new model

1. Create `models/<family>.py`. Implement the callables (plain functions;
   see `qwen35.py`). Build `should_quantize`'s suffix table and `fp8_modules`
   from the same module tables, so the sender's policy and the names user
   exclusions match against cannot drift apart.
2. If the model has routed MoE experts exported as stacked 3D tensors,
   return `StackedExperts(experts_base, projections, split_dim)` from
   `stacked_experts` — `projections` names what the tensor stacks, in order
   (`"gate_proj"`, `"up_proj"`, `"down_proj"`), and `split_dim` is the
   dimension that concatenates them (e.g. `gate_up_proj` splits in half along
   dim 1); use `None` when the tensor is a single projection. Models that
   export one tensor per expert (GLM-5) need nothing here.
3. Register it at module bottom and import the module in
   `models/__init__.py` (the import is what registers the spec):

   ```python
   MYMODEL_FP8_SPEC = register_fp8_spec(
       ModelFp8Spec(
           name="mymodel",
           matches=is_mymodel_config,
           should_quantize=is_quantizable_weight_shape,
           base_exclude_list=get_mymodel_base_exclude_list,
           fp8_modules=get_mymodel_fp8_modules,
           stacked_experts=get_mymodel_stacked_experts,  # only if experts are stacked
       )
   )
   ```

4. Add tests mirroring `tests/backends/skyrl_train/weight_sync/fp8/
   test_serialized_fp8.py` (quantize filter, base exclude list, MoE mapping) and
   `test_fp8_exclusions.py` / `test_fp8_exclusions_vllm.py` (exclusions, run
   through vLLM's own matching), and, for real coverage, an FP8 row in the
   GPU CI logprobs-roundtrip test.

No generic file changes are needed; checkpoints that match no registered
spec are rejected with the list of registered spec names.
