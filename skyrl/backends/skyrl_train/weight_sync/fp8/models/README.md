# Per-model specs for serialized FP8 weight sync

Serialized FP8 weight sync needs four model-specific answers, grouped in a
`ModelFp8Spec` (`base.py`) and resolved once per checkpoint with
`resolve_fp8_spec(hf_config)`:

| Field | Question it answers |
| --- | --- |
| `matches(hf_config)` | Does this spec support the checkpoint layout? |
| `should_quantize(name, shape)` | Should this exported HF weight be FP8 on the wire? (Linear weights yes; embeddings, norms, conv, router gates no.) |
| `ignored_layers(hf_config)` | Which vLLM module prefixes must stay unquantized? (Modules whose shards can't share a 128-block FP8 scheme.) |
| `moe_expert_spec(name)` | Is this a Megatron-Bridge *batched* expert tensor, and how does it map onto per-projection wire tensors? `None` for ordinary tensors. |

Everything else — blockwise casting, wire naming, the vLLM quantization
config, the receiver's fused-MoE loading — is generic and lives outside this
package. The receiver-side table (which fused vLLM parameter each expert
projection loads into) is **derived** from the specs via
`batched_moe_wire_targets()`, so the mapping is declared exactly once.

## Adding a new model

1. Create `models/<family>.py`. Implement the four callables (plain
   functions; see `qwen35.py` — suffix tables usually suffice for
   `should_quantize`).
2. If the model has routed MoE experts exported as batched 3D tensors,
   declare one `MoeProjection(hf_name, vllm_param, shard_id)` per projection
   and return `MoeExpertSpec(experts_base, projections, split_dim)` from
   `moe_expert_spec` — `split_dim` is the dimension that concatenates fused
   projections (e.g. `gate_up_proj` splits in half along dim 1); use `None`
   when the tensor is a single projection.
3. Register it at module bottom and import the module in
   `models/__init__.py` (the import is what registers the spec):

   ```python
   MYMODEL_FP8_SPEC = register_fp8_spec(
       ModelFp8Spec(
           name="mymodel",
           matches=is_mymodel_config,
           should_quantize=is_quantizable_weight_shape,
           ignored_layers=get_mymodel_fp8_ignored_layers,
           moe_expert_spec=batched_moe_expert_spec,
           moe_projections=(_MOE_GATE, _MOE_UP, _MOE_DOWN),
       )
   )
   ```

4. Add tests mirroring `tests/backends/skyrl_train/weight_sync/
   test_serialized_fp8.py` (quantize filter, ignored layers, MoE mapping)
   and, for real coverage, an FP8 row in the GPU CI logprobs-roundtrip test.

No generic file changes are needed; checkpoints that match no registered
spec are rejected with the list of registered spec names.
