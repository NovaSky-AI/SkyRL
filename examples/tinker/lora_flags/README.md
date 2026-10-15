# LoRA group flags (`train_attn` / `train_mlp`)

Tinker's `LoraConfig` lets you choose which parts of the model your LoRA adapter
covers. A narrower adapter trains fewer parameters and produces a smaller
checkpoint, at the cost of capacity.

```python
client = service.create_lora_training_client(
    base_model="Qwen/Qwen3-0.6B",
    rank=8,
    train_attn=True,
    train_mlp=False,   # attention-only adapter
)
```

Both flags default to `True`, which adapts every linear layer in the transformer
body. Setting one to `False` restricts the adapter to the other group:

| Flag | Covers | Typical module names |
|---|---|---|
| `train_attn` | attention projections | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| `train_mlp` | MLP projections | `gate_proj`, `up_proj`, `down_proj` |

The groups are disjoint and together make up the whole adapter. The modules are
resolved by inspecting the loaded model, so architecture-specific names are
handled without configuration.

The unembedding (`lm_head`, `embed_tokens`) is in neither group and is never
adapted on this backend -- see "Not supported" below.

**On MoE models the MLP group is usually empty.** Current `transformers` stores
expert projections as fused 3-D `nn.Parameter` tensors (Qwen3-MoE, Mixtral,
DeepSeek-V3) rather than `nn.Linear` modules, and PEFT can only adapt the
latter. Requesting `train_mlp=True, train_attn=False` on such a model raises
rather than returning an attention-only adapter. Expert projections are covered
only where an architecture exposes them as `nn.Linear`. The Megatron backend
does adapt MoE experts, with dedicated grouped-expert handling, but it rejects
these flags -- see "Not supported".

## Constraints

- **At least one of the two must be `True`.** An adapter covering no modules has
  nothing to train. The Tinker SDK rejects all three flags being `False` before
  the request leaves your process; setting only `train_attn` and `train_mlp` to
  `False` gets through, and then fails server-side with a `ValueError`.
- **Narrowing requires the default `target_modules`.** If you have set
  `trainer.policy.model.lora.target_modules` to an explicit module list, the
  flags have nothing well-defined to subtract from and the server raises a
  `ValueError`. Select the modules in that list instead.

Both errors are raised while the model is being built, not when
`create_lora_training_client` returns, so look for them in the server log.

## What narrowing costs you

This example trains one small supervised task three times -- attention-only,
MLP-only, and both -- holding the data, seed, learning rate and step count
fixed, so the only variable is which modules carry an adapter.

```bash
bash examples/tinker/lora_flags/run_demo.sh
```

It writes `lora_flags_results.json` and prints:

```
config        attn   mlp   step 0   step 3   step 5  step 10  steps<0.1     mean
both          True  True   1.6009   0.8665   0.5612   0.0014          9   0.2295
attn_only     True False   1.6009   1.0377   0.8607   0.3663         13   0.3413
mlp_only     False  True   1.6009   1.0188   0.8277   0.3602         12   0.3300

3 of 3 curves are distinct, so the flags reached PEFT.
Adapting both groups should converge fastest; it has the most adapted modules.
```

(One L4, `Qwen/Qwen3-0.6B`, rank 8, 30 steps.)

Reading the output: adapting both groups reaches the loss threshold in 9 steps
against 12-13 for either group alone, which is the trade-off the flags buy you.
Two columns cannot show a difference and are there for orientation only. **Step
0 is identical for every configuration** because LoRA initialises `B = 0`, so
the adapter contributes nothing at the first step regardless of what it covers;
the script warns if it ever sees a difference there, since that would mean the
runs were not comparable. **Final loss reaches 0 everywhere**, because eight
examples are memorised long before step 30. What separates the configurations is
how quickly they get there.

If all three curves come out identical, the flags are not reaching the adapter;
the script says so rather than printing a table you might read as a result.

The demo trains only (`forward_backward` + `optim_step`) and never samples, so
no inference engine is created and one GPU is enough.

## Checking what your own adapter covers

The loss curve tells you the configurations differ; the exported adapter tells
you exactly which modules each one touched. After training, look in the
per-model subdirectory under `trainer.policy.model.lora.lora_sync_path`
(default `/tmp/skyrl_lora_sync`):

```bash
jq .target_modules <path>/adapter_config.json
# ["k_proj","o_proj","q_proj","v_proj"]      # train_mlp=False
```

`adapter_model.safetensors` alongside it contains tensors for those modules
only. This is the direct check -- it does not depend on interpreting a training
curve. The same assertion runs in
`tests/tinker/skyrl_train/test_fsdp_lora_flags.py`.

## Running one configuration by hand

Two terminals:

```bash
# 1. server
bash examples/tinker/lora_flags/run_tinker_server.sh

# 2. client
uv run --extra tinker --extra fsdp python examples/tinker/lora_flags/lora_flags_demo.py \
    --label attn_only --train-attn --no-train-mlp
```

`BASE_MODEL`, `PORT` and `BACKEND_CONFIG` are environment variables.

The FSDP backend serves one adapter per server lifetime (multi-tenant LoRA is
Megatron-only), so a configuration change needs a fresh server. The sweep script
starts one per configuration on `PORT`, `PORT+1` and `PORT+2`, so a server left
over from an earlier run cannot be mistaken for a new one. If you reuse a server
by hand, a second client with different flags is rejected with a LoRA signature
mismatch.

## Not supported

- **`train_unembed` has no effect on this backend.** LoRA is applied through
  PEFT's `"all-linear"`, which excludes the output layer, so your adapter covers
  the attention and MLP projections only. Note the Tinker SDK **defaults
  `train_unembed` to `True`**, so this applies to clients that never set it. The
  server logs a warning when it is requested -- it is not returned to the
  client, so check the server output, not yours. Start the server with
  `--backend jax` if you need unembedding support.
- **Megatron rejects a non-default `train_attn` / `train_mlp`.** It builds its
  target list from fixed module names instead of inspecting the model. Use
  `--backend fsdp`, or name the modules directly in
  `trainer.policy.model.lora.target_modules`.
