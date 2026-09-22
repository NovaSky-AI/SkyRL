# TE FlashAttention 2 head_dim patch

Backport of [NVIDIA/TransformerEngine#3360](https://github.com/NVIDIA/TransformerEngine/pull/3360)
for the `transformer-engine==2.16.0` pin.

## The bug

`utils.py:750` in TE 2.16.0 rejects FA2 for `head_dim_qk > 192` unless the
compute capability is `(8,0), (9,0), (10,0), (12,0)`:

```python
or (
    head_dim_qk > 192
    and device_compute_capability not in ((8, 0), (9, 0), (10, 0), (12, 0))
)
```

Upstream removed this allowlist in #2836, #2629 reintroduced it by accident, and
#3360 removes it again, keeping only FA2's real limits (`<= 256`, `% 8 == 0`).
Consequence on an excluded arch: a head_dim of 256 (Gemma 2/3) drops to unfused
attention, which materializes the quadratic attention matrix — upstream reported
a 202 GiB allocation and OOM at 65k tokens.

## Minimal repro

Needs a GPU whose compute capability is **not** sm80/90/100/120 — sm103
(B300/GB300) is upstream's case; sm89 (L40S, L4) and sm86 (A10G, A40) hit the
identical branch, since the gate keys on compute capability alone.

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv   # must not be 8.0/9.0/10.0/12.0
uv run --isolated --extra megatron python \
    skyrl/backends/skyrl_train/patches/te/verify_fa2_head_dim.py
```

Phase A (backend selection) should show:

```
 head_dim    FA2 before     FA2 after
      128         2.8.3         2.8.3
      192         2.8.3         2.8.3
      200          None         2.8.3   <- fixed
      256          None         2.8.3   <- fixed
      264          None          None   <- still rejected, correctly
```

Phase B forces FA2 and diffs fwd+bwd against TE's unfused reference. This is the
part that proves the allowlist was stale rather than protecting a broken kernel.
Reference from a validated H100 run (`--force`, bf16, 4k causal): `max |out
diff| = 0.01562`, `max |dq diff| = 0.01562`, PASS. A crash or CUDA error instead
means #3360 itself is unsafe on that arch — report it upstream, don't ship.

On an allowlisted GPU the gate is dead code, so phase A shows no change; that
run only confirms the no-op guard.

## Why a Python patch

TE ships as a prebuilt wheel, and `uv run --isolated` builds a fresh venv per
invocation, so a site-packages edit does not survive a run (nor reach other Ray
nodes). The gate is an inline clause in a ~1000-line function, but
`get_attention_backend` is undecorated and its only caller resolves it as a
module attribute (`dpa_utils.get_attention_backend`) — so recompiling that one
function into TE's own `__dict__` is enough, with no reference chasing. The patch
also invalidates `_attention_backends`, which memoizes backend selection.

Guarded to no-op on sm80/90/100/120, idempotent, and matches the literal 2.16.0
source so it self-disables on a newer TE instead of misfiring.

## Wired in at

`MegatronWorker.make_megatron_module()`, before `provide_distributed_model()` —
the choke point both the policy and ref workers pass through, and the last hook
before any TE attention layer exists. Megatron only; FSDP uses HF transformers.
Inert on H100.

## Delete this when

The `transformer-engine` pin moves to a release containing #3360. It is a
temporary backport, and a stale copy goes quiet rather than loud — nothing will
remind you, so check this directory when bumping TE.

#3360 was approved but not yet merged as of 2026-08-13; re-check its final form
against this backport.
