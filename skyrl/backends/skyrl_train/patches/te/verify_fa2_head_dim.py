"""Verification harness for the TE FA2 head_dim patch (NVIDIA/TransformerEngine#3360).

Run this on a GPU whose compute capability is OUTSIDE TE 2.16.0's allowlist of
sm80 / sm90 / sm100 / sm120 -- e.g. sm103 (B300/GB300), or sm86/sm89
(A10, A40, L4, L40S, RTX 4090). On an allowlisted GPU the gate is dead code and
the patch is a deliberate no-op, so phase A shows no change; pass --force to
exercise the mechanics anyway.

    uv run --isolated --extra megatron python \
        skyrl/backends/skyrl_train/patches/te/verify_fa2_head_dim.py

Phase A checks backend *selection* (pure logic, no kernels).
Phase B actually runs FlashAttention 2 forward+backward at head_dim=256 and
compares against TE's unfused reference -- this is the part that proves the
allowlist was over-restrictive rather than protecting against a broken kernel.
"""

import argparse
import importlib.metadata
import os

import torch
import transformer_engine.pytorch as te
from transformer_engine.pytorch.attention.dot_product_attention import (
    dot_product_attention as dpa,
)
from transformer_engine.pytorch.attention.dot_product_attention import (
    utils as dpa_utils,
)

from skyrl.backends.skyrl_train.patches.te.patch_fa2_head_dim import (
    patch_fa2_head_dim_allowlist,
)

ALLOWLISTED = ((8, 0), (9, 0), (10, 0), (12, 0))
HEAD_DIMS = (128, 192, 200, 256, 264)


def fa2_backend(head_dim):
    """Return the FA backend version TE would allow, or None if it rejects FA2."""
    params = dpa_utils.AttentionParams(
        head_dim_qk=head_dim,
        head_dim_v=head_dim,
        max_seqlen_q=4096,
        max_seqlen_kv=4096,
        core_attention_bias_shape=None,  # the '1hss' default disables FA on its own
    )
    return dpa_utils.get_attention_backend(params)[1]


def run_attention(head_dim, force_backend, seqlen=4096, batch=1, heads=8):
    """Run one fwd+bwd through TE with a specific backend forced. Returns (out, dq)."""
    os.environ["NVTE_FLASH_ATTN"] = "1" if force_backend == "flash" else "0"
    os.environ["NVTE_FUSED_ATTN"] = "1" if force_backend == "fused" else "0"
    os.environ["NVTE_UNFUSED_ATTN"] = "1" if force_backend == "unfused" else "0"
    # Invalidate TE's memoized backend choice so the env change takes effect.
    dpa._attention_backends["attention_params"] = None
    dpa._attention_backends["backend_selection_requires_update"] = True

    torch.manual_seed(0)
    shape = (batch, seqlen, heads, head_dim)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3))
    layer = te.DotProductAttention(
        num_attention_heads=heads,
        kv_channels=head_dim,
        attention_dropout=0.0,
        qkv_format="bshd",
        attn_mask_type="causal",
    )
    out = layer(q, k, v)
    out.sum().backward()
    selected = {
        "flash": dpa._attention_backends["use_flash_attention"],
        "fused": dpa._attention_backends["use_fused_attention"],
        "unfused": dpa._attention_backends["use_unfused_attention"],
    }
    return out.detach().float(), q.grad.detach().float(), selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="patch even on an allowlisted GPU")
    parser.add_argument("--skip-numerics", action="store_true", help="phase A only")
    args = parser.parse_args()

    cc = dpa_utils.get_device_compute_capability()
    name = torch.cuda.get_device_name(0)
    affected = cc not in ALLOWLISTED
    print(f"GPU: {name}  compute capability: sm{'.'.join(map(str, cc))}")
    te_version = importlib.metadata.version("transformer_engine")
    print(f"TE: {te_version}   affected by the allowlist: {affected}")
    if not affected and not args.force:
        print("\nThis GPU is already allowlisted by TE 2.16.0 -- the patch is a no-op here.")
        print("Re-run with --force to exercise the mechanics, or use an sm103/sm86/sm89 box.")

    print("\n--- phase A: backend selection ---")
    before = {hd: fa2_backend(hd) for hd in HEAD_DIMS}
    applied = patch_fa2_head_dim_allowlist(force=args.force)
    after = {hd: fa2_backend(hd) for hd in HEAD_DIMS}
    print(f"patch applied: {applied}")
    print(f"{'head_dim':>9}  {'FA2 before':>12}  {'FA2 after':>12}")
    for hd in HEAD_DIMS:
        print(f"{hd:>9}  {str(before[hd]):>12}  {str(after[hd]):>12}")
    print("\nExpected on an affected GPU: 128/192 unchanged, 200 and 256 go None -> 2.8.3,")
    print("264 stays None (still above FA2's real limit).")

    if args.skip_numerics:
        return
    if after[256] is None:
        print("\nFA2 still rejected at head_dim=256; skipping numerics.")
        return

    print("\n--- phase B: FA2 fwd+bwd numerics at head_dim=256 ---")
    flash_out, flash_dq, flash_sel = run_attention(256, "flash")
    print(f"forced flash -> selection {flash_sel}")
    if not flash_sel["flash"]:
        print("FA2 was not actually selected; numerics comparison is meaningless. Investigate.")
        return
    ref_out, ref_dq, ref_sel = run_attention(256, "unfused")
    print(f"forced unfused -> selection {ref_sel}")

    out_err = (flash_out - ref_out).abs().max().item()
    dq_err = (flash_dq - ref_dq).abs().max().item()
    print(f"max |out  diff| vs unfused: {out_err:.5f}")
    print(f"max |dq   diff| vs unfused: {dq_err:.5f}")
    # bf16 attention at 4k context: differences here are accumulation-order noise.
    tol = 0.05
    print(f"PASS: {out_err < tol and dq_err < tol}  (tolerance {tol}, bf16)")


if __name__ == "__main__":
    main()
