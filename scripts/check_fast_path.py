#!/usr/bin/env python3
"""Check whether Qwen3.5 fast-path kernels (causal-conv1d + flash-linear-attention) are available."""

import sys


def check_fast_path():
    errors = []

    # 1. Check causal-conv1d
    try:
        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update

        print("[OK] causal-conv1d: causal_conv1d_fn and causal_conv1d_update imported")
    except ImportError as e:
        errors.append(f"causal-conv1d: {e}")
        print(f"[FAIL] causal-conv1d: {e}")

    # 2. Check flash-linear-attention (fla)
    try:
        from fla.modules import FusedRMSNormGated
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

        print("[OK] flash-linear-attention: chunk_gated_delta_rule and fused_recurrent_gated_delta_rule imported")
        print(f"[OK] flash-linear-attention: FusedRMSNormGated imported")
    except ImportError as e:
        errors.append(f"flash-linear-attention: {e}")
        print(f"[FAIL] flash-linear-attention: {e}")

    # 3. Check the transformers-level flag
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import is_fast_path_available

        if is_fast_path_available:
            print("[OK] transformers: is_fast_path_available = True")
        else:
            errors.append("transformers: is_fast_path_available = False")
            print("[FAIL] transformers: is_fast_path_available = False")
    except ImportError as e:
        errors.append(f"transformers: {e}")
        print(f"[FAIL] transformers: {e}")

    # Summary
    print()
    if errors:
        print(f"FAST PATH: DISABLED ({len(errors)} issue(s))")
        for err in errors:
            print(f"  - {err}")
        return 1
    else:
        print("FAST PATH: ENABLED")
        return 0


if __name__ == "__main__":
    sys.exit(check_fast_path())
