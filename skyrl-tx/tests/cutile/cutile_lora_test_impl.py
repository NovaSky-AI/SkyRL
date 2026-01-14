"""
Cutile-based implementation of ragged_dot for LoRA EP computation.

NOTE: This file is now a wrapper around the production code in tx/kernels/.
The actual implementation has been moved to the production package.

For production use, import from tx.kernels.cutile_lora instead.
This file is kept for backwards compatibility with test development.
"""

# Re-export from production code
from tx.kernels.cutile_lora import (
    cutile_ragged_dot as cutile_ragged_dot_test,
    jax_to_torch,
    torch_to_jax,
)
from tx.kernels.cutile_lora_kernels import (
    lora_align_tile_size,
    launch_cutile_lora_gemm,
    CUTILE_AVAILABLE,
)

# For backwards compatibility
__all__ = [
    "cutile_ragged_dot_test",
    "jax_to_torch",
    "torch_to_jax",
    "lora_align_tile_size",
    "launch_cutile_lora_gemm",
    "CUTILE_AVAILABLE",
    "check_environment",
]

# Original implementation moved to tx/kernels/ - see:
# - tx/kernels/cutile_lora.py (main API)
# - tx/kernels/cutile_lora_kernels.py (kernel implementations)
# - tx/kernels/cutile_config.py (configuration)


def check_environment():
    """Check if environment is suitable for running cutile."""
    import torch
    import jax

    issues = []

    if not CUTILE_AVAILABLE:
        issues.append("Cutile (cuda.tile) not available")

    if not torch.cuda.is_available():
        issues.append("PyTorch CUDA not available")

    try:
        jax.devices("gpu")
    except RuntimeError:
        issues.append("JAX GPU not available")

    if issues:
        print("Environment check FAILED:")
        for issue in issues:
            print(f"  ✗ {issue}")
        return False
    else:
        print("Environment check PASSED:")
        print("  ✓ Cutile available")
        print("  ✓ PyTorch CUDA available")
        print("  ✓ JAX GPU available")
        return True


if __name__ == "__main__":
    """Run environment check."""
    print("Cutile LoRA Test Implementation")
    print("=" * 50)
    check_environment()
