# CUDA FFI Extensions

## Building

The CUDA extension is built automatically when creating a wheel:

```bash
uv build
```

For development, build manually:

```bash
uv run build-ffi
```

### Environment Variables

- `CUTLASS_DIR` - Path to CUTLASS checkout (optional, clones automatically if not set)
- `NVCC_BIN` - Path to nvcc (default: `nvcc`)
- `NVCC_ARCH` - CUDA architecture (default: `90a` for H100)

### Notes

- Requires CUDA nvcc with C++17 support
- The FFI kernel expects bfloat16 inputs/outputs and int32 group metadata
