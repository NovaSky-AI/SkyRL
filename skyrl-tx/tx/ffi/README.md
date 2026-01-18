Build (Linux + CUDA)

1) Build the shared library:
```
export CUTLASS_DIR=/path/to/cutlass
tx/ffi/build_ragged_dot_ffi.sh
```

2) Make the shared library discoverable:
- Copy `tx/ffi/_build/libragged_dot_ffi.so` to `tx/ffi/libragged_dot_ffi.so`, or
- Set `TX_RAGGED_DOT_FFI_PATH=/path/to/libragged_dot_ffi.so`.

Notes:
- The FFI kernel expects bfloat16 inputs/outputs and int32 group metadata.
