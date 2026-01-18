#include <cuda_runtime.h>
#include <stdint.h>

#include <vector>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

namespace ffi = xla::ffi;

using Dtype = cutlass::bfloat16_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;
using Accum = float;
using Gemm = cutlass::gemm::device::Gemm<Dtype, LayoutA, Dtype, LayoutB, Dtype, LayoutC, Accum>;

static ffi::Error CudaError(const char* message) {
  return ffi::Error::Internal(message);
}

ffi::Error RaggedDotCudaImpl(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16> lhs,
    ffi::Buffer<ffi::BF16> rhs,
    ffi::Buffer<ffi::S32> group_sizes,
    ffi::Buffer<ffi::S32> group_offset,
    ffi::ResultBuffer<ffi::BF16> out) {
  auto lhs_dims = lhs.dimensions();
  auto rhs_dims = rhs.dimensions();
  auto group_sizes_dims = group_sizes.dimensions();
  auto group_offset_dims = group_offset.dimensions();

  if (lhs_dims.size() != 2 || rhs_dims.size() != 3 || group_sizes_dims.size() != 1 ||
      group_offset_dims.size() != 1) {
    return ffi::Error::InvalidArgument("Unexpected ragged_dot dimensions.");
  }

  int64_t m64 = lhs_dims[0];
  int64_t k64 = lhs_dims[1];
  int64_t g_local64 = rhs_dims[0];
  int64_t rhs_k64 = rhs_dims[1];
  int64_t n64 = rhs_dims[2];
  int64_t g64 = group_sizes_dims[0];

  if (k64 != rhs_k64) {
    return ffi::Error::InvalidArgument("lhs/rhs K dimension mismatch.");
  }

  if (m64 < 0 || k64 < 0 || n64 < 0 || g64 < 0 || g_local64 < 0) {
    return ffi::Error::InvalidArgument("Invalid dimensions.");
  }

  int32_t m = static_cast<int32_t>(m64);
  int32_t k = static_cast<int32_t>(k64);
  int32_t n = static_cast<int32_t>(n64);
  int32_t g = static_cast<int32_t>(g64);
  int32_t g_local = static_cast<int32_t>(g_local64);

  int32_t offset = 0;
  cudaError_t err = cudaMemcpyAsync(
      &offset, group_offset.typed_data(), sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    return CudaError("Failed to copy group_offset.");
  }

  std::vector<int32_t> sizes(static_cast<size_t>(g));
  if (g > 0) {
    err = cudaMemcpyAsync(
        sizes.data(),
        group_sizes.typed_data(),
        static_cast<size_t>(g) * sizeof(int32_t),
        cudaMemcpyDeviceToHost,
        stream);
    if (err != cudaSuccess) {
      return CudaError("Failed to copy group_sizes.");
    }
  }

  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    return CudaError("Failed to synchronize stream.");
  }

  if (offset < 0 || offset > g || offset + g_local > g) {
    return ffi::Error::InvalidArgument("group_offset out of range.");
  }

  std::vector<int32_t> offsets(static_cast<size_t>(g) + 1);
  offsets[0] = 0;
  for (int32_t i = 0; i < g; ++i) {
    offsets[static_cast<size_t>(i) + 1] = offsets[static_cast<size_t>(i)] + sizes[i];
  }
  if (offsets[static_cast<size_t>(g)] != m) {
    return ffi::Error::InvalidArgument("group_sizes sum does not match lhs rows.");
  }

  err = cudaMemsetAsync(out->typed_data(), 0, static_cast<size_t>(m) * n * sizeof(Dtype), stream);
  if (err != cudaSuccess) {
    return CudaError("Failed to zero output.");
  }

  if (g_local == 0 || m == 0 || n == 0 || k == 0) {
    return ffi::Error::Success();
  }

  Gemm gemm;
  size_t max_workspace = 0;
  for (int32_t gi = offset; gi < offset + g_local; ++gi) {
    int32_t rows = sizes[gi];
    if (rows == 0) {
      continue;
    }

    int32_t local = gi - offset;
    const Dtype* A = reinterpret_cast<const Dtype*>(lhs.typed_data()) +
        static_cast<int64_t>(offsets[gi]) * k;
    const Dtype* B = reinterpret_cast<const Dtype*>(rhs.typed_data()) +
        static_cast<int64_t>(local) * k * n;
    Dtype* C = reinterpret_cast<Dtype*>(out->typed_data()) +
        static_cast<int64_t>(offsets[gi]) * n;

    Gemm::Arguments args(
        {rows, n, k},
        {A, k},
        {B, n},
        {C, n},
        {C, n},
        {1.0f, 0.0f});

    size_t workspace_size = Gemm::get_workspace_size(args);
    if (workspace_size > max_workspace) {
      max_workspace = workspace_size;
    }
  }

  void* workspace = nullptr;
  if (max_workspace > 0) {
    err = cudaMalloc(&workspace, max_workspace);
    if (err != cudaSuccess) {
      return CudaError("Failed to allocate CUTLASS workspace.");
    }
  }

  for (int32_t gi = offset; gi < offset + g_local; ++gi) {
    int32_t rows = sizes[gi];
    if (rows == 0) {
      continue;
    }

    int32_t local = gi - offset;
    const Dtype* A = reinterpret_cast<const Dtype*>(lhs.typed_data()) +
        static_cast<int64_t>(offsets[gi]) * k;
    const Dtype* B = reinterpret_cast<const Dtype*>(rhs.typed_data()) +
        static_cast<int64_t>(local) * k * n;
    Dtype* C = reinterpret_cast<Dtype*>(out->typed_data()) +
        static_cast<int64_t>(offsets[gi]) * n;

    Gemm::Arguments args(
        {rows, n, k},
        {A, k},
        {B, n},
        {C, n},
        {C, n},
        {1.0f, 0.0f});

    cutlass::Status status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
      if (workspace != nullptr) {
        cudaFree(workspace);
      }
      return ffi::Error::Internal("cutlass cannot implement.");
    }

    status = gemm.initialize(args, workspace);
    if (status != cutlass::Status::kSuccess) {
      if (workspace != nullptr) {
        cudaFree(workspace);
      }
      return ffi::Error::Internal("cutlass cannot initialize.");
    }

    status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess) {
      if (workspace != nullptr) {
        cudaFree(workspace);
      }
      return ffi::Error::Internal("cutlass gemm failed.");
    }
  }

  if (workspace != nullptr) {
    cudaFree(workspace);
  }

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RaggedDotCuda,
    RaggedDotCudaImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::BF16>>()  // lhs
        .Arg<ffi::Buffer<ffi::BF16>>()  // rhs
        .Arg<ffi::Buffer<ffi::S32>>()   // group_sizes
        .Arg<ffi::Buffer<ffi::S32>>()   // group_offset
        .Ret<ffi::Buffer<ffi::BF16>>());  // out
