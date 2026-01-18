#include <cuda_runtime.h>
#include <stdint.h>

#include <array>
#include <vector>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

namespace ffi = xla::ffi;

using Dtype = cutlass::bfloat16_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;
using Accum = float;
using Gemm = cutlass::gemm::device::GemmGrouped<
    Dtype,
    LayoutA,
    Dtype,
    LayoutB,
    Dtype,
    LayoutC,
    Accum>;

static ffi::Error CudaError(const char* message) {
  return ffi::Error::Internal(message);
}

using Strides = std::array<int64_t, 3>;

__global__ void prepare_grouped_gemm_data(
    const Dtype* A,
    const Dtype* B,
    Dtype* output,
    const int32_t* offs,
    int32_t group_count,
    int32_t k,
    int32_t n,
    int64_t lda,
    int64_t ldb_group,
    int64_t ldout,
    const Strides tensor_ShapeA,
    Dtype** A_ptrs,
    Dtype** B_ptrs,
    Dtype** output_ptrs,
    cutlass::gemm::GemmCoord* problem_sizes) {
  int32_t tid = threadIdx.x;
  if (tid >= group_count) {
    return;
  }

  int32_t start = tid == 0 ? 0 : offs[tid - 1];
  int32_t end = offs[tid];
  int32_t m = end - start;
  if (m < 0) {
    return;
  }

  if (end > tensor_ShapeA[0]) {
    return;
  }

  A_ptrs[tid] = const_cast<Dtype*>(A) + static_cast<int64_t>(start) * lda;
  B_ptrs[tid] = const_cast<Dtype*>(B) + static_cast<int64_t>(tid) * ldb_group;
  output_ptrs[tid] = output + static_cast<int64_t>(start) * ldout;
  problem_sizes[tid] = cutlass::gemm::GemmCoord(m, n, k);
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

  if (g_local > 1024) {
    return ffi::Error::InvalidArgument("group_count must be <= 1024.");
  }

  int32_t shard_start = offsets[static_cast<size_t>(offset)];

  std::vector<int32_t> local_offs(static_cast<size_t>(g_local));
  int32_t running = 0;
  for (int32_t i = 0; i < g_local; ++i) {
    running += sizes[offset + i];
    local_offs[static_cast<size_t>(i)] = running;
  }

  const Dtype* A_base = reinterpret_cast<const Dtype*>(lhs.typed_data()) +
      static_cast<int64_t>(shard_start) * k;
  const Dtype* B_base = reinterpret_cast<const Dtype*>(rhs.typed_data());
  Dtype* out_base = reinterpret_cast<Dtype*>(out->typed_data()) +
      static_cast<int64_t>(shard_start) * n;

  int64_t lda = k;
  int64_t ldb_group = static_cast<int64_t>(k) * n;
  int64_t ldout = n;

  Strides shapeA = {m, k, 1};

  auto align_up = [](size_t value, size_t alignment) -> size_t {
    return (value + alignment - 1) & ~(alignment - 1);
  };

  size_t bytes = 0;
  bytes = align_up(bytes, 16);
  size_t offs_offset = bytes;
  bytes += sizeof(int32_t) * static_cast<size_t>(g_local);
  bytes = align_up(bytes, 16);
  size_t A_ptrs_offset = bytes;
  bytes += sizeof(Dtype*) * static_cast<size_t>(g_local);
  bytes = align_up(bytes, 16);
  size_t B_ptrs_offset = bytes;
  bytes += sizeof(Dtype*) * static_cast<size_t>(g_local);
  bytes = align_up(bytes, 16);
  size_t out_ptrs_offset = bytes;
  bytes += sizeof(Dtype*) * static_cast<size_t>(g_local);
  bytes = align_up(bytes, 16);
  size_t problem_sizes_offset = bytes;
  bytes += sizeof(cutlass::gemm::GemmCoord) * static_cast<size_t>(g_local);

  void* slab = nullptr;
  err = cudaMalloc(&slab, bytes);
  if (err != cudaSuccess) {
    return CudaError("Failed to allocate grouped GEMM slab.");
  }

  char* base = reinterpret_cast<char*>(slab);
  int32_t* d_offs = reinterpret_cast<int32_t*>(base + offs_offset);
  Dtype** d_A_ptrs = reinterpret_cast<Dtype**>(base + A_ptrs_offset);
  Dtype** d_B_ptrs = reinterpret_cast<Dtype**>(base + B_ptrs_offset);
  Dtype** d_out_ptrs = reinterpret_cast<Dtype**>(base + out_ptrs_offset);
  cutlass::gemm::GemmCoord* d_problem_sizes =
      reinterpret_cast<cutlass::gemm::GemmCoord*>(base + problem_sizes_offset);

  err = cudaMemcpyAsync(
      d_offs,
      local_offs.data(),
      sizeof(int32_t) * g_local,
      cudaMemcpyHostToDevice,
      stream);
  if (err != cudaSuccess) {
    cudaFree(slab);
    return CudaError("Failed to copy offs.");
  }

  prepare_grouped_gemm_data<<<1, g_local, 0, stream>>>(
      A_base,
      B_base,
      out_base,
      d_offs,
      g_local,
      k,
      n,
      lda,
      ldb_group,
      ldout,
      shapeA,
      d_A_ptrs,
      d_B_ptrs,
      d_out_ptrs,
      d_problem_sizes);

  Gemm gemm;
  typename Gemm::Arguments args(
      d_problem_sizes,
      g_local,
      {d_A_ptrs, lda},
      {d_B_ptrs, n},
      {d_out_ptrs, ldout},
      {d_out_ptrs, ldout},
      {1.0f, 0.0f});

  cutlass::Status status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    cudaFree(slab);
    return ffi::Error::Internal("cutlass cannot implement grouped gemm.");
  }

  size_t workspace_size = gemm.get_workspace_size(args);
  void* workspace = nullptr;
  if (workspace_size > 0) {
    err = cudaMalloc(&workspace, workspace_size);
    if (err != cudaSuccess) {
      cudaFree(slab);
      return CudaError("Failed to allocate CUTLASS workspace.");
    }
  }

  status = gemm(args, workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    if (workspace != nullptr) {
      cudaFree(workspace);
    }
    cudaFree(slab);
    return ffi::Error::Internal("cutlass grouped gemm failed.");
  }

  if (workspace != nullptr) {
    cudaFree(workspace);
  }
  cudaFree(slab);

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
