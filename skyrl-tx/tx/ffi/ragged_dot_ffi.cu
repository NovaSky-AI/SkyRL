#include <cuda_runtime.h>
#include <stdint.h>

#include <array>
#include <vector>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include <cutlass/cutlass.h>
#include <cutlass/version.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/memory.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/util/packed_stride.hpp>

#include <cute/tensor.hpp>
#include <cutlass/epilogue/threadblock/default_thread_map_tensor_op.h>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>

#if !defined(CUTLASS_MAJOR) || CUTLASS_MAJOR < 3
#error "This kernel requires CUTLASS >= 3.x (SM90 grouped GEMM)."
#endif

namespace ffi = xla::ffi;

using DtypeA = cutlass::bfloat16_t;
using DtypeB = cutlass::bfloat16_t;
using DtypeOutput = cutlass::bfloat16_t;
using DtypeAccum = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;
constexpr int AlignmentA = 8;
constexpr int AlignmentB = 8;
constexpr int AlignmentOutput = 8;

using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using TileShape = cute::Shape<cute::_128, cute::_128, cute::_64>;
using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
using ProblemShape = cutlass::gemm::GroupProblemShape<
    cute::Shape<int32_t, int32_t, int32_t>>;

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag,
        OperatorClass,
        TileShape,
        ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        DtypeAccum,
        DtypeAccum,
        void,
        LayoutOutput*,
        AlignmentOutput,
        DtypeOutput,
        LayoutOutput*,
        AlignmentOutput,
        EpilogueSchedule,
        cutlass::epilogue::fusion::LinearCombination<DtypeOutput, DtypeAccum>>::CollectiveOp;

using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag,
        OperatorClass,
        DtypeA,
        LayoutA*,
        AlignmentA,
        DtypeB,
        LayoutB*,
        AlignmentB,
        DtypeAccum,
        TileShape,
        ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    ProblemShape,
    CollectiveMainloop,
    CollectiveEpilogue>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideOutput = typename Gemm::GemmKernel::InternalStrideD;
using ProblemShapeType = ProblemShape::UnderlyingProblemShape;

static ffi::Error CudaError(const char* message) {
  return ffi::Error::Internal(message);
}

using Strides = std::array<int64_t, 3>;

__global__ void prepare_grouped_gemm_data(
    const DtypeA* A,
    const DtypeB* B,
    DtypeOutput* output,
    const int32_t* group_sizes,
    const int32_t* group_offset,
    int32_t num_groups,
    int32_t group_count,
    int32_t k,
    int32_t n,
    const Strides tensor_StrideA,
    const Strides tensor_StrideB,
    const Strides tensor_StrideOutput,
    const Strides tensor_ShapeA,
    DtypeA** A_ptrs,
    DtypeB** B_ptrs,
    DtypeOutput** output_ptrs,
    StrideA* stride_A,
    StrideB* stride_B,
    StrideOutput* stride_output,
    ProblemShapeType* problem_sizes) {
  int32_t tid = threadIdx.x;
  if (tid >= group_count) {
    return;
  }

  int32_t offset = group_offset[0];
  int32_t global = offset + tid;
  if (global < 0 || global >= num_groups) {
    return;
  }

  int32_t start = 0;
  for (int32_t i = 0; i < global; ++i) {
    start += group_sizes[i];
  }
  int32_t m = group_sizes[global];
  if (m < 0) {
    return;
  }

  if (start + m > tensor_ShapeA[0]) {
    return;
  }

  int64_t lda = tensor_StrideA[0];
  int64_t ldb = tensor_StrideB[1];
  int64_t ldoutput = tensor_StrideOutput[0];

  A_ptrs[tid] = const_cast<DtypeA*>(A) + static_cast<int64_t>(start) * lda;
  B_ptrs[tid] = const_cast<DtypeB*>(B) + static_cast<int64_t>(tid) * tensor_StrideB[0];
  output_ptrs[tid] = output + static_cast<int64_t>(start) * ldoutput;
  problem_sizes[tid] = ProblemShapeType(m, n, k);

  stride_A[tid] = cutlass::make_cute_packed_stride(StrideA{}, {lda, lda, 1});
  stride_B[tid] = cutlass::make_cute_packed_stride(StrideB{}, {ldb, ldb, 1});
  stride_output[tid] = cutlass::make_cute_packed_stride(StrideOutput{}, {m, ldoutput, 1});
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

  const int32_t* group_sizes_ptr = group_sizes.typed_data();
  const int32_t* group_offset_ptr = group_offset.typed_data();
  cudaError_t err;

  err = cudaMemsetAsync(
      out->typed_data(), 0, static_cast<size_t>(m) * n * sizeof(DtypeOutput), stream);
  if (err != cudaSuccess) {
    return CudaError("Failed to zero output.");
  }

  if (g_local == 0 || m == 0 || n == 0 || k == 0) {
    return ffi::Error::Success();
  }

  if (g_local > 1024) {
    return ffi::Error::InvalidArgument("group_count must be <= 1024.");
  }

  const DtypeA* A_base = reinterpret_cast<const DtypeA*>(lhs.typed_data());
  const DtypeB* B_base = reinterpret_cast<const DtypeB*>(rhs.typed_data());
  DtypeOutput* out_base = reinterpret_cast<DtypeOutput*>(out->typed_data());

  Strides strideA = {k, 1, 1};
  Strides strideB = {static_cast<int64_t>(k) * n, n, 1};
  Strides strideOut = {n, 1, 1};
  Strides shapeA = {m, k, 1};

  auto align_up = [](size_t value, size_t alignment) -> size_t {
    return (value + alignment - 1) & ~(alignment - 1);
  };

  size_t bytes = 0;
  bytes = align_up(bytes, 16);
  size_t A_ptrs_offset = bytes;
  bytes += sizeof(DtypeA*) * static_cast<size_t>(g_local);
  bytes = align_up(bytes, 16);
  size_t B_ptrs_offset = bytes;
  bytes += sizeof(DtypeB*) * static_cast<size_t>(g_local);
  bytes = align_up(bytes, 16);
  size_t out_ptrs_offset = bytes;
  bytes += sizeof(DtypeOutput*) * static_cast<size_t>(g_local);
  bytes = align_up(bytes, 16);
  size_t stride_A_offset = bytes;
  bytes += sizeof(StrideA) * static_cast<size_t>(g_local);
  bytes = align_up(bytes, 16);
  size_t stride_B_offset = bytes;
  bytes += sizeof(StrideB) * static_cast<size_t>(g_local);
  bytes = align_up(bytes, 16);
  size_t stride_output_offset = bytes;
  bytes += sizeof(StrideOutput) * static_cast<size_t>(g_local);
  bytes = align_up(bytes, 16);
  size_t problem_sizes_offset = bytes;
  bytes += sizeof(ProblemShape::UnderlyingProblemShape) * static_cast<size_t>(g_local);

  void* slab = nullptr;
  err = cudaMalloc(&slab, bytes);
  if (err != cudaSuccess) {
    return CudaError("Failed to allocate grouped GEMM slab.");
  }

  char* base = reinterpret_cast<char*>(slab);
  DtypeA** d_A_ptrs = reinterpret_cast<DtypeA**>(base + A_ptrs_offset);
  DtypeB** d_B_ptrs = reinterpret_cast<DtypeB**>(base + B_ptrs_offset);
  DtypeOutput** d_out_ptrs = reinterpret_cast<DtypeOutput**>(base + out_ptrs_offset);
  StrideA* d_stride_A = reinterpret_cast<StrideA*>(base + stride_A_offset);
  StrideB* d_stride_B = reinterpret_cast<StrideB*>(base + stride_B_offset);
  StrideOutput* d_stride_output = reinterpret_cast<StrideOutput*>(base + stride_output_offset);
  ProblemShape::UnderlyingProblemShape* d_problem_sizes =
      reinterpret_cast<ProblemShape::UnderlyingProblemShape*>(base + problem_sizes_offset);

  prepare_grouped_gemm_data<<<1, g_local, 0, stream>>>(
      A_base,
      B_base,
      out_base,
      group_sizes_ptr,
      group_offset_ptr,
      g,
      g_local,
      k,
      n,
      strideA,
      strideB,
      strideOut,
      shapeA,
      d_A_ptrs,
      d_B_ptrs,
      d_out_ptrs,
      d_stride_A,
      d_stride_B,
      d_stride_output,
      d_problem_sizes);

  Gemm gemm;
  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {g_local, d_problem_sizes, nullptr},
      {(const DtypeA**)d_A_ptrs, d_stride_A, (const DtypeB**)d_B_ptrs, d_stride_B},
      {{}, nullptr, d_stride_output, d_out_ptrs, d_stride_output}};

  args.epilogue.thread.alpha = 1.0f;
  args.epilogue.thread.dAlpha = {cute::_0{}, cute::_0{}, 0};

  cutlass::Status status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    cudaFree(slab);
    return ffi::Error::Internal("cutlass cannot implement grouped gemm.");
  }

  int device = 0;
  cudaDeviceProp props;
  if (cudaGetDevice(&device) == cudaSuccess && cudaGetDeviceProperties(&props, device) == cudaSuccess) {
    args.hw_info.sm_count = props.multiProcessorCount;
  }

  size_t workspace_size = Gemm::get_workspace_size(args);
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
