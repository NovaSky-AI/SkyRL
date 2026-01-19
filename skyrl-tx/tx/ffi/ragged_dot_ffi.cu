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

// Cache SM count per device to avoid repeated cudaGetDeviceProperties calls
static int g_sm_count[16] = {0};

static int get_sm_count() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= 16) {
    return 0;
  }
  if (g_sm_count[device] == 0) {
    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, device) == cudaSuccess) {
      g_sm_count[device] = props.multiProcessorCount;
    }
  }
  return g_sm_count[device];
}

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
// Tuned for Qwen3-30B-A3B MoE: small M per group, K=768/2048, N=768/2048/lora_rank
// Smaller M tile (64) handles small groups better, K=64 for memory bandwidth
using TileShape = cute::Shape<cute::_64, cute::_128, cute::_64>;
// Use 2x1x1 cluster on H100 for better L2 cache utilization
using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;
// Cooperative schedule for better work distribution across thread blocks
using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
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
    const int32_t* group_offsets_cumsum,  // Precomputed cumsum of group_sizes
    const int32_t* group_offset_ptr,      // Device pointer to first group index
    int32_t num_groups,
    int32_t group_count,
    int32_t k,
    int32_t n,
    int64_t lda,
    int64_t ldb,
    int64_t ldoutput,
    int32_t total_rows,
    DtypeA** A_ptrs,
    DtypeB** B_ptrs,
    DtypeOutput** output_ptrs,
    StrideA* stride_A,
    StrideB* stride_B,
    StrideOutput* stride_output,
    ProblemShapeType* problem_sizes) {
  // Use shared memory to broadcast first_group_idx to all threads
  __shared__ int32_t s_first_group_idx;
  if (threadIdx.x == 0) {
    s_first_group_idx = group_offset_ptr[0];
  }
  __syncthreads();

  int32_t tid = threadIdx.x;
  if (tid >= group_count) {
    return;
  }

  int32_t global = s_first_group_idx + tid;
  if (global < 0 || global >= num_groups) {
    return;
  }

  // Use precomputed cumsum: start = cumsum[global-1] if global > 0, else 0
  int32_t start = (global > 0) ? group_offsets_cumsum[global - 1] : 0;
  int32_t m = group_sizes[global];
  if (m < 0 || start + m > total_rows) {
    return;
  }

  A_ptrs[tid] = const_cast<DtypeA*>(A) + static_cast<int64_t>(start) * lda;
  B_ptrs[tid] = const_cast<DtypeB*>(B) + static_cast<int64_t>(tid) * ldb * k;
  output_ptrs[tid] = output + static_cast<int64_t>(start) * ldoutput;
  problem_sizes[tid] = ProblemShapeType(m, n, k);

  stride_A[tid] = cutlass::make_cute_packed_stride(StrideA{}, {lda, lda, 1});
  stride_B[tid] = cutlass::make_cute_packed_stride(StrideB{}, {ldb, ldb, 1});
  stride_output[tid] = cutlass::make_cute_packed_stride(StrideOutput{}, {m, ldoutput, 1});
}

ffi::Error RaggedDotCudaImpl(
    cudaStream_t stream,
    ffi::ScratchAllocator scratch,
    ffi::Buffer<ffi::BF16> lhs,
    ffi::Buffer<ffi::BF16> rhs,
    ffi::Buffer<ffi::S32> group_sizes,
    ffi::Buffer<ffi::S32> group_offset,
    ffi::Buffer<ffi::S32> group_offsets_cumsum,
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
  const int32_t* group_offsets_cumsum_ptr = group_offsets_cumsum.typed_data();

  // Strides for row-major layout
  int64_t lda = k;
  int64_t ldb = n;
  int64_t ldoutput = n;

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

  auto slab_or = scratch.Allocate(bytes);
  if (!slab_or.has_value()) {
    return ffi::Error::Internal("Failed to allocate grouped GEMM slab from scratch.");
  }
  void* slab = slab_or.value();

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
      group_offsets_cumsum_ptr,
      group_offset_ptr,
      g,
      g_local,
      k,
      n,
      lda,
      ldb,
      ldoutput,
      m,
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
    return ffi::Error::Internal("cutlass cannot implement grouped gemm.");
  }

  args.hw_info.sm_count = get_sm_count();

  size_t workspace_size = Gemm::get_workspace_size(args);
  void* workspace = nullptr;
  if (workspace_size > 0) {
    auto workspace_or = scratch.Allocate(workspace_size);
    if (!workspace_or.has_value()) {
      return ffi::Error::Internal("Failed to allocate CUTLASS workspace from scratch.");
    }
    workspace = workspace_or.value();
  }

  status = gemm(args, workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    return ffi::Error::Internal("cutlass grouped gemm failed.");
  }

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RaggedDotCuda,
    RaggedDotCudaImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Arg<ffi::Buffer<ffi::BF16>>()  // lhs
        .Arg<ffi::Buffer<ffi::BF16>>()  // rhs
        .Arg<ffi::Buffer<ffi::S32>>()   // group_sizes
        .Arg<ffi::Buffer<ffi::S32>>()   // group_offset
        .Arg<ffi::Buffer<ffi::S32>>()   // group_offsets_cumsum
        .Ret<ffi::Buffer<ffi::BF16>>());  // out
