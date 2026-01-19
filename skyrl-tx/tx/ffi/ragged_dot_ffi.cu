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
#include <cutlass/util/packed_stride.hpp>

#include <cute/tensor.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>

#if !defined(CUTLASS_MAJOR) || CUTLASS_MAJOR < 3
#error "This kernel requires CUTLASS >= 3.x (SM90 grouped GEMM)."
#endif

namespace ffi = xla::ffi;

static std::vector<cudaDeviceProp> g_device_props;

static int get_sm_count() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess || device < 0) {
    return 0;
  }
  if (static_cast<size_t>(device) >= g_device_props.size()) {
    g_device_props.resize(device + 1);
  }
  cudaDeviceProp& props = g_device_props[device];
  if (!props.multiProcessorCount) {
    cudaGetDeviceProperties(&props, device);
  }
  return props.multiProcessorCount;
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
using TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;
using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
using ProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int32_t, int32_t, int32_t>>;

using CollectiveEpilogue =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        DtypeAccum, DtypeAccum, void, LayoutOutput*, AlignmentOutput,
        DtypeOutput, LayoutOutput*, AlignmentOutput, EpilogueSchedule,
        cutlass::epilogue::fusion::LinearCombination<DtypeOutput, DtypeAccum>>::CollectiveOp;

using CollectiveMainloop =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, DtypeA, LayoutA*, AlignmentA,
        DtypeB, LayoutB*, AlignmentB, DtypeAccum, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename Gemm::GemmKernel::InternalStrideA;
using StrideB = typename Gemm::GemmKernel::InternalStrideB;
using StrideOutput = typename Gemm::GemmKernel::InternalStrideD;
using ProblemShapeType = ProblemShape::UnderlyingProblemShape;

// Backward kernel for d_rhs: computes lhs.T @ grad per group
// Uses ColumnMajor for A to interpret row-major lhs as transposed
using LayoutA_Bwd = cutlass::layout::ColumnMajor;

using CollectiveEpilogue_Bwd =
    typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, TileShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        DtypeAccum, DtypeAccum, void, LayoutOutput*, AlignmentOutput,
        DtypeOutput, LayoutOutput*, AlignmentOutput, EpilogueSchedule,
        cutlass::epilogue::fusion::LinearCombination<DtypeOutput, DtypeAccum>>::CollectiveOp;

using CollectiveMainloop_Bwd =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, DtypeA, LayoutA_Bwd*, AlignmentA,
        DtypeB, LayoutB*, AlignmentB, DtypeAccum, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue_Bwd::SharedStorage))>,
        KernelSchedule>::CollectiveOp;

using GemmKernel_Bwd = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop_Bwd, CollectiveEpilogue_Bwd>;
using Gemm_Bwd = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_Bwd>;
using StrideA_Bwd = typename Gemm_Bwd::GemmKernel::InternalStrideA;
using StrideB_Bwd = typename Gemm_Bwd::GemmKernel::InternalStrideB;
using StrideOutput_Bwd = typename Gemm_Bwd::GemmKernel::InternalStrideD;


__global__ void prepare_grouped_gemm_data(
    const DtypeA* A,
    const DtypeB* B,
    DtypeOutput* output,
    const int32_t* group_offsets_cumsum,
    const int32_t* group_offset_ptr,
    int32_t k,
    int32_t n,
    DtypeA** A_ptrs,
    DtypeB** B_ptrs,
    DtypeOutput** output_ptrs,
    StrideA* stride_A,
    StrideB* stride_B,
    StrideOutput* stride_output,
    ProblemShapeType* problem_sizes) {
  int32_t tid = threadIdx.x;
  int32_t global = group_offset_ptr[0] + tid;

  int32_t start = (global > 0) ? group_offsets_cumsum[global - 1] : 0;
  int32_t m = group_offsets_cumsum[global] - start;

  A_ptrs[tid] = const_cast<DtypeA*>(A) + static_cast<int64_t>(start) * k;
  B_ptrs[tid] = const_cast<DtypeB*>(B) + static_cast<int64_t>(tid) * n * k;
  output_ptrs[tid] = output + static_cast<int64_t>(start) * n;
  problem_sizes[tid] = ProblemShapeType(m, n, k);

  stride_A[tid] = cutlass::make_cute_packed_stride(StrideA{}, {k, k, 1});
  stride_B[tid] = cutlass::make_cute_packed_stride(StrideB{}, {n, n, 1});
  stride_output[tid] = cutlass::make_cute_packed_stride(StrideOutput{}, {m, n, 1});
}

// Backward kernel for d_rhs: lhs.T @ grad -> d_rhs[G, K, N]
// Memory pattern: A=lhs (ragged), B=grad (ragged), output=d_rhs (per-group)
__global__ void prepare_grouped_gemm_bwd_data(
    const DtypeA* lhs,           // [M_total, K] row-major (ragged)
    const DtypeB* grad,          // [M_total, N] row-major (ragged)
    DtypeOutput* d_rhs,          // [G, K, N] row-major (per-group)
    const int32_t* group_offsets_cumsum,
    const int32_t* group_offset_ptr,
    int32_t k,
    int32_t n,
    DtypeA** A_ptrs,
    DtypeB** B_ptrs,
    DtypeOutput** output_ptrs,
    StrideA_Bwd* stride_A,
    StrideB_Bwd* stride_B,
    StrideOutput_Bwd* stride_output,
    ProblemShapeType* problem_sizes) {
  int32_t tid = threadIdx.x;
  int32_t global = group_offset_ptr[0] + tid;

  int32_t start = (global > 0) ? group_offsets_cumsum[global - 1] : 0;
  int32_t m = group_offsets_cumsum[global] - start;

  // A = lhs slice, viewed as ColumnMajor [K, M_g] (transposed)
  A_ptrs[tid] = const_cast<DtypeA*>(lhs) + static_cast<int64_t>(start) * k;
  // B = grad slice [M_g, N]
  B_ptrs[tid] = const_cast<DtypeB*>(grad) + static_cast<int64_t>(start) * n;
  // Output = d_rhs[tid] with shape [K, N]
  output_ptrs[tid] = d_rhs + static_cast<int64_t>(tid) * k * n;

  // GEMM: [K, M_g] @ [M_g, N] = [K, N]
  problem_sizes[tid] = ProblemShapeType(k, n, m);

  // ColumnMajor stride for A (lhs viewed as transposed)
  stride_A[tid] = cutlass::make_cute_packed_stride(StrideA_Bwd{}, {m, m, 1});
  stride_B[tid] = cutlass::make_cute_packed_stride(StrideB_Bwd{}, {n, n, 1});
  stride_output[tid] = cutlass::make_cute_packed_stride(StrideOutput_Bwd{}, {k, n, 1});
}

ffi::Error RaggedDotCudaImpl(
    cudaStream_t stream,
    ffi::ScratchAllocator scratch,
    ffi::Buffer<ffi::BF16> lhs,
    ffi::Buffer<ffi::BF16> rhs,
    ffi::Buffer<ffi::S32> group_offset,
    ffi::Buffer<ffi::S32> group_offsets_cumsum,
    ffi::ResultBuffer<ffi::BF16> out) {
  auto lhs_dims = lhs.dimensions();
  auto rhs_dims = rhs.dimensions();
  auto group_offset_dims = group_offset.dimensions();

  if (lhs_dims.size() != 2 || rhs_dims.size() != 3 || group_offset_dims.size() != 1) {
    return ffi::Error::InvalidArgument("Unexpected ragged_dot dimensions.");
  }

  int64_t m64 = lhs_dims[0];
  int64_t k64 = lhs_dims[1];
  int64_t g_local64 = rhs_dims[0];
  int64_t rhs_k64 = rhs_dims[1];
  int64_t n64 = rhs_dims[2];

  if (k64 != rhs_k64) {
    return ffi::Error::InvalidArgument("lhs/rhs K dimension mismatch.");
  }

  int32_t m = static_cast<int32_t>(m64);
  int32_t k = static_cast<int32_t>(k64);
  int32_t n = static_cast<int32_t>(n64);
  int32_t g_local = static_cast<int32_t>(g_local64);

  const int32_t* group_offset_ptr = group_offset.typed_data();
  cudaError_t err;

  err = cudaMemsetAsync(
      out->typed_data(), 0, static_cast<size_t>(m) * n * sizeof(DtypeOutput), stream);
  if (err != cudaSuccess) {
    return ffi::Error::Internal("Failed to zero output.");
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

  size_t gl = static_cast<size_t>(g_local);
  size_t bytes = 7 * 16 +  // alignment padding
                 sizeof(DtypeA*) * gl + sizeof(DtypeB*) * gl + sizeof(DtypeOutput*) * gl +
                 sizeof(StrideA) * gl + sizeof(StrideB) * gl + sizeof(StrideOutput) * gl +
                 sizeof(ProblemShapeType) * gl;

  auto slab_or = scratch.Allocate(bytes);
  if (!slab_or.has_value()) {
    return ffi::Error::Internal("Failed to allocate grouped GEMM slab from scratch.");
  }

  auto align16 = [](char*& p) { p = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(p) + 15) & ~15); };
  char* p = reinterpret_cast<char*>(slab_or.value());
  align16(p); DtypeA** d_A_ptrs = reinterpret_cast<DtypeA**>(p); p += sizeof(DtypeA*) * gl;
  align16(p); DtypeB** d_B_ptrs = reinterpret_cast<DtypeB**>(p); p += sizeof(DtypeB*) * gl;
  align16(p); DtypeOutput** d_out_ptrs = reinterpret_cast<DtypeOutput**>(p); p += sizeof(DtypeOutput*) * gl;
  align16(p); StrideA* d_stride_A = reinterpret_cast<StrideA*>(p); p += sizeof(StrideA) * gl;
  align16(p); StrideB* d_stride_B = reinterpret_cast<StrideB*>(p); p += sizeof(StrideB) * gl;
  align16(p); StrideOutput* d_stride_output = reinterpret_cast<StrideOutput*>(p); p += sizeof(StrideOutput) * gl;
  align16(p); ProblemShapeType* d_problem_sizes = reinterpret_cast<ProblemShapeType*>(p);

  prepare_grouped_gemm_data<<<1, g_local, 0, stream>>>(
      A_base, B_base, out_base,
      group_offsets_cumsum_ptr, group_offset_ptr, k, n,
      d_A_ptrs, d_B_ptrs, d_out_ptrs,
      d_stride_A, d_stride_B, d_stride_output, d_problem_sizes);

  Gemm gemm;
  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {g_local, d_problem_sizes, nullptr},
      {(const DtypeA**)d_A_ptrs, d_stride_A, (const DtypeB**)d_B_ptrs, d_stride_B},
      {{}, nullptr, d_stride_output, d_out_ptrs, d_stride_output}};

  args.epilogue.thread.alpha = 1.0f;
  args.epilogue.thread.dAlpha = {cute::_0{}, cute::_0{}, 0};
  args.hw_info.sm_count = get_sm_count();

  cutlass::Status status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return ffi::Error::Internal("cutlass cannot implement grouped gemm.");
  }

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
        .Arg<ffi::Buffer<ffi::S32>>()   // group_offset
        .Arg<ffi::Buffer<ffi::S32>>()   // group_offsets_cumsum
        .Ret<ffi::Buffer<ffi::BF16>>());  // out

// Backward pass for d_rhs: computes lhs.T @ grad per group -> d_rhs[G, K, N]
ffi::Error RaggedDotBwdCudaImpl(
    cudaStream_t stream,
    ffi::ScratchAllocator scratch,
    ffi::Buffer<ffi::BF16> lhs,
    ffi::Buffer<ffi::BF16> grad,
    ffi::Buffer<ffi::S32> group_offset,
    ffi::Buffer<ffi::S32> group_offsets_cumsum,
    ffi::ResultBuffer<ffi::BF16> d_rhs) {
  auto lhs_dims = lhs.dimensions();
  auto grad_dims = grad.dimensions();
  auto d_rhs_dims = d_rhs->dimensions();

  if (lhs_dims.size() != 2 || grad_dims.size() != 2 || d_rhs_dims.size() != 3) {
    return ffi::Error::InvalidArgument("Unexpected ragged_dot_bwd dimensions.");
  }

  int64_t m64 = lhs_dims[0];
  int64_t k64 = lhs_dims[1];
  int64_t grad_m64 = grad_dims[0];
  int64_t n64 = grad_dims[1];
  int64_t g_local64 = d_rhs_dims[0];

  if (m64 != grad_m64) {
    return ffi::Error::InvalidArgument("lhs/grad M dimension mismatch.");
  }
  if (d_rhs_dims[1] != k64 || d_rhs_dims[2] != n64) {
    return ffi::Error::InvalidArgument("d_rhs shape must be [G, K, N].");
  }

  int32_t m = static_cast<int32_t>(m64);
  int32_t k = static_cast<int32_t>(k64);
  int32_t n = static_cast<int32_t>(n64);
  int32_t g_local = static_cast<int32_t>(g_local64);

  cudaError_t err;
  err = cudaMemsetAsync(
      d_rhs->typed_data(), 0, static_cast<size_t>(g_local) * k * n * sizeof(DtypeOutput), stream);
  if (err != cudaSuccess) {
    return ffi::Error::Internal("Failed to zero d_rhs output.");
  }

  if (g_local == 0 || m == 0 || n == 0 || k == 0) {
    return ffi::Error::Success();
  }

  if (g_local > 1024) {
    return ffi::Error::InvalidArgument("group_count must be <= 1024.");
  }

  const DtypeA* lhs_base = reinterpret_cast<const DtypeA*>(lhs.typed_data());
  const DtypeB* grad_base = reinterpret_cast<const DtypeB*>(grad.typed_data());
  DtypeOutput* d_rhs_base = reinterpret_cast<DtypeOutput*>(d_rhs->typed_data());
  const int32_t* group_offset_ptr = group_offset.typed_data();
  const int32_t* group_offsets_cumsum_ptr = group_offsets_cumsum.typed_data();

  size_t gl = static_cast<size_t>(g_local);
  size_t bytes = 7 * 16 +
                 sizeof(DtypeA*) * gl + sizeof(DtypeB*) * gl + sizeof(DtypeOutput*) * gl +
                 sizeof(StrideA_Bwd) * gl + sizeof(StrideB_Bwd) * gl + sizeof(StrideOutput_Bwd) * gl +
                 sizeof(ProblemShapeType) * gl;

  auto slab_or = scratch.Allocate(bytes);
  if (!slab_or.has_value()) {
    return ffi::Error::Internal("Failed to allocate grouped GEMM bwd slab from scratch.");
  }

  auto align16 = [](char*& p) { p = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(p) + 15) & ~15); };
  char* p = reinterpret_cast<char*>(slab_or.value());
  align16(p); DtypeA** d_A_ptrs = reinterpret_cast<DtypeA**>(p); p += sizeof(DtypeA*) * gl;
  align16(p); DtypeB** d_B_ptrs = reinterpret_cast<DtypeB**>(p); p += sizeof(DtypeB*) * gl;
  align16(p); DtypeOutput** d_out_ptrs = reinterpret_cast<DtypeOutput**>(p); p += sizeof(DtypeOutput*) * gl;
  align16(p); StrideA_Bwd* d_stride_A = reinterpret_cast<StrideA_Bwd*>(p); p += sizeof(StrideA_Bwd) * gl;
  align16(p); StrideB_Bwd* d_stride_B = reinterpret_cast<StrideB_Bwd*>(p); p += sizeof(StrideB_Bwd) * gl;
  align16(p); StrideOutput_Bwd* d_stride_output = reinterpret_cast<StrideOutput_Bwd*>(p); p += sizeof(StrideOutput_Bwd) * gl;
  align16(p); ProblemShapeType* d_problem_sizes = reinterpret_cast<ProblemShapeType*>(p);

  prepare_grouped_gemm_bwd_data<<<1, g_local, 0, stream>>>(
      lhs_base, grad_base, d_rhs_base,
      group_offsets_cumsum_ptr, group_offset_ptr, k, n,
      d_A_ptrs, d_B_ptrs, d_out_ptrs,
      d_stride_A, d_stride_B, d_stride_output, d_problem_sizes);

  Gemm_Bwd gemm;
  typename Gemm_Bwd::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {g_local, d_problem_sizes, nullptr},
      {(const DtypeA**)d_A_ptrs, d_stride_A, (const DtypeB**)d_B_ptrs, d_stride_B},
      {{}, nullptr, d_stride_output, d_out_ptrs, d_stride_output}};

  args.epilogue.thread.alpha = 1.0f;
  args.epilogue.thread.dAlpha = {cute::_0{}, cute::_0{}, 0};
  args.hw_info.sm_count = get_sm_count();

  cutlass::Status status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return ffi::Error::Internal("cutlass cannot implement grouped gemm bwd.");
  }

  size_t workspace_size = Gemm_Bwd::get_workspace_size(args);
  void* workspace = nullptr;
  if (workspace_size > 0) {
    auto workspace_or = scratch.Allocate(workspace_size);
    if (!workspace_or.has_value()) {
      return ffi::Error::Internal("Failed to allocate CUTLASS bwd workspace from scratch.");
    }
    workspace = workspace_or.value();
  }

  status = gemm(args, workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    return ffi::Error::Internal("cutlass grouped gemm bwd failed.");
  }

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RaggedDotBwdCuda,
    RaggedDotBwdCudaImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::ScratchAllocator>()
        .Arg<ffi::Buffer<ffi::BF16>>()  // lhs
        .Arg<ffi::Buffer<ffi::BF16>>()  // grad
        .Arg<ffi::Buffer<ffi::S32>>()   // group_offset
        .Arg<ffi::Buffer<ffi::S32>>()   // group_offsets_cumsum
        .Ret<ffi::Buffer<ffi::BF16>>());  // d_rhs
