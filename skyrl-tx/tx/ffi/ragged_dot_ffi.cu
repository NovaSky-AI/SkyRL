#include <cuda_runtime.h>
#include <stdint.h>

#include <optional>
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
using TileShape = cute::Shape<cute::_64, cute::_256, cute::_64>;
using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
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
using ProblemShapeType = ProblemShape::UnderlyingProblemShape;

// Backward kernel for d_rhs: computes lhs.T @ grad per group
// Uses ColumnMajor for A to interpret row-major lhs as transposed
using LayoutA_Bwd = cutlass::layout::ColumnMajor;

using CollectiveMainloop_Bwd =
    typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, DtypeA, LayoutA_Bwd*, AlignmentA,
        DtypeB, LayoutB*, AlignmentB, DtypeAccum, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule>::CollectiveOp;

using GemmKernel_Bwd = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop_Bwd, CollectiveEpilogue>;
using Gemm_Bwd = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_Bwd>;

template <typename T>
static T* carve_aligned(char*& p, size_t count) {
  T* out = reinterpret_cast<T*>((reinterpret_cast<uintptr_t>(p) + 15) & ~uintptr_t(15));
  p = reinterpret_cast<char*>(out + count);
  return out;
}

template <typename GemmT>
struct GroupedGemmData {
  using StrideA = typename GemmT::GemmKernel::InternalStrideA;
  using StrideB = typename GemmT::GemmKernel::InternalStrideB;
  using StrideOutput = typename GemmT::GemmKernel::InternalStrideD;

  const DtypeA** A_ptrs;
  const DtypeB** B_ptrs;
  DtypeOutput** out_ptrs;
  StrideA* stride_A;
  StrideB* stride_B;
  StrideOutput* stride_output;
  ProblemShapeType* problem_sizes;

  static std::optional<GroupedGemmData> Allocate(ffi::ScratchAllocator& scratch, size_t g) {
    size_t bytes = 7 * 16 +
                   sizeof(const DtypeA*) * g + sizeof(const DtypeB*) * g + sizeof(DtypeOutput*) * g +
                   sizeof(StrideA) * g + sizeof(StrideB) * g + sizeof(StrideOutput) * g +
                   sizeof(ProblemShapeType) * g;
    auto slab_or = scratch.Allocate(bytes);
    if (!slab_or.has_value()) {
      return std::nullopt;
    }
    GroupedGemmData data;
    char* p = reinterpret_cast<char*>(slab_or.value());
    data.A_ptrs = carve_aligned<const DtypeA*>(p, g);
    data.B_ptrs = carve_aligned<const DtypeB*>(p, g);
    data.out_ptrs = carve_aligned<DtypeOutput*>(p, g);
    data.stride_A = carve_aligned<StrideA>(p, g);
    data.stride_B = carve_aligned<StrideB>(p, g);
    data.stride_output = carve_aligned<StrideOutput>(p, g);
    data.problem_sizes = carve_aligned<ProblemShapeType>(p, g);
    return data;
  }

  typename GemmT::Arguments MakeArgs(int32_t g_local) const {
    typename GemmT::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {g_local, problem_sizes, nullptr},
        {A_ptrs, stride_A, B_ptrs, stride_B},
        {{}, nullptr, stride_output, out_ptrs, stride_output}};
    args.epilogue.thread.alpha = 1.0f;
    args.epilogue.thread.dAlpha = {cute::_0{}, cute::_0{}, 0};
    return args;
  }
};

enum class GemmDir { Fwd, Bwd };

// Unified prepare kernel for forward and backward passes
// Fwd: A[M,K] @ B[G,K,N] -> out[M,N] (ragged by group)
// Bwd: A.T[K,M] @ B[M,N] -> out[G,K,N] (lhs transposed via ColumnMajor layout)
// group_offsets_cumsum has length G+1 with cumsum[0]=0 (include_initial=True)
template <typename GemmT, GemmDir Dir>
__global__ void prepare_grouped_gemm(
    const DtypeA* A, const DtypeB* B, DtypeOutput* out,
    const int32_t* group_offsets_cumsum, const int32_t* group_offset_ptr,
    int32_t k, int32_t n, GroupedGemmData<GemmT> data) {
  using Data = GroupedGemmData<GemmT>;
  int32_t tid = threadIdx.x;
  int32_t global = group_offset_ptr[0] + tid;
  int32_t start = group_offsets_cumsum[global];
  int32_t m = group_offsets_cumsum[global + 1] - start;

  data.A_ptrs[tid] = A + static_cast<int64_t>(start) * k;
  if constexpr (Dir == GemmDir::Fwd) {
    data.B_ptrs[tid] = B + static_cast<int64_t>(tid) * n * k;
    data.out_ptrs[tid] = out + static_cast<int64_t>(start) * n;
    data.problem_sizes[tid] = ProblemShapeType(m, n, k);
    data.stride_A[tid] = cutlass::make_cute_packed_stride(typename Data::StrideA{}, {k, k, 1});
    data.stride_output[tid] = cutlass::make_cute_packed_stride(typename Data::StrideOutput{}, {m, n, 1});
  } else {
    data.B_ptrs[tid] = B + static_cast<int64_t>(start) * n;
    data.out_ptrs[tid] = out + static_cast<int64_t>(tid) * k * n;
    data.problem_sizes[tid] = ProblemShapeType(k, n, m);
    data.stride_A[tid] = cutlass::make_cute_packed_stride(typename Data::StrideA{}, {m, m, 1});
    data.stride_output[tid] = cutlass::make_cute_packed_stride(typename Data::StrideOutput{}, {k, n, 1});
  }
  data.stride_B[tid] = cutlass::make_cute_packed_stride(typename Data::StrideB{}, {n, n, 1});
}

template <typename GemmT, GemmDir Dir>
ffi::Error ExecuteGroupedGemm(
    cudaStream_t stream, ffi::ScratchAllocator& scratch,
    const DtypeA* A, const DtypeB* B, DtypeOutput* out,
    const int32_t* group_offsets_cumsum, const int32_t* group_offset,
    int32_t g_local, int32_t k, int32_t n) {
  if (g_local > 1024) return ffi::Error::InvalidArgument("group_count must be <= 1024.");

  auto data = GroupedGemmData<GemmT>::Allocate(scratch, g_local);
  if (!data) return ffi::Error::Internal("Failed to allocate grouped GEMM slab.");

  prepare_grouped_gemm<GemmT, Dir><<<1, g_local, 0, stream>>>(
      A, B, out, group_offsets_cumsum, group_offset, k, n, *data);

  GemmT gemm;
  auto args = data->MakeArgs(g_local);
  args.hw_info.sm_count = get_sm_count();

  if (gemm.can_implement(args) != cutlass::Status::kSuccess) {
    return ffi::Error::Internal("cutlass cannot implement grouped gemm.");
  }

  void* workspace = nullptr;
  if (size_t workspace_size = GemmT::get_workspace_size(args)) {
    auto workspace_or = scratch.Allocate(workspace_size);
    if (!workspace_or.has_value()) {
      return ffi::Error::Internal("Failed to allocate CUTLASS workspace.");
    }
    workspace = workspace_or.value();
  }

  if (gemm(args, workspace, stream) != cutlass::Status::kSuccess) {
    return ffi::Error::Internal("cutlass grouped gemm failed.");
  }

  return ffi::Error::Success();
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

  if (lhs_dims.size() != 2 || rhs_dims.size() != 3 || group_offset.dimensions().size() != 1) {
    return ffi::Error::InvalidArgument("Unexpected ragged_dot dimensions.");
  }
  if (lhs_dims[1] != rhs_dims[1]) {
    return ffi::Error::InvalidArgument("lhs/rhs K dimension mismatch.");
  }

  int32_t m = static_cast<int32_t>(lhs_dims[0]);
  int32_t k = static_cast<int32_t>(lhs_dims[1]);
  int32_t g_local = static_cast<int32_t>(rhs_dims[0]);
  int32_t n = static_cast<int32_t>(rhs_dims[2]);

  if (cudaMemsetAsync(out->typed_data(), 0, static_cast<size_t>(m) * n * sizeof(DtypeOutput), stream) != cudaSuccess) {
    return ffi::Error::Internal("Failed to zero output.");
  }

  return ExecuteGroupedGemm<Gemm, GemmDir::Fwd>(
      stream, scratch,
      reinterpret_cast<const DtypeA*>(lhs.typed_data()),
      reinterpret_cast<const DtypeB*>(rhs.typed_data()),
      reinterpret_cast<DtypeOutput*>(out->typed_data()),
      group_offsets_cumsum.typed_data(), group_offset.typed_data(), g_local, k, n);
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
  if (lhs_dims[0] != grad_dims[0]) {
    return ffi::Error::InvalidArgument("lhs/grad M dimension mismatch.");
  }
  if (d_rhs_dims[1] != lhs_dims[1] || d_rhs_dims[2] != grad_dims[1]) {
    return ffi::Error::InvalidArgument("d_rhs shape must be [G, K, N].");
  }

  int32_t k = static_cast<int32_t>(lhs_dims[1]);
  int32_t n = static_cast<int32_t>(grad_dims[1]);
  int32_t g_local = static_cast<int32_t>(d_rhs_dims[0]);

  if (cudaMemsetAsync(d_rhs->typed_data(), 0, static_cast<size_t>(g_local) * k * n * sizeof(DtypeOutput), stream) != cudaSuccess) {
    return ffi::Error::Internal("Failed to zero d_rhs output.");
  }

  return ExecuteGroupedGemm<Gemm_Bwd, GemmDir::Bwd>(
      stream, scratch,
      reinterpret_cast<const DtypeA*>(lhs.typed_data()),
      reinterpret_cast<const DtypeB*>(grad.typed_data()),
      reinterpret_cast<DtypeOutput*>(d_rhs->typed_data()),
      group_offsets_cumsum.typed_data(), group_offset.typed_data(), g_local, k, n);
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
