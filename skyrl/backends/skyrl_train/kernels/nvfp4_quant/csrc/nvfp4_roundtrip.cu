// Fused NVFP4 quantize -> dequantize (and pack) kernels for fake-NVFP4 QAT, sm_100.
//
// Every block of 16 is quantized with FlashInfer's own arithmetic (cvt_warp_fp16_to_fp4 in
// flashinfer/data/csrc/nv_internal/tensorrt_llm/kernels/quantization_utils.cuh, fast-math scale path,
// 16 elements per thread as built for sm_100 with CUDA >= 12.9), so the codes, scales and the 4/6 choice
// are bit-identical to FlashInfer's:
//
//   sf6 = g * (amax16 * rcp.approx(6)),  sf4 = sf6 * 1.5           (fp32, RN)
//   s   = e4m3(sf) (RN, satfinite),      os = rcp.approx(f32(s) * rcp.approx(g))
//   q   = e2m1(x * os) (RN, satfinite)
//   4/6 error (MSE, fast path): sum_i (f32(f16(q_i) * f16(s)) - x_i * g)^2, fp32 RN, elements 0..15 in order;
//   pick 4 iff err4 < err6.
//
// and the round trip is FlashInfer's decode, bf16((q * s) * alpha), q * s exact (computed as an fp16 product,
// exact: <= 6 significant bits, range [2^-10, 2688]).
//
// Instruction-level choices (B200):
//   - one thread per 16-element block: one 256-bit load and one 256-bit store per block, no shuffles;
//   - block amax on the raw bf16 bits: and 0x7fff7fff + vmax.u16x2 (integer order == |value| order);
//   - cvt.rn.satfinite.e2m1x2.f32 / cvt.rn.f16x2.e2m1x2 / cvt.rn.satfinite.e4m3x2.f32 / cvt.rn.f16x2.e4m3x2
//     hardware conversions (one instruction per element pair; no table or magic-number rounding);
//   - packed fp32 SIMD (mul/add.rn.f32x2, sm_100): x*os, x*g, the output scale, and the two 4/6 error
//     accumulators packed as one (err4, err6) pair, each still summed in element order (bit-exact with
//     FlashInfer's scalar sums);
//   - the 4/6 difference q*s - x*g as one mixed-precision fma.rn.f32.f16 per element and candidate (the f16
//     product is exact, so a single rounding equals FlashInfer's fsub of the decoded candidate), then only the
//     chosen candidate is decoded for the output;
//   - both candidate scales rounded to E4M3 and decoded back with one cvt each (e4m3x2).

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 1000

namespace {

// Inline PTX only where no intrinsic or plain C++ compiles to the same SASS:
//   - the 256-bit load / store: no 32-byte __ldg/__ldcs-style intrinsic carries .nc + L1::no_allocate, and a plain
//     C++ 32-byte store reschedules the per-token kernels into up to 8 more registers;
//   - rcp.approx.ftz.f32: __fdividef(1, x) is div.approx (MUFU.RCP + an extra FADD), __frcp_rn is IEEE;
//   - the non-ftz mul.rn.f32x2: in this -ftz=true build __fmul2_rn is the .ftz form;
//   - the E2M1 cvts: cuda_fp4.h's intrinsics pass each pair through a zero-extended 16-bit register, so packing /
//     unpacking 4 pairs costs extra masks and shifts (no byte-select / merge forms);
//   - fma.rn.f32.f16 (FHFMA): no intrinsic; f16->f32 converts + fmaf stay three instructions.

struct alignas(32) V8 {
  uint32_t d[8];
};

__device__ __forceinline__ V8 ld_nc_v8(const void* p) {
  V8 v;
  asm volatile("ld.global.nc.L1::no_allocate.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(v.d[0]), "=r"(v.d[1]), "=r"(v.d[2]), "=r"(v.d[3]), "=r"(v.d[4]), "=r"(v.d[5]),
                 "=r"(v.d[6]), "=r"(v.d[7])
               : "l"(p));
  return v;
}

__device__ __forceinline__ void st_v8(void* p, const V8& v) {
  asm volatile("st.global.v8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" ::"l"(p), "r"(v.d[0]), "r"(v.d[1]),
               "r"(v.d[2]), "r"(v.d[3]), "r"(v.d[4]), "r"(v.d[5]), "r"(v.d[6]), "r"(v.d[7])
               : "memory");
}

__device__ __forceinline__ float rcp_approx(float a) {
  float b;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(a));
  return b;
}

// Quantize-side packed fp32 ops flush subnormals, as FlashInfer's kernels do (built with -use_fast_math; with
// -ftz=true, __fmul2_rn / __fadd2_rn are mul/add.rn.ftz.f32x2); the output scale multiply does not (FlashInfer's
// bytes are decoded by a non-ftz kernel).
__device__ __forceinline__ float2 mul2_noftz(float2 a, float2 b) {
  float2 d;
  asm("mul.rn.f32x2 %0, %1, %2;"
      : "=l"(reinterpret_cast<unsigned long long&>(d))
      : "l"(reinterpret_cast<unsigned long long&>(a)), "l"(reinterpret_cast<unsigned long long&>(b)));
  return d;
}

__device__ __forceinline__ float2 bcast(float a) { return make_float2(a, a); }

// bf16 pair (low half = element 2i) -> fp32 pair (x = element 2i).
__device__ __forceinline__ float2 bf2_to_f2(uint32_t u) {
  return make_float2(__uint_as_float(u << 16), __uint_as_float(u & 0xffff0000u));
}

__device__ __forceinline__ uint32_t h2_bits(__half2 h) { return *reinterpret_cast<uint32_t*>(&h); }
__device__ __forceinline__ __half2 bits_h2(uint32_t u) { return *reinterpret_cast<__half2*>(&u); }

// 8 fp32 (4 pairs, x = even element) -> 8 E2M1 codes, element 2i in the low nibble of byte i.
__device__ __forceinline__ uint32_t e2m1x8(const float2* p) {
  uint32_t r;
  asm("{\n"
      ".reg .b8 b0, b1, b2, b3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
      "mov.b32 %0, {b0, b1, b2, b3};\n"
      "}"
      : "=r"(r)
      : "f"(p[0].x), "f"(p[0].y), "f"(p[1].x), "f"(p[1].y), "f"(p[2].x), "f"(p[2].y), "f"(p[3].x), "f"(p[3].y));
  return r;
}

// 8 E2M1 codes -> 4 f16x2 (exact).
__device__ __forceinline__ void e2m1_to_f16x8(uint32_t codes, __half2* h) {
  uint32_t h0, h1, h2, h3;
  asm("{\n"
      ".reg .b8 b0, b1, b2, b3;\n"
      "mov.b32 {b0, b1, b2, b3}, %4;\n"
      "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
      "cvt.rn.f16x2.e2m1x2 %1, b1;\n"
      "cvt.rn.f16x2.e2m1x2 %2, b2;\n"
      "cvt.rn.f16x2.e2m1x2 %3, b3;\n"
      "}"
      : "=r"(h0), "=r"(h1), "=r"(h2), "=r"(h3)
      : "r"(codes));
  h[0] = bits_h2(h0), h[1] = bits_h2(h1), h[2] = bits_h2(h2), h[3] = bits_h2(h3);
}

// 8 E2M1 codes -> 4 f16x2 values multiplied by the f16x2 scale s2 (exact), each as an fp32 pair.
__device__ __forceinline__ void decode8(uint32_t codes, __half2 s2, float2* out) {
  __half2 h[4];
  e2m1_to_f16x8(codes, h);
#pragma unroll
  for (int i = 0; i < 4; i++) out[i] = __half22float2(__hmul2_rn(h[i], s2));
}

// (x, y) = fma(f16 halves of q2, f16 s, (c.x, c.y)) in fp32, one rounding each.
__device__ __forceinline__ float2 fhfma2(__half2 q2, __half s, float2 c) {
  float2 d;
  asm("{\n"
      ".reg .b16 ql, qh;\n"
      "mov.b32 {ql, qh}, %2;\n"
      "fma.rn.f32.f16 %0, ql, %3, %4;\n"
      "fma.rn.f32.f16 %1, qh, %3, %5;\n"
      "}"
      : "=f"(d.x), "=f"(d.y)
      : "r"(h2_bits(q2)), "h"(__half_as_ushort(s)), "f"(c.x), "f"(c.y));
  return d;
}

// (lo, hi) fp32 -> two E4M3 bytes (hi << 8 | lo), and their decoded values as an f16x2 (lo, hi).
__device__ __forceinline__ __half2 e4m3x2(float lo, float hi, uint32_t& bytes) {
  __nv_fp8x2_storage_t const e = __nv_cvt_float2_to_fp8x2(make_float2(lo, hi), __NV_SATFINITE, __NV_E4M3);
  bytes = e;
  return __nv_cvt_fp8x2_to_halfraw2(e, __NV_E4M3);
}

// max |x| of a 16-element bf16 block, as bf16 bits (NaN propagates: its bits order above inf). The expert amax
// (torch's aminmax propagates NaN).
__device__ __forceinline__ uint32_t amax_bits(const V8& x) {
  uint32_t m = x.d[0] & 0x7fff7fffu;
#pragma unroll
  for (int i = 1; i < 8; i++) m = __vmaxu2(m, x.d[i] & 0x7fff7fffu);
  return max(m & 0xffffu, m >> 16);
}

// FlashInfer's NaN handling (__hmax2 / fmaxf return the non-NaN operand), from the NaN-propagating amax `fast`:
// the block amax the quantizer uses is NaN only if all 16 values are NaN, else the max over the non-NaN ones;
// the per-token row amax ignores NaN entirely. Rare slow path, taken only when the block holds a NaN.
__device__ __forceinline__ void amax_flashinfer(const V8& x, uint32_t fast, uint32_t& row_bits, uint32_t& blk_bits) {
  if (fast <= 0x7f80u) {
    row_bits = blk_bits = fast;
    return;
  }
  uint32_t m = 0, all = 0xffffffffu;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    uint32_t const a = x.d[i] & 0x7fff7fffu;
    uint32_t const nan = __vcmpgtu2(a, 0x7f807f80u);
    m = __vmaxu2(m, a & ~nan);
    all &= nan;
  }
  row_bits = max(m & 0xffffu, m >> 16);
  blk_bits = all == 0xffffffffu ? 0x7fc0u : row_bits;
}

// The decode's E4M3 scale as an f16x2 broadcast: FlashInfer's bytes are decoded by a kernel that reads the
// NaN byte 0x7f as 480 (exponent 15, mantissa 7), so the round trip does the same (only blocks whose scale is
// NaN: an all-NaN block, or a NaN / zero global scale).
__device__ __forceinline__ __half2 decode_scale2(uint32_t sf_byte, __half2 s2) {
  return sf_byte == 0x7fu ? bits_h2(0x5f805f80u) : s2;
}

struct BlockResult {
  uint32_t c0, c1;  // codes, elements 0..7 and 8..15
  uint32_t sf;      // E4M3 byte
  float2 v[8];      // q * s as exact fp32 pairs
};

// One 16-element block on FlashInfer's grid. g: the encode (global) scale, rg = rcp.approx(g),
// rcp6 = rcp.approx(6).
template <bool FOUR_SIX>
__device__ __forceinline__ void quant_block(const V8& x, uint32_t abits, float g, float rg, float rcp6,
                                            BlockResult& r) {
  float const amax = __uint_as_float(abits << 16);
  float2 xp[8];
#pragma unroll
  for (int i = 0; i < 8; i++) xp[i] = bf2_to_f2(x.d[i]);
  __half2 s2;  // the chosen candidate's decoded scale, broadcast
  if constexpr (!FOUR_SIX) {
    float const sfh = g * (amax * rcp6);
    uint32_t bytes;
    __half2 const h2 = e4m3x2(sfh, 0.0f, bytes);
    float const os = amax != 0.0f ? rcp_approx(__low2float(h2) * rg) : 0.0f;
    float2 q[8];
#pragma unroll
    for (int i = 0; i < 8; i++) q[i] = __fmul2_rn(xp[i], bcast(os));
    r.c0 = e2m1x8(q);
    r.c1 = e2m1x8(q + 4);
    r.sf = bytes & 0xffu;
    s2 = __low2half2(h2);
  } else {
    if (amax == 0.0f) {
      r.c0 = r.c1 = 0u;
      r.sf = 0u;
#pragma unroll
      for (int i = 0; i < 8; i++) r.v[i] = make_float2(0.0f, 0.0f);
      return;
    }
    float const sf6h = g * (amax * rcp6);
    float const sf4h = sf6h * 1.5f;
    uint32_t bytes;  // hi: candidate 4, lo: candidate 6
    __half2 const h2 = e4m3x2(sf6h, sf4h, bytes);
    float const sf4 = __high2float(h2), sf6 = __low2float(h2);
    float const os4 = rcp_approx(sf4 * rg), os6 = rcp_approx(sf6 * rg);
    float2 q4[8], q6[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
      q4[i] = __fmul2_rn(xp[i], bcast(os4));
      q6[i] = __fmul2_rn(xp[i], bcast(os6));
    }
    uint32_t const c40 = e2m1x8(q4), c41 = e2m1x8(q4 + 4);
    uint32_t const c60 = e2m1x8(q6), c61 = e2m1x8(q6 + 4);
    // diff = fma(f16 code, f16 scale, -x*g) in one rounding: the product is exact, so this is FlashInfer's
    // fsub(f32(f16(q) * f16(s)), x * g) bit for bit; then only the chosen candidate is decoded for the output.
    __half2 h4[8], h6[8];
    e2m1_to_f16x8(c40, h4);
    e2m1_to_f16x8(c41, h4 + 4);
    e2m1_to_f16x8(c60, h6);
    e2m1_to_f16x8(c61, h6 + 4);
    float2 err = make_float2(0.0f, 0.0f);  // (err4, err6)
#pragma unroll
    for (int i = 0; i < 8; i++) {
      float2 const nt = __fmul2_rn(xp[i], bcast(-g));
      float2 const d4 = fhfma2(h4[i], __high2half(h2), nt), d6 = fhfma2(h6[i], __low2half(h2), nt);
      float2 d = make_float2(d4.x, d6.x);
      err = __fadd2_rn(err, __fmul2_rn(d, d));
      d = make_float2(d4.y, d6.y);
      err = __fadd2_rn(err, __fmul2_rn(d, d));
    }
    bool const pick4 = err.x < err.y;
    r.c0 = pick4 ? c40 : c60;
    r.c1 = pick4 ? c41 : c61;
    r.sf = pick4 ? (bytes >> 8) : (bytes & 0xffu);
    s2 = pick4 ? __high2half2(h2) : __low2half2(h2);
  }
  s2 = decode_scale2(r.sf, s2);
  decode8(r.c0, s2, r.v);
  decode8(r.c1, s2, r.v + 4);
}

__device__ __forceinline__ V8 dequant_out(const BlockResult& r, float alpha) {
  V8 o;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    __nv_bfloat162 const b = __float22bfloat162_rn(mul2_noftz(r.v[i], bcast(alpha)));
    o.d[i] = *reinterpret_cast<const uint32_t*>(&b);
  }
  return o;
}

__device__ __forceinline__ int64_t swz_offset(int64_t m, int c, int kTiles) {
  // FlashInfer get_sf_out_offset_128x4: [mTile, kTile, 32, 4, 4]
  return (m >> 7) * (int64_t)kTiles * 512 + (int64_t)(c >> 2) * 512 + (m & 31) * 16 + ((m >> 5) & 3) * 4 + (c & 3);
}

__device__ __forceinline__ uint32_t warp_max(uint32_t v) { return __reduce_max_sync(0xffffffffu, v); }

// *dst = max of v over the CTA (THREADS threads), written by thread 0.
template <int THREADS>
__device__ __forceinline__ void store_block_max(uint32_t v, uint32_t* dst) {
  __shared__ uint32_t sm[THREADS / 32];
  v = warp_max(v);
  if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = v;
  __syncthreads();
  if (threadIdx.x < 32) {
    v = warp_max(threadIdx.x < THREADS / 32 ? sm[threadIdx.x] : 0u);
    if (threadIdx.x == 0) *dst = v;
  }
}

// Torch's per-expert global scale arithmetic (fake_experts.global_scales; normal-range values, so -ftz is moot):
// amax.clamp_min(1e-8); gs = 2688.0 / amax == reciprocal(amax) * 2688; alpha = 1.0 / gs == reciprocal(gs).
__device__ __forceinline__ void expert_scales(uint32_t abits, float& gs, float& alpha) {
  float amax = __uint_as_float(abits << 16);
  if (!(amax != amax)) amax = fmaxf(amax, 1e-8f);
  gs = __fmul_rn(__frcp_rn(amax), 2688.0f);
  alpha = __frcp_rn(gs);
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------
// Weights: (E, N, K) bf16 contiguous, per-expert global scale from the expert's amax.

#ifndef RT_THREADS
#define RT_THREADS 256
#endif
#ifndef RT_UNROLL
#define RT_UNROLL 1
#endif
constexpr int kThreads = RT_THREADS;

// Pass 1: per (expert, chunk) max |w| as bf16 bits.
__global__ void __launch_bounds__(kThreads) expert_amax_kernel(const uint8_t* __restrict__ x, int64_t nb,
                                                               uint32_t* __restrict__ partial) {
  int const e = blockIdx.y, chunk = blockIdx.x, chunks = gridDim.x;
  int64_t const per = (nb + chunks - 1) / chunks;
  int64_t const b0 = (int64_t)chunk * per, b1 = min(nb, b0 + per);
  const uint8_t* base = x + (int64_t)e * nb * 32;
  uint32_t m = 0;
  int64_t b = b0 + threadIdx.x;
  for (; b + 3 * kThreads < b1; b += 4 * kThreads) {
    V8 v0 = ld_nc_v8(base + b * 32), v1 = ld_nc_v8(base + (b + kThreads) * 32);
    V8 v2 = ld_nc_v8(base + (b + 2 * kThreads) * 32), v3 = ld_nc_v8(base + (b + 3 * kThreads) * 32);
#pragma unroll
    for (int i = 0; i < 8; i++) {
      m = __vmaxu2(m, v0.d[i] & 0x7fff7fffu);
      m = __vmaxu2(m, v1.d[i] & 0x7fff7fffu);
      m = __vmaxu2(m, v2.d[i] & 0x7fff7fffu);
      m = __vmaxu2(m, v3.d[i] & 0x7fff7fffu);
    }
  }
  for (; b < b1; b += kThreads) {
    V8 v0 = ld_nc_v8(base + b * 32);
#pragma unroll
    for (int i = 0; i < 8; i++) m = __vmaxu2(m, v0.d[i] & 0x7fff7fffu);
  }
  store_block_max<kThreads>(max(m & 0xffffu, m >> 16), partial + (int64_t)e * chunks + chunk);
}

// Pass 2: the round trip / pack. grid (chunks, E). partial: (E, pchunks) bf16-bit maxima from pass 1, or, with
// spec != nullptr, the speculative amax (E,) bf16 bits; then this pass also records its own maxima into
// partial (E, chunks) for the check pass.
template <bool FOUR_SIX>
__global__ void __launch_bounds__(kThreads) expert_rt_kernel(
    const uint8_t* __restrict__ x, int64_t nb, int kb, const uint32_t* __restrict__ partial_in, int pchunks,
    const uint32_t* __restrict__ spec, uint32_t* __restrict__ partial_out, float* __restrict__ gs_out,
    float* __restrict__ alpha_out, uint8_t* __restrict__ out, uint8_t* __restrict__ codes,
    uint8_t* __restrict__ sf_lin, uint8_t* __restrict__ sf_swz, int n_rows, const uint32_t* __restrict__ skip) {
  int const e = blockIdx.y, chunk = blockIdx.x, chunks = gridDim.x;
  int const lane = threadIdx.x & 31;
  uint32_t abits;
  if (spec != nullptr) {
    abits = spec[e];
  } else {
    uint32_t m = 0;
    for (int j = lane; j < pchunks; j += 32) m = max(m, partial_in[(int64_t)e * pchunks + j]);
    abits = warp_max(m);
    if (skip != nullptr && abits == skip[e]) return;  // check pass: the speculation was right
  }
  float gs, alpha;
  expert_scales(abits, gs, alpha);
  if (chunk == 0 && threadIdx.x == 0) {
    if (gs_out) gs_out[e] = gs;
    if (alpha_out) alpha_out[e] = alpha;
  }
  float const rg = rcp_approx(gs), rcp6 = rcp_approx(6.0f);
  int64_t const per = (nb + chunks - 1) / chunks;
  int64_t const b0 = (int64_t)chunk * per, b1 = min(nb, b0 + per);
  int64_t const eb = (int64_t)e * nb;
  int const kTiles = (kb + 3) >> 2;
  uint32_t mx = 0;
  for (int64_t b = b0 + threadIdx.x; b < b1; b += RT_UNROLL * kThreads) {
    V8 xv[RT_UNROLL];
#pragma unroll
    for (int u = 0; u < RT_UNROLL; u++)
      if (b + u * kThreads < b1) xv[u] = ld_nc_v8(x + (eb + b + u * kThreads) * 32);
#pragma unroll
    for (int u = 0; u < RT_UNROLL; u++) {
      int64_t const bb = b + u * kThreads;
      if (bb >= b1) break;
      uint32_t const ab = amax_bits(xv[u]);
      mx = max(mx, ab);
      uint32_t ab_row, ab_blk;
      amax_flashinfer(xv[u], ab, ab_row, ab_blk);
      BlockResult r;
      quant_block<FOUR_SIX>(xv[u], ab_blk, gs, rg, rcp6, r);
      int64_t const gb = eb + bb;
      if (out) st_v8(out + gb * 32, dequant_out(r, alpha));
      if (codes) *reinterpret_cast<uint2*>(codes + gb * 8) = make_uint2(r.c0, r.c1);
      if (sf_lin) sf_lin[gb] = (uint8_t)r.sf;
      if (sf_swz) {
        // E*N*K/16 < 2^32 (checked by the caller): a 32-bit division
        uint32_t const row = (uint32_t)gb / (uint32_t)kb;
        int const c = (int)((uint32_t)gb - row * (uint32_t)kb);
        sf_swz[swz_offset(row, c, kTiles)] = (uint8_t)r.sf;
      }
    }
  }
  if (partial_out != nullptr) store_block_max<kThreads>(mx, partial_out + (int64_t)e * chunks + chunk);
}

// ---------------------------------------------------------------------------------------------------------
// Per-token activations: rows of K bf16, row amax -> t = amax * (1/2688), g = rcp.approx(t) (FlashInfer's
// nvfp4QuantAndPerTokenScaleKernel fast path). RPC rows per CTA, WPR warps per row, BPL blocks of 16 per lane,
// all held in registers between the row amax and the quantization. Outputs (any subset): the bf16 round trip
// (may alias x), and the packed operand at row dst[row] (identity without dst): codes, linear and/or 128x4
// swizzled scales, decode scale t.
// x may alias out (in place): each 32-byte block is read once, by the thread that later stores it.
template <bool FOUR_SIX, int WPR, int BPL, int RPC>
__global__ void __launch_bounds__(32 * WPR * RPC) per_token_kernel(
    const uint8_t* x, int64_t m_rows, int kb, int64_t x_stride, float inv_scale, uint8_t* out,
    int64_t out_stride, const int64_t* __restrict__ dst, uint8_t* __restrict__ codes, uint8_t* __restrict__ sf_lin,
    uint8_t* __restrict__ sf_swz, float* __restrict__ t_out) {
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int const rloc = warp / WPR, wr = warp % WPR;
  int64_t const row = (int64_t)blockIdx.x * RPC + rloc;
  bool const row_ok = row < m_rows;
  const uint8_t* xr = x + row * x_stride * 2;
  V8 v[BPL];
  uint32_t ab[BPL];
  uint32_t mx = 0;
#pragma unroll
  for (int i = 0; i < BPL; i++) {
    int const j = (wr * 32 + lane) + i * (32 * WPR);
    if (row_ok && j < kb) v[i] = ld_nc_v8(xr + (int64_t)j * 32);
  }
#pragma unroll
  for (int i = 0; i < BPL; i++) {
    int const j = (wr * 32 + lane) + i * (32 * WPR);
    if (row_ok && j < kb) {
      ab[i] = amax_bits(v[i]);
      mx = max(mx, ab[i]);
    }
  }
  if (mx > 0x7f80u) {  // a NaN in this thread's blocks: FlashInfer's NaN-skipping amaxes (rare)
    mx = 0;
#pragma unroll
    for (int i = 0; i < BPL; i++) {
      int const j = (wr * 32 + lane) + i * (32 * WPR);
      if (row_ok && j < kb) {
        uint32_t row_bits;
        amax_flashinfer(v[i], ab[i], row_bits, ab[i]);
        mx = max(mx, row_bits);
      }
    }
  }
  mx = warp_max(mx);
  if constexpr (WPR > 1) {
    __shared__ uint32_t sm[WPR * RPC];
    if (lane == 0) sm[warp] = mx;
    __syncthreads();
    mx = 0;
#pragma unroll
    for (int w = 0; w < WPR; w++) mx = max(mx, sm[rloc * WPR + w]);
  }
  if (!row_ok) return;
  float const t = __uint_as_float(mx << 16) * inv_scale;
  float const g = t != 0.0f ? rcp_approx(t) : 0.0f;
  float const rg = rcp_approx(g), rcp6 = rcp_approx(6.0f);
  int64_t const drow = dst != nullptr ? dst[row] : row;
  if (t_out != nullptr && wr == 0 && lane == 0) t_out[drow] = t;
  uint8_t* orow = out ? out + row * out_stride * 2 : nullptr;
  int const kTiles = (kb + 3) >> 2;
#pragma unroll
  for (int i = 0; i < BPL; i++) {
    int const j = (wr * 32 + lane) + i * (32 * WPR);
    if (j < kb) {
      BlockResult r;
      quant_block<FOUR_SIX>(v[i], ab[i], g, rg, rcp6, r);
      if (orow) st_v8(orow + (int64_t)j * 32, dequant_out(r, t));
      if (codes) *reinterpret_cast<uint2*>(codes + (drow * kb + j) * 8) = make_uint2(r.c0, r.c1);
      if (sf_lin) sf_lin[drow * kb + j] = (uint8_t)r.sf;
      if (sf_swz) sf_swz[swz_offset(drow, j, kTiles)] = (uint8_t)r.sf;
    }
  }
}

// ---------------------------------------------------------------------------------------------------------
// C launchers (ctypes). Pointers as integers; 0 = absent. Return 0 on success, else a CUDA error code.

extern "C" {

int nvfp4rt_expert_amax(const void* x, int64_t e, int64_t nb, int chunks, void* partial, void* stream) {
  dim3 grid(chunks, (unsigned)e);
  expert_amax_kernel<<<grid, kThreads, 0, (cudaStream_t)stream>>>((const uint8_t*)x, nb, (uint32_t*)partial);
  return (int)cudaGetLastError();
}

int nvfp4rt_expert_rt(int four_six, const void* x, int64_t e, int64_t nb, int kb, const void* partial_in,
                      int pchunks, const void* spec, void* partial_out, int chunks, void* gs, void* alpha, void* out,
                      void* codes, void* sf_lin, void* sf_swz, int n_rows, const void* skip, void* stream) {
  dim3 grid(chunks, (unsigned)e);
  auto k = four_six ? expert_rt_kernel<true> : expert_rt_kernel<false>;
  k<<<grid, kThreads, 0, (cudaStream_t)stream>>>(
      (const uint8_t*)x, nb, kb, (const uint32_t*)partial_in, pchunks, (const uint32_t*)spec, (uint32_t*)partial_out,
      (float*)gs, (float*)alpha, (uint8_t*)out, (uint8_t*)codes, (uint8_t*)sf_lin, (uint8_t*)sf_swz, n_rows,
      (const uint32_t*)skip);
  return (int)cudaGetLastError();
}

#define PT_CASE(W, B, R)                                                                                    \
  if (wpr == W && bpl == B && rpc == R) {                                                                  \
    auto k = four_six ? per_token_kernel<true, W, B, R> : per_token_kernel<false, W, B, R>;               \
    dim3 grid((unsigned)((m + R - 1) / R));                                                                \
    k<<<grid, 32 * W * R, 0, (cudaStream_t)stream>>>((const uint8_t*)x, m, kb, x_stride, inv_scale,        \
                                                     (uint8_t*)out, out_stride, (const int64_t*)dst,       \
                                                     (uint8_t*)codes, (uint8_t*)sf_lin, (uint8_t*)sf_swz,  \
                                                     (float*)t);                                           \
    return (int)cudaGetLastError();                                                                        \
  }

int nvfp4rt_per_token(int four_six, int wpr, int bpl, int rpc, const void* x, int64_t m, int kb, int64_t x_stride,
                      float inv_scale, void* out, int64_t out_stride, const void* dst, void* codes, void* sf_lin,
                      void* sf_swz, void* t, void* stream) {
  PT_CASE(1, 1, 4)
  PT_CASE(1, 2, 4)
  PT_CASE(1, 4, 4)
  PT_CASE(1, 4, 2)
  PT_CASE(2, 2, 2)
  PT_CASE(4, 1, 1)
  PT_CASE(4, 2, 1)
  PT_CASE(4, 3, 1)
  PT_CASE(4, 4, 1)
  PT_CASE(6, 2, 1)
  PT_CASE(8, 2, 1)
  PT_CASE(12, 1, 1)
  PT_CASE(16, 1, 1)
  return -1;
}

}  // extern "C"

#endif
