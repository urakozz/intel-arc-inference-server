// Probe (ruling A21): int4-g64 straight through sycl-tla's BMG mixed-input
// mainloop, bf16 A x int4 B, group 64, GPTQ v1 zero = 8.
//
// The question is whether the loader's EXISTING layout-0 weights
// (`qweight [K/8][N]` u32 + `scales f16 [K/64][N]`) can feed
// `MainloopIntelXeXMX16MixedPrecision` with no per-chunk bridge, and at what
// rate. Predictions are pre-registered in
// docs/probe-gemm-int4-mixed-2026-09-05.md and were committed before this file
// was built or run.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <sycl/sycl.hpp>

#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/util/packed_stride.hpp"
#include <cute/tensor.hpp>

// cute's printing headers map printf onto SYCL's device printf; this probe is
// host code and its output is markdown.
#undef printf

#include "common/bf16.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/quant.h"
#include "loader/safetensors.h"
#include "loader/snapshot.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/context_sycl.h"
#include "tla_pin.h"

using namespace cute;

namespace {

constexpr int kReplays = 8;
constexpr int kDroppedReplays = 3;
constexpr int kTimedEnqueues = 4;
constexpr size_t kSamples = 4096;
constexpr uint32_t kGroup = 64;

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error(what); }

const char* env_or_unset(const char* name) {
  const char* value = std::getenv(name);
  return value ? value : "(unset)";
}

uint32_t xorshift(uint32_t& value) {
  value ^= value << 13;
  value ^= value >> 17;
  value ^= value << 5;
  return value;
}

// A raw bf16 word with exponent 124: magnitude in [0.125, 0.25), so no dot
// product over K = 17408 can reach the fp32 range. Same generator as P2's.
uint16_t random_bf16(uint32_t& state) {
  return uint16_t((xorshift(state) & 0x807fu) | 0x3e00u);
}

// ---------------------------------------------------------------------------
// The instantiation under test.
//
// A is bf16 RowMajor [M][K] with XE_2D_U16x32x32_LD_V and B is 4-bit
// ColumnMajor -- i.e. [N][K] with K contiguous -- with XE_2D_U4x32x16_LD_T,
// the transposing 4-bit block load Intel's own u4 example selects
// (examples/02_bmg_gemm_mixed_dtype/02_bmg_gemm_f16_u4_f16.cpp:580-599).
// The tile, MMA atom and subgroup layout are the ones P2 measured and
// src/sycl/xe_gemm_config.h ships; they are also the bf16xs8 example's
// (02_bmg_gemm_bf16_s8_bf16.cpp:503-508). The epilogue is P2's corrected one:
// ElementC = void, so no C is read.
// ---------------------------------------------------------------------------
using ElementAccumulator = float;
using ElementComputeEpilogue = float;
using ElementInputA = cute::bfloat16_t;
using ElementOutput = float;
using ElementScale = cutlass::half_t;                      // the checkpoint's f16 scales
using StrideScale = cute::Stride<cute::_1, int64_t, int64_t>;   // == scales [K/64][N]
using ElementZero = int8_t;                                // integer, so (q - z) stays exact
using StrideZero = StrideScale;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

using GmemTiledCopyA = XE_2D_U16x32x32_LD_V;
using GmemTiledCopyB = XE_2D_U4x32x16_LD_T;

using TileShape = cute::Shape<_256, _256, _32>;
// The LEGACY 8x16x16 bf16 DPAS atom, which is what BOTH mixed-dtype examples
// use (02_bmg_gemm_bf16_s8_bf16.cpp:507 at this very tile and subgroup layout).
// It is the same instruction as P2's XE_DPAS_TT<8,float,bf16>, but a different
// CuTe fragment description, and the difference is load-bearing: the new atom
// gives a per-thread B fragment of shape (2,8) (VNNI-structured), and
// make_fragment_layout then refuses every legacy 4-bit copy atom --
// copy_traits_xe_legacy.hpp:530, "MMA atom be bigger than copy atom in one
// dimension and smaller in other dimension", with CopyVals (1,32) against
// MmaVals (8,2). The legacy atom's BLayout is Shape<_16,_16> Stride<_1,_16>,
// i.e. a FLAT 16-value per-thread fragment, so MmaVals is (1,16) and the
// constraint holds. This is a property of the two atoms alone: no tile shape
// or subgroup layout can repair the new atom's pairing with XE_2D_U4*.
using TiledMma = typename cutlass::gemm::TiledMMAHelper<
    MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>, Layout<TileShape>,
    Layout<cute::Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

constexpr int kPipelineStages = 3;
using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16MixedPrecision<kPipelineStages>;
using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeGeneric;
using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementOutput, ElementComputeEpilogue, ElementAccumulator, ElementAccumulator,
    cutlass::FloatRoundStyle::round_to_nearest>;
using FusionCallbacks = cutlass::epilogue::fusion::FusionCallbacks<
    EpilogueDispatchPolicy, EpilogueOp, TileShape, decltype(tile_shape(TiledMma()))>;
using CollectiveEpilogue = cutlass::epilogue::collective::CollectiveEpilogue<
    EpilogueDispatchPolicy, TileShape, void, void, cutlass::gemm::TagToStrideC_t<LayoutC>,
    ElementOutput, cutlass::gemm::TagToStrideC_t<LayoutD>, FusionCallbacks, void, void>;
static_assert(cute::is_void_v<typename CollectiveEpilogue::ElementC>,
              "this probe is D = A*B: the epilogue must not read C");

template <class BOptionalTuple>
struct Chain {
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      GEMMDispatchPolicy, TileShape, ElementInputA, cutlass::gemm::TagToStrideA_t<LayoutA>,
      BOptionalTuple, cutlass::gemm::TagToStrideB_t<LayoutB>, TiledMma,
      GmemTiledCopyA, void, void, cute::identity,
      GmemTiledCopyB, void, void, cute::identity>;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using ElementB = typename CollectiveMainloop::ElementB;
  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
  using StrideC = typename GemmKernel::StrideC;
  using StrideD = typename GemmKernel::StrideD;
  using Arguments = typename GemmKernel::Arguments;
};

// ConvertAndScale over SIGNED int4: the repack stores q^8, and
// sign_extend4(q^8) == q-8 for every q in [0,16), so GPTQ v1's zero point of 8
// is carried with no zero tensor at all (gemv.cl:53 relies on the same
// identity and gemv_test proves it bit-identical).
using ScaleChain = Chain<cute::tuple<cutlass::int4b_t, ElementScale, StrideScale>>;
// The stock-semantics control: unsigned nibbles verbatim and an explicit
// int8 zero tensor of the constant 8.
using ZeroChain =
    Chain<cute::tuple<cutlass::uint4b_t, ElementScale, StrideScale, ElementZero, StrideZero>>;

static_assert(ScaleChain::CollectiveMainloop::IsATransformed == false, "B is the quantised operand");
static_assert(std::is_same_v<typename ScaleChain::CollectiveMainloop::ElementMMA, cute::bfloat16_t>,
              "the MMA operand type must be bf16");

// ---------------------------------------------------------------------------
// Layout arithmetic, host side.
// ---------------------------------------------------------------------------

// Layout 0's 4-bit address: nibble i of word (r, n) is k = 8r + i, LSB first.
inline uint32_t layout0_nibble(const uint32_t* qw, uint32_t N, uint32_t k, uint32_t n) {
  return (qw[size_t(k / 8) * N + n] >> (4 * (k % 8))) & 0xFu;
}

// The repack the mainloop's ColumnMajor B needs: element (n, k) at 4-bit index
// n*K + k. Because BOTH layouts pack eight consecutive k LSB-first into 32
// bits, the repack is exactly a u32 TRANSPOSE of qweight -- no nibble is ever
// moved within a word. `xor8` additionally applies the signed-int4 shift.
void repack_to_nk(const uint32_t* qw, uint32_t K, uint32_t N, bool xor8, uint32_t* out) {
  const uint32_t words_per_row = K / 8;
  const uint32_t mask = xor8 ? 0x88888888u : 0u;
  for (uint32_t n = 0; n < N; ++n)
    for (uint32_t r = 0; r < words_per_row; ++r)
      out[size_t(n) * words_per_row + r] = qw[size_t(r) * N + n] ^ mask;
}

// tools/oracle/dequant.py:44-45, verbatim: the product in fp32, ONE cast to
// bf16. `unsigned_q` is the checkpoint nibble, before any repack.
inline uint16_t oracle_bf16(uint32_t unsigned_q, uint16_t f16_scale) {
  return common::f32_to_bf16((float(unsigned_q) - 8.0f) * common::f16_to_f32(f16_scale));
}

// ---------------------------------------------------------------------------
// One int4 matrix, host side, in every form the probe needs.
// ---------------------------------------------------------------------------
struct Int4Matrix {
  uint32_t K = 0, N = 0;
  std::vector<uint32_t> qweight;   // layout 0: [K/8][N]
  std::vector<uint16_t> scales;    // f16 [K/64][N]
  const char* origin = "";
};

Int4Matrix synthetic_matrix(uint32_t K, uint32_t N, uint32_t seed) {
  Int4Matrix m;
  m.K = K;
  m.N = N;
  m.origin = "synthetic";
  m.qweight.resize(size_t(K / 8) * N);
  for (uint32_t& w : m.qweight) w = xorshift(seed);
  m.scales.resize(size_t(K / kGroup) * N);
  // 0.02 .. 0.08, the oracle fixture's range: no group can overflow bf16.
  for (uint16_t& s : m.scales)
    s = common::f32_to_f16(0.02f + 0.06f * float(xorshift(seed) & 0xFFFFu) / 65535.0f);
  return m;
}

// The one real matrix: the fused GDN in_proj_qkv || in_proj_z of layer 0,
// K = 5120, N = 16384 -- exactly the brief's correctness shape, assembled the
// way model::Qwen35's table says the loader fuses it (Concat along N).
Int4Matrix real_qkvz(const std::string& snapshot) {
  loader::SafetensorsSet set(snapshot);
  auto find_prefix = [&](const char* suffix) -> std::string {
    const std::string want = std::string(suffix) + ".qweight";
    for (const auto& [name, unused] : set.tensors()) {
      (void)unused;
      if (name.size() >= want.size() && name.compare(name.size() - want.size(), want.size(), want) == 0)
        return name.substr(0, name.size() - 8);  // strip ".qweight"
    }
    fail(std::string("no tensor named *") + want + " in " + snapshot);
  };
  const loader::LinearSrc qkv = loader::LinearSrc::classify(set, find_prefix("layers.0.linear_attn.in_proj_qkv"));
  const loader::LinearSrc z = loader::LinearSrc::classify(set, find_prefix("layers.0.linear_attn.in_proj_z"));
  if (qkv.kind != loader::WKind::Int4 || z.kind != loader::WKind::Int4) fail("qkv/z are not int4");
  if (qkv.K != z.K) fail("qkv and z disagree on K");
  Int4Matrix m;
  m.K = qkv.K;
  m.N = qkv.N + z.N;
  m.origin = "checkpoint layers.0.linear_attn.in_proj_qkv || in_proj_z";
  m.qweight.resize(size_t(m.K / 8) * m.N);
  m.scales.resize(size_t(m.K / kGroup) * m.N);
  for (uint32_t r = 0; r < m.K / 8; ++r) {
    std::memcpy(&m.qweight[size_t(r) * m.N], &qkv.qweight[size_t(r) * qkv.N], size_t(qkv.N) * 4);
    std::memcpy(&m.qweight[size_t(r) * m.N + qkv.N], &z.qweight[size_t(r) * z.N], size_t(z.N) * 4);
  }
  for (uint32_t g = 0; g < m.K / kGroup; ++g) {
    std::memcpy(&m.scales[size_t(g) * m.N], &qkv.scales[size_t(g) * qkv.N], size_t(qkv.N) * 2);
    std::memcpy(&m.scales[size_t(g) * m.N + qkv.N], &z.scales[size_t(g) * z.N], size_t(z.N) * 2);
  }
  return m;
}

// ---------------------------------------------------------------------------
// Device-side plumbing.
// ---------------------------------------------------------------------------
struct DeviceMatrix {
  std::unique_ptr<l0::Mem> b4;      // the repacked [N][K] 4-bit weights
  std::unique_ptr<l0::Mem> scales;  // f16 [K/64][N], the checkpoint's own bytes
  std::unique_ptr<l0::Mem> zeros;   // int8 [K/64][N] of 8 (ZeroChain only)
};

std::unique_ptr<l0::Mem> upload(l0::Context& ctx, const void* src, size_t bytes) {
  auto mem = std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, bytes);
  l0::CmdList list = l0::CmdList::immediate(ctx);
  list.copy(mem->ptr(), src, bytes);
  return mem;
}

template <class C>
typename C::Arguments make_arguments(uint32_t m, uint32_t k, uint32_t n, const uint16_t* a,
                                     const void* b4, const uint16_t* scales, const int8_t* zeros,
                                     float* out, const cutlass::KernelHardwareInfo& hw) {
  using Element = typename C::ElementB;
  const typename C::GemmKernel::ProblemShape problem{int(m), int(n), int(k), 1};
  const auto dA = cutlass::make_cute_packed_stride(typename C::StrideA{}, make_shape(int(m), int(k), 1));
  // ColumnMajor B: Stride<int64_t,_1,int64_t> over (N,K,L), so get<0> = K.
  const auto dB = cutlass::make_cute_packed_stride(typename C::StrideB{}, make_shape(int(n), int(k), 1));
  const auto dC = cutlass::make_cute_packed_stride(typename C::StrideC{}, make_shape(int(m), int(n), 1));
  const auto dD = cutlass::make_cute_packed_stride(typename C::StrideD{}, make_shape(int(m), int(n), 1));
  const auto dS = cutlass::make_cute_packed_stride(StrideScale{},
                                                   make_shape(int(n), int(k / kGroup), 1));
  using MainloopArgs = typename C::CollectiveMainloop::Arguments;
  MainloopArgs mainloop{};
  mainloop.ptr_A = reinterpret_cast<const ElementInputA*>(a);
  mainloop.dA = dA;
  mainloop.ptr_B = reinterpret_cast<const Element*>(b4);
  mainloop.dB = dB;
  mainloop.ptr_S = reinterpret_cast<const ElementScale*>(scales);
  mainloop.dS = dS;
  if constexpr (!cute::is_void_v<typename C::CollectiveMainloop::ElementZero>) {
    mainloop.ptr_Z = zeros;
    mainloop.dZ = cutlass::make_cute_packed_stride(StrideZero{},
                                                   make_shape(int(n), int(k / kGroup), 1));
  } else {
    (void)zeros;
  }
  mainloop.group_size = int(kGroup);
  return {cutlass::gemm::GemmUniversalMode::kGemm, problem, mainloop,
          {{1.f, 0.f}, out, dC, out, dD}, hw};
}

template <class C>
struct Prepared {
  typename C::Gemm op;
  size_t workspace = 0;
  cutlass::Status can_implement = cutlass::Status::kErrorInternal;
};

template <class C>
Prepared<C> prepare(uint32_t m, uint32_t k, uint32_t n, const uint16_t* a, const void* b4,
                    const uint16_t* scales, const int8_t* zeros, float* out,
                    const cutlass::KernelHardwareInfo& hw, sycl::queue& queue) {
  Prepared<C> prepared;
  const auto args = make_arguments<C>(m, k, n, a, b4, scales, zeros, out, hw);
  prepared.workspace = C::Gemm::get_workspace_size(args);
  prepared.can_implement = prepared.op.can_implement(args);
  if (prepared.workspace != 0)
    fail("get_workspace_size returned " + std::to_string(prepared.workspace) + " bytes");
  if (prepared.can_implement != cutlass::Status::kSuccess) return prepared;
  const cutlass::Status ok = prepared.op.initialize(args, nullptr, &queue);
  if (ok != cutlass::Status::kSuccess)
    fail("Gemm::initialize failed with status " + std::to_string(int(ok)));
  return prepared;
}

template <class C>
void run_once(Prepared<C>& prepared, sycl::queue& queue) {
  const cutlass::Status status = prepared.op.run(&queue);
  if (status != cutlass::Status::kSuccess)
    fail("Gemm::run failed with status " + std::to_string(int(status)));
  queue.wait_and_throw();
}

template <class C>
double time_replays(Prepared<C>& prepared, sycl::queue& queue) {
  std::array<double, kReplays - kDroppedReplays> samples{};
  for (int replay = 0; replay < kReplays; ++replay) {
    const auto begin = std::chrono::steady_clock::now();
    for (int enqueue = 0; enqueue < kTimedEnqueues; ++enqueue) {
      const cutlass::Status status = prepared.op.run(&queue);
      if (status != cutlass::Status::kSuccess)
        fail("timed run failed with status " + std::to_string(int(status)));
    }
    queue.wait_and_throw();
    if (replay >= kDroppedReplays)
      samples[size_t(replay - kDroppedReplays)] =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count() /
          kTimedEnqueues;
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

// ---------------------------------------------------------------------------
// Stage 1 -- the layout verdict, and the bit-exact dequant check.
//
// A = I (M = K), so C[k][n] is the DPAS sum of K terms of which K-1 are 0*w
// and one is 1*w: it equals float(bf16 W[k][n]) EXACTLY. Every returned word
// is therefore compared bitwise against tools/oracle/dequant.py's value.
// ---------------------------------------------------------------------------
struct ExactResult {
  size_t compared = 0;
  size_t mismatched = 0;   // differs bitwise AND is not a signed-zero pair
  size_t signed_zero = 0;  // both values are zero, the sign bit differs
  std::string first;
};

template <class C>
ExactResult identity_extract(l0::Context& l0ctx, runtime::prefill::Context& cx,
                             const Int4Matrix& m, const void* b4_host, size_t b4_bytes,
                             const cutlass::KernelHardwareInfo& hw) {
  const uint32_t K = m.K, N = m.N;
  std::vector<uint16_t> identity(size_t(K) * K, 0);
  for (uint32_t i = 0; i < K; ++i) identity[size_t(i) * K + i] = common::f32_to_bf16(1.0f);
  std::vector<int8_t> zeros(size_t(K / kGroup) * N, int8_t(8));

  auto d_a = upload(l0ctx, identity.data(), identity.size() * 2);
  auto d_b = upload(l0ctx, b4_host, b4_bytes);
  auto d_s = upload(l0ctx, m.scales.data(), m.scales.size() * 2);
  auto d_z = upload(l0ctx, zeros.data(), zeros.size());
  l0::Mem d_c(l0ctx, l0::MemKind::Device, size_t(K) * N * sizeof(float));

  sycl::queue& queue = runtime::prefill::sycl_queue(cx);
  auto gemm = prepare<C>(K, K, N, d_a->as<uint16_t>(), d_b->ptr(), d_s->as<uint16_t>(),
                         d_z->as<int8_t>(), d_c.as<float>(), hw, queue);
  if (gemm.can_implement != cutlass::Status::kSuccess)
    fail("identity extraction cannot implement: status " + std::to_string(int(gemm.can_implement)));
  run_once<C>(gemm, queue);

  ExactResult result;
  const size_t rows_per_chunk = std::max<size_t>(1, (64ul << 20) / (size_t(N) * sizeof(float)));
  l0::Mem host_c(l0ctx, l0::MemKind::Host, rows_per_chunk * N * sizeof(float));
  for (uint32_t k0 = 0; k0 < K; k0 += uint32_t(rows_per_chunk)) {
    const uint32_t rows = std::min<uint32_t>(uint32_t(rows_per_chunk), K - k0);
    l0::CmdList copy = l0::CmdList::immediate(l0ctx);
    copy.copy(host_c.ptr(), static_cast<const char*>(d_c.ptr()) + size_t(k0) * N * sizeof(float),
              size_t(rows) * N * sizeof(float));
    const float* got = host_c.as<float>();
    for (uint32_t r = 0; r < rows; ++r) {
      const uint32_t k = k0 + r;
      for (uint32_t n = 0; n < N; ++n) {
        const uint16_t want = oracle_bf16(layout0_nibble(m.qweight.data(), N, k, n),
                                          m.scales[size_t(k / kGroup) * N + n]);
        const float want_f = common::bf16_to_f32(want);
        const float got_f = got[size_t(r) * N + n];
        ++result.compared;
        if (std::memcmp(&want_f, &got_f, sizeof(float)) != 0) {
          // A = I makes C[k][n] the fp32 sum of K-1 terms of 0*w and one 1*w.
          // That is exact for every w EXCEPT w = -0: IEEE (+0) + (-0) = +0, so
          // the accumulator cannot carry a negative zero. It is a limit of this
          // extraction, not of the mainloop, so it is counted separately rather
          // than silently tolerated.
          if (want_f == 0.0f && got_f == 0.0f) {
            ++result.signed_zero;
            continue;
          }
          if (result.mismatched == 0) {
            char buf[192];
            std::snprintf(buf, sizeof(buf), "k=%u n=%u got=%.9g want=%.9g (q=%u scale=0x%04X)", k, n,
                          double(got_f), double(want_f),
                          layout0_nibble(m.qweight.data(), N, k, n),
                          m.scales[size_t(k / kGroup) * N + n]);
            result.first = buf;
          }
          ++result.mismatched;
        }
      }
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Stage 2 -- the production-shape GEMM against a CPU reference built from the
// oracle's bf16 weights, accumulated in fp64.
// ---------------------------------------------------------------------------
struct Verification {
  double max_abs_err = 0.0;
  double tolerance = 0.0;
  bool bitwise = false;
};

Verification verify_sampled(l0::Context& l0ctx, const l0::Mem& first, const l0::Mem& second,
                            const std::vector<uint16_t>& a, const Int4Matrix& m, uint32_t M) {
  const uint32_t K = m.K, N = m.N;
  const size_t elements = size_t(M) * N;
  const size_t bytes = elements * sizeof(float);
  const size_t chunk = std::min<size_t>(64ul << 20, bytes);
  l0::Mem first_host(l0ctx, l0::MemKind::Host, chunk);
  l0::Mem second_host(l0ctx, l0::MemKind::Host, chunk);
  std::vector<size_t> samples;
  samples.reserve(kSamples);
  uint32_t seed = 0x7f4a7c15u;
  for (size_t i = 0; i < kSamples; ++i) samples.push_back(size_t(xorshift(seed)) % elements);
  std::sort(samples.begin(), samples.end());

  bool bitwise = true;
  double max_abs_err = 0.0, max_abs_ref = 0.0;
  size_t sample = 0;
  for (size_t offset = 0; offset < bytes; offset += chunk) {
    const size_t now = std::min(chunk, bytes - offset);
    l0::CmdList copy = l0::CmdList::immediate(l0ctx);
    copy.copy(first_host.ptr(), static_cast<const char*>(first.ptr()) + offset, now);
    copy.copy(second_host.ptr(), static_cast<const char*>(second.ptr()) + offset, now);
    bitwise = bitwise && std::memcmp(first_host.ptr(), second_host.ptr(), now) == 0;
    const size_t first_element = offset / sizeof(float);
    const size_t end_element = first_element + now / sizeof(float);
    const float* got = first_host.as<float>();
    while (sample < samples.size() && samples[sample] < end_element) {
      const size_t index = samples[sample++];
      const uint32_t mi = uint32_t(index / N), ni = uint32_t(index % N);
      double reference = 0.0;
      for (uint32_t k = 0; k < K; ++k) {
        const uint16_t w = oracle_bf16(layout0_nibble(m.qweight.data(), N, k, ni),
                                       m.scales[size_t(k / kGroup) * N + ni]);
        reference += double(common::bf16_to_f32(a[size_t(mi) * K + k])) *
                     double(common::bf16_to_f32(w));
      }
      max_abs_ref = std::max(max_abs_ref, std::fabs(reference));
      max_abs_err = std::max(max_abs_err, std::fabs(double(got[index - first_element]) - reference));
    }
  }
  return {max_abs_err, 5e-3 * max_abs_ref + 1e-4, bitwise};
}

// ---------------------------------------------------------------------------
// Stage 3 -- the rate matrix.
// ---------------------------------------------------------------------------
struct ProbeShape {
  uint32_t K, N;
  const char* name;
};
constexpr ProbeShape kShapes[] = {
    {5120, 16384, "qkv‖z"},   {6144, 5120, "out/o_proj"}, {5120, 34816, "gate‖up"},
    {17408, 5120, "down"},    {5120, 14336, "q‖k‖v"},     {5120, 248320, "lm_head"},
};
constexpr uint32_t kMs[] = {512, 1024, 2048, 4096};

struct RateResult {
  double ms = 0.0, tflops = 0.0;
  size_t workspace = 0;
  int can_implement = 0;
  bool bitwise = false;
};

template <class C>
RateResult rate_cell(l0::Context& l0ctx, runtime::prefill::Context& cx, const void* b4,
                     const uint16_t* scales, const int8_t* zeros, uint32_t M, uint32_t K,
                     uint32_t N, const cutlass::KernelHardwareInfo& hw, bool timed) {
  std::vector<uint16_t> a(size_t(M) * K);
  uint32_t seed = 0x9e3779b9u ^ M ^ K ^ N;
  for (uint16_t& v : a) v = random_bf16(seed);
  auto d_a = upload(l0ctx, a.data(), a.size() * 2);
  l0::Mem first(l0ctx, l0::MemKind::Device, size_t(M) * N * sizeof(float));
  l0::Mem second(l0ctx, l0::MemKind::Device, size_t(M) * N * sizeof(float));

  sycl::queue& queue = runtime::prefill::sycl_queue(cx);
  auto one = prepare<C>(M, K, N, d_a->as<uint16_t>(), b4, scales, zeros, first.as<float>(), hw, queue);
  if (one.can_implement != cutlass::Status::kSuccess)
    return {0.0, 0.0, one.workspace, int(one.can_implement), false};
  run_once<C>(one, queue);
  auto two = prepare<C>(M, K, N, d_a->as<uint16_t>(), b4, scales, zeros, second.as<float>(), hw, queue);
  run_once<C>(two, queue);

  bool bitwise = true;
  const size_t bytes = size_t(M) * N * sizeof(float);
  const size_t chunk = std::min<size_t>(64ul << 20, bytes);
  l0::Mem ha(l0ctx, l0::MemKind::Host, chunk), hb(l0ctx, l0::MemKind::Host, chunk);
  for (size_t offset = 0; offset < bytes; offset += chunk) {
    const size_t now = std::min(chunk, bytes - offset);
    l0::CmdList copy = l0::CmdList::immediate(l0ctx);
    copy.copy(ha.ptr(), static_cast<const char*>(first.ptr()) + offset, now);
    copy.copy(hb.ptr(), static_cast<const char*>(second.ptr()) + offset, now);
    bitwise = bitwise && std::memcmp(ha.ptr(), hb.ptr(), now) == 0;
  }

  const double ms = timed ? time_replays<C>(one, queue) : 0.0;
  const double flops = 2.0 * double(M) * K * N;
  return {ms, timed ? flops / (ms * 1e9) : 0.0, one.workspace, int(one.can_implement), bitwise};
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string stage = "all";
    const char* home = std::getenv("HOME");
    std::string snapshot =
        std::string(home ? home : ".") + "/models/qwen38-27b-w4g64-rtn/Qwen3.8-27B-w4g64";
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg.rfind("--stage=", 0) == 0) stage = arg.substr(8);
      else if (arg.rfind("--ckpt=", 0) == 0) snapshot = arg.substr(7);
      else fail("usage: probe_gemm_int4_mixed [--stage=layout|gemm|rate|all] [--ckpt=DIR]");
    }
    const bool do_layout = stage == "all" || stage == "layout";
    const bool do_gemm = stage == "all" || stage == "gemm";
    const bool do_rate = stage == "all" || stage == "rate";

    std::printf("sycl-tla sha %s (pin %s)\n", B70_SYCL_TLA_SHA, B70_SYCL_TLA_PIN);
    std::printf("env: ZE_AFFINITY_MASK=%s | IGC_allowDecompose2DBlockFuncs=%s\n",
                env_or_unset("ZE_AFFINITY_MASK"), env_or_unset("IGC_allowDecompose2DBlockFuncs"));
    std::printf("# Configuration: MainloopIntelXeXMX16MixedPrecision<%d>; TileShape 256x256x32; "
                "TiledMma XE_8x16x16_F32BF16BF16F32_TT (legacy atom), subgroup layout 8x4x1 "
                "(32 subgroups); "
                "A bf16 RowMajor / XE_2D_U16x32x32_LD_V; B 4-bit ColumnMajor ([N][K], K "
                "contiguous) / XE_2D_U4x32x16_LD_T; ElementScale = half_t, StrideScale = "
                "(_1,int64,int64); group 64; epilogue IntelXeGeneric with ElementC = void; "
                "TileScheduler = PersistentScheduler (data-parallel, no split-K atomics).\n",
                kPipelineStages);

    l0::Context l0ctx(0);
    std::printf("L0 device: %s\n", l0ctx.name().c_str());
    runtime::prefill::Context cx(l0ctx);
    cutlass::KernelHardwareInfo hw{};
    hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    std::printf("# scheduler sm_count = %d\n\n", hw.sm_count);
    {
      const auto dB = cutlass::make_cute_packed_stride(typename ScaleChain::StrideB{},
                                                       make_shape(16, 64, 1));
      std::printf("# StrideB over (N,K,L) at (N=16,K=64): ");
      cute::print(dB);
      std::printf("  -- get<0> is the ldb, so B is [N][K] with K contiguous\n");
    }

    Int4Matrix real;
    if (do_layout || do_gemm) {
      const std::string dir = loader::resolve_snapshot(snapshot);
      const auto t0 = std::chrono::steady_clock::now();
      real = real_qkvz(dir);
      const double load_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      std::printf("# real weights: %s, K=%u N=%u (%zu qweight words, %zu scales) read in %.1f ms\n",
                  real.origin, real.K, real.N, real.qweight.size(), real.scales.size(), load_ms);
    }

    if (do_layout) {
      std::printf("\n## Layout verdict (measured)\n\n");
      const size_t b4_bytes = size_t(real.K) * real.N / 2;
      std::vector<uint32_t> repacked(b4_bytes / 4);
      const auto t0 = std::chrono::steady_clock::now();
      repack_to_nk(real.qweight.data(), real.K, real.N, /*xor8=*/true, repacked.data());
      const double repack_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      std::printf("# host repack of K=%u N=%u (%zu MB): %.1f ms (%.2f GB/s), one-time\n", real.K,
                  real.N, b4_bytes >> 20, repack_ms, double(b4_bytes) * 2.0 / (repack_ms * 1e6));

      // Control: the SAME element values (q^8), left in the checkpoint's own
      // [K/8][N] arrangement. Any mismatch here is purely a layout mismatch.
      std::vector<uint32_t> untransposed(real.qweight.size());
      for (size_t i = 0; i < untransposed.size(); ++i) untransposed[i] = real.qweight[i] ^ 0x88888888u;

      const ExactResult direct =
          identity_extract<ScaleChain>(l0ctx, cx, real, untransposed.data(), b4_bytes, hw);
      std::printf("| B bytes fed | compared | mismatching | +0/-0 only | first real mismatch |\n");
      std::printf("|---|---:|---:|---:|---|\n");
      std::printf("| layout 0 verbatim (`qweight [K/8][N]`, q^8) | %zu | %zu | %zu | %s |\n",
                  direct.compared, direct.mismatched, direct.signed_zero,
                  direct.mismatched ? direct.first.c_str() : "(none)");
      const ExactResult repack =
          identity_extract<ScaleChain>(l0ctx, cx, real, repacked.data(), b4_bytes, hw);
      std::printf("| repacked `[N][K]` 4-bit, K contiguous, q^8 | %zu | %zu | %zu | %s |\n",
                  repack.compared, repack.mismatched, repack.signed_zero,
                  repack.mismatched ? repack.first.c_str() : "(none)");

      // The stock-semantics control: unsigned nibbles + an int8 zero tensor.
      std::vector<uint32_t> repacked_unsigned(b4_bytes / 4);
      repack_to_nk(real.qweight.data(), real.K, real.N, /*xor8=*/false, repacked_unsigned.data());
      const ExactResult zero_mode =
          identity_extract<ZeroChain>(l0ctx, cx, real, repacked_unsigned.data(), b4_bytes, hw);
      std::printf("| same repack, unsigned nibbles + int8 zero=8 tensor | %zu | %zu | %zu | %s |\n",
                  zero_mode.compared, zero_mode.mismatched, zero_mode.signed_zero,
                  zero_mode.mismatched ? zero_mode.first.c_str() : "(none)");
      std::printf("\n# The `+0/-0 only` column is this extraction's own limit, not the "
                  "mainloop's: with A = I the fp32 accumulator sums K-1 terms of 0*w, and IEEE "
                  "(+0)+(-0) = +0, so a weight whose oracle value is -0 (q = 8 with a negative "
                  "f16 scale) can only come back as +0.\n");
      std::fflush(stdout);
    }

    if (do_gemm) {
      std::printf("\n## Correctness at the brief's shape (measured)\n\n");
      const size_t b4_bytes = size_t(real.K) * real.N / 2;
      std::vector<uint32_t> repacked(b4_bytes / 4);
      repack_to_nk(real.qweight.data(), real.K, real.N, true, repacked.data());
      auto d_b = upload(l0ctx, repacked.data(), b4_bytes);
      auto d_s = upload(l0ctx, real.scales.data(), real.scales.size() * 2);
      std::vector<int8_t> zeros(size_t(real.K / kGroup) * real.N, int8_t(8));
      auto d_z = upload(l0ctx, zeros.data(), zeros.size());

      const uint32_t M = 64;
      std::vector<uint16_t> a(size_t(M) * real.K);
      uint32_t seed = 0x85ebca6bu;
      for (uint16_t& v : a) v = random_bf16(seed);
      auto d_a = upload(l0ctx, a.data(), a.size() * 2);
      l0::Mem first(l0ctx, l0::MemKind::Device, size_t(M) * real.N * sizeof(float));
      l0::Mem second(l0ctx, l0::MemKind::Device, size_t(M) * real.N * sizeof(float));
      sycl::queue& queue = runtime::prefill::sycl_queue(cx);
      auto one = prepare<ScaleChain>(M, real.K, real.N, d_a->as<uint16_t>(), d_b->ptr(),
                                     d_s->as<uint16_t>(), d_z->as<int8_t>(), first.as<float>(), hw,
                                     queue);
      if (one.can_implement != cutlass::Status::kSuccess)
        fail("M=64 real-weight GEMM cannot implement: " + std::to_string(int(one.can_implement)));
      run_once<ScaleChain>(one, queue);
      auto two = prepare<ScaleChain>(M, real.K, real.N, d_a->as<uint16_t>(), d_b->ptr(),
                                     d_s->as<uint16_t>(), d_z->as<int8_t>(), second.as<float>(), hw,
                                     queue);
      run_once<ScaleChain>(two, queue);
      const Verification v = verify_sampled(l0ctx, first, second, a, real, M);
      std::printf("| M | K | N | can_implement | workspace | max abs err | tol | two runs |\n");
      std::printf("|---:|---:|---:|---:|---:|---:|---:|---|\n");
      std::printf("| %u | %u | %u | %d | %zu B | %.6g | %.6g | %s |\n", M, real.K, real.N,
                  int(one.can_implement), one.workspace, v.max_abs_err, v.tolerance,
                  v.bitwise ? "bitwise identical" : "**DIFFER**");
      if (v.max_abs_err > v.tolerance) fail("M=64 real-weight GEMM exceeded the tolerance");
      std::fflush(stdout);
    }

    if (do_rate) {
      std::printf("\n## TFLOP/s (measured, iterate grade)\n\n");
      std::printf("# %d replays, first %d discarded, median of the last %d, %d enqueues/replay; a "
                  "discarded M=512 ramp control precedes each shape.\n",
                  kReplays, kDroppedReplays, kReplays - kDroppedReplays, kTimedEnqueues);
      std::printf("| shape | K×N | M | can_implement | ms | TFLOP/s | workspace | two runs |\n");
      std::printf("|---|---:|---:|---:|---:|---:|---:|---|\n");
      constexpr size_t kShapeCount = sizeof(kShapes) / sizeof(kShapes[0]);
      constexpr size_t kMCount = sizeof(kMs) / sizeof(kMs[0]);
      std::array<std::array<double, kMCount>, kShapeCount> rates{};
      bool all_clean = true;
      double gate_up_2048_zero = 0.0;
      for (size_t si = 0; si < kShapeCount; ++si) {
        const ProbeShape& shape = kShapes[si];
        const Int4Matrix m = synthetic_matrix(shape.K, shape.N, 0xC0FFEEu ^ shape.K ^ shape.N);
        const size_t b4_bytes = size_t(shape.K) * shape.N / 2;
        std::vector<uint32_t> repacked(b4_bytes / 4);
        repack_to_nk(m.qweight.data(), shape.K, shape.N, true, repacked.data());
        auto d_b = upload(l0ctx, repacked.data(), b4_bytes);
        auto d_s = upload(l0ctx, m.scales.data(), m.scales.size() * 2);
        std::vector<int8_t> zeros(size_t(shape.K / kGroup) * shape.N, int8_t(8));
        auto d_z = upload(l0ctx, zeros.data(), zeros.size());
        const RateResult warm = rate_cell<ScaleChain>(l0ctx, cx, d_b->ptr(), d_s->as<uint16_t>(),
                                                      d_z->as<int8_t>(), kMs[0], shape.K, shape.N,
                                                      hw, true);
        std::printf("# ramp control (discarded): %s M=%u, %.3f ms, %.2f TFLOP/s\n", shape.name,
                    kMs[0], warm.ms, warm.tflops);
        for (size_t mi = 0; mi < kMCount; ++mi) {
          const RateResult r = rate_cell<ScaleChain>(l0ctx, cx, d_b->ptr(), d_s->as<uint16_t>(),
                                                     d_z->as<int8_t>(), kMs[mi], shape.K, shape.N,
                                                     hw, true);
          rates[si][mi] = r.tflops;
          all_clean = all_clean && r.bitwise && r.workspace == 0 &&
                      r.can_implement == int(cutlass::Status::kSuccess);
          std::printf("| %s | %u×%u | %u | %d | %.3f | %.2f | %zu B | %s |\n", shape.name, shape.K,
                      shape.N, kMs[mi], r.can_implement, r.ms, r.tflops, r.workspace,
                      r.bitwise ? "identical" : "**DIFFER**");
          std::fflush(stdout);
        }
        if (std::strcmp(shape.name, "gate‖up") == 0) {
          std::vector<uint32_t> unsigned_pack(b4_bytes / 4);
          repack_to_nk(m.qweight.data(), shape.K, shape.N, false, unsigned_pack.data());
          auto d_bu = upload(l0ctx, unsigned_pack.data(), b4_bytes);
          const RateResult z = rate_cell<ZeroChain>(l0ctx, cx, d_bu->ptr(), d_s->as<uint16_t>(),
                                                    d_z->as<int8_t>(), 2048, shape.K, shape.N, hw,
                                                    true);
          gate_up_2048_zero = z.tflops;
          std::printf("# ConvertAndScaleWithZeroPoint control, gate‖up M=2048: %.3f ms, %.2f "
                      "TFLOP/s, workspace %zu B, two runs %s\n",
                      z.ms, z.tflops, z.workspace, z.bitwise ? "identical" : "**DIFFER**");
        }
      }
      std::printf("\n### TFLOP/s matrix (ConvertAndScale, signed int4, q^8)\n\n");
      std::printf("| shape | M=512 | M=1024 | M=2048 | M=4096 |\n|---|---:|---:|---:|---:|\n");
      for (size_t si = 0; si < kShapeCount; ++si)
        std::printf("| %s | %.2f | %.2f | %.2f | %.2f |\n", kShapes[si].name, rates[si][0],
                    rates[si][1], rates[si][2], rates[si][3]);
      std::printf("\ngate‖up M=2048: **%.2f TFLOP/s** vs the pre-registered bar of 128 "
                  "(%.1f%%); ConvertAndScaleWithZeroPoint control %.2f TFLOP/s.\n", rates[2][2],
                  rates[2][2] / 128.0 * 100.0, gate_up_2048_zero);
      std::printf("Determinism: %s; workspace 0 B in every cell; no split-K atomics.\n",
                  all_clean ? "two runs bitwise equal for every cell" : "**FAILED - see rows**");
      return all_clean ? 0 : 1;
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "probe_gemm_int4_mixed FAILED: %s\n", e.what());
    return 1;
  }
}
