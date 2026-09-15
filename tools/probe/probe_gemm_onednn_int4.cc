// P-C: oneDNN's `bf16_int4` W4A16 matmul -- the exact primitive vLLM's own
// prefill GEMM path uses (`int4_gemm_w4a16 -> dnnl matmul -> gemmstone`,
// Phase 1 §1.1). REFERENCE TARGET, not a candidate for adoption (addendum A0):
// its purpose is to set the rate an own in-kernel int4 GEMM would need to beat
// the two-pass dequant+bf16-GEMM path.
//
// Pre-registration: docs/probe-prefill-vllm-parity-2026-09-14.md addendum
// §A1 / §A3.3, committed before this file was built or run.
//
// This probe calls oneDNN's plain C++ primitive API directly (no vllm-xpu-
// kernels machinery: no GpuEngineManager/primitive cache, no torch) --
// `dnnl::sycl_interop::make_engine/make_stream` over our OWN
// `runtime::prefill::Context`'s sycl queue/context, exactly the integration
// pattern Phase 1 §1.4 read out of vllm-xpu-kernels' `onednn_runtime.h`. The
// weight layout is the K-contiguous-per-N repack Phase 1 §1.3 / A23 already
// priced (`out[n·K/8+r] = qweight[r·N+n]`, no XOR: oneDNN carries GPTQ v1's
// zero=8 as a scalar zero-point ATTRIBUTE, not a bit-flip on the weight).
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

// dnnl.hpp does not touch printf, but keep the same discipline every other
// SYCL probe in this tree uses.
#undef printf

#include "common/bf16.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/quant.h"
#include "loader/safetensors.h"
#include "loader/snapshot.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/context_sycl.h"

using namespace dnnl;

namespace {

constexpr int kReplays = 8;
constexpr int kDroppedReplays = 3;
constexpr int kTimedEnqueues = 4;
constexpr uint32_t kGroup = 64;

[[noreturn]] void fail(const std::string& s) { throw std::runtime_error(s); }

uint32_t xorshift(uint32_t& v) {
  v ^= v << 13;
  v ^= v >> 17;
  v ^= v << 5;
  return v;
}

struct Shape {
  uint32_t K, N;
  const char* name;
};
constexpr Shape kShapes[] = {
    {5120, 16384, "qkv‖z"},  {6144, 5120, "out/o_proj"},
    {5120, 34816, "gate‖up"}, {17408, 5120, "down"},
    {5120, 14336, "q‖k‖v"},
};
constexpr uint32_t kMs[] = {1024, 2048, 4096};

// ---------------------------------------------------------------------------
// Weight layout, host side. Layout 0 (our loader's native format, and the
// starting shape both sycl-tla's mixed mainloop and oneDNN's W4A16 path share,
// Phase 1 §1.3): `qweight [K/8][N]` u32, N-contiguous, unsigned nibbles.
// oneDNN's `int4_gemm_w4a16.h` wants K-contiguous PER N -- `ldb =
// mat2.strides()[dim-1]*8` only produces `ldb == K` if the physical byte order
// is already K-contiguous. That repack is EXACTLY A23's u32 transpose, minus
// the `^8` XOR (oneDNN's scalar zero-point attribute carries GPTQ v1's zero=8
// itself; sycl-tla's mixed mainloop needed the XOR because IT used a SIGNED
// int4b_t operand with no zero tensor at all -- two different ways to the same
// dequant, priced once at A23).
// ---------------------------------------------------------------------------
std::vector<uint32_t> repack_k_contig(const uint32_t* qw, uint32_t K, uint32_t N) {
  const uint32_t words_per_row = K / 8;
  std::vector<uint32_t> out(size_t(N) * words_per_row);
  for (uint32_t n = 0; n < N; ++n)
    for (uint32_t r = 0; r < words_per_row; ++r) out[size_t(n) * words_per_row + r] = qw[size_t(r) * N + n];
  return out;
}

// tools/oracle/dequant.py:44-45, verbatim (A23's S7): fp32 product, ONE cast
// to bf16. `unsigned_q` is the checkpoint nibble as stored, before any repack.
inline uint16_t oracle_bf16(uint32_t unsigned_q, uint16_t f16_scale) {
  return common::f32_to_bf16((float(unsigned_q) - 8.0f) * common::f16_to_f32(f16_scale));
}

uint16_t random_bf16(uint32_t& s) { return uint16_t((xorshift(s) & 0x807fu) | 0x3e00u); }

struct Int4Matrix {
  uint32_t K = 0, N = 0;
  std::vector<uint32_t> qweight;  // layout 0: [K/8][N]
  std::vector<uint16_t> scales;   // f16 [K/64][N]
};

Int4Matrix synthetic_matrix(uint32_t K, uint32_t N, uint32_t seed) {
  Int4Matrix m;
  m.K = K;
  m.N = N;
  m.qweight.resize(size_t(K / 8) * N);
  for (uint32_t& w : m.qweight) w = xorshift(seed);
  m.scales.resize(size_t(K / kGroup) * N);
  for (uint16_t& s : m.scales)
    s = common::f32_to_f16(0.02f + 0.06f * float(xorshift(seed) & 0xFFFFu) / 65535.0f);
  return m;
}

// The one real production matrix (A23's own choice, for direct comparability):
// layer 0's fused `in_proj_qkv || in_proj_z`, K=5120, N=16384.
Int4Matrix real_qkvz(const std::string& snapshot) {
  loader::SafetensorsSet set(snapshot);
  auto find_prefix = [&](const char* suffix) -> std::string {
    const std::string want = std::string(suffix) + ".qweight";
    for (const auto& [name, unused] : set.tensors()) {
      (void)unused;
      if (name.size() >= want.size() &&
          name.compare(name.size() - want.size(), want.size(), want) == 0)
        return name.substr(0, name.size() - 8);
    }
    fail(std::string("no tensor named *") + want + " in " + snapshot);
  };
  const loader::LinearSrc qkv =
      loader::LinearSrc::classify(set, find_prefix("layers.0.linear_attn.in_proj_qkv"));
  const loader::LinearSrc z =
      loader::LinearSrc::classify(set, find_prefix("layers.0.linear_attn.in_proj_z"));
  if (qkv.kind != loader::WKind::Int4 || z.kind != loader::WKind::Int4) fail("qkv/z are not int4");
  if (qkv.K != z.K) fail("qkv and z disagree on K");
  Int4Matrix m;
  m.K = qkv.K;
  m.N = qkv.N + z.N;
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
// oneDNN plumbing.
// ---------------------------------------------------------------------------
struct Prim {
  matmul::primitive_desc pd;
  matmul m;
};

// `bf16_int4`: bf16 activations, u4 weights, bf16 dst -- the exact joint dtype
// vLLM selects (onednn_ext.h's `onednn_types_mapper<bf16_int4>`). Weight md:
// LOGICAL dims {K,N}, physical strides {1,K} -- K-contiguous per N, the
// repacked buffer's actual byte layout, no reinterpretation needed. Scale md:
// the checkpoint's own compact shape {K/64,N}, f16, row-major -- identical to
// what our loader already carries (A21's S3: no scale bridge). Zero point:
// scalar s8, the GPTQ v1 symmetric convention (mask 0, no groups).
Prim make_matmul(engine& eng, uint32_t M, uint32_t K, uint32_t N) {
  memory::desc src_md({M, K}, memory::data_type::bf16, {K, 1});
  memory::desc wei_md({K, N}, memory::data_type::u4, {1, K});
  memory::desc dst_md({M, N}, memory::data_type::bf16, {N, 1});

  primitive_attr attr;
  attr.set_scratchpad_mode(scratchpad_mode::user);
  attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) + (1 << 1), {kGroup, 1}, memory::data_type::f16);
  attr.set_zero_points(DNNL_ARG_WEIGHTS, /*mask*/ 0, {}, memory::data_type::s8);
  attr.set_fpmath_mode(fpmath_mode::bf16, true);

  matmul::primitive_desc pd(eng, src_md, wei_md, dst_md, attr);
  // P-C follow-up (docs/probe-prefill-vllm-parity-2026-09-14.md, "P-C
  // follow-up" section): the original probe never recorded which oneDNN
  // implementation was actually dispatched -- print it so a JIT gemmstone
  // kernel can be told apart from a reference/fallback path.
  std::printf("# impl_info_str M=%u K=%u N=%u: %s\n", M, K, N, pd.impl_info_str());
  std::fflush(stdout);
  return {pd, matmul(pd)};
}

struct Run {
  double ms = 0.0;
};

void execute_once(Prim& p, stream& strm, engine& eng, void* src, void* wei, void* scale,
                  void* zp, void* dst, void* scratch) {
  std::unordered_map<int, memory> args;
  args[DNNL_ARG_SRC] = sycl_interop::make_memory(p.pd.src_desc(), eng, sycl_interop::memory_kind::usm, src);
  args[DNNL_ARG_WEIGHTS] =
      sycl_interop::make_memory(p.pd.weights_desc(), eng, sycl_interop::memory_kind::usm, wei);
  args[DNNL_ARG_DST] = sycl_interop::make_memory(p.pd.dst_desc(), eng, sycl_interop::memory_kind::usm, dst);
  args[DNNL_ARG_SCRATCHPAD] =
      sycl_interop::make_memory(p.pd.scratchpad_desc(), eng, sycl_interop::memory_kind::usm, scratch);
  const memory::desc scale_md({int64_t(p.pd.weights_desc().get_dims()[0]) / kGroup,
                               p.pd.weights_desc().get_dims()[1]},
                              memory::data_type::f16, memory::format_tag::ab);
  args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS] =
      sycl_interop::make_memory(scale_md, eng, sycl_interop::memory_kind::usm, scale);
  const memory::desc zp_md({1}, memory::data_type::s8, memory::format_tag::a);
  args[DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS] =
      sycl_interop::make_memory(zp_md, eng, sycl_interop::memory_kind::usm, zp);
  p.m.execute(strm, args);
}

double time_replays(Prim& p, stream& strm, sycl::queue& q, engine& eng, void* src, void* wei,
                    void* scale, void* zp, void* dst, void* scratch) {
  std::array<double, kReplays - kDroppedReplays> samples{};
  for (int replay = 0; replay < kReplays; ++replay) {
    const auto begin = std::chrono::steady_clock::now();
    for (int e = 0; e < kTimedEnqueues; ++e) execute_once(p, strm, eng, src, wei, scale, zp, dst, scratch);
    q.wait_and_throw();
    if (replay >= kDroppedReplays)
      samples[size_t(replay - kDroppedReplays)] =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
              .count() /
          kTimedEnqueues;
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

void* usm_upload(sycl::queue& q, const void* host, size_t bytes) {
  void* p = sycl::malloc_device(bytes, q);
  q.memcpy(p, host, bytes).wait();
  return p;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::string snapshot = "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ";
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg.rfind("--ckpt=", 0) == 0) snapshot = arg.substr(7);
    }
    l0::Context l0ctx(0);
    std::printf("# P-C: oneDNN bf16_int4 matmul (vLLM's real prefill GEMM path, gemmstone JIT)\n");
    std::printf("# L0 device: %s\n", l0ctx.name().c_str());
    runtime::prefill::Context cx(l0ctx);
    sycl::queue& q = runtime::prefill::sycl_queue(cx);
    engine eng = sycl_interop::make_engine(q.get_device(), q.get_context());
    stream strm = sycl_interop::make_stream(eng, q);
    std::printf("# oneDNN engine over our own prefill::Context queue (sycl_interop, USM)\n");

    // --- correctness: identity extraction vs the oracle, real checkpoint ---
    bool correctness_ok = true;
    {
      const std::string dir = loader::resolve_snapshot(snapshot);
      const Int4Matrix real = real_qkvz(dir);
      std::printf("\n## correctness: identity extraction vs tools/oracle/dequant.py\n");
      std::printf("# real matrix: layer 0 in_proj_qkv||in_proj_z, K=%u N=%u (checkpoint %s)\n",
                  real.K, real.N, dir.c_str());
      std::vector<uint32_t> repacked = repack_k_contig(real.qweight.data(), real.K, real.N);
      std::vector<uint16_t> identity(size_t(real.K) * real.K, 0);
      for (uint32_t i = 0; i < real.K; ++i) identity[size_t(i) * real.K + i] = common::f32_to_bf16(1.0f);
      std::vector<int8_t> zp = {int8_t(8)};

      void* d_a = usm_upload(q, identity.data(), identity.size() * 2);
      void* d_b = usm_upload(q, repacked.data(), repacked.size() * 4);
      void* d_s = usm_upload(q, real.scales.data(), real.scales.size() * 2);
      void* d_z = usm_upload(q, zp.data(), zp.size());
      void* d_c = sycl::malloc_device(size_t(real.K) * real.N * 2, q);

      Prim p = make_matmul(eng, real.K, real.K, real.N);
      void* d_scratch = sycl::malloc_device(p.pd.scratchpad_desc().get_size(), q);
      execute_once(p, strm, eng, d_a, d_b, d_s, d_z, d_c, d_scratch);
      q.wait_and_throw();

      std::vector<uint16_t> got(size_t(real.K) * real.N);
      q.memcpy(got.data(), d_c, got.size() * 2).wait();

      size_t compared = 0, mismatched = 0, signed_zero = 0;
      std::string first_mismatch;
      for (uint32_t k = 0; k < real.K; ++k) {
        for (uint32_t n = 0; n < real.N; ++n) {
          const uint32_t word = real.qweight[size_t(k / 8) * real.N + n];
          const uint32_t nibble = (word >> (4 * (k % 8))) & 0xFu;
          const uint16_t want = oracle_bf16(nibble, real.scales[size_t(k / kGroup) * real.N + n]);
          const uint16_t have = got[size_t(k) * real.N + n];
          ++compared;
          if (want != have) {
            // A23's D2 class: both zero, sign bit differs (q=8, negative scale).
            if ((want & 0x7fffu) == 0 && (have & 0x7fffu) == 0) {
              ++signed_zero;
            } else {
              ++mismatched;
              if (first_mismatch.empty())
                first_mismatch = "k=" + std::to_string(k) + " n=" + std::to_string(n) + " want=0x" +
                                 std::to_string(want) + " have=0x" + std::to_string(have);
            }
          }
        }
      }
      std::printf("compared %zu weights: %zu mismatched, %zu signed-zero (A23's D2 class, "
                  "%.3f%%)%s\n",
                  compared, mismatched, signed_zero, 100.0 * double(signed_zero) / double(compared),
                  mismatched ? (" first: " + first_mismatch).c_str() : "");
      correctness_ok = mismatched == 0;
      sycl::free(d_a, q);
      sycl::free(d_b, q);
      sycl::free(d_s, q);
      sycl::free(d_z, q);
      sycl::free(d_c, q);
      sycl::free(d_scratch, q);
    }

    // --- rate: synthetic weights (values do not affect TFLOP/s), every shape
    // and M, same protocol as every other prefill probe.
    std::printf("\n## TFLOP/s (measured, iterate grade)\n\n");
    std::printf("# %d replays, first %d discarded, median of last %d, %d enqueues/replay.\n",
                kReplays, kDroppedReplays, kReplays - kDroppedReplays, kTimedEnqueues);
    std::printf("| shape | K×N | M | ms | TFLOP/s | %% of 150 pre-registered |\n");
    std::printf("|---|---:|---:|---:|---:|---:|\n");
    std::array<std::array<double, sizeof(kMs) / sizeof(kMs[0])>, sizeof(kShapes) / sizeof(kShapes[0])>
        rates{};
    for (size_t si = 0; si < sizeof(kShapes) / sizeof(kShapes[0]); ++si) {
      const Shape& shape = kShapes[si];
      const Int4Matrix m = synthetic_matrix(shape.K, shape.N, 0xC0FFEEu ^ shape.K ^ shape.N);
      std::vector<uint32_t> repacked = repack_k_contig(m.qweight.data(), shape.K, shape.N);
      std::vector<int8_t> zp = {int8_t(8)};
      void* d_b = usm_upload(q, repacked.data(), repacked.size() * 4);
      void* d_s = usm_upload(q, m.scales.data(), m.scales.size() * 2);
      void* d_z = usm_upload(q, zp.data(), zp.size());

      for (size_t mi = 0; mi < sizeof(kMs) / sizeof(kMs[0]); ++mi) {
        const uint32_t M = kMs[mi];
        std::vector<uint16_t> a(size_t(M) * shape.K);
        uint32_t seed = 0x9e3779b9u ^ M ^ shape.K ^ shape.N;
        for (uint16_t& v : a) v = random_bf16(seed);
        void* d_a = usm_upload(q, a.data(), a.size() * 2);
        void* d_c = sycl::malloc_device(size_t(M) * shape.N * 2, q);
        Prim p = make_matmul(eng, M, shape.K, shape.N);
        void* d_scratch = sycl::malloc_device(p.pd.scratchpad_desc().get_size(), q);
        execute_once(p, strm, eng, d_a, d_b, d_s, d_z, d_c, d_scratch);  // warm-up
        q.wait_and_throw();
        const double ms = time_replays(p, strm, q, eng, d_a, d_b, d_s, d_z, d_c, d_scratch);
        const double tflops = 2.0 * double(M) * shape.K * shape.N / (ms * 1e9);
        rates[si][mi] = tflops;
        std::printf("| %s | %u×%u | %u | %.3f | %.2f | %.1f%% |\n", shape.name, shape.K, shape.N, M,
                    ms, tflops, tflops / 150.0 * 100.0);
        std::fflush(stdout);
        sycl::free(d_a, q);
        sycl::free(d_c, q);
        sycl::free(d_scratch, q);
      }
      sycl::free(d_b, q);
      sycl::free(d_s, q);
      sycl::free(d_z, q);
    }
    std::printf("\n### TFLOP/s matrix\n\n| shape | M=1024 | M=2048 | M=4096 |\n|---|---:|---:|---:|\n");
    for (size_t si = 0; si < sizeof(kShapes) / sizeof(kShapes[0]); ++si)
      std::printf("| %s | %.2f | %.2f | %.2f |\n", kShapes[si].name, rates[si][0], rates[si][1],
                  rates[si][2]);

    const double gate_up_2048 = rates[2][1];
    std::printf(
        "\ngate‖up M=2048: **%.2f TFLOP/s** vs pre-registered ~150 (band 135-175, hard floor "
        "96.0). correctness: %s.\n",
        gate_up_2048, correctness_ok ? "0 mismatches" : "**MISMATCHES -- see above**");
    if (gate_up_2048 < 96.0) {
      std::printf("R < 96.0: pre-registration's falsifier -- probe defect suspected, not a "
                  "finding (§A1).\n");
    }
    return correctness_ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "probe_gemm_onednn_int4 FAILED: %s\n", e.what());
    return 1;
  }
}
