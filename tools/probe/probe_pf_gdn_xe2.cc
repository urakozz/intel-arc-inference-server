// P-D: Intel's Xe2 chunked GDN kernel, torch-free. REFERENCE TARGET, not a
// candidate (addendum A0) -- its purpose is to set the TIME an own OpenCL/
// DPAS GDN rewrite would need to beat, and to record whether GDN (not GEMM)
// is where vLLM's remaining advantage lives.
//
// Pre-registration: docs/probe-prefill-vllm-parity-2026-09-14.md addendum
// §A3.4, committed before this file was built or run. Phase 1 §2.1 found
// `chunk_gated_delta_rule_kernels_xe2.hpp`'s torch dependency confined to its
// OUTER wrapper (`chunk_gated_delta_rule_impl_xe2`, the file's last ~130
// lines): everything above it, including the `kernel_launcher<T, StateT>`
// template this probe calls DIRECTLY, is raw pointers/SYCL only.
//
// The vendor header is included UNMODIFIED (read-only checkout,
// B70_VLLM_XPU_KERNELS_DIR). It still unconditionally `#include
// <torch/all.h>` and its non-template wrapper still names `torch::Tensor`,
// so it still needs SOMETHING at that path to typecheck -- but nothing calls
// that wrapper. `src/sycl/vllm_shim/torch/all.h` is that something: a
// minimal stand-in for exactly the `torch::`/`at::`/`TORCH_CHECK` surface the
// wrapper's SIGNATURE and body mention (enumerated by grep), each backed by
// just enough behaviour to compile; its `torch::zeros` throws if ever called,
// which it is not.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include <cute/tensor.hpp>

// A second, probe-side environment shim (the first is
// src/sycl/vllm_shim/torch/all.h). vllm-xpu-kernels' own `gemm.hpp` (included
// by chunk_gated_delta_rule_kernels_xe2.hpp below) declares `constexpr
// SPIRVScope barrier_scope = ScopeWorkgroup;` and calls
// `barrier_arrive(barrier_scope)` / `barrier_wait(barrier_scope)` --
// unqualified calls resolved via its own `using namespace cute;`. On this
// box's icpx/sycl-tla pin, `barrier_scope`'s actual argument type at those
// call sites is `int` (confirmed: `cute::barrier_arrive`/`barrier_wait` take
// `SPIRVScope`, an unscoped enum, and the overload resolution error names the
// passed type as `const int`, not `SPIRVScope` -- a real, reproducible
// mismatch between what vllm-xpu-kernels' gemm.hpp was written against and
// this project's pinned sycl-tla's `cute/util/xe_split_barrier.hpp`, found by
// isolated compilation of the unmodified vendor file, independent of torch).
// `ScopeWorkgroup`'s value is genuinely 2 either way, so these `int`
// overloads -- found first by ordinary lookup in the same `cute` namespace,
// an exact match for the `int` argument where the enum overloads need a
// conversion -- restore the intended call with a value-preserving cast, with
// no change to vendor code.
namespace cute {
inline void barrier_arrive(int scope, int memory_semantics = SemanticsNone) {
  barrier_arrive(static_cast<SPIRVScope>(scope), memory_semantics);
}
inline void barrier_wait(int scope, int memory_semantics = SemanticsNone) {
  barrier_wait(static_cast<SPIRVScope>(scope), memory_semantics);
}
}  // namespace cute

#include "chunk_gated_delta_rule_kernels_xe2.hpp"  // vendor, read-only, unmodified
#include "gdn_ref.h"                               // our own CPU reference (tests/kernels)
#include "tla_pin.h"

#undef printf

namespace {

namespace G = gdn_ref;

constexpr uint32_t kM = 2048;  // PrefillScratch::kC -- one full chunk
constexpr uint32_t kNumKHeads = 16, kNumVHeads = 48, kHeadDim = 128;
constexpr uint32_t kChunkSize = 64;  // gdn::chunk_size_xe2 (gdn_attn_utils.h)

double now_ms(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// loader::kGdnBlockBytes/4 without pulling in loader/small_layout.h (this
// probe reads no checkpoint): the same 41,120-float size gdn_chunk_test.cc
// uses, sized generously past kDtBiasOff + kHeads so nothing runs off the
// end (the gated-norm tail is unused here -- P-D compares state only, per
// §A3.4's own citation of A22's STATE band).
size_t loader_kGdnBlockFloats() { return G::kDtBiasOff + G::kHeads + G::kDim; }

// The identical fixture `gdn_chunk_test.cc`'s `make_fixture()` builds: a
// gate in the long-memory regime the real checkpoint is in (negA in
// [-4,-1], dt_bias = -4), not a degenerate one.
struct Fixture {
  std::vector<float> qkvz, ab, small;
};
Fixture make_fixture() {
  std::mt19937 rng(0x6d31u);
  std::normal_distribution<float> q(0.0f, 0.5f), a(0.0f, 1.0f), conv(0.0f, 0.5f);
  std::uniform_real_distribution<float> negA(-4.0f, -1.0f);
  Fixture f;
  f.qkvz.resize(size_t(kM) * G::kQkvzN);
  for (auto& e : f.qkvz) e = q(rng);
  f.ab.assign(size_t(kM) * G::kAbStride, 0.0f);
  for (uint32_t m = 0; m < kM; ++m)
    for (uint32_t h = 0; h < 2 * G::kHeads; ++h) f.ab[size_t(m) * G::kAbStride + h] = a(rng);
  f.small.assign(loader_kGdnBlockFloats(), 0.0f);
  for (size_t i = 0; i < size_t(G::kConvRows) * G::kConvTaps; ++i) f.small[i] = conv(rng);
  for (uint32_t h = 0; h < G::kHeads; ++h) {
    f.small[G::kNegAOff + h] = negA(rng);
    f.small[G::kDtBiasOff + h] = -4.0f;
  }
  return f;
}

// Depthwise causal conv1d + SiLU, replicated from `gdn_ref::step`'s inline
// block (tests/kernels/gdn_ref.h) so Q/K/V can be packed as THREE separate
// dense tensors -- Intel's kernel takes them as freestanding, fully-packed
// [seq][heads][dim] buffers with no producer-side strides, unlike our own
// `gdn_chunk`, which reads the fused `qkvz_partials` directly. A fresh,
// zeroed ring (this chunk is position 0) is used only here, independent of
// the CPU reference's own ring below -- both start from identical zeroed
// history, so both derive bit-identical xb.
struct Packed {
  std::vector<uint16_t> q, k, v;  // [M][16][128], [M][16][128], [M][48][128] bf16
};
Packed conv_silu_pack(const Fixture& f) {
  Packed p;
  p.q.resize(size_t(kM) * kNumKHeads * kHeadDim);
  p.k.resize(size_t(kM) * kNumKHeads * kHeadDim);
  p.v.resize(size_t(kM) * kNumVHeads * kHeadDim);
  std::vector<uint16_t> ring(size_t(G::kRing) * G::kConvRows, 0);
  for (uint32_t ch = 0; ch < G::kConvRows; ++ch) {
    const float* w = f.small.data() + size_t(ch) * G::kConvTaps;
    float win[4] = {0, 0, 0, 0};
    for (uint32_t m = 0; m < kM; ++m) {
      const uint16_t raw_b = G::rne(f.qkvz[size_t(m) * G::kQkvzN + ch]);
      ring[size_t(m % G::kRing) * G::kConvRows + ch] = raw_b;
      win[3] = G::f32(raw_b);
      float acc = 0.0f;
      for (uint32_t t = 0; t < G::kConvTaps; ++t) acc = std::fma(w[t], win[t], acc);
      const uint16_t out = G::rne(G::silu_f32(acc));
      if (ch < G::kKOff) {
        p.q[size_t(m) * kNumKHeads * kHeadDim + (ch - G::kQOff)] = out;
      } else if (ch < G::kVOff) {
        p.k[size_t(m) * kNumKHeads * kHeadDim + (ch - G::kKOff)] = out;
      } else {
        p.v[size_t(m) * kNumVHeads * kHeadDim + (ch - G::kVOff)] = out;
      }
      win[0] = win[1];
      win[1] = win[2];
      win[2] = win[3];
    }
  }

  // L2-norm Q and K, per position per k-head (gdn_ref::step's rq/rk block,
  // gdn_step.cl's l2norm): NOT done inside Intel's kernel -- grepping the
  // whole 1634-line file for rsqrt/sum-of-squares/l2norm_eps found nothing;
  // the three "l2norm for q, k" comments there are stale/descriptive, not
  // code. Confirmed the hard way: feeding raw (pre-norm) conv output gave a
  // finite-but-astronomical state (max rel 5.137e+24) after one call --
  // one defect, fixed here, not tuned. Q gets `kQScale` folded in
  // (1/sqrt(128)), K does not, matching gdn_ref.h exactly; the bf16 round
  // happens once, on the scaled value (this kernel's own rounding point,
  // not gdn_ref's fp32-after-round-before-scale -- a harmless difference for
  // a banded, not bitwise, reference-target comparison).
  for (uint32_t m = 0; m < kM; ++m) {
    for (uint32_t kh = 0; kh < kNumKHeads; ++kh) {
      uint16_t* qrow = &p.q[(size_t(m) * kNumKHeads + kh) * kHeadDim];
      uint16_t* krow = &p.k[(size_t(m) * kNumKHeads + kh) * kHeadDim];
      double sq = 0.0, sk = 0.0;
      for (uint32_t d = 0; d < kHeadDim; ++d) {
        const double qv = G::f32(qrow[d]), kv = G::f32(krow[d]);
        sq += qv * qv;
        sk += kv * kv;
      }
      const float inv_q = 1.0f / std::sqrt(float(sq) + 1e-6f);
      const float inv_k = 1.0f / std::sqrt(float(sk) + 1e-6f);
      for (uint32_t d = 0; d < kHeadDim; ++d) {
        qrow[d] = G::rne(G::f32(qrow[d]) * inv_q * G::kQScale);
        krow[d] = G::rne(G::f32(krow[d]) * inv_k);
      }
    }
  }
  return p;
}

void* usm_zeroed(sycl::queue& q, size_t bytes) {
  void* p = sycl::malloc_device(bytes, q);
  q.memset(p, 0, bytes).wait();
  return p;
}
template <class T>
void* usm_upload(sycl::queue& q, const std::vector<T>& host) {
  void* p = sycl::malloc_device(host.size() * sizeof(T), q);
  q.memcpy(p, host.data(), host.size() * sizeof(T)).wait();
  return p;
}

// RMS-floored relative difference, gdn_chunk_test.cc's own metric (§A3.4
// cites its band): |got-ref| / max(|ref|, rms).
struct Band {
  double max_rel = 0.0, mean_rel = 0.0, rms = 0.0;
};
Band band(const std::vector<float>& got, const std::vector<float>& ref) {
  double ss = 0;
  for (float v : ref) ss += double(v) * v;
  Band b;
  b.rms = std::sqrt(ss / double(ref.size()));
  const double floor = b.rms > 0 ? b.rms : 1.0;
  double sum = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double r = std::fabs(double(got[i] - ref[i])) / std::max(std::fabs(double(ref[i])), floor);
    sum += r;
    b.max_rel = std::max(b.max_rel, r);
  }
  b.mean_rel = sum / double(ref.size());
  return b;
}

}  // namespace

int main() {
  try {
    std::printf("# P-D: Intel's Xe2 chunked GDN kernel (chunk_gated_delta_rule_impl_xe2's own\n"
                "#      kernel_launcher, called directly -- torch-free shim, no wrapper call)\n"
                "# sycl-tla pin %s (%s)\n", B70_SYCL_TLA_PIN, B70_SYCL_TLA_SHA);

    const Fixture f = make_fixture();
    const Packed p = conv_silu_pack(f);

    // --- the CPU reference: same fixture, gdn_ref::step, ONE call over the
    // whole chunk (the conv window is local state carried across `m` WITHIN
    // one call, so one call at n_act=2048 is the same serial recurrence as
    // 2048 calls at n_act=1 -- gdn_chunk_test.cc's own case-1 reference). ---
    std::vector<uint16_t> ring_cpu(size_t(G::kRing) * G::kConvRows, 0);
    std::vector<float> state_cpu(size_t(G::kHeads) * G::kDim * G::kDim, 0.0f);
    std::vector<float> o_cpu(size_t(kM) * G::kHeads * G::kDim);
    G::step(0, kM, kM, f.qkvz.data(), f.ab.data(), f.small.data(), ring_cpu.data(), state_cpu.data(),
            o_cpu.data());
    std::printf("# CPU reference (gdn_ref::step, one call, n_act=%u): state computed\n", kM);

    // --- device setup -----------------------------------------------------
    sycl::queue queue{sycl::gpu_selector_v};
    std::printf("# SYCL device: %s\n", queue.get_device().get_info<sycl::info::device::name>().c_str());

    using T = cutlass::bfloat16_t;
    using StateT = float;

    void* d_q = usm_upload(queue, p.q);
    void* d_k = usm_upload(queue, p.k);
    void* d_v = usm_upload(queue, p.v);
    void* d_core_out = usm_zeroed(queue, size_t(kM) * G::kHeads * G::kDim * sizeof(uint16_t));

    // b/a: Intel wants [num_v_heads][total_virtual_seqlen] fp32, RAW (pre-
    // sigmoid/softplus -- chunk_prepare_kernel applies its own activation,
    // §gemm.hpp's "l2norm for q,k" comment at three call sites confirms this
    // kernel does its own internal q/k l2norm too, so Q/K above are fed PRE-
    // norm on purpose, matching this kernel's internal "l2norm for q, k"
    // stage rather than our own gdn_step's external qf/kf).
    std::vector<float> b_host(size_t(G::kHeads) * kM), a_host(size_t(G::kHeads) * kM);
    for (uint32_t h = 0; h < G::kHeads; ++h)
      for (uint32_t m = 0; m < kM; ++m) {
        a_host[size_t(h) * kM + m] = f.ab[size_t(m) * G::kAbStride + h];
        b_host[size_t(h) * kM + m] = f.ab[size_t(m) * G::kAbStride + G::kBOff + h];
      }
    void* d_b = usm_upload(queue, b_host);
    void* d_a = usm_upload(queue, a_host);

    std::vector<float> a_log_host(G::kHeads);
    std::vector<uint16_t> dt_bias_host(G::kHeads);
    for (uint32_t h = 0; h < G::kHeads; ++h) {
      a_log_host[h] = std::log(-f.small[G::kNegAOff + h]);  // negA = -exp(A_log)
      dt_bias_host[h] = G::rne(f.small[G::kDtBiasOff + h]);
    }
    void* d_A_log = usm_upload(queue, a_log_host);
    void* d_dt_bias = usm_upload(queue, dt_bias_host);

    const int batch_size = 1;
    const int padding_size = batch_size * (int(kChunkSize) - 1);
    const size_t A_elems = size_t(G::kHeads) * (kM + padding_size) * kChunkSize;
    const size_t w_elems = size_t(G::kHeads) * (kM + padding_size) * kHeadDim;
    const size_t u_elems = size_t(G::kHeads) * (kM + padding_size) * kHeadDim;
    void* d_A = usm_zeroed(queue, A_elems * sizeof(uint16_t));
    void* d_w = usm_zeroed(queue, w_elems * sizeof(uint16_t));
    void* d_u = usm_zeroed(queue, u_elems * sizeof(uint16_t));

    const int ssm_state_stride_0 = int(G::kHeads * G::kDim * G::kDim);
    void* d_state = usm_zeroed(queue, size_t(G::kHeads) * G::kDim * G::kDim * sizeof(float));

    std::vector<int32_t> qsl_host = {0, int32_t(kM)};
    std::vector<int32_t> cache_idx_host = {0};
    std::vector<uint8_t> has_init_host = {0};  // bool: false, this is a fresh chunk
    void* d_qsl = usm_upload(queue, qsl_host);
    void* d_cache_idx = usm_upload(queue, cache_idx_host);
    void* d_has_init = usm_upload(queue, has_init_host);

    auto run_once = [&] {
      queue.memset(d_state, 0, size_t(G::kHeads) * G::kDim * G::kDim * sizeof(float)).wait();
      queue.memset(d_A, 0, A_elems * sizeof(uint16_t)).wait();
      queue.memset(d_w, 0, w_elems * sizeof(uint16_t)).wait();
      queue.memset(d_u, 0, u_elems * sizeof(uint16_t)).wait();
      gdn::kernel_launcher<T, StateT>(
          queue, static_cast<T*>(d_core_out), static_cast<const T*>(d_q), static_cast<const T*>(d_k),
          static_cast<const T*>(d_v), static_cast<T*>(d_A), static_cast<T*>(d_w),
          static_cast<T*>(d_u), static_cast<const float*>(d_b), static_cast<float*>(d_a),
          static_cast<const float*>(d_A_log), static_cast<const T*>(d_dt_bias),
          static_cast<StateT*>(d_state), ssm_state_stride_0, static_cast<const int*>(d_qsl),
          static_cast<const int*>(d_cache_idx), reinterpret_cast<const bool*>(d_has_init), nullptr,
          batch_size, int(kM), int(kNumKHeads), int(kHeadDim), int(kNumVHeads), int(kHeadDim));
      queue.wait_and_throw();
    };

    // --- correctness (band, not bitwise -- pre-registered, A25/A26 precedent
    // for a reduction-order change): ssm_state [v][k] vs our state [k][v] --
    // TRANSPOSED conventions (Intel's own comment: "[.., head_v_dim,
    // head_k_dim]"; ours: "S[k*128+v] ... k-major"). ---
    run_once();
    std::vector<float> state_dev(size_t(G::kHeads) * G::kDim * G::kDim);
    queue.memcpy(state_dev.data(), d_state, state_dev.size() * sizeof(float)).wait();
    std::vector<float> state_dev_kmajor(state_dev.size());
    for (uint32_t h = 0; h < G::kHeads; ++h)
      for (uint32_t k = 0; k < G::kDim; ++k)
        for (uint32_t v = 0; v < G::kDim; ++v)
          state_dev_kmajor[(size_t(h) * G::kDim + k) * G::kDim + v] =
              state_dev[(size_t(h) * G::kDim + v) * G::kDim + k];
    const Band st_band = band(state_dev_kmajor, state_cpu);
    std::printf("\n## numerics (band, not bitwise -- A25/A26 precedent: a reduction-order change "
                "re-measures its own band)\n");
    std::printf("gdn_state vs CPU reference (gdn_ref::step): max rel %.3e, mean rel %.3e (A22's band "
                "was 3.506e-02 / 1.197e-03 for OUR kernel against the SAME kind of reference; not "
                "carried over -- Intel's kernel is different code)\n",
                st_band.max_rel, st_band.mean_rel);
    bool finite_ok = true;
    for (float v : state_dev) if (!std::isfinite(v)) finite_ok = false;
    std::printf("finite: %s\n", finite_ok ? "yes" : "**NO -- NaN/Inf in state**");

    // --- timing: 8 replays, drop 3, median of 5, this project's protocol ---
    constexpr int kReplays = 8, kDropped = 3;
    std::vector<double> samples;
    for (int r = 0; r < kReplays; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      run_once();
      const double ms = now_ms(t0);
      if (r >= kDropped) samples.push_back(ms);
    }
    std::sort(samples.begin(), samples.end());
    const double median_ms = samples[samples.size() / 2];
    std::printf("\n## time (measured, RECORD grade if preflight was clean; 8 replays, drop 3, "
                "median of 5, one discarded warm-up folded into replay 0)\n");
    std::printf("chunk_gated_delta_rule_impl_xe2's kernel_launcher, C=%u, %u v-heads: **%.3f ms**\n",
                kM, G::kHeads, median_ms);
    std::printf("vs our own gdn_chunk's GDN total: 303 ms/chunk measured (A30). Pre-registered band: "
                "150-300 ms, central 220.\n");
    if (median_ms >= 303.0)
      std::printf("t_Intel >= 303: GDN is not where vLLM wins; the residual gap is GEMM + dequant "
                  "(pre-registered interpretation, §A3.4).\n");
    else
      std::printf("delta = 303 - %.3f = %.3f ms/chunk: the prize an own DPAS/vector GDN rewrite "
                  "would need to price against.\n", median_ms, 303.0 - median_ms);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "probe_pf_gdn_xe2 FAILED: %s\n", e.what());
    return 1;
  }
}
