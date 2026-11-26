#include "cpu/expert_int4.h"

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

#include "common/bf16.h"

namespace cpu {

Int4Matrix pack_int4(const uint8_t* codes, const uint16_t* scales, uint32_t K, uint32_t N) {
  if (K % kGroup) throw std::invalid_argument("pack_int4: K is not a multiple of 64");
  const uint32_t G = K / kGroup;
  Int4Matrix m;
  m.K = K;
  m.N = N;
  m.q.resize(size_t(N) * G * 32);
  m.s.resize(size_t(N) * G);
  for (uint32_t n = 0; n < N; ++n)
    for (uint32_t g = 0; g < G; ++g) {
      const uint32_t sl = g / kSliceGroups, gs = slice_groups(G, sl), gi = g % kSliceGroups;
      const size_t at = size_t(sl) * kSliceGroups * N + size_t(n) * gs + gi;
      const uint8_t* c = codes + size_t(n) * K + size_t(g) * kGroup;
      uint8_t* d = m.q.data() + at * 32;
      for (uint32_t j = 0; j < 32; ++j) d[j] = uint8_t((c[j] & 0xF) | ((c[32 + j] & 0xF) << 4));
      m.s[at] = scales[size_t(n) * G + g];
    }
  return m;
}

// ---- pool -------------------------------------------------------------------------------------------------------
Pool::Pool(unsigned threads, std::vector<int> cpus) : n_(threads ? threads : 1) {
  auto pin_to = [](int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
  };
  const bool pin = cpus.size() >= n_;
  if (pin) pin_to(cpus[0]);
  for (unsigned t = 1; t < n_; ++t) workers_.emplace_back([this, t, pin, cpu = pin ? cpus[t] : 0, pin_to] {
    if (pin) pin_to(cpu);
    worker(t);
  });
}
Pool::~Pool() {
  stop_.store(true, std::memory_order_release);
  gen_.fetch_add(1, std::memory_order_release);
  gen_.notify_all();
  for (auto& w : workers_) w.join();
}
void Pool::worker(unsigned tid) {
  uint64_t seen = 0;
  for (;;) {
    // Spin, then yield, then (idle past ~2 ms: between requests) sleep until run() posts the next job.
    unsigned spins = 0;
    auto idle_since = std::chrono::steady_clock::now();
    while (gen_.load(std::memory_order_acquire) == seen) {
      if (++spins < 4000) {
        _mm_pause();
      } else if (std::chrono::steady_clock::now() - idle_since < std::chrono::milliseconds(2)) {
        std::this_thread::yield();
      } else {
        sleepers_.fetch_add(1, std::memory_order_acq_rel);
        gen_.wait(seen, std::memory_order_acquire);
        sleepers_.fetch_sub(1, std::memory_order_acq_rel);
        spins = 0;
        idle_since = std::chrono::steady_clock::now();
      }
    }
    seen = gen_.load(std::memory_order_acquire);
    if (stop_.load(std::memory_order_acquire)) return;
    fn_(ctx_, tid, n_);
    done_.fetch_add(1, std::memory_order_release);
  }
}
void Pool::run(void (*fn)(void*, unsigned, unsigned), void* ctx) {
  if (n_ == 1) return fn(ctx, 0, 1);
  fn_ = fn;
  ctx_ = ctx;
  done_.store(0, std::memory_order_relaxed);
  gen_.fetch_add(1, std::memory_order_release);
  if (sleepers_.load(std::memory_order_acquire)) gen_.notify_all();
  fn(ctx, 0, n_);
  unsigned spins = 0;
  while (done_.load(std::memory_order_acquire) != n_ - 1) {
    if (++spins < 4000) _mm_pause();
    else std::this_thread::yield();
  }
}

// ---- kernels ----------------------------------------------------------------------------------------------------
namespace {

// The card's epilogue: g_b = bf16(gate), u_b = bf16(up), h = bf16(bf16(silu(g_b)) * u_b); returned as the fp32 of that bf16.
inline float silu_up(float gate, float up) {
  const float g = common::bf16_to_f32(common::f32_to_bf16(gate));
  const float u = common::bf16_to_f32(common::f32_to_bf16(up));
  const float s = common::bf16_to_f32(common::f32_to_bf16(g / (1.0f + std::exp(-g))));
  return common::bf16_to_f32(common::f32_to_bf16(s * u));
}

inline float hsum256(__m256 v) {
  __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  a = _mm_add_ps(a, _mm_movehl_ps(a, a));
  a = _mm_add_ss(a, _mm_shuffle_ps(a, a, 1));
  return _mm_cvtss_f32(a);
}

// One K slice (gs groups) of C adjacent columns with M rows of x (fp32, row stride K), added into part[c][m]. q / s: the
// slice's first column; a column is gs groups on. The weight is (c - 8) * scale, exact in fp32 (a 4-bit integer times an
// f16), so one fma per weight vector and row sums the products straight into the accumulator; the card sums each group
// first and scales the sum, so the two differ by fp32 summation-order noise only. M * C accumulators hide the fma latency
// (5 cycles, 2 a cycle); the slice keeps x (M x 512 floats, 16 KB at M = 8) in L1 across the columns of a block.
template <int M, int C>
inline void acc_cols_avx2(const uint8_t* q, const uint16_t* s, uint32_t gs, const float* x, uint32_t K, __m256* part) {
  const __m256i m0f = _mm256_set1_epi32(0x0F);
  __m256 acc[C][M];
  for (int c = 0; c < C; ++c)
    for (int m = 0; m < M; ++m) acc[c][m] = part[c * M + m];
  for (uint32_t g = 0; g < gs; ++g) {
    float sc[C], nsc[C];   // scale and -8 x scale: w = fma(code, scale, -8 scale), exact
    for (int c = 0; c < C; ++c) {
      sc[c] = _cvtsh_ss(s[size_t(c) * gs + g]);   // F16C
      nsc[c] = -8.0f * sc[c];
    }
    const float* xg = x + size_t(g) * kGroup;
    for (int t = 0; t < 4; ++t) {   // bytes 8t..8t+7: low nibbles k = 8t.., high nibbles k = 32 + 8t..
      __m256 xl[M], xh[M];
      for (int m = 0; m < M; ++m) {
        xl[m] = _mm256_loadu_ps(xg + size_t(m) * K + 8 * t);
        xh[m] = _mm256_loadu_ps(xg + size_t(m) * K + 32 + 8 * t);
      }
      for (int c = 0; c < C; ++c) {
        const __m256i b = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + (size_t(c) * gs + g) * 32 + 8 * t)));
        const __m256 scv = _mm256_broadcast_ss(&sc[c]), nscv = _mm256_broadcast_ss(&nsc[c]);
        const __m256 wl = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_and_si256(b, m0f)), scv, nscv);
        const __m256 wh = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(b, 4)), scv, nscv);
        for (int m = 0; m < M; ++m) {
          acc[c][m] = _mm256_fmadd_ps(wl, xl[m], acc[c][m]);
          acc[c][m] = _mm256_fmadd_ps(wh, xh[m], acc[c][m]);
        }
      }
    }
  }
  for (int c = 0; c < C; ++c)
    for (int m = 0; m < M; ++m) part[c * M + m] = acc[c][m];
}

#define B70_AVX512 __attribute__((target("avx512f,avx512bw")))
// The same over 16 lanes. VNNI (int8 x int8) is not used: it needs the activations in int8, which is not the card's math.
template <int M, int C>
B70_AVX512 inline void acc_cols_avx512(const uint8_t* q, const uint16_t* s, uint32_t gs, const float* x, uint32_t K, __m512* part) {
  const __m512i m0f = _mm512_set1_epi32(0x0F);
  __m512 acc[C][M];
  for (int c = 0; c < C; ++c)
    for (int m = 0; m < M; ++m) acc[c][m] = part[c * M + m];
  for (uint32_t g = 0; g < gs; ++g) {
    float sc[C], nsc[C];
    for (int c = 0; c < C; ++c) {
      sc[c] = _cvtsh_ss(s[size_t(c) * gs + g]);
      nsc[c] = -8.0f * sc[c];
    }
    const float* xg = x + size_t(g) * kGroup;
    for (int t = 0; t < 2; ++t) {   // bytes 16t..16t+15: low nibbles k = 16t.., high nibbles k = 32 + 16t..
      __m512 xl[M], xh[M];
      for (int m = 0; m < M; ++m) {
        xl[m] = _mm512_loadu_ps(xg + size_t(m) * K + 16 * t);
        xh[m] = _mm512_loadu_ps(xg + size_t(m) * K + 32 + 16 * t);
      }
      for (int c = 0; c < C; ++c) {
        const __m512i b = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(q + (size_t(c) * gs + g) * 32 + 16 * t)));
        const __m512 scv = _mm512_set1_ps(sc[c]), nscv = _mm512_set1_ps(nsc[c]);
        const __m512 wl = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_and_si512(b, m0f)), scv, nscv);
        const __m512 wh = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_srli_epi32(b, 4)), scv, nscv);
        for (int m = 0; m < M; ++m) {
          acc[c][m] = _mm512_fmadd_ps(wl, xl[m], acc[c][m]);
          acc[c][m] = _mm512_fmadd_ps(wh, xh[m], acc[c][m]);
        }
      }
    }
  }
  for (int c = 0; c < C; ++c)
    for (int m = 0; m < M; ++m) part[c * M + m] = acc[c][m];
}

// Columns per pass by rows: M * C accumulators, 8..12 of the 16 ymm registers (the AVX-512 kernel has 32 and keeps the table).
constexpr int cols_for(int M) { return M == 1 ? 8 : M == 2 ? 4 : M == 3 ? 4 : M <= 6 ? 2 : 1; }
constexpr uint32_t kBlock = 16;   // columns a thread keeps partial sums for while it walks the K slices

// res[c][m] for columns [n0, n0 + nb), nb <= kBlock: the slices outermost, so a block's weights are one sequential run
// per slice and x's slice stays in L1 across its columns. Phase 1 calls it for the gate and the up columns, phase 2 for down.
#define B70_RANGE(NAME, ACC, VEC, HSUM, ATTR)                                                                           \
  template <int M>                                                                                                      \
  ATTR void NAME##_dots(const Int4Matrix& w, uint32_t n0, uint32_t nb, const float* x, float* res) {                    \
    constexpr int C = cols_for(M);                                                                                      \
    const uint32_t G = w.K / kGroup, slices = (G + kSliceGroups - 1) / kSliceGroups;                                    \
    VEC part[kBlock * M];                                                                                               \
    for (uint32_t i = 0; i < nb * M; ++i) part[i] = VEC{};                                                              \
    for (uint32_t sl = 0; sl < slices; ++sl) {                                                                          \
      const uint32_t gs = slice_groups(G, sl);                                                                          \
      const size_t base = size_t(sl) * kSliceGroups * w.N;                                                              \
      const uint8_t* q = w.q.data() + (base + size_t(n0) * gs) * 32;                                                    \
      const uint16_t* sp = w.s.data() + base + size_t(n0) * gs;                                                         \
      const float* xs = x + size_t(sl) * kSliceGroups * kGroup;                                                         \
      uint32_t c = 0;                                                                                                   \
      for (; c + C <= nb; c += C) ACC<M, C>(q + size_t(c) * gs * 32, sp + size_t(c) * gs, gs, xs, w.K, part + c * M);   \
      for (; c < nb; ++c) ACC<M, 1>(q + size_t(c) * gs * 32, sp + size_t(c) * gs, gs, xs, w.K, part + c * M);           \
    }                                                                                                                   \
    for (uint32_t i = 0; i < nb * M; ++i) res[i] = HSUM(part[i]);                                                       \
  }                                                                                                                     \
  template <int M>                                                                                                      \
  ATTR void NAME##_gate_up(const ExpertInt4& e, const float* x, float* h, uint32_t c0, uint32_t c1) {                   \
    const uint32_t inter = e.inter();                                                                                   \
    float g[kBlock * M], u[kBlock * M];                                                                                 \
    for (uint32_t i = c0; i < c1; i += kBlock) {                                                                        \
      const uint32_t nb = std::min(kBlock, c1 - i);                                                                     \
      NAME##_dots<M>(e.gate_up, i, nb, x, g);                                                                           \
      NAME##_dots<M>(e.gate_up, inter + i, nb, x, u);                                                                   \
      for (uint32_t c = 0; c < nb; ++c)                                                                                 \
        for (int m = 0; m < M; ++m) h[size_t(m) * inter + i + c] = silu_up(g[c * M + m], u[c * M + m]);                 \
    }                                                                                                                   \
  }                                                                                                                     \
  template <int M>                                                                                                      \
  ATTR void NAME##_down(const ExpertInt4& e, const float* h, float* out, uint32_t c0, uint32_t c1) {                    \
    const uint32_t N = e.down.N;                                                                                        \
    float r[kBlock * M];                                                                                                \
    for (uint32_t n = c0; n < c1; n += kBlock) {                                                                        \
      const uint32_t nb = std::min(kBlock, c1 - n);                                                                     \
      NAME##_dots<M>(e.down, n, nb, h, r);                                                                              \
      for (uint32_t c = 0; c < nb; ++c)                                                                                 \
        for (int m = 0; m < M; ++m) out[size_t(m) * N + n + c] = r[c * M + m];                                          \
    }                                                                                                                   \
  }
B70_RANGE(avx2, acc_cols_avx2, __m256, hsum256, )
B70_RANGE(avx512, acc_cols_avx512, __m512, _mm512_reduce_add_ps, B70_AVX512)

#define B70_TABLE(NAME, PH) \
  {NAME##_##PH<1>, NAME##_##PH<2>, NAME##_##PH<3>, NAME##_##PH<4>, NAME##_##PH<5>, NAME##_##PH<6>, NAME##_##PH<7>, NAME##_##PH<8>}

using GuFn = void (*)(const ExpertInt4&, const float*, float*, uint32_t, uint32_t);
using DnFn = void (*)(const ExpertInt4&, const float*, float*, uint32_t, uint32_t);

GuFn gu_fn(Isa isa, uint32_t M) {
  static const GuFn a2[] = B70_TABLE(avx2, gate_up);
  static const GuFn a5[] = B70_TABLE(avx512, gate_up);
  return (isa == Isa::Avx512 ? a5 : a2)[M - 1];
}
DnFn dn_fn(Isa isa, uint32_t M) {
  static const DnFn a2[] = B70_TABLE(avx2, down);
  static const DnFn a5[] = B70_TABLE(avx512, down);
  return (isa == Isa::Avx512 ? a5 : a2)[M - 1];
}

// The columns go out in chunks from a shared counter, not one static range a thread: the box's cores are shared, and a
// spinning barrier waits for the slowest thread.
struct Job {
  const ExpertInt4* e;
  const float* in;
  float* outp;
  uint32_t cols, chunk;
  size_t in_n;
  void (*fn)(const ExpertInt4&, const float*, float*, uint32_t, uint32_t);
  alignas(64) std::atomic<uint32_t> next{0};   // its own cache line: every thread reads the fields above
};
void run_job(void* p, unsigned, unsigned) {
  Job& j = *static_cast<Job*>(p);
  thread_local std::vector<float> mine;   // a private copy of the activations: the caller / other cores wrote the shared one
  mine.assign(j.in, j.in + j.in_n);
  for (uint32_t i; (i = j.next.fetch_add(j.chunk, std::memory_order_relaxed)) < j.cols;)
    j.fn(*j.e, mine.data(), j.outp, i, std::min(i + j.chunk, j.cols));
}

}  // namespace

const char* isa_name(Isa i) { return i == Isa::Avx512 ? "avx512" : "avx2"; }
bool cpu_has_avx512() {
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw");
}

void expert_forward(const ExpertInt4& e, const uint16_t* x, uint32_t M, float* out, ExpertScratch& sc, Pool* pool, Isa isa) {
  if (M < 1 || M > kMaxRows) throw std::invalid_argument("expert_forward: M must be 1..8");
  if (isa == Isa::Avx512 && !cpu_has_avx512()) throw std::runtime_error("expert_forward: this CPU has no AVX-512");
  const uint32_t hidden = e.hidden(), inter = e.inter();
  sc.x.resize(size_t(M) * hidden);
  sc.h.resize(size_t(M) * inter);
  for (size_t i = 0; i < sc.x.size(); ++i) sc.x[i] = common::bf16_to_f32(x[i]);
  Job p1{&e, sc.x.data(), sc.h.data(), inter, 8, size_t(M) * hidden, gu_fn(isa, M)};
  Job p2{&e, sc.h.data(), out, hidden, 16, size_t(M) * inter, dn_fn(isa, M)};
  if (pool) {
    pool->run(run_job, &p1);
    pool->run(run_job, &p2);
  } else {
    run_job(&p1, 0, 1);
    run_job(&p2, 0, 1);
  }
}

}  // namespace cpu
