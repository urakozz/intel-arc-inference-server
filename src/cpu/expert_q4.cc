#include "cpu/expert_q4.h"

#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "common/bf16.h"

namespace cpu {
namespace {

constexpr uint32_t kTile = 136;                  // u32 a (n-tile, k-group): 128 of nibbles, 8 of f16 scales
constexpr uint32_t kGu = kQ4Hidden / 64;         // 40 k-groups of gate||up
constexpr uint32_t kUpKs = 4;                    // q4_moe.cl's gate||up K slices (src/kernels/CMakeLists.txt UP_KS)
constexpr uint32_t kDnTiles = kQ4Hidden / 16;    // 160 down n-tiles
// The down's K slices at an expert width (DN_KS): 2 at the whole 640 (10 k-groups), 1 at a tp card's half 320 (5).
inline uint32_t dn_ks(uint32_t inter) { return inter == kQ4Inter ? 2 : 1; }

inline float rf(float f) { return common::bf16_to_f32(common::f32_to_bf16(f)); }

// q4_moe.cl's exp_torch, step for step (tests/kernels/eager_attn_ref.h's twin).
float exp_torch(float d) {
  const int q = int(std::nearbyint(d * 1.442695040888963407359924681001892137426645954152985934135449406931f));
  const float qf = float(q);
  float s = std::fma(qf, -0.693145751953125f, d);
  s = std::fma(qf, -1.428606765330187045e-06f, s);
  float u = 0.000198527617612853646278381f;
  u = std::fma(u, s, 0.00139304355252534151077271f);
  u = std::fma(u, s, 0.00833336077630519866943359f);
  u = std::fma(u, s, 0.0416664853692054748535156f);
  u = std::fma(u, s, 0.166666671633720397949219f);
  u = std::fma(u, s, 0.5f);
  const float ss = s * s;
  u = 1.0f + std::fma(ss, u, s);
  if (d < -104.0f) return 0.0f;
  if (100.0f < d) return INFINITY;
  const auto pow2i = [](int e) {
    const uint32_t b = uint32_t(e + 127) << 23;
    float f;
    std::memcpy(&f, &b, 4);
    return f;
  };
  u = u * pow2i(q >> 1);
  return u * pow2i(q - (q >> 1));
}
inline uint16_t silu_up(float gate, float up) {
  const float g = rf(gate), u = rf(up);
  return common::f32_to_bf16(rf(g / (1.0f + exp_torch(-g))) * u);
}

// One n-tile's 16 columns over k-groups [0, ng) from `tile` (its group g0), R rows of fp32 x (row stride ld, at
// k = 64 g0): res[r][lane]. Lanes 0..7 then 8..15, a ymm each. The first product of a word is a multiply where the
// card adds it to 0: equal except for the sign of a zero, which the group sum (from +0) absorbs.
template <int R>
inline void tile_dot(const uint32_t* tile, uint32_t ng, const float* x, size_t ld, float* res) {
  const __m256i m0f = _mm256_set1_epi32(0xF), eight = _mm256_set1_epi32(8);
  for (int h = 0; h < 2; ++h) {
    __m256 acc[R];
    for (int r = 0; r < R; ++r) acc[r] = _mm256_setzero_ps();
    const uint32_t* t = tile;
    for (uint32_t g = 0; g < ng; ++g, t += kTile) {
      const __m256 sc = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(t + 128 + h * 4)));
      __m256 gacc[R];
      for (int r = 0; r < R; ++r) gacc[r] = _mm256_setzero_ps();
      for (int j = 0; j < 8; ++j) {
        __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(t + j * 16 + h * 8));
        __m256 c[8];
        for (int i = 0; i < 8; ++i) {
          c[i] = _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_and_si256(w, m0f), eight));
          w = _mm256_srli_epi32(w, 4);
        }
        for (int r = 0; r < R; ++r) {
          const float* xv = x + size_t(r) * ld + size_t(g) * 64 + j * 8;
          __m256 a = _mm256_mul_ps(c[0], _mm256_broadcast_ss(xv));
          for (int i = 1; i < 8; ++i) a = _mm256_fmadd_ps(c[i], _mm256_broadcast_ss(xv + i), a);
          gacc[r] = _mm256_add_ps(gacc[r], a);
        }
      }
      for (int r = 0; r < R; ++r) acc[r] = _mm256_add_ps(acc[r], _mm256_mul_ps(gacc[r], sc));
    }
    for (int r = 0; r < R; ++r) _mm256_storeu_ps(res + r * 16 + h * 8, acc[r]);
  }
}

#define B70_AVX512 __attribute__((target("avx512f")))
// The same, the 16 lanes in one zmm: each lane's operations in the same order, so the same bits. A code becomes its
// float (c - 8) by a 16-entry table lookup (vpermps reads an index's low 4 bits: a shift, no mask, sub or convert).
// VNNI / AVX512-BF16 dot products are not used: they take int8 or bf16-pair inputs and sum in another order.
template <int R>
B70_AVX512 inline void tile_dot512(const uint32_t* tile, uint32_t ng, const float* x, size_t ld, float* res) {
  const __m512 lut = _mm512_setr_ps(-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7);
  __m512 acc[R];
  for (int r = 0; r < R; ++r) acc[r] = _mm512_setzero_ps();
  const uint32_t* t = tile;
  for (uint32_t g = 0; g < ng; ++g, t += kTile) {
    const __m512 sc = _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(t + 128)));
    __m512 gacc[R];
    for (int r = 0; r < R; ++r) gacc[r] = _mm512_setzero_ps();
    for (int j = 0; j < 8; ++j) {
      const __m512i w = _mm512_loadu_si512(t + j * 16);
      __m512 c[8];
      for (int i = 0; i < 8; ++i) c[i] = _mm512_permutexvar_ps(_mm512_srli_epi32(w, 4 * i), lut);
      for (int r = 0; r < R; ++r) {
        const float* xv = x + size_t(r) * ld + size_t(g) * 64 + j * 8;
        __m512 a = _mm512_mul_ps(c[0], _mm512_set1_ps(xv[0]));
        for (int i = 1; i < 8; ++i) a = _mm512_fmadd_ps(c[i], _mm512_set1_ps(xv[i]), a);
        gacc[r] = _mm512_add_ps(gacc[r], a);
      }
    }
    for (int r = 0; r < R; ++r) acc[r] = _mm512_add_ps(acc[r], _mm512_mul_ps(gacc[r], sc));
  }
  for (int r = 0; r < R; ++r) _mm512_storeu_ps(res + r * 16, acc[r]);
}
template <int R, bool W>
inline void tile(const uint32_t* t, uint32_t ng, const float* x, size_t ld, float* res) {
  if constexpr (W) tile_dot512<R>(t, ng, x, ld, res);
  else tile_dot<R>(t, ng, x, ld, res);
}

// gate||up unit p (intermediate columns [16 p, 16 p + 16)): work-group blk = p / 2 of q4_moe_gate_up, its gate tile
// 4 blk + 2 (p % 2) and the up tile after it; the four K slices, then (s0 + s2) + (s1 + s3).
template <int R, bool W>
void gate_up_unit(const uint32_t* gu, uint32_t p, const float* x, uint16_t* h, uint32_t inter) {
  const uint32_t gt = (p / 2) * 4 + 2 * (p % 2);
  float s[2][kUpKs][R * 16];
  for (uint32_t u = 0; u < 2; ++u)
    for (uint32_t q = 0; q < kUpKs; ++q) {
      const uint32_t g0 = q * (kGu / kUpKs);
      tile<R, W>(gu + (size_t(gt + u) * kGu + g0) * kTile, kGu / kUpKs, x + size_t(g0) * 64, kQ4Hidden, s[u][q]);
    }
  for (int r = 0; r < R; ++r)
    for (uint32_t l = 0; l < 16; ++l) {
      float v[2];
      for (uint32_t u = 0; u < 2; ++u) {
        float a[kUpKs];
        for (uint32_t q = 0; q < kUpKs; ++q) a[q] = s[u][q][r * 16 + l];
        for (uint32_t st = kUpKs / 2; st > 0; st >>= 1)
          for (uint32_t q = 0; q < st; ++q) a[q] += a[q + st];
        v[u] = a[0];
      }
      h[size_t(r) * inter + p * 16 + l] = silu_up(v[0], v[1]);
    }
}
// down n-tile nt: the K slices, s0 + s1 (one slice at a tp card's half: s0).
template <int R, bool W>
void down_unit(const uint32_t* dn, uint32_t nt, const float* h, float* const* out, uint32_t inter) {
  const uint32_t ng = inter / 64, ks = dn_ks(inter);
  float s[2][R * 16];
  for (uint32_t q = 0; q < ks; ++q) {
    const uint32_t g0 = q * (ng / ks);
    tile<R, W>(dn + (size_t(nt) * ng + g0) * kTile, ng / ks, h + size_t(g0) * 64, inter, s[q]);
  }
  for (int r = 0; r < R; ++r)
    for (uint32_t l = 0; l < 16; ++l) {
      float a[2];
      for (uint32_t q = 0; q < ks; ++q) a[q] = s[q][r * 16 + l];
      for (uint32_t st = ks / 2; st > 0; st >>= 1)
        for (uint32_t q = 0; q < st; ++q) a[q] += a[q + st];
      out[r][nt * 16 + l] = a[0];
    }
}

// A phase's units go out from a shared counter (the box's cores are shared: no static split), over every expert of the
// batch (expert-major). A thread copies an expert's activations into its own buffer when it takes a unit of another
// expert than its last (expert_int4.cc's lesson: reading lines another core wrote cost 2-4x at 22 threads).
constexpr uint32_t kChunk = 4;   // down n-tiles a unit
struct Phase {
  const Q4Job* jobs;
  uint32_t count = 0, inter = kQ4Inter;
  uint16_t* h;   // bf16 [count][kMaxRows][inter]: phase 1 writes it, phase 2 reads it
  bool down = false;
  alignas(64) std::atomic<uint32_t> next{0};
};
template <bool W, int R>
void unit(const Phase& ph, const Q4Job& j, uint16_t* h, uint32_t u, const float* mine) {
  if (!ph.down) {
    gate_up_unit<R, W>(j.gu, u, mine, h, ph.inter);
  } else {
    for (uint32_t t = u * kChunk; t < std::min(u * kChunk + kChunk, kDnTiles); ++t)
      down_unit<R, W>(j.dn, t, mine, j.out, ph.inter);
  }
}
template <bool W>
void run_phase(void* p, unsigned, unsigned) {
  Phase& ph = *static_cast<Phase*>(p);
  thread_local std::vector<float> mine;
  const uint32_t per = ph.down ? (kDnTiles + kChunk - 1) / kChunk : ph.inter / 16;   // 16 columns a gate||up unit
  uint32_t have = ~0u;
  for (uint32_t g; (g = ph.next.fetch_add(1, std::memory_order_relaxed)) < ph.count * per;) {
    const uint32_t e = g / per, u = g % per;
    const Q4Job& j = ph.jobs[e];
    uint16_t* h = ph.h + size_t(e) * kMaxRows * ph.inter;
    if (e != have) {
      have = e;
      if (!ph.down) {
        mine.resize(size_t(j.n) * kQ4Hidden);
        for (uint32_t r = 0; r < j.n; ++r)
          for (uint32_t k = 0; k < kQ4Hidden; ++k) mine[size_t(r) * kQ4Hidden + k] = common::bf16_to_f32(j.x[r][k]);
      } else {
        mine.resize(size_t(j.n) * ph.inter);
        for (size_t i = 0; i < size_t(j.n) * ph.inter; ++i) mine[i] = common::bf16_to_f32(h[i]);
      }
    }
    switch (j.n) {
      case 1: unit<W, 1>(ph, j, h, u, mine.data()); break;
      case 2: unit<W, 2>(ph, j, h, u, mine.data()); break;
      case 3: unit<W, 3>(ph, j, h, u, mine.data()); break;
      case 4: unit<W, 4>(ph, j, h, u, mine.data()); break;
      case 5: unit<W, 5>(ph, j, h, u, mine.data()); break;
      case 6: unit<W, 6>(ph, j, h, u, mine.data()); break;
      case 7: unit<W, 7>(ph, j, h, u, mine.data()); break;
      default: unit<W, 8>(ph, j, h, u, mine.data()); break;
    }
  }
}

}  // namespace

void q4_experts(const Q4Job* jobs, uint32_t count, Pool* pool, Isa isa, uint32_t inter) {
  for (uint32_t e = 0; e < count; ++e)
    if (jobs[e].n < 1 || jobs[e].n > kMaxRows) throw std::invalid_argument("q4_experts: 1..8 rows");
  if (inter != kQ4Inter && inter != kQ4TpInter) throw std::invalid_argument("q4_experts: inter 640 or 320");
  if (isa == Isa::Avx512 && !cpu_has_avx512()) throw std::runtime_error("q4_experts: this CPU has no AVX-512");
  if (count == 0) return;
  thread_local std::vector<uint16_t> h;
  h.resize(size_t(count) * kMaxRows * inter);
  const auto fn = isa == Isa::Avx512 ? run_phase<true> : run_phase<false>;
  Phase p1;
  p1.jobs = jobs;
  p1.count = count;
  p1.inter = inter;
  p1.h = h.data();
  Phase p2;
  p2.jobs = jobs;
  p2.count = count;
  p2.inter = inter;
  p2.h = h.data();
  p2.down = true;
  if (pool) {
    pool->run(fn, &p1);
    pool->run(fn, &p2);
  } else {
    fn(&p1, 0, 1);
    fn(&p2, 0, 1);
  }
}

void q4_expert(const uint32_t* gu, const uint32_t* dn, const uint16_t* const* x, uint32_t n, float* const* out,
               Pool* pool, Isa isa, uint32_t inter) {
  if (n < 1 || n > kMaxRows) throw std::invalid_argument("q4_expert: 1..8 rows");
  Q4Job j;
  j.gu = gu;
  j.dn = dn;
  j.n = n;
  for (uint32_t r = 0; r < n; ++r) {
    j.x[r] = x[r];
    j.out[r] = out[r];
  }
  q4_experts(&j, 1, pool, isa, inter);
}

}  // namespace cpu
