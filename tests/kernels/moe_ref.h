#pragma once
// CPU reference for src/kernels/moe.cl (spec 15c) - the same op chain in the same
// order, every line with a twin in the kernel; the two must be edited together.
//
//   route()        moe_route: bf16-round the router logits, max (exact), exp, the
//                  pairwise Σ tree, p = ex / Σ, rank = #{beaters} with ties to the
//                  lower expert id, the renormalised top-k weights (ascending Σ of the
//                  top-k p) rounded to bf16, the shared gate's sigmoid rounded to bf16.
//   split_dot()    the int4 layout-1 GEMV of one column as moe_gate_up / moe_down run
//                  it: gemv.cl's per-group order (8 dot8 into a group sum, times the
//                  f16 scale) over `ks` K slices, merged by the fixed pairwise tree.
//   silu_up()      moe_gate_up's epilogue: rne(f32(rne(silu(f32(g_b)))) x f32(u_b)).
//   combine()      moe_down's epilogue: the top-k terms rne(f32(d_b) x w_k) summed in
//                  fp32 in ASCENDING SLOT ORDER, rounded once; the shared expert's term
//                  by its gate; their bf16 sum; the residual add.
//   block()        the whole block for one token, through the functions above.
//
// What is NOT bit-exact between this and the device: `exp` (3 ulp in OpenCL; the
// softmax, SiLU and sigmoid) and the GEMV's multiply-adds (OpenCL may contract them
// to fma). So tests/kernels/moe_test.cc holds the GEMV-fed values to a tolerance and
// the route's ids exactly (inputs chosen with clear gaps, or exact ties); the ordering
// and rounding chain itself is exact here and is what tests/kernels/moe_ref_test.cc
// checks on the host against an independent double-precision MoE.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "common/bf16.h"

namespace moe_ref {

inline float f32(uint16_t h) { return common::bf16_to_f32(h); }
inline uint16_t rne(float f) { return common::f32_to_bf16(f); }
inline float rf(float f) { return f32(rne(f)); }   // a value rounded to bf16, as fp32
inline float silu_f32(float x) { return x / (1.0f + std::exp(-x)); }
inline float sigmoid_f32(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// The route row (moe.cl R_*, kernels::moe_route).
constexpr uint32_t kWords = 32, kIds = 0, kWeights = 8, kSharedGate = 16, kProbs = 17, kProb9 = 25;

struct Shape {
  uint32_t experts, top_k, hidden, inter, router_n;
  uint32_t slots() const { return top_k + 1; }
  uint32_t shared_block() const { return experts; }
  // u32 words of one expert's gate||up / down layout-1 block (loader/moe_layout.h).
  size_t gate_up_words() const { return size_t(2 * inter / 16) * (hidden / 64) * 136; }
  size_t down_words() const { return size_t(hidden / 16) * (inter / 64) * 136; }
};
inline Shape ornith_shape() { return {256, 8, 2048, 512, 272}; }

struct Route {
  uint32_t ids[8] = {};
  float w[8] = {};   // bf16 values
  float p[8] = {};
  float p9 = 0;      // p of rank top_k
  float sg = 0;      // the shared gate, a bf16 value
};

// moe_route for one token's logits row [router_n] (fp32, the GEMV's unrounded output).
inline Route route(const float* logits, const Shape& s) {
  const uint32_t E = s.experts;
  std::vector<float> l(E), ex(E), red(E), p(E);
  for (uint32_t e = 0; e < E; ++e) l[e] = rf(logits[e]);
  float mx = l[0];
  for (uint32_t j = 1; j < E; ++j) mx = std::fmax(mx, l[j]);
  for (uint32_t e = 0; e < E; ++e) red[e] = ex[e] = std::exp(l[e] - mx);
  for (uint32_t stride = E / 2; stride > 0; stride >>= 1)
    for (uint32_t e = 0; e < stride; ++e) red[e] += red[e + stride];
  for (uint32_t e = 0; e < E; ++e) p[e] = ex[e] / red[0];
  Route r;
  std::vector<uint32_t> top_id(s.slots());
  std::vector<float> top_p(s.slots(), 0.0f);
  for (uint32_t k = 0; k < s.slots(); ++k) top_id[k] = k;
  for (uint32_t e = 0; e < E; ++e) {
    uint32_t rank = 0;
    for (uint32_t j = 0; j < E; ++j) rank += (p[j] > p[e] || (p[j] == p[e] && j < e)) ? 1u : 0u;
    if (rank < s.slots()) {
      top_id[rank] = e;
      top_p[rank] = p[e];
    }
  }
  float s8 = 0.0f;
  for (uint32_t k = 0; k < s.top_k; ++k) s8 += top_p[k];
  for (uint32_t k = 0; k < s.top_k; ++k) {
    r.ids[k] = top_id[k];
    r.p[k] = top_p[k];
    r.w[k] = rf(top_p[k] / s8);
  }
  r.p9 = top_p[s.top_k];
  r.sg = rf(sigmoid_f32(rf(logits[E])));
  return r;
}

// One layout-1 block's column n over k-groups [g0, g1): moe.cl's tile_dot.
inline float tile_dot(const uint32_t* blk, uint32_t K, uint32_t n, const uint16_t* x, uint32_t g0,
                      uint32_t g1) {
  const uint32_t G = K / 64;
  float acc = 0.f;
  for (uint32_t g = g0; g < g1; ++g) {
    const uint32_t* tile = blk + (size_t(n / 16) * G + g) * 136;
    const uint32_t sw = tile[128 + (n % 16) / 2];
    const float scale = common::f16_to_f32(uint16_t(n % 2 == 0 ? sw & 0xFFFFu : sw >> 16));
    float gacc = 0.f;
    for (uint32_t j = 0; j < 8; ++j) {
      const uint32_t word = tile[j * 16 + n % 16];
      float a = 0.f;
      for (uint32_t i = 0; i < 8; ++i)
        a += float(int((word >> (4 * i)) & 0xFu) - 8) * f32(x[g * 64 + j * 8 + i]);
      gacc += a;
    }
    acc += gacc * scale;
  }
  return acc;
}
// `ks` K slices (a power of 2 dividing K / 64) merged by the pairwise tree.
inline float split_dot(const uint32_t* blk, uint32_t K, uint32_t n, const uint16_t* x, uint32_t ks) {
  const uint32_t per = K / 64 / ks;
  std::vector<float> r(ks);
  for (uint32_t q = 0; q < ks; ++q) r[q] = tile_dot(blk, K, n, x, q * per, (q + 1) * per);
  for (uint32_t stride = ks / 2; stride > 0; stride >>= 1)
    for (uint32_t q = 0; q < stride; ++q) r[q] += r[q + stride];
  return r[0];
}

inline uint16_t silu_up(float gate_sum, float up_sum) {
  const uint16_t g_b = rne(gate_sum), u_b = rne(up_sum);
  const uint16_t s_b = rne(silu_f32(f32(g_b)));
  return rne(f32(s_b) * f32(u_b));
}

// gate||up's interleave16 columns of intermediate column i (common::cols_interleave16).
inline uint32_t gate_col(uint32_t i) { return (i / 16) * 32 + i % 16; }
inline uint32_t up_col(uint32_t i) { return gate_col(i) + 16; }

// moe_down's epilogue for one hidden column: `down` the SLOTS fp32 down sums in slot
// order (the shared expert last), the route's weights and gate, the residual in.
inline uint16_t combine(const float* down, const Route& r, const Shape& s, uint16_t resid) {
  float sum = 0.0f;
  for (uint32_t k = 0; k < s.top_k; ++k) sum += f32(rne(f32(rne(down[k])) * r.w[k]));
  const uint16_t r_b = rne(sum);
  const uint16_t sh_b = rne(f32(rne(down[s.top_k])) * r.sg);
  const uint16_t o_b = rne(f32(r_b) + f32(sh_b));
  return rne(f32(resid) + f32(o_b));
}

// h for every slot of one token: [slots][inter] bf16 (moe_gate_up).
inline std::vector<uint16_t> gate_up(const Route& r, const uint16_t* x, const uint32_t* gu,
                                     const Shape& s, uint32_t up_ks) {
  std::vector<uint16_t> h(size_t(s.slots()) * s.inter);
  for (uint32_t slot = 0; slot < s.slots(); ++slot) {
    const uint32_t e = slot < s.top_k ? r.ids[slot] : s.shared_block();
    const uint32_t* blk = gu + e * s.gate_up_words();
    for (uint32_t i = 0; i < s.inter; ++i)
      h[size_t(slot) * s.inter + i] = silu_up(split_dot(blk, s.hidden, gate_col(i), x, up_ks),
                                              split_dot(blk, s.hidden, up_col(i), x, up_ks));
  }
  return h;
}

// The residual row after moe_down (resid in, [hidden] bf16).
inline std::vector<uint16_t> down(const Route& r, const std::vector<uint16_t>& h, const uint32_t* dn,
                                  const Shape& s, const uint16_t* resid, uint32_t dn_ks) {
  std::vector<uint16_t> out(s.hidden);
  std::vector<float> d(s.slots());
  for (uint32_t n = 0; n < s.hidden; ++n) {
    for (uint32_t slot = 0; slot < s.slots(); ++slot) {
      const uint32_t e = slot < s.top_k ? r.ids[slot] : s.shared_block();
      d[slot] = split_dot(dn + e * s.down_words(), s.inter, n, h.data() + size_t(slot) * s.inter,
                          dn_ks);
    }
    out[n] = combine(d.data(), r, s, resid[n]);
  }
  return out;
}

}  // namespace moe_ref
