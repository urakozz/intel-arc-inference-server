#pragma once
// CPU references for spec 21d's Qwen3.8-Flash-Next prefill kernels (src/kernels/qwen4exp/q4_pf_moe.cl,
// q4_pf_attn.cl, q4_pf_ple.cl and q4_qsa.cl / q4_ple.cl's prefill forms) - each function with a twin on the
// device; edit the two together. Built on 21c's tests/kernels/qwen4exp_ref.h (the decode chains: route, combine,
// the PLE block, the indexer keys and the selection, the attentions) and spec 15d's pf_moe_ref.h (the grouped
// layout, dequant_block), which it includes and does not change - so "prefill combines by decode's rule" is a
// comparison between two functions of this directory (qwen4exp_pf_ref_test.cc), not a claim.
//
//   tmax()             the padded tile count of a C-row chunk: floor((C x 10 + 512 x 31) / 32) + ceil(C / 32)
//                      (1200 at C = 2048; runtime::qwen4exp::pf_tiles is the host's home of it)
//   sort()             q4_pf_sort: expert-major rows, ascending token inside an expert, tiles of 32, the shared
//                      expert's block 512 last with its C tokens in order, pair_row, the header
//   gather()           q4_pf_gather: row r of the sorted A operand (zeros for padding)
//   dequant_gu/_dn     q4_pf_dequant_gu / _dn: block e < 512 the int4 layout-1 dequant (pf_moe_ref's), block 512
//                      the shared expert - its int4 layout-1 blocks dequantised (ours) or its bf16 gemv_bf16 tiles
//                      copied into row-major [K][N] (Intel's)
//   combine()          q4_pf_moe_combine: q4_moe_down's epilogue over sorted rows - the 10 routed terms in RANK
//                      order (the route row's slot order: grouped_mm's), each rf(rf(y) x w) then added (no fma),
//                      r_b = rne, + rne(rf(y_shared) x sg), one rounding; NOT folded into H (the next combine does)
//   dense_rows()       the rows of a chunk at `pos` whose position p <= 2050 (<= 2051 visible positions: QSA is
//                      exactly causal attention there - spec 21 §2.3) - pf_flash_attn's rows; the rest sparse
//   qsa_prep_chunk()   q4_qsa_prep at M = C (-DQSA_PF) + q4_qsa_ring: the chunk's query heads, every block the
//                      chunk completes (its raw keys from the chunk, positions before pos from the tail ring),
//                      then the ring's update - the last min(8, C) raw keys (decode's ring after the same ids)
//   sparse_attn_fp64() q4_pf_sparse_attn's walk in fp64: per row over its own list in tiles of kKt list entries
//                      (ascending), the online softmax at 1/16 - O before the gate ([24][rows][256] layout)
//   sparse_attn_eager() the reference's rounding points over the same list (s rounded twice, p once, o once:
//                      q4ref::qsa_attn_eager's chain without the gate, which pf_attn's gate applies)
//   ple_ids_chunk()    q4_ple_gather at M = C (-DPLE_PF): every row's history from the chunk's ids (positions >=
//                      pos) or the position-indexed id ring (before pos) - no ring write in the gather
//   ple_gate_row()     q4_pf_ple_gate: decode's block up to `gated` and its norm `gn` (one row, 4 streams)
//   ple_conv_row()     q4_pf_ple_conv: the dilated conv over gn of p - 9, p - 6, p - 3 and p, silu, H += out
//   ple_chunk()        the three launches over a chunk (gate, conv, ring) - Review Focus 4: a row's conv must not
//                      read a ring slot this chunk overwrites, so the ring is written last
//
// What is NOT bit-exact between these and the device: the grouped GEMMs' sums (DPAS order), the sparse flash's
// exp / sum order (a cosine bar), and `exp` itself. The sort, gather, dequants, the combine, the prep, the PLE
// chain and the ring updates are exact.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "common/bf16.h"
#include "kernels/k2_ref.h"        // eager_softmax (torch's fp32 softmax)
#include "kernels/pf_moe_ref.h"    // dequant_block (layout 1)
#include "kernels/qwen4exp_ref.h"

namespace q4pf {

using q4ref::f32;
using q4ref::rf;
using q4ref::rne;
constexpr uint32_t kTm = 32, kNone = 0xFFFFFFFFu, kC = 2048;
constexpr uint32_t kE = q4ref::kExperts, kK = q4ref::kTopK, kH = q4ref::kHidden, kI = q4ref::kInter;
constexpr uint32_t kShared = kE;   // the shared expert's block in a weight batch
constexpr uint32_t kW = q4ref::kWords;
// The header (q4_pf_moe.cl H_*, kernels::qwen4exp::pf_hdr): [0] tiles used, [1] the shared expert's first row,
// [2] rows used (32 x tiles), [3] C, [4 + e] expert e's rows (e < 512), [4 + 512] C.
constexpr uint32_t kHdrTiles = 0, kHdrSharedRow = 1, kHdrRows = 2, kHdrC = 3, kHdrCount = 4;
inline constexpr uint32_t hdr_words() { return (kHdrCount + kE + 1 + 15) / 16 * 16; }   // 528
// The last position whose row sees at most 2051 positions (n = (p + 1) / 4 <= 512 complete blocks).
constexpr uint32_t kDenseLast = q4ref::kTopBlocks * q4ref::kBlock + 2;   // 2050
constexpr uint32_t kKt = 32;          // q4_pf_sparse_attn's list entries per tile
constexpr uint32_t kListRow = 2064, kCountWord = 2052;   // kernels::qwen4exp::kListRow / kCountWord

inline uint32_t tmax(uint32_t C) { return (C * kK + kE * (kTm - 1)) / kTm + (C + kTm - 1) / kTm; }

struct Sorted {
  std::vector<uint32_t> hdr, tiles, row_tok, pair_row;
  uint32_t tiles_used = 0;
  uint32_t rows_used() const { return hdr[kHdrRows]; }
  uint32_t shared_row() const { return hdr[kHdrSharedRow]; }
};

// q4_pf_sort over route rows [C][32] (ids at kIds, rank order; an id past 511 clamps to it, as the kernel's
// ushort staging does - q4_route never writes one).
inline Sorted sort(const uint32_t* route, uint32_t C) {
  const uint32_t T = tmax(C);
  std::vector<uint32_t> id(size_t(C) * kK);
  for (uint32_t t = 0; t < C; ++t)
    for (uint32_t j = 0; j < kK; ++j) id[size_t(t) * kK + j] = std::min(route[size_t(t) * kW + q4ref::kIds + j], kE - 1);
  std::vector<uint32_t> cnt(kE, 0), toff(kE + 1, 0);
  for (uint32_t v : id) ++cnt[v];
  for (uint32_t e = 0; e < kE; ++e) toff[e + 1] = toff[e] + (cnt[e] + kTm - 1) / kTm;
  const uint32_t ts = toff[kE], st = (C + kTm - 1) / kTm, ntiles = ts + st, rs = ts * kTm;
  Sorted o;
  o.hdr.assign(hdr_words(), 0);
  o.tiles.assign(size_t(T) * 2, 0);
  o.row_tok.assign(size_t(T) * kTm, kNone);
  o.pair_row.assign(size_t(C) * kK, kNone);
  for (uint32_t t = 0; t < T; ++t) o.tiles[2 * t] = kNone;
  for (uint32_t e = 0; e < kE; ++e) {
    const uint32_t t0 = toff[e], r0 = t0 * kTm;
    for (uint32_t i = 0; i < toff[e + 1] - t0; ++i) {
      o.tiles[2 * (t0 + i)] = e;
      o.tiles[2 * (t0 + i) + 1] = r0 + i * kTm;
    }
    uint32_t p = 0;
    for (uint32_t t = 0; t < C; ++t)
      for (uint32_t j = 0; j < kK; ++j)
        if (id[size_t(t) * kK + j] == e) {
          o.row_tok[r0 + p] = t;
          o.pair_row[size_t(t) * kK + j] = r0 + p;
          ++p;
        }
  }
  for (uint32_t i = 0; i < st; ++i) {
    o.tiles[2 * (ts + i)] = kShared;
    o.tiles[2 * (ts + i) + 1] = rs + i * kTm;
  }
  for (uint32_t i = 0; i < st * kTm; ++i) o.row_tok[rs + i] = i < C ? i : kNone;
  o.tiles_used = ntiles;
  o.hdr[kHdrTiles] = ntiles;
  o.hdr[kHdrSharedRow] = rs;
  o.hdr[kHdrRows] = ntiles * kTm;
  o.hdr[kHdrC] = C;
  for (uint32_t e = 0; e < kE; ++e) o.hdr[kHdrCount + e] = cnt[e];
  o.hdr[kHdrCount + kE] = C;
  return o;
}

// Row r of the sorted A operand: x row row_tok[r] (bf16 [C][width]), zeros for a padding row.
inline std::vector<uint16_t> gather(const Sorted& st, const uint16_t* x, uint32_t width = kH) {
  std::vector<uint16_t> g(size_t(st.rows_used()) * width, 0);
  for (uint32_t r = 0; r < st.rows_used(); ++r)
    if (st.row_tok[r] != kNone)
      std::copy(x + size_t(st.row_tok[r]) * width, x + size_t(st.row_tok[r] + 1) * width, g.begin() + size_t(r) * width);
  return g;
}

// A gemv_bf16-tiled {K, N} weight as row-major bf16 [K][N] (Intel's bf16 shared expert, block 512).
inline std::vector<uint16_t> untile_bf16(const uint16_t* tiles, uint32_t K, uint32_t N) {
  std::vector<uint16_t> out(size_t(K) * N);
  for (uint32_t k = 0; k < K; ++k)
    for (uint32_t n = 0; n < N; ++n) out[size_t(k) * N + n] = tiles[q4ref::tile_index(K, k, n)];
  return out;
}
// q4_pf_dequant_gu: block e of the gate||up batch as bf16 [2560][1280] (interleave16 kept); block 512 the shared
// expert - int4 layout-1 (sh4, ours) or bf16 tiles (shb, Intel's): exactly one of the two is non-null.
inline std::vector<uint16_t> dequant_gu(const uint32_t* gu, const uint32_t* sh4, const uint16_t* shb, uint32_t e) {
  if (e < kE) return pf_moe_ref::dequant_block(gu + size_t(e) * q4ref::gate_up_block_words(), kH, 2 * kI);
  return sh4 ? pf_moe_ref::dequant_block(sh4, kH, 2 * kI) : untile_bf16(shb, kH, 2 * kI);
}
// q4_pf_dequant_dn: block e of the down batch as bf16 [640][2560].
inline std::vector<uint16_t> dequant_dn(const uint32_t* dn, const uint32_t* sh4, const uint16_t* shb, uint32_t e) {
  if (e < kE) return pf_moe_ref::dequant_block(dn + size_t(e) * q4ref::down_block_words(), kI, kH);
  return sh4 ? pf_moe_ref::dequant_block(sh4, kI, kH) : untile_bf16(shb, kI, kH);
}

// q4_pf_moe_combine for every token of the chunk into y_out [C][2560]: `y` the down GEMM's sorted rows (rne(Σ),
// pitch 2560), the route rows [C][32]: q4ref::combine with each down sum read instead of computed.
inline void combine(const uint32_t* route, const Sorted& s, const uint16_t* y, uint16_t* out, uint32_t C) {
  float d[q4ref::kSlots], w[kK];
  for (uint32_t t = 0; t < C; ++t) {
    const uint32_t* rr = route + size_t(t) * kW;
    for (uint32_t j = 0; j < kK; ++j) w[j] = q4ref::as_f32(rr[q4ref::kWeights + j]);
    const float sg = q4ref::as_f32(rr[q4ref::kSharedGate]);
    for (uint32_t n = 0; n < kH; ++n) {
      for (uint32_t j = 0; j < kK; ++j) d[j] = f32(y[size_t(s.pair_row[size_t(t) * kK + j]) * kH + n]);
      d[kK] = f32(y[size_t(s.shared_row() + t) * kH + n]);
      out[size_t(t) * kH + n] = q4ref::combine(d, w, sg);
    }
  }
}

// --- QSA over a chunk ----------------------------------------------------------------------------------------------
// The rows of a chunk at `pos` with at most 2051 visible positions (p <= 2050): rows [0, dense_rows) run dense
// flash, rows [dense_rows, C) the sparse kernel over their own lists.
inline uint32_t dense_rows(uint32_t pos, uint32_t C) { return pos > kDenseLast ? 0u : std::min(C, kDenseLast + 1 - pos); }

// q4_qsa_prep at M = C (-DQSA_PF) then q4_qsa_ring, over the indexer GEMM's fp32 rows idx [C][ld]: q [C][4][128]
// out; every block b with 4b + 3 in [pos, pos + C) formed into keys [blocks][128] (its raw keys from the chunk, or
// from the tail ring for positions before pos); then tail[p % 8] = raw(p) for the last min(8, C) rows. `rope` the
// loader's table [positions][2][32], wq / wk the indexer's (1 + w) norms [128].
inline void qsa_prep_chunk(const float* idx, uint32_t ld, uint32_t pos, uint32_t C, const float* wq, const float* wk,
                           const float* rope, uint16_t* tail, uint16_t* keys, float* q) {
  const uint32_t D = q4ref::kIdxDim, B = q4ref::kBlock;
  std::vector<uint16_t> raw(size_t(C) * D);
  for (uint32_t m = 0; m < C; ++m) {
    q4ref::qsa_q(idx + size_t(m) * ld, wq, rope + size_t(pos + m) * 2 * q4ref::kRotHalf, q + size_t(m) * q4ref::kIdxHeads * D);
    q4ref::qsa_raw_key(idx + size_t(m) * ld, raw.data() + size_t(m) * D);
  }
  for (uint32_t m = 0; m < C; ++m) {
    const uint32_t p = pos + m;
    if ((p + 1) % B != 0) continue;
    const uint32_t b = (p + 1) / B - 1;
    const uint16_t* r4[4];
    for (uint32_t j = 0; j < B; ++j) {
      const uint32_t qp = b * B + j;
      r4[j] = qp >= pos ? raw.data() + size_t(qp - pos) * D : tail + size_t(qp % q4ref::kTailSlots) * D;
    }
    q4ref::qsa_block_key(r4, wk, rope + size_t(b * B) * 2 * q4ref::kRotHalf, keys + size_t(b) * D);
  }
  for (uint32_t m = C > q4ref::kTailSlots ? C - q4ref::kTailSlots : 0; m < C; ++m)
    std::copy(raw.begin() + size_t(m) * D, raw.begin() + size_t(m + 1) * D, tail + size_t((pos + m) % q4ref::kTailSlots) * D);
}

// q4_pf_sparse_attn's walk in fp64 for one (row, q head h) over its list [0, count): tiles of kKt list entries
// (ascending), the online softmax with m_safe (a row's first tile always holds a key: count >= 2048), O / l at the
// end. q bf16 values [24][256] (the row's), caches [rows][2][256] by position.
inline std::vector<double> sparse_row_fp64(const float* q, const uint16_t* kc, const uint16_t* vc, const uint32_t* list,
                                           uint32_t count, uint32_t h) {
  const uint32_t j = h / q4ref::kGqa, HD = q4ref::kHd, KVH = q4ref::kKvHeads;
  double m = -INFINITY, l = 0.0;
  std::vector<double> o(HD, 0.0);
  for (uint32_t t0 = 0; t0 < count; t0 += kKt) {
    double s[kKt];
    double tm = -INFINITY;
    for (uint32_t i = 0; i < kKt; ++i) {
      if (t0 + i >= count) {
        s[i] = -INFINITY;
        continue;
      }
      const uint16_t* k = kc + (size_t(list[t0 + i]) * KVH + j) * HD;
      double a = 0;
      for (uint32_t d = 0; d < HD; ++d) a += double(q[size_t(h) * HD + d]) * f32(k[d]);
      s[i] = a * double(q4ref::kAttnScale);
      tm = std::max(tm, s[i]);
    }
    const double mnew = std::max(m, tm), msafe = mnew == -INFINITY ? 0.0 : mnew;
    const double corr = std::exp(m - msafe);
    double psum = 0;
    for (uint32_t d = 0; d < HD; ++d) o[d] *= corr;
    for (uint32_t i = 0; i < kKt; ++i) {
      const double pi = std::exp(s[i] - msafe);
      if (pi == 0.0) continue;
      psum += pi;
      const uint16_t* v = vc + (size_t(list[t0 + i]) * KVH + j) * HD;
      for (uint32_t d = 0; d < HD; ++d) o[d] += pi * f32(v[d]);
    }
    l = l * corr + psum;
    m = mnew;
  }
  for (double& v : o) v /= l;
  return o;
}
// Every row r in [r0, C) of a chunk, every q head: o [24][C][256] fp64 (pf_o's layout; rows below r0 untouched).
// q [C][24][256] bf16 values, lists [C][kListRow] (positions [0, count)), counts [C].
inline void sparse_attn_fp64(const float* q, const uint16_t* kc, const uint16_t* vc, const uint32_t* lists,
                             const uint32_t* counts, uint32_t r0, uint32_t C, double* o) {
  const uint32_t HD = q4ref::kHd, QH = q4ref::kQHeads;
  for (uint32_t r = r0; r < C; ++r)
    for (uint32_t h = 0; h < QH; ++h) {
      const std::vector<double> v =
          sparse_row_fp64(q + size_t(r) * QH * HD, kc, vc, lists + size_t(r) * kListRow, counts[r], h);
      std::copy(v.begin(), v.end(), o + (size_t(h) * C + r) * HD);
    }
}
// The reference's rounding points for one (row, head) over the list: s_i = rne(rne(q_b . k_i) x 1/16) (q_b the
// bf16 q, one fma chain), p = torch's fp32 softmax rounded once (k2_ref::eager_softmax), o_d = rne(sum_i p_i v_i)
// (one fma chain, i ascending) - q4ref::qsa_attn_eager's chain before the gate.
inline std::vector<uint16_t> sparse_row_eager(const float* q, const uint16_t* kc, const uint16_t* vc,
                                              const uint32_t* list, uint32_t count, uint32_t h) {
  const uint32_t j = h / q4ref::kGqa, HD = q4ref::kHd, KVH = q4ref::kKvHeads;
  std::vector<float> s(count), p(count);
  for (uint32_t i = 0; i < count; ++i) {
    const uint16_t* k = kc + (size_t(list[i]) * KVH + j) * HD;
    float a = 0.0f;
    for (uint32_t d = 0; d < HD; ++d) a = std::fma(rf(q[size_t(h) * HD + d]), f32(k[d]), a);
    s[i] = rf(rf(a) * q4ref::kAttnScale);
  }
  k2_ref::eager_softmax(s.data(), count, p.data());
  std::vector<uint16_t> o(HD);
  for (uint32_t d = 0; d < HD; ++d) {
    float acc = 0.0f;
    for (uint32_t i = 0; i < count; ++i) acc = std::fma(p[i], f32(vc[(size_t(list[i]) * KVH + j) * HD + d]), acc);
    o[d] = rne(acc);
  }
  return o;
}

// --- PLE over a chunk (spec 21 §2.4; Review Focus 4) ----------------------------------------------------------------
// The history of every row of a chunk at `pos` (ids [C] the chunk's tokens): positions >= pos from the chunk,
// positions before pos from the id ring (slot q % 16, written by earlier chunks / decode steps).
inline std::vector<loader::Q4PleHistory> ple_ids_chunk(const uint32_t* ids, uint32_t pos, uint32_t C,
                                                       const uint32_t* id_ring) {
  std::vector<loader::Q4PleHistory> h(C);
  for (uint32_t m = 0; m < C; ++m)
    h[m] = q4ref::ple_history(pos + m, [&](uint32_t qp) { return qp >= pos ? ids[qp - pos] : id_ring[qp % q4ref::kPleRing]; });
  return h;
}

// q4_pf_ple_gate for one row: q4ref::ple_block's chain up to the gated value and its conv norm. H [10240] (read
// only), kv [12800] the key||value GEMM's fp32 row; gated / gn [10240] out.
inline q4ref::PleTrace ple_gate_row(const uint16_t* H, const float* kv, const float* wk, const float* wq, const float* wc,
                                    uint16_t* gated, uint16_t* gn) {
  const uint32_t HN = q4ref::kHcN, D = q4ref::kHidden;
  q4ref::PleTrace t;
  std::vector<uint16_t> key(HN), keyn(HN), qn(HN);
  for (uint32_t c = 0; c < HN; ++c) key[c] = rne(kv[c]);
  q4ref::grouped_norm(key.data(), wk, keyn.data());
  q4ref::grouped_norm(H, wq, qn.data());
  const uint16_t eps_b = rne(1e-6f);
  for (uint32_t s = 0; s < q4ref::kHc; ++s) {
    std::vector<float> red(256, 0.0f);
    for (uint32_t i = 0; i < 256; ++i) {
      float acc = 0.0f;
      for (uint32_t k = i; k < D; k += 256) acc += rf(f32(keyn[s * D + k]) * f32(qn[s * D + k]));
      red[i] = acc;
    }
    for (uint32_t stride = 128; stride > 0; stride >>= 1)
      for (uint32_t i = 0; i < stride; ++i) red[i] += red[i + stride];
    const float sb = rf(rf(red[0]) / q4ref::kSqrtHidden);
    const float a = std::fabs(sb);
    const float c = a < f32(eps_b) ? f32(eps_b) : a;
    const float r = rf(std::sqrt(c));
    const float g = sb > 0.0f ? r : (sb < 0.0f ? -r : 0.0f);
    const float sg = rf(q4ref::sigmoid_t(g));
    t.s[s] = sb;
    t.gate[s] = g;
    t.sg[s] = sg;
    for (uint32_t k = 0; k < D; ++k) gated[s * D + k] = rne(sg * rf(kv[HN + k]));
  }
  q4ref::grouped_norm(gated, wc, gn);
  return t;
}
// q4_pf_ple_conv for one row: hist[3] the normed gated rows of p - 9, p - 6, p - 3 (null: before position 0), gn
// the row's own; H += rne(gated + rne(silu(rne(sum_t w[c][t] x_t)))) (one fma chain, t ascending).
inline void ple_conv_row(uint16_t* H, const uint16_t* gated, const uint16_t* gn, const uint16_t* const hist[3],
                         const float* taps) {
  for (uint32_t c = 0; c < q4ref::kHcN; ++c) {
    const float xs[4] = {hist[0] ? f32(hist[0][c]) : 0.0f, hist[1] ? f32(hist[1][c]) : 0.0f,
                         hist[2] ? f32(hist[2][c]) : 0.0f, f32(gn[c])};
    float acc = 0.0f;
    for (uint32_t j = 0; j < 4; ++j) acc = std::fma(xs[j], taps[size_t(c) * 4 + j], acc);
    const float conv = rf(q4ref::silu_t(rf(acc)));
    const float out = rf(f32(gated[c]) + conv);
    H[c] = rne(f32(H[c]) + out);
  }
}
// The three launches over a chunk: gate (every row), conv (every row: gn of p - 3k from this chunk when >= pos, else
// the conv ring's slot (p - 3k) % 16), then the rings - the last min(16, C) rows' gn and ids. H [C][10240] in / out,
// kv [C][12800], ring [16][10240], id_ring [16].
inline void ple_chunk(uint16_t* H, const float* kv, const uint32_t* ids, uint32_t pos, uint32_t C, const float* wk,
                      const float* wq, const float* wc, const float* taps, uint16_t* ring, uint32_t* id_ring) {
  const uint32_t HN = q4ref::kHcN, R = q4ref::kPleRing;
  std::vector<uint16_t> gated(size_t(C) * HN), gn(size_t(C) * HN);
  for (uint32_t m = 0; m < C; ++m)
    ple_gate_row(H + size_t(m) * HN, kv + size_t(m) * (HN + q4ref::kHidden), wk, wq, wc, gated.data() + size_t(m) * HN,
                 gn.data() + size_t(m) * HN);
  for (uint32_t m = 0; m < C; ++m) {
    const uint32_t p = pos + m;
    const uint16_t* hist[3];
    for (uint32_t t = 0; t < 3; ++t) {
      const uint32_t back = 9 - 3 * t;
      if (p < back) hist[t] = nullptr;
      else if (p - back >= pos) hist[t] = gn.data() + size_t(p - back - pos) * HN;
      else hist[t] = ring + size_t((p - back) % R) * HN;
    }
    ple_conv_row(H + size_t(m) * HN, gated.data() + size_t(m) * HN, gn.data() + size_t(m) * HN, hist, taps);
  }
  for (uint32_t m = C > R ? C - R : 0; m < C; ++m) {
    std::copy(gn.begin() + size_t(m) * HN, gn.begin() + size_t(m + 1) * HN, ring + size_t((pos + m) % R) * HN);
    id_ring[(pos + m) % R] = ids[m];
  }
}

}  // namespace q4pf
