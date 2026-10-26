// Spec 25d Task 2 (B2 on prefill): Nemotron 3.5 Bolt's prefill kernels on the card, at the real shapes, against tests/
// kernels/bolt_pf_ref.h. Needs a B70, no checkpoint (synthetic weights). Every kernel is fed its own upstream outputs and
// compared on them; the card build rounds division and sqrt correctly and the twin's transcendentals are the kernels' own,
// so every check is BITWISE but DPAS's sums (C B^T's DPAS build, the grouped GEMMs, the slab GEMM) and flash attention's.
//
//   1. nb_pf_seed / _conv / _dt / _cb (plain) / _ssd (fp32 and _B16) / _gnorm at (pos, C) = (0, 181), (2085, 300) and
//      (37, 2048) - bitwise; Review Focus 4: both tilings (W16_R16, W8_R4) ten times each, finite and equal
//   2. nb_pf_cb_L64_DPAS against the plain build: the worst error relative to Σ_n |C B| (bar 2^-16, PROPOSED: a 128-term fp32
//      sum's worst case is 128 x 2^-24 = 2^-17 of Σ|terms|, two orders at most twice that) and the scan's y / state through
//      it (y bf16 differences counted, the state's row-relative error printed)
//   3. the grouped MoE at I 1536 / 2560: nb_pf_sort, nb_pf_gather, nb_pf_dequant_up / _dn and nb_pf_combine bitwise; the
//      grouped GEMMs (nb_pf_moe_gemm up relu2 / down) BITWISE pf_gemm_T0 on the busiest expert's rows + the host epilogue,
//      and against the twin's decode-order rows at cosine >= 0.9999 (PROPOSED); every tile of the table run twice, equal
//   4. the linears' slabs: pf_dequant_slab (N 4096, both layouts) and k2_pf_dequant_slab (in_proj 18560, q||k||v 4608,
//      fc1||shared_up 6400: the zero-padded tail) exactly the GPTQ columns; in_proj's tail slab through pf_gemm_T0 lands
//      zeros in the padding columns [18560, 18688)
//   5. 25c's per-row kernels at M 2048 (the norm folds _NF / _S1 / _S1_T1, the router, the route, the shared act, the
//      attention prep) and nb_pf_embed_gather - bitwise the twin over 300 rows
//   6. attention (Decision 15: spec 23d's mg_pf_flash_attn at Bolt's full NoPE layer): flash against fp64 at cosine >= 0.99999
//      at depths 3000 / 65536 / 200000 (a 64-row chunk ending there); the eager build bitwise decode's eager rows (23d)
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/repack.h"
#include "kernels/bolt_kernels.h"
#include "kernels/bolt_pf_ref.h"
#include "kernels/kernels.h"
#include "kernels/prefill/pf_kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "runtime/control.h"
#include "runtime/prefill/context.h"
#include "runtime/prefill/gemm.h"
#include "runtime/prefill/gemm_l0.h"
#include "runtime/prefill/kernels.h"

namespace {

namespace kb = kernels::bolt;
namespace br = boltref;
namespace bp = boltpf;
using br::f32;
using br::rne;
using runtime::prefill::arg_val;
using runtime::prefill::KernelArg;
using runtime::prefill::PtrArg;

static_assert(kb::kPfC == bp::kPfC && kb::kPfSub == bp::kSub && kb::kPfTm == bp::kTm && kb::kPfInLd == bp::kInLd &&
                  kb::kPfMaxSub == bp::kMaxSub,
              "kernels::bolt's prefill constants and boltpf's disagree");
static_assert(kb::pf_hdr::words() == bp::kHdrWords && kb::pf_hdr::kRows == bp::kHdrRows &&
                  kb::pf_hdr::kCount == bp::kHdrCount && kb::pf_hdr::kTiles == bp::kHdrTiles,
              "the sort's header moved");

constexpr uint32_t X = bp::kX, LD = bp::kInLd, IN = bp::kInner, NH = bp::kHeads, H = br::kHidden, C2 = bp::kPfC;

float fval(bool coarse, uint32_t t, uint32_t i, uint32_t seed, float amp = 1.0f) {
  return br::fixture_val(coarse, t, i, seed, 0.125f, amp);
}
double cosine(const double* a, const float* b, size_t n) {
  double ab = 0, aa = 0, bb = 0;
  for (size_t i = 0; i < n; ++i) {
    ab += a[i] * b[i];
    aa += a[i] * a[i];
    bb += double(b[i]) * b[i];
  }
  return (aa == 0 && bb == 0) ? 1.0 : ab / std::sqrt(aa * bb);
}
bool bits_eq(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
}

struct Dev {
  l0::Context ctx{0};
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  runtime::prefill::Context cx{ctx};
  runtime::prefill::KernelCache kc{ctx};
  template <class T>
  l0::Mem up(const std::vector<T>& v) {
    l0::Mem m(ctx, l0::MemKind::Device, std::max<size_t>(v.size() * sizeof(T), 64));
    imm.copy(m.ptr(), v.data(), v.size() * sizeof(T));
    return m;
  }
  l0::Mem zeros(size_t bytes) {
    l0::Mem m(ctx, l0::MemKind::Device, std::max<size_t>(bytes, 64));
    imm.fill(m.ptr(), 0u, std::max<size_t>(bytes, 64));
    return m;
  }
  template <class T>
  std::vector<T> rd(const l0::Mem& m, size_t n, size_t off_elems = 0) {
    std::vector<T> v(n);
    imm.copy(v.data(), static_cast<const T*>(m.ptr()) + off_elems, n * sizeof(T));
    return v;
  }
  void go(const std::string& v, const char* entry, uint32_t gx, uint32_t gy, uint32_t gz,
          std::initializer_list<KernelArg> args) {
    cx.launch(kc(v, entry), gx, gy, gz, args);
    cx.wait();
  }
};

// --- 1-2: the Mamba chain ------------------------------------------------------------------------------------------
struct MambaIn {
  std::vector<float> P, cw, cb, A, D, dt_bias, gw, S0;
  std::vector<uint16_t> ring;
};
MambaIn mamba_inputs(uint32_t rows) {
  MambaIn m;
  m.P.assign(size_t(rows) * LD, 0.0f);
  for (uint32_t t = 0; t < rows; ++t) {
    float* r = m.P.data() + size_t(t) * LD;
    for (uint32_t i = 0; i < IN; ++i) r[i] = fval(false, 80, t * IN + i, 13, 4.0f);
    for (uint32_t i = 0; i < X; ++i) r[br::kXbcOff + i] = fval(false, 81, t * X + i, 13, 2.0f);
    for (uint32_t h = 0; h < NH; ++h) r[br::kDtOff + h] = fval(false, 82, t * NH + h, 13, 2.0f);
  }
  m.cw.resize(size_t(X) * 4);
  m.cb.resize(X);
  for (uint32_t i = 0; i < X * 4; ++i) m.cw[i] = fval(false, 83, i, 13, 0.5f);
  for (uint32_t i = 0; i < X; ++i) m.cb[i] = fval(false, 84, i, 13, 0.5f);
  m.A.resize(NH);
  m.D.resize(NH);
  m.dt_bias.resize(NH);
  for (uint32_t h = 0; h < NH; ++h) {
    m.dt_bias[h] = fval(false, 63, h, 12, 4.0f);
    m.A[h] = -br::exp_t(fval(false, 64, h, 12, 4.0f));
    m.D[h] = fval(false, 65, h, 12, 2.0f);
  }
  m.dt_bias[0] = -10.5f;                 // softplus far below 0.001: no clamp
  m.dt_bias[6] = -11.6875f;
  m.A[3] = -br::exp_t(br::rf(4.41f));    // decay e^-82 per unit dt
  m.A[4] = -br::exp_t(br::rf(8.81f));    // the checkpoint's largest A_log
  m.gw.resize(IN);
  for (uint32_t i = 0; i < IN; ++i) m.gw[i] = br::rf(1.0f + fval(false, 85, i, 13, 0.5f));
  m.S0.resize(size_t(NH) * bp::kP * bp::kN);
  for (size_t i = 0; i < m.S0.size(); ++i) m.S0[i] = br::fixture_f32(86, uint32_t(i), 13, 0.5f);
  m.ring.resize(size_t(br::kRing) * X);
  for (size_t i = 0; i < m.ring.size(); ++i) m.ring[i] = rne(fval(false, 77, uint32_t(i), 5, 3.0f));
  return m;
}
struct MambaOut {
  std::vector<uint16_t> xc, ring, y, y16, out;
  std::vector<float> dt, cum, cb, state, state16w;
};
MambaOut mamba_device(Dev& dv, const MambaIn& m, uint32_t pos, uint32_t C, uint32_t W, uint32_t R, bool dpas = false) {
  const std::string v = kb::pf_mamba_variant(false, W, R), v16 = kb::pf_mamba_variant(true, W, R);
  const uint32_t ns = bp::n_sub(pos, C);
  l0::Mem P = dv.up(std::vector<float>(m.P.begin(), m.P.begin() + size_t(C) * LD)), ring = dv.up(m.ring),
          seed = dv.zeros(size_t(3) * X * 2), cw = dv.up(m.cw), cb = dv.up(m.cb), xc = dv.zeros(size_t(C) * X * 2),
          A = dv.up(m.A), D = dv.up(m.D), bias = dv.up(m.dt_bias), dt = dv.zeros(size_t(C) * NH * 4),
          cum = dv.zeros(size_t(C) * NH * 4), cbo = dv.zeros(size_t(ns) * 8 * 64 * 64 * 4), st = dv.up(m.S0),
          y = dv.zeros(size_t(C) * IN * 2), y16 = dv.zeros(size_t(C) * IN * 2), gw = dv.up(m.gw),
          out = dv.zeros(size_t(C) * IN * 2);
  std::vector<uint16_t> s16(m.S0.size());
  for (size_t i = 0; i < s16.size(); ++i) s16[i] = rne(m.S0[i]);
  l0::Mem st16 = dv.up(s16);
  dv.go(v, "nb_pf_seed", X / kb::kPfWg, 1, 1, {PtrArg(ring.ptr()), PtrArg(seed.ptr()), arg_val(pos)});
  dv.go(v, "nb_pf_conv", X / kb::kPfWg, (C + R - 1) / R, 1,
        {PtrArg(P.ptr()), PtrArg(seed.ptr()), PtrArg(cw.ptr()), PtrArg(cb.ptr()), PtrArg(ring.ptr()), PtrArg(xc.ptr()),
         arg_val(pos), arg_val(C)});
  dv.go(v, "nb_pf_dt", 1, ns, 1,
        {PtrArg(P.ptr()), PtrArg(A.ptr()), PtrArg(bias.ptr()), PtrArg(dt.ptr()), PtrArg(cum.ptr()), arg_val(pos), arg_val(C)});
  if (dpas)
    dv.go(kb::pf_cb_dpas_variant(), "nb_pf_cb", ns, 8, 1, {PtrArg(xc.ptr()), PtrArg(cbo.ptr()), arg_val(pos), arg_val(C)});
  else
    dv.go(v, "nb_pf_cb", ns, 8, 1, {PtrArg(xc.ptr()), PtrArg(cbo.ptr()), arg_val(pos), arg_val(C)});
  for (int b = 0; b < 2; ++b)
    dv.go(b ? v16 : v, "nb_pf_ssd", NH * (64 / W), 1, 1,
          {PtrArg(xc.ptr()), PtrArg(dt.ptr()), PtrArg(cum.ptr()), PtrArg(cbo.ptr()), PtrArg(D.ptr()),
           PtrArg(b ? st16.ptr() : st.ptr()), PtrArg(b ? y16.ptr() : y.ptr()), arg_val(pos), arg_val(C)});
  dv.go(v, "nb_pf_gnorm", 8, C, 1, {PtrArg(P.ptr()), PtrArg(y.ptr()), PtrArg(gw.ptr()), PtrArg(out.ptr())});
  MambaOut o;
  o.xc = dv.rd<uint16_t>(xc, size_t(C) * X);
  o.ring = dv.rd<uint16_t>(ring, m.ring.size());
  o.dt = dv.rd<float>(dt, size_t(C) * NH);
  o.cum = dv.rd<float>(cum, size_t(C) * NH);
  o.cb = dv.rd<float>(cbo, size_t(ns) * 8 * 64 * 64);
  o.state = dv.rd<float>(st, m.S0.size());
  o.y = dv.rd<uint16_t>(y, size_t(C) * IN);
  o.y16 = dv.rd<uint16_t>(y16, size_t(C) * IN);
  o.out = dv.rd<uint16_t>(out, size_t(C) * IN);
  const std::vector<uint16_t> sw = dv.rd<uint16_t>(st16, s16.size());
  o.state16w.resize(sw.size());
  for (size_t i = 0; i < sw.size(); ++i) o.state16w[i] = f32(sw[i]);
  return o;
}
bool same_out(const MambaOut& a, const MambaOut& b) {
  return a.xc == b.xc && a.ring == b.ring && bits_eq(a.dt, b.dt) && bits_eq(a.cum, b.cum) && bits_eq(a.cb, b.cb) &&
         a.y == b.y && a.y16 == b.y16 && a.out == b.out && bits_eq(a.state, b.state) && bits_eq(a.state16w, b.state16w);
}
bool finite(const MambaOut& o) {
  for (float v : o.state)
    if (!std::isfinite(v)) return false;
  for (uint16_t v : o.y)
    if (!std::isfinite(f32(v))) return false;
  return true;
}

void mamba(Dev& dv) {
  struct Case {
    uint32_t pos, C;
  };
  const MambaIn m = mamba_inputs(C2);
  for (const Case k : {Case{0, 181}, Case{2085, 300}, Case{37, 2048}}) {
    const MambaOut d = mamba_device(dv, m, k.pos, k.C, kb::kPfSsdW, kb::kPfConvRows);
    std::vector<uint16_t> ring = m.ring;
    const std::vector<uint16_t> sd = bp::seed(ring.data(), k.pos);
    CHECK(d.xc == bp::conv_chunk(m.P.data(), k.C, sd.data(), m.cw.data(), m.cb.data(), ring.data(), k.pos));
    CHECK(d.ring == ring);
    const bp::DtCum dc = bp::dt_cum(m.P.data(), k.C, k.pos, m.A.data(), m.dt_bias.data());
    CHECK(bits_eq(d.dt, dc.dt) && bits_eq(d.cum, dc.cum));
    CHECK(bits_eq(d.cb, bp::cb(d.xc.data(), k.C, k.pos)));
    std::vector<float> st = m.S0;
    std::vector<uint16_t> y(size_t(k.C) * IN, 0);
    bp::ssd(d.xc.data(), k.C, k.pos, dc, d.cb.data(), m.D.data(), st.data(), y.data());
    CHECK(d.y == y && bits_eq(d.state, st));
    std::vector<uint16_t> s16(m.S0.size()), y16(size_t(k.C) * IN, 0);
    for (size_t i = 0; i < s16.size(); ++i) s16[i] = rne(m.S0[i]);
    bp::ssd(d.xc.data(), k.C, k.pos, dc, d.cb.data(), m.D.data(), nullptr, y16.data(), 0, NH, s16.data());
    std::vector<float> s16w(s16.size());
    for (size_t i = 0; i < s16.size(); ++i) s16w[i] = f32(s16[i]);
    CHECK(d.y16 == y16 && bits_eq(d.state16w, s16w));
    CHECK(d.out == bp::gnorm_chunk(m.P.data(), k.C, d.y.data(), m.gw.data()));
    // Review Focus 4: both tilings, ten times each, finite and equal
    for (int rep = 0; rep < 10; ++rep)
      for (const auto& wr : {std::pair<uint32_t, uint32_t>{16, 16}, std::pair<uint32_t, uint32_t>{8, 4}}) {
        const MambaOut r = mamba_device(dv, m, k.pos, k.C, wr.first, wr.second);
        CHECK(finite(r) && same_out(r, d));
      }
    // 2. C B^T by DPAS: against the plain build, and the scan through it
    const MambaOut q = mamba_device(dv, m, k.pos, k.C, kb::kPfSsdW, kb::kPfConvRows, true);
    double worst = 0.0;
    const uint32_t ns = bp::n_sub(k.pos, k.C), off = k.pos % 64;
    for (uint32_t kk = 0; kk < ns; ++kk)
      for (uint32_t g = 0; g < 8; ++g)
        for (uint32_t i = 0; i < 64; ++i)
          for (uint32_t j = 0; j < 64; ++j) {
            const int ti = int(kk * 64 + i) - int(off), tj = int(kk * 64 + j) - int(off);
            if (ti < 0 || tj < 0 || ti >= int(k.C) || tj >= int(k.C)) continue;
            double mag = 0.0;
            for (uint32_t n = 0; n < 128; ++n)
              mag += std::fabs(double(f32(d.xc[size_t(ti) * X + br::kCOff + g * 128 + n])) *
                               f32(d.xc[size_t(tj) * X + br::kBOff + g * 128 + n]));
            const size_t at = bp::cb_at(kk, g, i, j);
            if (mag > 0) worst = std::max(worst, std::fabs(double(q.cb[at]) - d.cb[at]) / mag);
          }
    size_t ydiff = 0;
    for (size_t i = 0; i < d.y.size(); ++i) ydiff += d.y[i] != q.y[i];
    double srel = 0.0;
    for (uint32_t r = 0; r < NH * 64; ++r) {
      double mx = 0.0, dd = 0.0;
      for (uint32_t n = 0; n < 128; ++n) {
        mx = std::max(mx, double(std::fabs(d.state[size_t(r) * 128 + n])));
        dd = std::max(dd, double(std::fabs(d.state[size_t(r) * 128 + n] - q.state[size_t(r) * 128 + n])));
      }
      if (mx > 0) srel = std::max(srel, dd / mx);
    }
    CHECK(worst <= std::ldexp(1.0, -16));
    std::printf("1. the Mamba chain at (pos %u, C %u, %u sub-chunks): seed, conv, ring, dt, cum, C B^T, y, the state (fp32 "
                "and _B16), gnorm bitwise the twin; W16_R16 and W8_R4 x10 equal and finite\n", k.pos, k.C, ns);
    std::printf("2. nb_pf_cb by DPAS vs plain: worst %.3g of sum |C B| (bar 2^-16, PROPOSED); the scan through it: y %zu of %zu "
                "bf16 values differ, the state row-relative %.3g\n", worst, ydiff, d.y.size(), srel);
  }
}

// --- 3: the grouped MoE ---------------------------------------------------------------------------------------------
std::vector<uint32_t> routes_of(uint32_t C, uint32_t E, uint32_t seed) {
  std::vector<uint32_t> rt(size_t(C) * br::kWords);
  std::vector<float> lg(E), bias(E);
  for (uint32_t e = 0; e < E; ++e) bias[e] = 4.5f + br::fixture_f32(93, e, seed) * 0.04f;
  for (uint32_t t = 0; t < C; ++t) {
    for (uint32_t e = 0; e < E; ++e) lg[e] = br::fixture_f32(94, t * E + e, seed, 4.0f);
    const br::Route r = br::route(lg.data(), bias.data(), E);
    std::copy(r.row, r.row + br::kWords, rt.begin() + size_t(t) * br::kWords);
  }
  return rt;
}
std::vector<uint32_t> blocks_of(uint32_t E, uint32_t I, uint32_t seed) {
  const size_t bw = br::expert_block_words(I);
  std::vector<uint32_t> b(size_t(E) * bw);
  for (size_t i = 0; i < b.size(); ++i) {
    if ((i % bw) % 136 < 128) {
      b[i] = br::fixture_hash(90, uint32_t(i), seed);
    } else {
      auto sc = [&](uint32_t h) {
        return uint32_t(common::f32_to_f16((1.0f + float((h >> 4) % 8) / 8.0f) * 0.015625f * ((h & 1) ? -1.0f : 1.0f)));
      };
      b[i] = sc(br::fixture_hash(91, uint32_t(i), seed)) | (sc(br::fixture_hash(92, uint32_t(i), seed)) << 16);
    }
  }
  return b;
}
void moe(Dev& dv) {
  for (const uint32_t I : {1536u, 2560u}) {
    const uint32_t C = 181, E = I == 1536 ? 240 : 496;
    const std::vector<uint32_t> rt = routes_of(C, E, I);
    const std::vector<uint32_t> blocks = blocks_of(E, I, I);
    std::vector<uint16_t> lat(size_t(C) * br::kLatent);
    for (size_t i = 0; i < lat.size(); ++i) lat[i] = rne(fval(false, 95, uint32_t(i), I));
    const bp::Sorted s = bp::sort(rt.data(), C, E);
    const uint32_t T = bp::tmax(C, E), R = T * bp::kTm + 256;   // + 256: the dense GEMM's padded rows stay inside
    const std::string v = kb::pf_moe_variant(I);
    l0::Mem drt = dv.up(rt), dl = dv.up(lat), db = dv.up(blocks), hdr = dv.zeros(size_t(bp::kHdrWords) * 4),
            tiles = dv.zeros(size_t(2) * T * 4), row_tok = dv.zeros(size_t(R) * 4),
            pair_row = dv.zeros(size_t(C) * br::kTopK * 4), xg = dv.zeros(size_t(R) * br::kLatent * 2),
            ha = dv.zeros(size_t(R) * I * 2), yg = dv.zeros(size_t(R) * br::kLatent * 2);
    dv.go(v, "nb_pf_sort", 1, 1, 1,
          {PtrArg(drt.ptr()), PtrArg(hdr.ptr()), PtrArg(tiles.ptr()), PtrArg(row_tok.ptr()), PtrArg(pair_row.ptr()),
           arg_val(C), arg_val(T), arg_val(E)});
    const std::vector<uint32_t> gh = dv.rd<uint32_t>(hdr, bp::kHdrWords);
    CHECK(std::equal(s.hdr.begin(), s.hdr.begin() + bp::kHdrCount + E, gh.begin()));
    CHECK(dv.rd<uint32_t>(tiles, size_t(2) * T) == s.tiles);
    CHECK(dv.rd<uint32_t>(row_tok, s.rows_used()) == std::vector<uint32_t>(s.row_tok.begin(), s.row_tok.begin() + s.rows_used()));
    CHECK(dv.rd<uint32_t>(pair_row, size_t(C) * br::kTopK) == s.pair_row);
    dv.go(v, "nb_pf_gather", T * bp::kTm, 1, 1, {PtrArg(dl.ptr()), PtrArg(hdr.ptr()), PtrArg(row_tok.ptr()), PtrArg(xg.ptr())});
    const std::vector<uint16_t> xgv = dv.rd<uint16_t>(xg, size_t(s.rows_used()) * br::kLatent);
    CHECK(xgv == bp::gather(s, lat.data()));
    // the weight batches (the walk's: kPfBatchBytes / block bytes experts a batch) and the grouped GEMMs
    const uint32_t per = uint32_t(kb::kPfBatchBytes / (size_t(br::kLatent) * I * 2));
    l0::Mem wu = dv.zeros(size_t(per) * br::kLatent * I * 2), wd = dv.zeros(size_t(per) * I * br::kLatent * 2);
    for (int pass = 0; pass < 2; ++pass) {   // every tile run twice: the second pass must rewrite the same bits
      for (uint32_t b0 = 0; b0 < E; b0 += per) {
        const uint32_t b1 = std::min(E, b0 + per);
        dv.go(v, "nb_pf_dequant_up", I / 16, br::kLatent / 64, b1 - b0,
              {PtrArg(db.ptr()), PtrArg(hdr.ptr()), PtrArg(wu.ptr()), arg_val(b0), arg_val(b1), arg_val(E)});
        if (pass == 0 && b0 == 0)
          for (uint32_t e = b0; e < std::min(b1, b0 + 8); ++e)
            if (s.hdr[bp::kHdrCount + e])
              CHECK(dv.rd<uint16_t>(wu, size_t(br::kLatent) * I, size_t(e - b0) * br::kLatent * I) ==
                    bp::dequant_up(blocks.data(), e, I));
        dv.go(kb::pf_up_gemm_variant(I), "nb_pf_moe_gemm", T, I / 256, 1,
              {PtrArg(tiles.ptr()), PtrArg(xg.ptr()), PtrArg(wu.ptr()), PtrArg(ha.ptr()), arg_val(b0), arg_val(b1)});
      }
      for (uint32_t b0 = 0; b0 < E; b0 += per) {
        const uint32_t b1 = std::min(E, b0 + per);
        dv.go(v, "nb_pf_dequant_dn", br::kLatent / 16, I / 64, b1 - b0,
              {PtrArg(db.ptr()), PtrArg(hdr.ptr()), PtrArg(wd.ptr()), arg_val(b0), arg_val(b1), arg_val(E)});
        if (pass == 0 && b0 == 0)
          for (uint32_t e = b0; e < std::min(b1, b0 + 8); ++e)
            if (s.hdr[bp::kHdrCount + e])
              CHECK(dv.rd<uint16_t>(wd, size_t(I) * br::kLatent, size_t(e - b0) * I * br::kLatent) ==
                    bp::dequant_dn(blocks.data(), e, I));
        dv.go(kb::pf_down_gemm_variant(I), "nb_pf_moe_gemm", T, br::kLatent / 256, 1,
              {PtrArg(tiles.ptr()), PtrArg(ha.ptr()), PtrArg(wd.ptr()), PtrArg(yg.ptr()), arg_val(b0), arg_val(b1)});
      }
    }
    const std::vector<uint16_t> a = dv.rd<uint16_t>(ha, size_t(s.rows_used()) * I);
    const std::vector<uint16_t> y = dv.rd<uint16_t>(yg, size_t(s.rows_used()) * br::kLatent);
    // GROUPED == DENSE for the busiest expert: its rows through pf_gemm_T0 over the same bf16 blocks + the host epilogue
    uint32_t eb = 0;
    for (uint32_t e = 1; e < E; ++e)
      if (s.hdr[bp::kHdrCount + e] > s.hdr[bp::kHdrCount + eb]) eb = e;
    uint32_t r0 = 0;
    while (s.tiles[2 * (r0 / bp::kTm)] != eb) r0 += bp::kTm;
    const uint32_t n = s.hdr[bp::kHdrCount + eb];
    {
      const std::vector<uint16_t> bu = bp::dequant_up(blocks.data(), eb, I), bd = bp::dequant_dn(blocks.data(), eb, I);
      l0::Mem du = dv.up(bu), dd = dv.up(bd), cu = dv.zeros(size_t(256) * I * 4), cd = dv.zeros(size_t(256) * br::kLatent * 4);
      const runtime::prefill::GemmBatch gu{n, br::kLatent, I, 1, br::kLatent, I, I, 0, 0, 0};
      runtime::prefill::gemm_l0(dv.cx, dv.kc, gu, xg.as<uint16_t>() + size_t(r0) * br::kLatent, du.as<uint16_t>(),
                                cu.as<float>(), false);
      dv.cx.wait();
      const std::vector<float> cuv = dv.rd<float>(cu, size_t(n) * I);
      for (size_t i = 0; i < cuv.size(); ++i) CHECK_EQ(a[size_t(r0) * I + i], br::relu2(rne(cuv[i])));
      const runtime::prefill::GemmBatch gd{n, I, br::kLatent, 1, I, br::kLatent, br::kLatent, 0, 0, 0};
      runtime::prefill::gemm_l0(dv.cx, dv.kc, gd, ha.as<uint16_t>() + size_t(r0) * I, dd.as<uint16_t>(), cd.as<float>(), false);
      dv.cx.wait();
      const std::vector<float> cdv = dv.rd<float>(cd, size_t(n) * br::kLatent);
      for (size_t i = 0; i < cdv.size(); ++i) CHECK_EQ(y[size_t(r0) * br::kLatent + i], rne(cdv[i]));
    }
    // against the twin's decode-order rows (the GEMM's sum order differs: a cosine)
    const std::vector<uint16_t> ta = bp::grouped_up(s, xgv.data(), blocks.data(), I);
    double cos_a = 1.0;
    for (uint32_t r = 0; r < s.rows_used(); r += 7) {
      if (s.row_tok[r] == bp::kNone) continue;
      std::vector<double> w(I);
      std::vector<float> g(I);
      for (uint32_t i = 0; i < I; ++i) {
        w[i] = f32(ta[size_t(r) * I + i]);
        g[i] = f32(a[size_t(r) * I + i]);
      }
      cos_a = std::min(cos_a, cosine(w.data(), g.data(), I));
    }
    CHECK(cos_a >= 0.9999);
    // the combine on the device's down rows
    l0::Mem dr = dv.zeros(size_t(C) * br::kLatent * 2);
    dv.go(v, "nb_pf_combine", br::kLatent / kb::kPfWg, C, 1,
          {PtrArg(drt.ptr()), PtrArg(pair_row.ptr()), PtrArg(yg.ptr()), PtrArg(dr.ptr()), arg_val(C)});
    CHECK(dv.rd<uint16_t>(dr, size_t(C) * br::kLatent) == bp::combine(rt.data(), s, y.data(), C));
    std::printf("3. the grouped MoE at I %u, E %u, %u rows: sort / gather / dequant / combine bitwise; grouped == dense "
                "pf_gemm_T0 for expert %u (%u rows), up relu2 and down bitwise; up vs the twin's decode order: worst row "
                "cosine %.7f (bar 0.9999); every tile twice, the same bits\n", I, E, C, eb, n, cos_a);
  }
}

// --- 4: the linears' slabs ------------------------------------------------------------------------------------------
void slabs(Dev& dv) {
  struct L {
    uint32_t K, N, layout;
  };
  for (const L c : {L{4096, 18560, 0}, L{4096, 4608, 0}, L{4096, 6400, 0}, L{8192, 4096, 1}, L{4096, 4096, 1},
                    L{1024, 4096, 1}, L{5376, 4096, 1}, L{8192, 4096, 0}}) {
    const br::Gptq g = br::gptq_of(200 + c.N % 97, c.K, c.N, 6);
    std::vector<uint32_t> words;
    std::vector<uint16_t> scales;
    if (c.layout == 1) {
      words.resize(size_t(c.N / 16) * (c.K / 64) * 136);
      common::repack_int4_layout1(g.qweight.data(), g.scales.data(), c.K, c.N, words.data());
    } else {
      words = g.qweight;
      scales = g.scales;
    }
    const model::GemvShape sh{c.K, c.N, 1, c.layout};
    const std::string v = kb::pf_slab_variant(sh);
    l0::Mem w = dv.up(words), sc = dv.up(scales.empty() ? std::vector<uint16_t>(32, 0) : scales),
            slab = dv.zeros(size_t(c.K) * kb::kPfSlab * 2);
    auto at = [&](uint32_t k, uint32_t n) {
      const int q = int((g.qweight[size_t(k / 8) * c.N + n] >> (4 * (k % 8))) & 0xF) - 8;
      return rne(float(q) * common::f16_to_f32(g.scales[size_t(k / 64) * c.N + n]));
    };
    for (uint32_t n0 = 0; n0 < c.N; n0 += kb::kPfSlab) {
      const uint32_t ns = c.N - n0 >= kb::kPfSlab ? kb::kPfSlab : runtime::prefill::pad256(c.N - n0);
      if (kb::pf_slab_tail(sh))
        dv.go(v, "k2_pf_dequant_slab", ns / 16, c.K / 64, 1,
              {PtrArg(w.ptr()), PtrArg(sc.ptr()), PtrArg(slab.ptr()), arg_val(n0), arg_val(ns)});
      else
        dv.go(v, "pf_dequant_slab", kb::kPfSlab / 16, c.K / 64, 1,
              {PtrArg(w.ptr()), PtrArg(sc.ptr()), PtrArg(slab.ptr()), arg_val(n0)});
      if (n0 != 0 && n0 + kb::kPfSlab < c.N) continue;   // the first and the last (tail) slab checked
      const std::vector<uint16_t> got = dv.rd<uint16_t>(slab, size_t(c.K) * ns);
      for (uint32_t k = 0; k < c.K; k += 29)
        for (uint32_t cc = 0; cc < ns; ++cc) CHECK_EQ(got[size_t(k) * ns + cc], n0 + cc < c.N ? at(k, n0 + cc) : uint16_t(0));
    }
    std::printf("4. %s: the first and the last slab exactly the GPTQ columns%s\n", v.c_str(),
                c.N % kb::kPfSlab ? " (the tail's padding columns zeros)" : "");
  }
}

// --- 5: 25c's per-row kernels at M 2048 and the gather -----------------------------------------------------------------
void rows(Dev& dv) {
  const uint32_t C = 300;
  std::vector<float> w(H), s1(size_t(C2) * H, 0.0f), s2(size_t(C2) * H, 0.0f);
  for (uint32_t k = 0; k < H; ++k) w[k] = br::rf(1.0f + fval(false, 1, k, 3, 0.5f));
  std::vector<uint16_t> h(size_t(C2) * H, 0);
  for (size_t i = 0; i < size_t(C) * H; ++i) {
    h[i] = rne(fval(false, 98, uint32_t(i), 3, 2.0f));
    s1[i] = fval(false, 99, uint32_t(i), 3);
    s2[i] = fval(false, 100, uint32_t(i), 3);
  }
  l0::Mem dh = dv.up(h), d1 = dv.up(s1), d2 = dv.up(s2), dw = dv.up(w), ho = dv.zeros(size_t(C2) * H * 2),
          xo = dv.zeros(size_t(C2) * H * 2);
  for (const auto& ss : {std::pair<uint32_t, uint32_t>{0, 0}, std::pair<uint32_t, uint32_t>{1, 0},
                         std::pair<uint32_t, uint32_t>{1, 1}}) {
    dv.go(kb::pf_norm_variant(ss.first, ss.second), "nb_res_norm", kb::kNormWgs, C, 1,
          {PtrArg(dh.ptr()), PtrArg(d1.ptr()), PtrArg(d2.ptr()), PtrArg(dw.ptr()), PtrArg(ho.ptr()), PtrArg(xo.ptr())});
    const br::NormOut tw = br::res_norm(h.data(), s1.data(), ss.first, s2.data(), ss.second, C2, w.data());
    CHECK(dv.rd<uint16_t>(xo, size_t(C) * H) == std::vector<uint16_t>(tw.x.begin(), tw.x.begin() + size_t(C) * H));
    if (ss.first)
      CHECK(dv.rd<uint16_t>(ho, size_t(C) * H) == std::vector<uint16_t>(tw.h_out.begin(), tw.h_out.begin() + size_t(C) * H));
  }
  const uint32_t E = 496;
  std::vector<uint16_t> x(size_t(C2) * H, 0), gw(size_t(E) * H);
  for (size_t i = 0; i < size_t(C) * H; ++i) x[i] = rne(fval(false, 101, uint32_t(i), 3));
  for (size_t i = 0; i < gw.size(); ++i) gw[i] = rne(fval(false, 102, uint32_t(i), 3, 0.05f));
  std::vector<uint16_t> tiles(size_t(H) * E);
  common::repack_bf16_tiled(gw.data(), H, E, tiles.data());
  std::vector<float> bias(E);
  for (uint32_t e = 0; e < E; ++e) bias[e] = 4.5f + br::fixture_f32(103, e, 3) * 0.04f;
  l0::Mem dx = dv.up(x), dt = dv.up(tiles), dlg = dv.zeros(size_t(C2) * E * 4), db = dv.up(bias),
          drt = dv.zeros(size_t(C2) * br::kWords * 4);
  dv.go(kb::pf_router_variant(), "nb_router", E / 16, C, 1, {PtrArg(dx.ptr()), PtrArg(dt.ptr()), PtrArg(dlg.ptr()), arg_val(E)});
  dv.go(kb::pf_route_variant(), "nb_route", 1, C, 1, {PtrArg(dlg.ptr()), PtrArg(db.ptr()), PtrArg(drt.ptr()), arg_val(E)});
  const std::vector<float> lg = dv.rd<float>(dlg, size_t(C) * E);
  const std::vector<uint32_t> rt = dv.rd<uint32_t>(drt, size_t(C) * br::kWords);
  for (uint32_t m = 0; m < C; ++m) {
    const std::vector<float> want = br::router(x.data() + size_t(m) * H, tiles.data(), E);
    CHECK(std::memcmp(want.data(), lg.data() + size_t(m) * E, size_t(E) * 4) == 0);
    const br::Route r = br::route(lg.data() + size_t(m) * E, bias.data(), E);
    CHECK(std::equal(r.row, r.row + br::kWords, rt.begin() + size_t(m) * br::kWords));
  }
  std::vector<float> sl(size_t(C2) * br::kFc1N, 0.0f), sq(size_t(C2) * br::kQkvN, 0.0f);
  for (size_t i = 0; i < size_t(C) * br::kFc1N; ++i) sl[i] = fval(false, 104, uint32_t(i), 3, 3.0f);
  for (size_t i = 0; i < size_t(C) * br::kQkvN; ++i) sq[i] = fval(false, 105, uint32_t(i), 3);
  l0::Mem dsl = dv.up(sl), dlat = dv.zeros(size_t(C2) * br::kLatent * 2), dsh = dv.zeros(size_t(C2) * br::kShared * 2);
  dv.go(kb::pf_sh_act_variant(), "nb_sh_act", (br::kFc1N + 255) / 256, C, 1, {PtrArg(dsl.ptr()), PtrArg(dlat.ptr()), PtrArg(dsh.ptr())});
  std::vector<uint16_t> lat(size_t(C2) * br::kLatent), sh(size_t(C2) * br::kShared);
  br::sh_act(sl.data(), 1, C2, lat.data(), sh.data());
  CHECK(dv.rd<uint16_t>(dlat, size_t(C) * br::kLatent) == std::vector<uint16_t>(lat.begin(), lat.begin() + size_t(C) * br::kLatent));
  CHECK(dv.rd<uint16_t>(dsh, size_t(C) * br::kShared) == std::vector<uint16_t>(sh.begin(), sh.begin() + size_t(C) * br::kShared));
  const uint32_t pos = 4090;
  l0::Mem ctrl(dv.ctx, l0::MemKind::Shared, sizeof(runtime::Control));
  std::memset(ctrl.ptr(), 0, sizeof(runtime::Control));
  ctrl.as<runtime::Control>()->pos = pos;
  ctrl.as<runtime::Control>()->n_active = C;
  l0::Mem dsq = dv.up(sq), dq = dv.zeros(size_t(C2) * 4096 * 4), dk = dv.zeros(size_t(pos + C) * 256 * 2),
          dvv = dv.zeros(size_t(pos + C) * 256 * 2);
  dv.go(kb::pf_attn_prep_variant(), "nb_attn_prep", (br::kQkvN + 255) / 256, C, 1,
        {PtrArg(ctrl.ptr()), PtrArg(dsq.ptr()), PtrArg(dq.ptr()), PtrArg(dk.ptr()), PtrArg(dvv.ptr())});
  const std::vector<float> q = dv.rd<float>(dq, size_t(C) * 4096);
  const std::vector<uint16_t> k = dv.rd<uint16_t>(dk, size_t(pos + C) * 256), vv = dv.rd<uint16_t>(dvv, size_t(pos + C) * 256);
  for (uint32_t m = 0; m < C; ++m) {
    const br::Prep p = br::attn_prep(sq.data(), 1, C2, m);
    CHECK(std::equal(p.q.begin(), p.q.end(), q.begin() + size_t(m) * 4096));
    CHECK(std::equal(p.k.begin(), p.k.end(), k.begin() + size_t(pos + m) * 256));
    CHECK(std::equal(p.v.begin(), p.v.end(), vv.begin() + size_t(pos + m) * 256));
  }
  {   // the gather from the chunk's ids (ids < 64: four tiles of the table)
    std::vector<uint16_t> et(size_t(64) * H);
    for (size_t i = 0; i < et.size(); ++i) et[i] = uint16_t(br::fixture_hash(97, uint32_t(i), 1));
    std::vector<uint32_t> ids(37);
    for (uint32_t m = 0; m < 37; ++m) ids[m] = (m * 7 + 3) % 64;
    l0::Mem di = dv.up(ids), det = dv.up(et), dho = dv.zeros(size_t(37) * H * 2);
    dv.go(kb::pf_embed_variant(), "nb_pf_embed_gather", 1, 37, 1,
          {PtrArg(di.ptr()), PtrArg(det.ptr()), PtrArg(dho.ptr()), arg_val(37u)});
    const std::vector<uint16_t> got = dv.rd<uint16_t>(dho, size_t(37) * H);
    for (uint32_t m = 0; m < 37; ++m)
      CHECK(std::equal(got.begin() + size_t(m) * H, got.begin() + size_t(m + 1) * H, br::embed_row(et.data(), ids[m]).begin()));
  }
  std::printf("5. 25c's kernels at M 2048 over 300 rows bitwise the twin: nb_res_norm _NF / _S1 / _S1_T1, nb_router + nb_route "
              "(E 496), nb_sh_act, nb_attn_prep (pos 4090); nb_pf_embed_gather bitwise\n");
}

// --- 6: attention -----------------------------------------------------------------------------------------------------
void attention(Dev& dv) {
  for (const uint32_t depth : {3000u, 65536u, 200000u}) {
    const uint32_t C = 64, pos = depth - C;
    std::vector<uint16_t> kc(size_t(depth) * 256), vc(size_t(depth) * 256), q16(size_t(C) * 4096);
    for (size_t i = 0; i < kc.size(); ++i) {
      kc[i] = rne(fval(false, 71, uint32_t(i), depth));
      vc[i] = rne(fval(false, 72, uint32_t(i), depth));
    }
    std::vector<float> q(size_t(C2) * 4096, 0.0f);
    for (size_t i = 0; i < q16.size(); ++i) {
      q16[i] = rne(fval(false, 70, uint32_t(i), depth));
      q[i] = f32(q16[i]);
    }
    l0::Mem dq = dv.up(q), dk = dv.up(kc), dvv = dv.up(vc), fo = dv.zeros(size_t(C) * 4096 * 2),
            eo = dv.zeros(size_t(C) * 4096 * 2);
    dv.go(kb::pf_flash_variant(false), "mg_pf_flash_attn", (C + kb::kPfFlashRpw - 1) / kb::kPfFlashRpw, 2, 16 / kb::kPfFlashHpw,
          {PtrArg(dq.ptr()), PtrArg(dk.ptr()), PtrArg(dvv.ptr()), PtrArg(fo.ptr()), arg_val(pos), arg_val(C)});
    dv.go(kb::pf_flash_variant(true), "mg_pf_flash_attn", 32, C, 1,
          {PtrArg(dq.ptr()), PtrArg(dk.ptr()), PtrArg(dvv.ptr()), PtrArg(eo.ptr()), arg_val(pos), arg_val(C)});
    const std::vector<uint16_t> f = dv.rd<uint16_t>(fo, size_t(C) * 4096), e = dv.rd<uint16_t>(eo, size_t(C) * 4096);
    double worst = 1.0;
    size_t eager_diff = 0;
    for (uint32_t t = 0; t < C; t += (depth > 65536 ? 9 : 3)) {
      std::vector<double> want(4096);
      br::attention_fp64(q16.data() + size_t(t) * 4096, kc.data(), vc.data(), pos + t, want.data());
      std::vector<float> got(4096);
      for (uint32_t i = 0; i < 4096; ++i) got[i] = f32(f[size_t(t) * 4096 + i]);
      for (uint32_t hh = 0; hh < 32; ++hh) worst = std::min(worst, cosine(want.data() + hh * 128, got.data() + hh * 128, 128));
      std::vector<uint16_t> eg(4096);
      br::attention_eager(q16.data() + size_t(t) * 4096, kc.data(), vc.data(), pos + t, eg.data());
      for (uint32_t i = 0; i < 4096; ++i) eager_diff += eg[i] != e[size_t(t) * 4096 + i];
    }
    CHECK(worst >= 0.99999);
    CHECK(eager_diff == 0);
    std::printf("6. mg_pf_flash_attn at depth %u (64 rows): flash vs fp64 worst head cosine %.7f (bar 0.99999); the eager "
                "build bitwise decode's eager rows\n", depth, worst);
  }
}

}  // namespace

int main() {
  Dev dv;
  mamba(dv);
  moe(dv);
  slabs(dv);
  rows(dv);
  attention(dv);
  std::printf("bolt_pf_kernels_test OK\n");
  return 0;
}
