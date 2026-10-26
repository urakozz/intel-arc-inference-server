// Spec 25c Task 3 (B2): Nemotron 3.5 Bolt's decode kernels on the card, at the real shapes, against tests/kernels/
// bolt_ref.h (and, where the fixture has them, 25a's reference's own values). Needs a B70, no checkpoint (synthetic
// weights). Every binary here is one a decode list binds (kernels::bolt names), plus the block's test-only M = 4 builds
// (the M-row form). The twin's transcendentals are the kernels' own (Sleef's exp, fdlibm's
// log1p) and the card build rounds division and sqrt correctly, so every check is BITWISE but flash attention's.
//
//   1. nb_res_norm _NF on the fixture's rows (vs torch too), _S1, the MoE block's two-source _S1_T1 (vs torch's fold),
//      the M = 4 build's 4 rows each the M = 1 twin's row (Review Focus 4: every work-group's Σ the same tree)
//   2. nb_conv (Review Focus 1): the fixture's 18 positions from a garbage ring as M = 1 launches and as 1 / 2 / 3 / 4-row
//      launches of the M4 build - the ring holding the RAW xBC at pos % 16
//   3. nb_ssm fp32 / bf16 state, plain and the arm conv `_C`, M = 1 and M = 4 (4 rows a launch) over 8 positions crossing
//      nothing, then 14 .. 17 (the ring wraps); dt_bias -10.5 at head 0 (no clamp); y, the state and the ring bitwise
//   4. nb_gnorm on the fixture's 4 rows (vs torch too)
//   5. nb_router (E 496 / 240) and nb_res_norm _S1_R's logits; nb_route on torch's logits (Review Focus 2) and the crafted
//      rows (ties at the 9th / 10th to the lower id incl. the second lane slot, all-equal, a reordering bias, E 496's 495)
//   6. nb_moe_up / nb_moe_down at I 1536 / 2048 / 2304 / 2560 over int4 experts with signed scales (each slot's id from
//      the route row, ascending); the arm `route` (_R) against nb_route + nb_moe_up (Review Focus 5)
//   7. nb_sh_act, nb_attn_prep at 0 / 262143, nb_embed_gather, the argmax at 131072 (maxima at 0 and 131071, ties low)
//   8. attention (Decision 15: spec 23c's binaries, a ones gate): mg_attn_eager against the twin at 6 / 300 / 3000 keys;
//      mg_attn (flash) against fp64 at cosine >= 0.99999 at 1, 2, 300, 4096, 65536, 262143 keys
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/exp_torch.h"
#include "common/repack.h"
#include "kernels/bolt_fixture.h"
#include "kernels/bolt_kernels.h"
#include "kernels/bolt_ref.h"
#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "runtime/control.h"

namespace {

namespace kb = kernels::bolt;
namespace br = boltref;
namespace fx = bolt_fixture;
using br::f32;
using br::rf;
using br::rne;

static_assert(kb::route::kWords == br::kWords && kb::route::kIds == br::kIds && kb::route::kW == br::kW &&
                  kb::route::kSel == br::kSel && kb::route::kSum == br::kSum && kb::route::kSelMin == br::kSelMin,
              "the Bolt route row: nb_moe.cl R_*, kernels::bolt::route and boltref disagree");
static_assert(kb::kUpKs == br::kMoeUpKs && kb::kDnKs == br::kMoeDnKs, "the MoE K splits: host and twin disagree");

constexpr uint32_t H = br::kHidden, NIN = br::kInProjN, XBC = br::kXbc, INNER = br::kInner;

struct Dev {
  l0::Context ctx{0};
  l0::Queue q{ctx};
  l0::Fence f{q};
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  std::vector<std::unique_ptr<l0::Module>> mods;
  l0::Mem upload(const void* p, size_t bytes) {
    l0::Mem m(ctx, l0::MemKind::Device, bytes);
    imm.copy(m.ptr(), p, bytes);
    return m;
  }
  template <class T>
  l0::Mem up(const std::vector<T>& v) { return upload(v.data(), v.size() * sizeof(T)); }
  l0::Mem zeros(size_t bytes) {
    l0::Mem m(ctx, l0::MemKind::Device, bytes);
    imm.fill(m.ptr(), 0u, bytes);
    return m;
  }
  template <class T>
  std::vector<T> read(const l0::Mem& m, size_t n, size_t off = 0) {
    std::vector<T> v(n);
    imm.copy(v.data(), static_cast<const uint8_t*>(m.ptr()) + off, n * sizeof(T));
    return v;
  }
  template <class T>
  void write(l0::Mem& m, const std::vector<T>& v) { imm.copy(m.ptr(), v.data(), v.size() * sizeof(T)); }
  l0::Module& mod(const std::string& variant) {
    mods.push_back(std::make_unique<l0::Module>(ctx, kernels::path(variant)));
    return *mods.back();
  }
  void run(l0::Kernel& k, uint32_t gx, uint32_t gy = 1) {
    l0::CmdList list = l0::CmdList::regular(ctx);
    list.launch(k, gx, gy);
    list.close();
    q.execute(list, &f);
    f.wait();
  }
};
struct Ctrl {
  l0::Mem m;
  explicit Ctrl(Dev& d) : m(d.ctx, l0::MemKind::Shared, sizeof(runtime::Control)) {
    std::memset(m.ptr(), 0, sizeof(runtime::Control));
  }
  runtime::Control* c() { return m.as<runtime::Control>(); }
  void set(uint32_t pos, uint32_t n) {
    c()->pos = pos;
    c()->n_active = n;
  }
};

// Arguments: an allocation, a raw device pointer, or a uint32_t value - in order.
inline void set1(l0::Kernel& k, uint32_t i, const l0::Mem& m) { k.arg_ptr(i, m.ptr()); }
inline void set1(l0::Kernel& k, uint32_t i, uint32_t v) { k.arg<uint32_t>(i, v); }
template <class... A>
void launch(Dev& d, l0::Module& m, const char* entry, uint32_t wg, uint32_t gx, uint32_t gy, const A&... a) {
  l0::Kernel k = m.kernel(entry);
  k.group_size(wg);
  uint32_t i = 0;
  (set1(k, i++, a), ...);
  d.run(k, gx, gy);
}

std::vector<float> vals(bool coarse, uint32_t t, uint32_t n, uint32_t seed, float step = 0.125f, float amp = 1.0f,
                        uint32_t base = 0) {
  std::vector<float> v(n);
  for (uint32_t i = 0; i < n; ++i) v[i] = br::fixture_val(coarse, t, base + i, seed, step, amp);
  return v;
}
std::vector<uint16_t> bf(const std::vector<float>& v) {
  std::vector<uint16_t> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = rne(v[i]);
  return o;
}
std::vector<uint32_t> bits(const std::vector<float>& v) {
  std::vector<uint32_t> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = br::as_u32(v[i]);
  return o;
}
template <class T>
std::vector<T> sub(const std::vector<T>& v, size_t a, size_t n) {
  return std::vector<T>(v.begin() + a, v.begin() + a + n);
}

// --- 1. the norm ------------------------------------------------------------------------------------------------------
void test_norm(Dev& d) {
  std::vector<float> w = vals(false, 1, H, 1, 0.125f, 0.5f);
  for (float& x : w) x = rf(1.0f + x);
  l0::Mem wb = d.up(w);
  const std::vector<uint16_t> fxx = br::hex16(fx::kNormX), fh = br::hex16(fx::kNormH);
  {   // _NF: the fixture's rows 0-3
    l0::Module& m = d.mod(kb::res_norm_variant(1, 0, 0, false));
    const std::vector<std::vector<float>> rows = {vals(true, 0, H, 10), vals(false, 0, H, 11, 0.125f, 4.0f),
                                                  vals(true, 0, H, 12, 128.0f), vals(true, 0, H, 13, 0x1p-20f)};
    for (uint32_t r = 0; r < 4; ++r) {
      const std::vector<uint16_t> h = bf(rows[r]);
      l0::Mem hb = d.up(h), sb = d.zeros(64), ob = d.zeros(H * 2), xb = d.zeros(H * 2);
      launch(d, m, "nb_res_norm", kb::kNormWg, kb::kNormWgs, 1, hb, sb, sb, wb, ob, xb);
      const br::NormOut tw = br::res_norm(h.data(), nullptr, 0, nullptr, 0, 1, w.data());
      CHECK(d.read<uint16_t>(xb, H) == tw.x);
      if (r != 1) CHECK(d.read<uint16_t>(xb, H) == sub(fxx, size_t(r) * H, H));   // coarse: torch's own bits
    }
  }
  {   // _S1_T1: the fixture's rows 6-7 (fc2 at S 1, shared_down at S 2 - the fixture's; the card's S 1 build reads one
      // shared_down slice: the two S 2 slices are summed on the host into one, which is exact for coarse slices)
    l0::Module& m = d.mod(kb::res_norm_variant(1, 1, 1, false));
    for (uint32_t r = 6; r < 8; ++r) {
      const std::vector<uint16_t> h = bf(vals(false, 0, H, 14 + (r - 4), 0.125f, 2.0f));
      const std::vector<float> s1 = vals(true, 2, H, 20 + r), b0 = vals(true, 3, H, 20 + r), b1 = vals(true, 4, H, 20 + r);
      std::vector<float> s2(H);
      for (uint32_t k = 0; k < H; ++k) s2[k] = b0[k] + b1[k];
      l0::Mem hb = d.up(h), sb = d.up(s1), s2b = d.up(s2), ob = d.zeros(H * 2), xb = d.zeros(H * 2);
      launch(d, m, "nb_res_norm", kb::kNormWg, kb::kNormWgs, 1, hb, sb, s2b, wb, ob, xb);
      const br::NormOut tw = br::res_norm(h.data(), s1.data(), 1, s2.data(), 1, 1, w.data());
      CHECK(d.read<uint16_t>(ob, H) == tw.h_out);
      CHECK(d.read<uint16_t>(xb, H) == tw.x);
      CHECK(d.read<uint16_t>(ob, H) == sub(fh, size_t(r - 4) * H, H));   // torch's fold, bitwise
    }
  }
  {   // M = 4, two sources: every row the twin's (one launch, 4 rows)
    l0::Module& m = d.mod("nb_res_norm_M4_K4096_W8_S1_T1");
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-2.f, 2.f);
    std::vector<float> s1(4 * H), s2(4 * H), hf(4 * H);
    for (float& v : s1) v = u(rng);
    for (float& v : s2) v = u(rng);
    for (float& v : hf) v = u(rng);
    const std::vector<uint16_t> h = bf(hf);
    l0::Mem hb = d.up(h), sb = d.up(s1), s2b = d.up(s2), ob = d.zeros(4 * H * 2), xb = d.zeros(4 * H * 2);
    launch(d, m, "nb_res_norm", kb::kNormWg, kb::kNormWgs, 4, hb, sb, s2b, wb, ob, xb);
    const br::NormOut tw = br::res_norm(h.data(), s1.data(), 1, s2.data(), 1, 4, w.data());
    CHECK(d.read<uint16_t>(ob, 4 * H) == tw.h_out);
    CHECK(d.read<uint16_t>(xb, 4 * H) == tw.x);
  }
  std::printf("  1. nb_res_norm: _NF (torch's coarse rows bitwise), the two-source fold (torch's bitwise), M = 4 - bitwise\n");
}

// --- 2. the conv -------------------------------------------------------------------------------------------------------
void test_conv(Dev& d) {
  const uint32_t T = fx::kConvT;
  const std::vector<uint16_t> X = bf(vals(false, 20, T * XBC, 5, 0.125f, 2.0f));
  const std::vector<float> w = vals(false, 21, XBC * 4, 5, 0.125f, 0.5f), b = vals(false, 22, XBC, 5, 0.125f, 0.5f);
  l0::Mem wb = d.up(w), bb = d.up(b);
  std::vector<uint16_t> ring(size_t(br::kRing) * XBC);
  for (size_t i = 0; i < ring.size(); ++i) ring[i] = uint16_t(0x3F80 + (i * 7919) % 4000);   // garbage: never read
  for (const std::vector<uint32_t>& ms : {std::vector<uint32_t>{1}, std::vector<uint32_t>{1, 2, 3, 4}}) {
    const bool multi = ms.size() > 1;
    l0::Module& m = d.mod(multi ? "nb_conv_M4_C10240_S1" : kb::conv_variant(1, 1));
    l0::Mem rb = d.up(ring), xb = d.zeros(size_t(4) * XBC * 2);
    std::vector<uint16_t> ring_t = ring;
    Ctrl c(d);
    uint32_t p = 0;
    for (size_t i = 0; p < T; ++i) {
      const uint32_t M = std::min(ms[i % ms.size()], T - p);
      std::vector<float> s(size_t(M) * NIN, 0.0f);
      for (uint32_t r = 0; r < M; ++r)
        for (uint32_t ch = 0; ch < XBC; ++ch) s[size_t(r) * NIN + br::kXbcOff + ch] = f32(X[size_t(p + r) * XBC + ch]);
      // the M4 build reads [S][4][N]: rows past M are never read (n_active = M)
      std::vector<float> s4(size_t(multi ? 4 : 1) * NIN, 0.0f);
      std::copy(s.begin(), s.end(), s4.begin());
      l0::Mem sb = d.up(s4);
      c.set(p, M);
      launch(d, m, "nb_conv", kb::kConvWg, XBC / kb::kConvWg, 1, c.m, sb, wb, bb, rb, xb);
      const std::vector<uint16_t> tw = br::conv_step(s.data(), 1, M, ring_t.data(), w.data(), b.data(), p);
      CHECK(d.read<uint16_t>(xb, size_t(M) * XBC) == tw);
      p += M;
    }
    CHECK(d.read<uint16_t>(rb, ring.size()) == ring_t);
  }
  std::printf("  2. nb_conv: 18 positions (the ring wraps at 16) as M = 1 launches and 1 / 2 / 3 / 4-row launches - bitwise\n");
}

// --- 3. the SSM step ---------------------------------------------------------------------------------------------------
struct MambaIn {
  std::vector<float> cw, cb, A, D, dt_bias, S0, slices;   // slices [T][18560]
};
MambaIn mamba_inputs(uint32_t T) {
  MambaIn in;
  std::mt19937 rng(17);
  std::uniform_real_distribution<float> u(-2.f, 2.f);
  in.cw = vals(false, 21, XBC * 4, 5, 0.125f, 0.5f);
  in.cb = vals(false, 22, XBC, 5, 0.125f, 0.5f);
  in.dt_bias = vals(false, 32, br::kHeads, 6, 0.125f, 4.0f);
  in.dt_bias[0] = -10.5f;
  in.dt_bias[1] = 8.5f;
  const std::vector<float> A_log = vals(false, 33, br::kHeads, 6, 0.125f, 4.0f);
  in.A.resize(br::kHeads);
  for (uint32_t h = 0; h < br::kHeads; ++h) in.A[h] = -common::exp_torch(A_log[h]);
  in.D = vals(false, 34, br::kHeads, 6, 0.125f, 2.0f);
  in.S0.resize(size_t(br::kHeads) * br::kHeadDim * br::kState);
  for (size_t i = 0; i < in.S0.size(); ++i) in.S0[i] = br::fixture_f32(35, uint32_t(i), 6);
  in.slices.resize(size_t(T) * NIN);
  for (float& v : in.slices) v = rf(u(rng));
  return in;
}
void test_ssm(Dev& d) {
  const uint32_t T = 18;
  const MambaIn in = mamba_inputs(T);
  l0::Mem Ab = d.up(in.A), Db = d.up(in.D), bb = d.up(in.dt_bias), cwb = d.up(in.cw), cbb = d.up(in.cb);
  for (int b16 = 0; b16 < 2; ++b16)
    for (int conv = 0; conv < 2; ++conv)
      for (uint32_t MM : {1u, 4u}) {
        l0::Module& m = d.mod(MM == 1 ? kb::ssm_variant(1, 1, conv, b16)
                                      : std::string("nb_ssm_M4_H128_P64_N128_G8_S1") + (conv ? "_C" : "") + (b16 ? "_B16" : ""));
        std::vector<uint16_t> ring_t(size_t(br::kRing) * XBC, 0), st16(in.S0.size());
        std::vector<float> st = in.S0;
        for (size_t i = 0; i < st16.size(); ++i) st16[i] = rne(st[i]);
        l0::Mem stb = b16 ? d.up(st16) : d.up(st), rb = d.up(ring_t), yb = d.zeros(size_t(4) * INNER * 2),
                xb = d.zeros(size_t(4) * XBC * 2);
        Ctrl c(d);
        uint32_t p = 0;
        for (size_t i = 0; p < T; ++i) {
          const uint32_t M = std::min<uint32_t>(MM == 1 ? 1 : uint32_t(1 + i % 4), T - p);
          std::vector<float> s(size_t(MM) * NIN, 0.0f);
          std::copy(in.slices.begin() + size_t(p) * NIN, in.slices.begin() + size_t(p + M) * NIN, s.begin());
          const std::vector<uint16_t> xc = br::conv_step(s.data(), 1, M, ring_t.data(), in.cw.data(), in.cb.data(), p);
          const std::vector<uint16_t> yt = br::ssm_step(s.data(), 1, M, xc.data(), in.A.data(), in.D.data(),
                                                        in.dt_bias.data(), st.data(), b16 ? st16.data() : nullptr);
          l0::Mem sb = d.up(s);
          c.set(p, M);
          if (conv) {
            launch(d, m, "nb_ssm", kb::kSsmWg, kb::ssm_groups(), 1, c.m, sb, cwb, cbb, rb, Ab, Db, bb, stb, yb);
          } else {
            std::vector<uint16_t> xc4(size_t(MM) * XBC, 0);
            std::copy(xc.begin(), xc.end(), xc4.begin());
            d.write(xb, xc4);
            launch(d, m, "nb_ssm", kb::kSsmWg, kb::ssm_groups(), 1, c.m, sb, xb, Ab, Db, bb, stb, yb);
          }
          CHECK(d.read<uint16_t>(yb, size_t(M) * INNER) == yt);
          p += M;
        }
        if (b16)
          CHECK(d.read<uint16_t>(stb, st16.size()) == st16);
        else
          CHECK(bits(d.read<float>(stb, st.size())) == bits(st));
        if (conv) CHECK(d.read<uint16_t>(rb, ring_t.size()) == ring_t);
        std::printf("  3. nb_ssm%s %s, M %u: 18 positions (the ring wraps; head 0 at dt_bias -10.5) - y, the state%s "
                    "bitwise\n", conv ? "_C" : "", b16 ? "bf16 state" : "fp32 state", MM, conv ? ", the ring" : "");
      }
}

// --- 4. the gated norm --------------------------------------------------------------------------------------------------
void test_gnorm(Dev& d) {
  std::vector<float> y = vals(false, 40, 4 * INNER, 7, 0.125f, 2.0f);
  y[size_t(1) * INNER + 2 * 1024 + 5] = 60.0f;
  const std::vector<uint16_t> yb_ = bf(y), z = bf(vals(false, 41, 4 * INNER, 7, 0.125f, 4.0f));
  std::vector<float> w = vals(false, 42, INNER, 7, 0.125f, 0.5f);
  for (float& x : w) x = rf(1.0f + x);
  l0::Module& m = d.mod(kb::gnorm_variant(1, 1));
  std::vector<uint16_t> got;
  for (uint32_t r = 0; r < 4; ++r) {
    std::vector<float> s(NIN, 0.0f);
    for (uint32_t c = 0; c < INNER; ++c) s[c] = f32(z[size_t(r) * INNER + c]);
    l0::Mem sb = d.up(s), yy = d.up(sub(yb_, size_t(r) * INNER, INNER)), wb = d.up(w), ob = d.zeros(INNER * 2);
    launch(d, m, "nb_gnorm", kb::kGnormWg, INNER / br::kGnormGroup, 1, sb, yy, wb, ob);
    const std::vector<uint16_t> o = d.read<uint16_t>(ob, INNER);
    CHECK(o == br::gnorm(s.data(), 1, 1, yb_.data() + size_t(r) * INNER, w.data()));
    got.insert(got.end(), o.begin(), o.end());
  }
  const std::vector<uint16_t> want = br::hex16(fx::kGnorm);
  size_t diff = 0;
  for (size_t i = 0; i < want.size(); ++i) diff += got[i] != want[i];
  std::printf("  4. nb_gnorm: 4 rows bitwise the twin; %zu of %zu off torch's rmsnorm_gated (the gate's exp)\n", diff,
              want.size());
}

// --- 5. the router and the route -----------------------------------------------------------------------------------------
void test_route(Dev& d) {
  l0::Module& mr = d.mod(kb::router_variant(1));
  l0::Module& mt = d.mod(kb::route_variant(1));
  {   // the router at E 496 and 240 against the twin; the arm `router` (_S1_R) gives the same logits
    l0::Module& mnr = d.mod(kb::res_norm_variant(1, 1, 0, true));
    l0::Module& mn = d.mod(kb::res_norm_variant(1, 1, 0, false));
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    for (uint32_t E : {496u, 240u}) {
      std::vector<float> sl(H), gwf(size_t(E) * H), w(H);
      for (float& v : sl) v = u(rng);
      for (float& v : gwf) v = u(rng) * 0.05f;
      for (float& v : w) v = rf(1.0f + 0.5f * u(rng));
      const std::vector<uint16_t> h = bf(vals(false, 0, H, 77, 0.125f, 2.0f)), gw = bf(gwf);
      std::vector<uint16_t> tiles(size_t(H) * E);
      common::repack_bf16_tiled(gw.data(), H, E, tiles.data());
      l0::Mem hb = d.up(h), sb = d.up(sl), wb = d.up(w), tb = d.up(tiles), o1 = d.zeros(H * 2), x1 = d.zeros(H * 2),
              o2 = d.zeros(H * 2), x2 = d.zeros(H * 2), l1 = d.zeros(E * 4), l2 = d.zeros(E * 4);
      launch(d, mnr, "nb_res_norm", kb::kNormWg, kb::kNormWgs + E / 16, 1, hb, sb, sb, wb, o1, x1, tb, l1, E);
      launch(d, mn, "nb_res_norm", kb::kNormWg, kb::kNormWgs, 1, hb, sb, sb, wb, o2, x2);
      launch(d, mr, "nb_router", kb::kRouterWg, E / 16, 1, x2, tb, l2, E);
      const std::vector<uint16_t> x = d.read<uint16_t>(x2, H);
      CHECK(d.read<uint16_t>(x1, H) == x && d.read<uint16_t>(o1, H) == d.read<uint16_t>(o2, H));
      CHECK(bits(d.read<float>(l1, E)) == bits(d.read<float>(l2, E)));
      CHECK(bits(d.read<float>(l2, E)) == bits(br::router(x.data(), tiles.data(), E)));
    }
  }
  struct C {
    uint32_t E, seed;
    const char *logits, *ids, *sel;
  };
  for (const C& c : {C{496, 8, fx::kRouteLogits496, fx::kRouteIds496, fx::kRouteSel496},
                     C{240, 9, fx::kRouteLogits240, fx::kRouteIds240, fx::kRouteSel240}}) {
    std::vector<float> bias(c.E);
    for (uint32_t e = 0; e < c.E; ++e) bias[e] = br::fixture_f32(52, e, c.seed) * 0.04f + 4.5f;
    const std::vector<uint32_t> lg = br::hex32(c.logits), ids = br::hex32(c.ids), sel = br::hex32(c.sel);
    l0::Mem bb = d.up(bias);
    for (uint32_t r = 0; r < 32; ++r) {
      std::vector<float> l(c.E);
      for (uint32_t e = 0; e < c.E; ++e) l[e] = br::as_f32(lg[size_t(r) * c.E + e]);
      l0::Mem lb = d.up(l), ro = d.zeros(br::kWords * 4);
      launch(d, mt, "nb_route", kb::kRouteWg, 1, 1, lb, bb, ro, c.E);
      const std::vector<uint32_t> row = d.read<uint32_t>(ro, br::kWords);
      const br::Route rt = br::route(l.data(), bias.data(), c.E);
      CHECK(row == std::vector<uint32_t>(rt.row, rt.row + br::kWords));
      for (uint32_t k = 0; k < 9; ++k) CHECK_EQ(row[br::kIds + k], ids[size_t(r) * 9 + k]);    // torch's set
      for (uint32_t k = 0; k < 10; ++k) CHECK_EQ(row[br::kSel + k], sel[size_t(r) * 10 + k]);  // torch's sel
    }
  }
  const std::vector<uint32_t> cl = br::hex32(fx::kCraftLogits), cbias = br::hex32(fx::kCraftBias);
  for (uint32_t r = 0; r < fx::kCraftN; ++r) {
    const uint32_t E = fx::kCraftE[r];
    std::vector<float> l(E), b(E);
    for (uint32_t e = 0; e < E; ++e) {
      l[e] = br::as_f32(cl[size_t(r) * 496 + e]);
      b[e] = br::as_f32(cbias[size_t(r) * 496 + e]);
    }
    l0::Mem lb = d.up(l), bb = d.up(b), ro = d.zeros(br::kWords * 4);
    launch(d, mt, "nb_route", kb::kRouteWg, 1, 1, lb, bb, ro, E);
    const br::Route rt = br::route(l.data(), b.data(), E);
    CHECK(d.read<uint32_t>(ro, br::kWords) == std::vector<uint32_t>(rt.row, rt.row + br::kWords));
  }
  {   // M = 4 rows of one launch, each the twin's (the logits' row stride is E)
    l0::Module& m4 = d.mod("nb_route_M4_T9");
    const uint32_t E = 496;
    std::vector<float> bias(E);
    for (uint32_t e = 0; e < E; ++e) bias[e] = br::fixture_f32(52, e, 8) * 0.04f + 4.5f;
    const std::vector<uint32_t> lg = br::hex32(fx::kRouteLogits496);
    std::vector<float> l(4 * E);
    for (size_t i = 0; i < l.size(); ++i) l[i] = br::as_f32(lg[i]);
    l0::Mem lb = d.up(l), bb = d.up(bias), ro = d.zeros(4 * br::kWords * 4);
    launch(d, m4, "nb_route", kb::kRouteWg, 1, 4, lb, bb, ro, E);
    const std::vector<uint32_t> rows = d.read<uint32_t>(ro, 4 * br::kWords);
    for (uint32_t r = 0; r < 4; ++r) {
      const br::Route rt = br::route(l.data() + size_t(r) * E, bias.data(), E);
      CHECK(sub(rows, size_t(r) * br::kWords, br::kWords) == std::vector<uint32_t>(rt.row, rt.row + br::kWords));
    }
  }
  std::printf("  5. nb_router (E 496 / 240) = the twin = the _R arm's logits; nb_route on torch's logits (its ids and sel), "
              "the crafted rows, M = 4 - bitwise\n");
}

// --- 6. the experts ------------------------------------------------------------------------------------------------------
void test_moe(Dev& d) {
  const uint32_t L = br::kLatent, E = 240;
  std::mt19937 rng(23);
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  for (uint32_t I : {1536u, 2048u, 2304u, 2560u}) {
    // nine distinct ids, ascending, and their weights; only their blocks are filled
    const uint32_t ids[9] = {0, 17, 42, 99, 120, 150, 200, 230, 239};
    std::vector<uint32_t> blocks(size_t(E) * br::expert_block_words(I), 0u);
    for (uint32_t e : ids)
      br::expert_block(br::gptq_of(500 + 2 * e + I, L, I, 7), br::gptq_of(501 + 2 * e + I, I, L, 7),
                       blocks.data() + size_t(e) * br::expert_block_words(I));
    std::vector<float> latf(L);
    for (float& v : latf) v = u(rng);
    const std::vector<uint16_t> lat = bf(latf);
    br::Route rt;
    std::vector<uint32_t> row(br::kWords, 0u);
    for (uint32_t k = 0; k < 9; ++k) {
      rt.ids[k] = ids[k];
      rt.w[k] = 0.3f + 0.1f * float(k);
      row[br::kIds + k] = ids[k];
      row[br::kW + k] = br::as_u32(rt.w[k]);
    }
    l0::Module& mu = d.mod(kb::moe_up_variant(1, I, false));
    l0::Module& md = d.mod(kb::moe_down_variant(1, I));
    l0::Mem rb = d.up(row), lb = d.up(lat), wb = d.up(blocks), ab = d.zeros(size_t(9) * I * 2), ob = d.zeros(L * 2);
    launch(d, mu, "nb_moe_up", kb::moe_up_wg(), kb::moe_up_groups(I), 1, rb, lb, wb, ab, E);
    launch(d, md, "nb_moe_down", kb::moe_down_wg(), kb::moe_down_groups(), 1, rb, ab, wb, ob, E);
    const br::MoeOut o = br::moe(rt, lat.data(), blocks.data(), I);
    CHECK(d.read<uint16_t>(ab, size_t(9) * I) == o.a);
    CHECK(d.read<uint16_t>(ob, L) == o.r);
    // The arm route: _R from logits that rank these ids top; its route row = nb_route's and its a = nb_moe_up's.
    std::vector<float> logits(E, -6.0f), bias(E, 0.0f);
    for (uint32_t k = 0; k < 9; ++k) logits[ids[k]] = 1.0f + 0.25f * float(k);
    l0::Module& mr = d.mod(kb::moe_up_variant(1, I, true));
    l0::Module& mt = d.mod(kb::route_variant(1));
    l0::Mem lgb = d.up(logits), bsb = d.up(bias), r1 = d.zeros(br::kWords * 4), r2 = d.zeros(br::kWords * 4),
            a1 = d.zeros(size_t(9) * I * 2), a2 = d.zeros(size_t(9) * I * 2);
    launch(d, mr, "nb_moe_up", kb::moe_up_wg(), kb::moe_up_groups(I), 1, lgb, bsb, r1, lb, wb, a1, E);
    launch(d, mt, "nb_route", kb::kRouteWg, 1, 1, lgb, bsb, r2, E);
    launch(d, mu, "nb_moe_up", kb::moe_up_wg(), kb::moe_up_groups(I), 1, r2, lb, wb, a2, E);
    CHECK(d.read<uint32_t>(r1, br::kWords) == d.read<uint32_t>(r2, br::kWords));
    CHECK(d.read<uint16_t>(a1, size_t(9) * I) == d.read<uint16_t>(a2, size_t(9) * I));
    std::printf("  6. I %u: nb_moe_up (9 slots, relu2) and nb_moe_down (the ascending-id combine) bitwise the twin; _R = "
                "nb_route + nb_moe_up bitwise\n", I);
  }
}

// --- 7. the small kernels and the head ---------------------------------------------------------------------------------
void test_small(Dev& d) {
  std::mt19937 rng(5);
  std::uniform_real_distribution<float> u(-3.f, 3.f);
  {
    std::vector<float> sl(br::kFc1N);
    for (float& v : sl) v = u(rng);
    l0::Module& m = d.mod(kb::sh_act_variant(1, 1));
    l0::Mem sb = d.up(sl), lb = d.zeros(br::kLatent * 2), shb = d.zeros(br::kShared * 2);
    launch(d, m, "nb_sh_act", kb::kActWg, (br::kFc1N + 255) / 256, 1, sb, lb, shb);
    std::vector<uint16_t> lat(br::kLatent), sh(br::kShared);
    br::sh_act(sl.data(), 1, 1, lat.data(), sh.data());
    CHECK(d.read<uint16_t>(lb, br::kLatent) == lat && d.read<uint16_t>(shb, br::kShared) == sh);
  }
  {
    std::vector<float> sl(br::kQkvN);
    for (float& v : sl) v = u(rng);
    l0::Module& m = d.mod(kb::attn_prep_variant(1, 1));
    l0::Mem sb = d.up(sl);
    for (uint32_t pos : {0u, 262143u}) {
      Ctrl c(d);
      c.set(pos, 1);
      l0::Mem qb = d.zeros(4096 * 4), kk = d.zeros(size_t(pos + 1) * 512), vv = d.zeros(size_t(pos + 1) * 512);
      launch(d, m, "nb_attn_prep", kb::kPrepWg, (br::kQkvN + 255) / 256, 1, c.m, sb, qb, kk, vv);
      const br::Prep want = br::attn_prep(sl.data(), 1, 1, 0);
      CHECK(bits(d.read<float>(qb, 4096)) == bits(want.q));
      CHECK(d.read<uint16_t>(kk, 256, size_t(pos) * 512) == want.k && d.read<uint16_t>(vv, 256, size_t(pos) * 512) == want.v);
    }
  }
  {
    std::vector<uint16_t> tiles(size_t(H) * br::kVocab);
    for (uint16_t& t : tiles) t = uint16_t(rng());
    l0::Module& m = d.mod(kb::embed_variant(1));
    l0::Mem tb = d.up(tiles);
    for (uint32_t id : {0u, 7u, 131071u}) {
      Ctrl c(d);
      c.set(0, 1);
      c.c()->cur_token[0] = id;
      l0::Mem hb = d.zeros(H * 2);
      launch(d, m, "nb_embed_gather", kb::kEmbedWg, 1, 1, c.m, tb, hb);
      CHECK(d.read<uint16_t>(hb, H) == br::embed_row(tiles.data(), id));
    }
  }
  {   // the argmax at 131072 (every id real): maxima at 0 and 131071; a tie at the top to the lower id
    l0::Module& m1 = d.mod(kb::argmax1_variant(1));
    l0::Module& m2 = d.mod(kb::argmax2_variant());
    std::vector<float> lg = vals(false, 90, br::kVocab, 4);
    for (int pass = 0; pass < 3; ++pass) {
      std::vector<float> l = lg;
      if (pass == 0) l[0] = 50.0f;
      if (pass == 1) l[br::kVocab - 1] = 100.0f;
      if (pass == 2) l[7] = l[9] = 200.0f;
      l0::Mem lb = d.up(l), pb = d.zeros(size_t(kb::argmax_groups()) * 2 * 4);
      Ctrl c(d);
      c.set(5, 1);
      launch(d, m1, "argmax_stage1", kb::kArgmaxWg, kb::argmax_groups(), 1, lb, pb);
      launch(d, m2, "argmax_stage2", kb::kArgmaxWg, 1, 1, c.m, pb);
      CHECK_EQ(c.c()->out_token[0], br::argmax(l.data()));
      CHECK_EQ(c.c()->out_token[0], pass == 0 ? 0u : pass == 1 ? br::kVocab - 1 : 7u);
      CHECK_EQ(c.c()->pos, 6u);
    }
  }
  std::printf("  7. nb_sh_act, nb_attn_prep at 0 / 262143, nb_embed_gather, the argmax at 131072 - bitwise\n");
}

// --- 8. attention (spec 23c's binaries, Decision 15) -------------------------------------------------------------------------
void test_attention(Dev& d) {
  const uint32_t HD = br::kHd, KV = br::kKvHeads, QH = br::kQHeads;
  const std::vector<uint16_t> ones(size_t(QH) * HD, uint16_t(0x3F80));
  l0::Mem gb = d.up(ones);
  l0::Module& me = d.mod(kb::attn_variant(1, true));
  for (uint32_t ci = 0; ci < 3; ++ci) {
    const uint32_t L = fx::kAttnL[ci];
    const std::vector<uint16_t> q = bf(vals(true, 70, QH * HD, ci, 0.5f));
    std::vector<float> qf(q.size());
    for (size_t i = 0; i < q.size(); ++i) qf[i] = f32(q[i]);
    const std::vector<uint16_t> kk = bf(vals(true, 71, L * KV * HD, 9)), vv = bf(vals(true, 72, L * KV * HD, 9));
    Ctrl c(d);
    c.set(L - 1, 1);
    l0::Mem qb = d.up(qf), kb_ = d.up(kk), vb = d.up(vv), sb = d.zeros(size_t(QH) * L * 4), ob = d.zeros(size_t(QH) * HD * 2);
    launch(d, me, "mg_attn_eager", kb::kAttnWg, QH, 1, c.m, qb, kb_, vb, gb, sb, ob, L);
    std::vector<uint16_t> tw(size_t(QH) * HD);
    br::attention_eager(q.data(), kk.data(), vv.data(), L - 1, tw.data());
    CHECK(d.read<uint16_t>(ob, size_t(QH) * HD) == tw);
  }
  l0::Module& mf = d.mod(kb::attn_variant(1, false));
  for (uint32_t L : {1u, 2u, 300u, 4096u, 65536u, 262143u}) {
    std::mt19937 rng(L);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    std::vector<float> qf(size_t(QH) * HD);
    for (float& v : qf) v = rf(u(rng) * 2.0f);
    std::vector<uint16_t> q(qf.size()), kk(size_t(L) * KV * HD), vv(size_t(L) * KV * HD);
    for (size_t i = 0; i < q.size(); ++i) q[i] = rne(qf[i]);
    for (uint16_t& x : kk) x = rne(u(rng));
    for (uint16_t& x : vv) x = rne(u(rng));
    Ctrl c(d);
    c.set(L - 1, 1);
    l0::Mem qb = d.up(qf), kb_ = d.up(kk), vb = d.up(vv), part = d.zeros(size_t(QH) * kb::kAttnTgt * kb::kAttnPart * 4),
            ob = d.zeros(size_t(QH) * HD * 2);
    launch(d, mf, "mg_attn_decode", kb::kAttnWg, KV, kb::kAttnTgt, c.m, qb, kb_, vb, part);
    launch(d, mf, "mg_attn_reduce", kb::kAttnWg, QH, 1, c.m, part, gb, ob);
    std::vector<double> want(size_t(QH) * HD);
    br::attention_fp64(q.data(), kk.data(), vv.data(), L - 1, want.data());
    const std::vector<uint16_t> got = d.read<uint16_t>(ob, size_t(QH) * HD);
    double worst = 1.0;
    for (uint32_t h = 0; h < QH; ++h) {
      double dot = 0, na = 0, nb = 0;
      for (uint32_t dd = 0; dd < HD; ++dd) {
        const double a = f32(got[size_t(h) * HD + dd]), b = want[size_t(h) * HD + dd];
        dot += a * b;
        na += a * a;
        nb += b * b;
      }
      worst = std::min(worst, dot / std::sqrt(na * nb + 1e-300));
    }
    CHECK(worst >= 0.99999);
    std::printf("  8. mg_attn (flash, a ones gate) at %u keys: worst head cosine %.7f against fp64 (bar 0.99999)\n", L, worst);
  }
  std::printf("  8. mg_attn_eager (a ones gate) at 6 / 300 / 3000 keys bitwise the twin\n");
}

}  // namespace

int main() {
  Dev d;
  std::printf("bolt_kernels_test on %s\n", d.ctx.name().c_str());
  test_norm(d);
  test_conv(d);
  test_ssm(d);
  test_gnorm(d);
  test_route(d);
  test_moe(d);
  test_small(d);
  test_attention(d);
  std::printf("bolt_kernels_test OK\n");
  return 0;
}
