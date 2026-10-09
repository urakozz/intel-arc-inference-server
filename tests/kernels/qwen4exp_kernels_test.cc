// Spec 21c Task 2 Step 4 (F2) and Task 5 Step 1: Qwen3.8-Flash-Next's decode kernels on the card, at the real
// shapes, against tests/kernels/qwen4exp_ref.h. Needs a B70, no checkpoint (synthetic weights); exits 77 (SKIP)
// without a GPU. Every binary of the CMake block's spec 21c section (kernels::qwen4exp names) - BITWISE unless said:
//
//   1. q4_hc_combine_norm _E, _S4, _S1, _Y, _X, _Y_NN (H and xn; _NN leaves xn untouched); q4_hc_up_mix _I and
//      plain (x and the inject weights)
//   2. q4_ple_gather _BF16 and _F32 from REAL host-USM ranges (16 per table + 16 of scales, l0::MemKind::Host at
//      2 MiB, the device pointer table u64 [32] as loader/qwen4exp_ple_usm.cc builds it; the real multipliers,
//      head sizes q4_ple_primes(100000, 16, 0)): positions 0..20 one launch each, EOS 248044 and id 248319 in the
//      sequence - the ids against loader::q4_ple_ids through q4ref::ple_history, the rows against
//      q4ref::ple_row, the id ring; q4_ple_check: a tag per 2 MiB page then the data's first words, read back
//      through the pointer table; q4_ple_block over 12 positions through the 16-slot conv ring
//   3. q4_qsa_prep at 0..3, 2044..2051, 8996..8999 (blocks 0, 511, 512, 2249: q, the 8-slot tail, the compressed
//      keys; loader::q4_rope_table); q4_qsa_score at 600 blocks and at 65536 (max_len 262144: the grid spans
//      every block, the early-out past n); q4_qsa_select - Review Focus 2's rows: n 512 at tails 0..3 (p 2047 ..
//      2050: 2049, 2050 identity), n 513 at tails 0..3 (p 2051 the first cut with (p + 1) % 4 == 0 - p's own
//      block a candidate, losing and winning - up to 2054, tail 3), exact ties across the 512 / 513 cut (the
//      lower block), the cut inside zeros, the fixture's four crafted rows, n 600, n 65536 once: the list
//      ascending, the count 2048 + tail, the diagnostic
//   4. q4_route: random rows, the fixture's four (a tie for the 10th, all -1e30 but ten), a crafted tie - the
//      whole route row; q4_moe _SH4 and _SHB over 512 real expert blocks (1.34 GB of random int4): every slot,
//      each routed slot alone and the shared slot alone (the others' weights zero, Review Focus 4), a replay
//   5. q4_qsa_attn_eager over q4ref::qsa_select's lists at p 1, 2050 (identity), 2051 (the first cut), 2400, 9000
//   6. prep_gated_head_M1_SIG: random and the fixture's (torch's) case
//   7. flash (q4_qsa_attn + q4_qsa_reduce) against q4ref::qsa_attn_fp64 x rne(sigmoid(gate)) at cosine >= 0.99999
//      (spec 6 K1's bar) at p 1, 2050, 2051, 9000, 131072 (caches [max_len][2][256] bf16); and at p 100, 1000,
//      2050 (identity lists) against attn_v2_M1_T32_Q24KV2's dense result within the same bar - the gate
//      +30 on both sides (sigmoid 1: attn_v2 multiplies by the fp32 sigmoid, q4_qsa_reduce by its bf16 rounding,
//      so with the gate saturated both outputs are rne(o))
//
// Opt-in modes (each runs alone):
//   --bench-attn       decision 10's evidence: the sparse kernel (q4_qsa_attn + q4_qsa_reduce over the identity
//                      list) against attn_v2 (attn_decode_v2 + attn_reduce_v2) at depths 512 / 1024 / 2048 -
//                      lists of 100 launch pairs, 3 warm-up replays each, then 3 rounds of 20 replays with A and B
//                      interleaved replay by replay; the median round, printed in us per launch pair
//   --ple-rate [GB]    plan Task 5 Step 1: 16 + 16 host-USM ranges per 21b's rule (head sizes the real primes
//                      q4_ple_primes(20000000, 16, 0) scaled so int8 rows + bf16 scales total GB; default 16, 51.8
//                      the real table), refused by 21b's host-fit rule (exit 77); q4_ple_gather_M1_BF16 alone in
//                      a replayed regular list at M = 1, 10^4 replays with a random token written into Control's
//                      cur_token (and pos advanced) between replays, the readback of ids and e checked after
//                      every replay; device 0, then device 1 (l0::Context(c0, 1), the same host ranges), then both
//                      concurrently (each replay submits to both queues before waiting on either). Prints the
//                      median us per token, the implied bus rate (16 rows x 162 B a token) and whether every row read
//                      back equal. "wide": 16 launches per replay, each with its own Control block, id ring and e
//                      row (16 tokens = 256 distinct rows a replay; the in-order list runs them back to back), the
//                      readback of one launch a replay (launch i % 16)
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "kernels/kernels.h"
#include "kernels/qwen4exp_fixture.h"
#include "kernels/qwen4exp_kernels.h"
#include "kernels/qwen4exp_ref.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "loader/qwen4exp_ple.h"
#include "loader/qwen4exp_ple_hash.h"
#include "loader/qwen4exp_repack.h"
#include "model/qwen4exp.h"
#include "runtime/control.h"

namespace {

namespace qk = kernels::qwen4exp;
namespace qr = q4ref;
namespace fx = qwen4exp_fixture;
using qr::f32;
using qr::rf;
using qr::rne;

static_assert(qk::route::kWords == qr::kWords && qk::route::kIds == qr::kIds && qk::route::kWeights == qr::kWeights &&
                  qk::route::kSharedGate == qr::kSharedGate && qk::route::kP10 == qr::kP10 && qk::route::kP11 == qr::kP11,
              "the route row: q4_moe.cl R_*, kernels::qwen4exp::route and q4ref disagree");
static_assert(qk::kListMax == qr::kListMax && qk::kTailSlots == qr::kTailSlots && qk::kTopBlocks == qr::kTopBlocks &&
                  qk::kSlots == qr::kSlots && qk::kPleRing == qr::kPleRing,
              "kernels::qwen4exp and q4ref disagree on a device constant");

constexpr uint32_t D = qk::kHidden, HCN = qk::kHcN, HD = qk::kHd, QH = qk::kQHeads, KVH = qk::kKvHeads;
constexpr uint32_t QN = QH * HD, KVN = KVH * HD;

int ulps16(uint16_t a, uint16_t b) {
  auto key = [](uint16_t v) { return (v & 0x8000) ? -int(v & 0x7FFF) : int(v); };
  return std::abs(key(a) - key(b));
}
std::vector<uint16_t> random_bf16(size_t n, float lo, float hi, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<uint16_t> v(n);
  for (uint16_t& e : v) e = rne(d(rng));
  return v;
}
std::vector<float> random_f32(size_t n, float lo, float hi, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<float> v(n);
  for (float& e : v) e = d(rng);
  return v;
}
std::vector<float> bf16_vals(size_t n, float lo, float hi, uint32_t seed) {   // fp32 holding bf16 values
  const std::vector<uint16_t> b = random_bf16(n, lo, hi, seed);
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = f32(b[i]);
  return v;
}
std::vector<uint32_t> random_blocks(size_t words, uint32_t seed) {
  std::vector<uint32_t> v(words);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> sd(0.01f, 0.05f);
  for (size_t t = 0; t < words / 136; ++t) {
    uint32_t* tile = v.data() + t * 136;
    for (int i = 0; i < 128; ++i) tile[i] = rng();
    for (int i = 0; i < 8; ++i)
      tile[128 + i] = uint32_t(common::f32_to_f16(sd(rng))) | (uint32_t(common::f32_to_f16(sd(rng))) << 16);
  }
  return v;
}

int failures = 0;
// Bitwise comparisons: "bitwise (n values)" or the first differing index and how many differ.
struct Diff {
  size_t n = 0, differ = 0, first = SIZE_MAX;
  uint64_t got_first = 0, want_first = 0;
  void add(size_t i, uint64_t got, uint64_t want) {
    ++n;
    if (got == want) return;
    if (differ++ == 0) {
      first = i;
      got_first = got;
      want_first = want;
    }
  }
  template <class T>
  void all(const std::vector<T>& got, const std::vector<T>& want, size_t base = 0) {
    if (got.size() != want.size()) {
      add(base, got.size(), want.size());
      return;
    }
    for (size_t i = 0; i < got.size(); ++i) add(base + i, uint64_t(got[i]), uint64_t(want[i]));
  }
  void all(const std::vector<float>& got, const std::vector<float>& want, size_t base = 0) {
    std::vector<uint32_t> g(got.size()), w(want.size());
    for (size_t i = 0; i < got.size(); ++i) g[i] = qr::as_u32(got[i]);
    for (size_t i = 0; i < want.size(); ++i) w[i] = qr::as_u32(want[i]);
    all(g, w, base);
  }
};
void report(const std::string& what, const Diff& d) {
  if (d.differ == 0) {
    std::printf("  %s: bitwise (%zu values)\n", what.c_str(), d.n);
    return;
  }
  std::printf("  FAIL %s: %zu of %zu differ, the first at index %zu (got 0x%llx, want 0x%llx)\n", what.c_str(), d.differ, d.n,
              d.first, (unsigned long long)d.got_first, (unsigned long long)d.want_first);
  ++failures;
}
void report_bool(const std::string& what, bool ok, const std::string& detail = "") {
  std::printf("  %s%s: %s%s\n", ok ? "" : "FAIL ", what.c_str(), ok ? "exact" : "differs", detail.c_str());
  if (!ok) ++failures;
}

struct Dev {
  l0::Context ctx{0};
  l0::Queue q{ctx};
  l0::Fence f{q};
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  l0::Mem upload(const void* p, size_t bytes) {
    l0::Mem m(ctx, l0::MemKind::Device, bytes);
    imm.copy(m.ptr(), p, bytes);
    return m;
  }
  template <class T>
  l0::Mem upload(const std::vector<T>& v) {
    return upload(v.data(), v.size() * sizeof(T));
  }
  l0::Mem zeros(size_t bytes) {
    l0::Mem m(ctx, l0::MemKind::Device, bytes);
    imm.fill(m.ptr(), 0u, bytes);
    return m;
  }
  l0::Mem filled(size_t bytes, uint32_t pattern) {
    l0::Mem m(ctx, l0::MemKind::Device, bytes);
    imm.fill(m.ptr(), pattern, bytes);
    return m;
  }
  template <class T>
  void write(const l0::Mem& m, const std::vector<T>& v) {
    imm.copy(m.ptr(), v.data(), v.size() * sizeof(T));
  }
  template <class T>
  std::vector<T> read(const l0::Mem& m, size_t n, size_t off = 0) {
    std::vector<T> v(n);
    imm.copy(v.data(), static_cast<const uint8_t*>(m.ptr()) + off, n * sizeof(T));
    return v;
  }
  void run(l0::Kernel& k, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1) {
    l0::CmdList list = l0::CmdList::regular(ctx);
    list.launch(k, gx, gy, gz);
    list.close();
    q.execute(list, &f);
    f.wait();
  }
};

// A kernel argument: a device / host / shared allocation (its pointer) or a u32 by value.
struct A {
  bool ptr;
  const void* p;
  uint32_t u;
  A(const l0::Mem& m) : ptr(true), p(m.ptr()), u(0) {}   // NOLINT
  A(uint32_t v) : ptr(false), p(nullptr), u(v) {}        // NOLINT
};
l0::Kernel bind(l0::Module& m, const char* name, uint32_t wg, std::initializer_list<A> args) {
  l0::Kernel k = m.kernel(name);
  k.group_size(wg);
  uint32_t i = 0;
  for (const A& a : args) {
    if (a.ptr) k.arg_ptr(i, a.p);
    else k.arg<uint32_t>(i, a.u);
    ++i;
  }
  return k;
}
void launch(Dev& d, l0::Module& m, const char* name, uint32_t wg, std::array<uint32_t, 3> grid, std::initializer_list<A> args) {
  l0::Kernel k = bind(m, name, wg, args);
  d.run(k, grid[0], grid[1], grid[2]);
}

// The shared Control block (runtime::Control: pos word 0, n_active 1, cur_token 2.. - the CTRL_* defines).
struct Ctl {
  l0::Mem m;
  runtime::Control* c;
  explicit Ctl(l0::Context& ctx) : m(ctx, l0::MemKind::Shared, sizeof(runtime::Control)), c(m.as<runtime::Control>()) {
    std::memset(c, 0, sizeof(*c));
    c->n_active = 1;
  }
  void at(uint32_t pos, uint32_t n_active = 1) {
    c->pos = pos;
    c->n_active = n_active;
  }
};

// ---- 1. hyper-connections ------------------------------------------------------------------------------------------
void hc_tests(Dev& d) {
  Ctl ctl(d.ctx);
  ctl.at(7);
  const std::vector<uint16_t> H0 = random_bf16(HCN, -2.f, 2.f, 1);
  const std::vector<float> w = bf16_vals(HCN, 0.5f, 1.5f, 2);
  const std::vector<float> inj = {rf(0.75f), rf(1.25f), rf(0.5f), rf(1.875f)};
  const std::vector<float> slices = random_f32(size_t(4) * D, -0.5f, 0.5f, 3);
  const std::vector<uint16_t> ybf = random_bf16(D, -1.f, 1.f, 4), emb = random_bf16(D, -0.1f, 0.1f, 5);
  l0::Mem ib = d.upload(inj), wb = d.upload(w), sl = d.upload(slices), yb = d.upload(ybf), eb = d.upload(emb);
  struct V {
    qk::HcSrc src;
    uint32_t S;
    bool norm;
  };
  for (const V& v : {V{qk::HcSrc::Embed, 0, true}, V{qk::HcSrc::Slices, 4, true}, V{qk::HcSrc::Slices, 1, true},
                     V{qk::HcSrc::Y, 0, true}, V{qk::HcSrc::None, 0, true}, V{qk::HcSrc::Y, 0, false}}) {
    const std::string name = qk::hc_combine_norm_variant(1, v.src, v.S, v.norm);
    l0::Module m(d.ctx, kernels::path(name));
    l0::Mem hb = d.upload(H0), xb = d.zeros(size_t(HCN) * 2);
    const l0::Mem& src = v.src == qk::HcSrc::Embed ? eb : v.src == qk::HcSrc::Slices ? sl : yb;
    launch(d, m, "q4_hc_combine_norm", qk::kHcWg, {qk::kHc, 1, 1}, {ctl.m, hb, src, ib, wb, xb});
    std::vector<uint16_t> y(D), Hh = H0, xh(HCN, 0);
    if (v.src == qk::HcSrc::Embed) y = emb;
    else if (v.src == qk::HcSrc::Y) y = ybf;
    else if (v.src == qk::HcSrc::Slices)
      for (uint32_t k = 0; k < D; ++k) y[k] = qr::y_of_slices(slices.data(), v.S, 1, 0, D, k);
    const qr::HcSrc hs = v.src == qk::HcSrc::Embed ? qr::HcSrc::Embed : v.src == qk::HcSrc::None ? qr::HcSrc::None : qr::HcSrc::Y;
    qr::hc_combine_norm(Hh.data(), hs, y.data(), inj.data(), w.data(), v.norm ? xh.data() : nullptr);
    Diff dh, dx;
    dh.all(d.read<uint16_t>(hb, HCN), Hh);
    dx.all(d.read<uint16_t>(xb, HCN), xh);
    report(name + " H", dh);
    report(name + (v.norm ? " xn" : " xn untouched"), dx);
  }
  for (bool inject : {true, false}) {
    const uint32_t rows = inject ? qk::kHcDownN : qk::kHcLow;
    const std::string name = qk::hc_up_mix_variant(1, inject);
    l0::Module m(d.ctx, kernels::path(name));
    const std::vector<float> down = random_f32(rows, -6.f, 6.f, 10 + inject);
    const std::vector<uint16_t> up = random_bf16(size_t(qk::kHcLow) * HCN, -0.15f, 0.15f, 12);
    const std::vector<uint16_t> xn = random_bf16(HCN, -2.f, 2.f, 13);
    l0::Mem db = d.upload(down), ub = d.upload(up), xnb = d.upload(xn), xb = d.zeros(size_t(D) * 2), ijb = d.zeros(16);
    launch(d, m, "q4_hc_up_mix", qk::kUpMixWg, {D / 16, 1, 1}, {ctl.m, db, ub, xnb, xb, ijb});
    std::vector<uint16_t> xh(D);
    std::vector<float> ih(4, 0.0f);
    qr::hc_up_mix(down.data(), inject, up.data(), xn.data(), xh.data(), ih.data());
    Diff dx, di;
    dx.all(d.read<uint16_t>(xb, D), xh);
    di.all(d.read<float>(ijb, 4), ih);
    report(name + " x", dx);
    report(name + (inject ? " inj" : " inj untouched"), di);
  }
}

// ---- 2. PLE --------------------------------------------------------------------------------------------------------
// The pages q4_ple_check reads in a range of `bytes` (the loader's rule: a page whose first u64 fits).
uint32_t pages_of(size_t bytes) { return bytes < 8 ? 0u : uint32_t((bytes - 8) / loader::kQ4PlePage + 1); }

void ple_tests(Dev& d) {
  const model::Qwen4ExpDesc& md = model::qwen4exp();
  const std::array<uint64_t, 3> mult = loader::q4_ple_multipliers(md.vocab, md.ngram, 0, md.ple_seed);
  const std::vector<uint64_t> sizes = loader::q4_ple_primes(100000, 16, 0), offs = loader::q4_ple_offsets(sizes);
  std::vector<uint64_t> kc(qk::kPleConsts);
  for (uint32_t i = 0; i < 3; ++i) kc[i] = mult[i];
  for (uint32_t h = 0; h < 16; ++h) {
    kc[3 + h] = sizes[h];
    kc[3 + 16 + h] = offs[h];
  }
  l0::Mem kcb = d.upload(kc);
  std::vector<uint32_t> seq(21);
  {
    std::mt19937 rng(51);
    for (uint32_t& t : seq) t = rng() % qk::kVocab;
    seq[0] = seq[13] = 248319;
    seq[4] = seq[9] = seq[10] = qk::kPleEos;
  }
  // the q ranges (shared by both scale forms), random int8 rows
  std::vector<std::unique_ptr<l0::Mem>> q(16);
  for (uint32_t h = 0; h < 16; ++h) {
    q[h] = std::make_unique<l0::Mem>(d.ctx, l0::MemKind::Host, size_t(sizes[h]) * qk::kPleDim, loader::kQ4PlePage);
    std::mt19937 rng(60 + h);
    int8_t* p = q[h]->as<int8_t>();
    for (size_t i = 0; i < q[h]->size(); ++i) p[i] = int8_t(int(rng() % 255) - 127);
  }
  for (bool bf16_scale : {true, false}) {
    const size_t sb = bf16_scale ? 2 : 4;
    std::vector<std::unique_ptr<l0::Mem>> s(16);
    for (uint32_t h = 0; h < 16; ++h) {
      s[h] = std::make_unique<l0::Mem>(d.ctx, l0::MemKind::Host, size_t(sizes[h]) * sb, loader::kQ4PlePage);
      const std::vector<float> sc = random_f32(sizes[h], 0.001f, 0.05f, 70 + h);
      for (uint64_t r = 0; r < sizes[h]; ++r) {
        if (bf16_scale) s[h]->as<uint16_t>()[r] = rne(sc[r]);
        else s[h]->as<float>()[r] = sc[r];
      }
    }
    const auto scale_of = [&](uint32_t h, uint64_t r) {
      return bf16_scale ? f32(s[h]->as<uint16_t>()[r]) : s[h]->as<float>()[r];
    };
    const auto range = [&](uint32_t r) -> l0::Mem& { return r < 16 ? *q[r] : *s[r - 16]; };
    std::vector<uint64_t> ptrs(32);
    for (uint32_t r = 0; r < 32; ++r) ptrs[r] = reinterpret_cast<uint64_t>(range(r).ptr());
    l0::Mem pb = d.upload(ptrs);
    const std::string name = qk::ple_gather_variant(1, bf16_scale);
    l0::Module m(d.ctx, kernels::path(name));
    // the gather: positions 0..20, one launch each
    {
      Ctl ctl(d.ctx);
      l0::Mem ring = d.filled(16 * 4, 0xDEADBEEFu), e = d.zeros(size_t(D) * 2), ids = d.zeros(16 * 8);
      Diff di, de, dr;
      for (uint32_t pos = 0; pos <= 20; ++pos) {
        ctl.at(pos);
        ctl.c->cur_token[0] = seq[pos];
        launch(d, m, "q4_ple_gather", qk::kPleWg, {qk::kPleHeads, 1, 1}, {ctl.m, ring, pb, kcb, e, ids});
        const loader::Q4PleHistory hh = qr::ple_history(pos, [&](uint32_t t) { return seq[t]; });
        const std::array<uint64_t, 16> want = loader::q4_ple_ids(hh.t0, hh.t1, hh.t2, mult, sizes, offs);
        const std::vector<uint64_t> got = d.read<uint64_t>(ids, 16);
        const std::vector<uint16_t> eg = d.read<uint16_t>(e, D);
        std::vector<uint16_t> ew(D);
        for (uint32_t h = 0; h < 16; ++h) {
          di.add(pos * 16 + h, got[h], want[h]);
          const uint64_t r = want[h] - offs[h];
          qr::ple_row(q[h]->as<int8_t>() + r * qk::kPleDim, scale_of(h, r), ew.data() + h * qk::kPleDim);
        }
        de.all(eg, ew, size_t(pos) * D);
        dr.add(pos, d.read<uint32_t>(ring, 1, (pos % 16) * 4)[0], seq[pos]);
      }
      report(name + " ids (positions 0..20 x 16 heads, host USM)", di);
      report(name + " rows", de);
      report(name + " the id ring", dr);
    }
    // q4_ple_check: the tags, then the data's first words, through the pointer table
    {
      std::vector<uint32_t> first(33, 0);
      for (uint32_t r = 0; r < 32; ++r) first[r + 1] = first[r] + pages_of(range(r).size());
      const uint32_t pages = first[32];
      l0::Mem fb = d.upload(first), ob = d.zeros(size_t(pages) * 8);
      for (int pass = 0; pass < 2; ++pass) {
        std::vector<std::vector<uint8_t>> saved;
        if (pass == 0)   // tags over the data's first words (restored after)
          for (uint32_t r = 0; r < 32; ++r)
            for (uint32_t p = 0; p < pages_of(range(r).size()); ++p) {
              uint8_t* at = static_cast<uint8_t*>(range(r).ptr()) + size_t(p) * loader::kQ4PlePage;
              saved.emplace_back(at, at + 8);
              const uint64_t tag = loader::q4_ple_tag(r, p);
              std::memcpy(at, &tag, 8);
            }
        launch(d, m, "q4_ple_check", qk::kPleCheckWg, {(pages + qk::kPleCheckWg - 1) / qk::kPleCheckWg, 1, 1},
               {pb, fb, ob, pages});
        std::vector<uint64_t> want;
        for (uint32_t r = 0; r < 32; ++r)
          for (uint32_t p = 0; p < pages_of(range(r).size()); ++p) {
            uint64_t w = 0;
            std::memcpy(&w, static_cast<const uint8_t*>(range(r).ptr()) + size_t(p) * loader::kQ4PlePage, 8);
            want.push_back(w);
          }
        Diff dc;
        dc.all(d.read<uint64_t>(ob, pages), want);
        report(qk::ple_check_variant(bf16_scale) + std::string(pass == 0 ? " q4_ple_check tags (" : " q4_ple_check data words (") +
                   std::to_string(pages) + " pages)",
               dc);
        if (pass == 0) {
          size_t i = 0;
          for (uint32_t r = 0; r < 32; ++r)
            for (uint32_t p = 0; p < pages_of(range(r).size()); ++p)
              std::memcpy(static_cast<uint8_t*>(range(r).ptr()) + size_t(p) * loader::kQ4PlePage, saved[i++].data(), 8);
        }
      }
    }
  }
  // q4_ple_block: 12 positions through the conv ring
  {
    const std::string name = qk::ple_block_variant(1);
    l0::Module m(d.ctx, kernels::path(name));
    std::vector<float> pw = bf16_vals(size_t(3) * HCN, 0.5f, 1.5f, 80);
    const std::vector<float> taps = random_f32(size_t(HCN) * 4, -0.5f, 0.5f, 81);
    pw.insert(pw.end(), taps.begin(), taps.end());
    l0::Mem pwb = d.upload(pw), ring = d.zeros(size_t(qk::kPleRing) * HCN * 2), hb = d.zeros(size_t(HCN) * 2),
            kvb = d.zeros(size_t(qk::kPleKvN) * 4);
    std::vector<std::vector<uint16_t>> hring(16, std::vector<uint16_t>(HCN, 0));
    Ctl ctl(d.ctx);
    Diff dh, dr;
    for (uint32_t pos = 0; pos < 12; ++pos) {
      const std::vector<float> kv = random_f32(qk::kPleKvN, -2.f, 2.f, 90 + pos);
      const std::vector<uint16_t> H = random_bf16(HCN, -2.f, 2.f, 110 + pos);
      ctl.at(pos);
      d.write(hb, H);
      d.write(kvb, kv);
      launch(d, m, "q4_ple_block", qk::kPleBlockWg, {qk::kHc, 1, 1}, {ctl.m, hb, kvb, pwb, ring});
      std::vector<uint16_t> Hh = H, out(HCN);
      const uint16_t* hist[3] = {pos >= 9 ? hring[(pos - 9) % 16].data() : nullptr, pos >= 6 ? hring[(pos - 6) % 16].data() : nullptr,
                                 pos >= 3 ? hring[(pos - 3) % 16].data() : nullptr};
      qr::ple_block(Hh.data(), kv.data(), pw.data(), pw.data() + HCN, pw.data() + 2 * HCN, taps.data(), hist, out.data());
      hring[pos % 16] = out;
      dh.all(d.read<uint16_t>(hb, HCN), Hh, size_t(pos) * HCN);
      dr.all(d.read<uint16_t>(ring, HCN, size_t(pos % 16) * HCN * 2), out, size_t(pos) * HCN);
    }
    report(name + " H (positions 0..11)", dh);
    report(name + " conv ring rows", dr);
  }
}

// ---- 3. the indexer and the selection ------------------------------------------------------------------------------
// Distinct scores 0.01 x (a permutation of 1..n); the blocks of descending ranks [lo, hi) then all set to the score of
// rank lo: an exact tie group straddling the 512 / 513 cut when lo < 512 < hi.
std::vector<float> tied_scores(uint32_t n, uint32_t lo, uint32_t hi, uint32_t seed) {
  std::vector<uint32_t> perm(n);
  for (uint32_t b = 0; b < n; ++b) perm[b] = b + 1;
  std::mt19937 rng(seed);
  std::shuffle(perm.begin(), perm.end(), rng);
  std::vector<float> s(n);
  for (uint32_t b = 0; b < n; ++b) s[b] = 0.01f * float(perm[b]);
  std::vector<uint32_t> idx(n);
  for (uint32_t b = 0; b < n; ++b) idx[b] = b;
  std::sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) { return s[a] > s[b]; });
  if (lo < n) {
    const float v = s[idx[lo]];
    for (uint32_t r = lo; r < hi && r < n; ++r) s[idx[r]] = v;
  }
  return s;
}

// One q4_qsa_select launch for row p over `scores` against q4ref::qsa_select: the list, the count word, the words
// past the count untouched, the diagnostic - all exact.
bool select_case(Dev& d, l0::Module& m, const std::vector<float>& scores, uint32_t p, std::string* why) {
  Ctl ctl(d.ctx);
  ctl.at(p);
  l0::Mem sb = d.upload(scores), lb = d.filled(size_t(qk::kListRow) * 4, 0xFFFFFFFFu), db = d.zeros(8);
  launch(d, m, "q4_qsa_select", qk::kSelectWg, {1, 1, 1}, {ctl.m, sb, uint32_t(scores.size()), lb, db});
  const qr::Selection sel = qr::qsa_select(scores.data(), p);
  const std::vector<uint32_t> l = d.read<uint32_t>(lb, qk::kListRow);
  const std::vector<float> dg = d.read<float>(db, 2);
  const uint32_t c = sel.count();
  if (l[qk::kCountWord] != c || c != qr::qsa_count(p)) {
    *why = "count " + std::to_string(l[qk::kCountWord]) + ", want " + std::to_string(c);
    return false;
  }
  for (uint32_t i = 0; i < c; ++i)
    if (l[i] != sel.list[i]) {
      *why = "list[" + std::to_string(i) + "] " + std::to_string(l[i]) + ", want " + std::to_string(sel.list[i]);
      return false;
    }
  for (uint32_t i = c; i < qk::kListRow; ++i)
    if (i != qk::kCountWord && l[i] != 0xFFFFFFFFu) {
      *why = "word " + std::to_string(i) + " past the count written";
      return false;
    }
  if (qr::as_u32(dg[0]) != qr::as_u32(sel.s512) || qr::as_u32(dg[1]) != qr::as_u32(sel.s513)) {
    *why = "diag {" + std::to_string(dg[0]) + ", " + std::to_string(dg[1]) + "}, want {" + std::to_string(sel.s512) + ", " +
           std::to_string(sel.s513) + "}";
    return false;
  }
  return true;
}

void qsa_tests(Dev& d) {
  const std::string name = qk::qsa_variant(1);
  l0::Module m(d.ctx, kernels::path(name));
  const model::Qwen4ExpDesc& md = model::qwen4exp();
  // -- prep
  {
    const uint32_t ML = 9004;
    const std::vector<float> rope = loader::q4_rope_table(md, ML);
    const std::vector<float> small = bf16_vals(768, 0.5f, 1.5f, 200);
    l0::Mem rb = d.upload(rope), smb = d.upload(small), qb = d.zeros(512 * 4), tb = d.zeros(size_t(qk::kTailSlots) * 128 * 2),
            kb = d.zeros(size_t(ML / 4) * 128 * 2), ib = d.zeros(640 * 4);
    Ctl ctl(d.ctx);
    std::vector<std::vector<uint16_t>> raw(ML);
    Diff dq, dt, dk;
    std::string blocks;
    for (uint32_t pos : {0u, 1u, 2u, 3u, 2044u, 2045u, 2046u, 2047u, 2048u, 2049u, 2050u, 2051u, 8996u, 8997u, 8998u, 8999u}) {
      const std::vector<float> idx = random_f32(qk::kIdxN, -3.f, 3.f, 300 + pos);
      ctl.at(pos);
      d.write(ib, idx);
      launch(d, m, "q4_qsa_prep", qk::kPrepWg, {qk::kIdxHeads + 1, 1, 1}, {ctl.m, ib, smb, rb, qb, tb, kb});
      std::vector<float> qh(512);
      qr::qsa_q(idx.data(), small.data() + 512, rope.data() + size_t(pos) * 64, qh.data());
      dq.all(d.read<float>(qb, 512), qh, size_t(pos) * 512);
      raw[pos].resize(128);
      qr::qsa_raw_key(idx.data(), raw[pos].data());
      dt.all(d.read<uint16_t>(tb, 128, size_t(pos % qk::kTailSlots) * 128 * 2), raw[pos], size_t(pos) * 128);
      if ((pos + 1) % qk::kBlock == 0) {
        const uint32_t b = (pos + 1) / 4 - 1;
        const uint16_t* r4[4] = {raw[4 * b].data(), raw[4 * b + 1].data(), raw[4 * b + 2].data(), raw[4 * b + 3].data()};
        std::vector<uint16_t> key(128);
        qr::qsa_block_key(r4, small.data() + 640, rope.data() + size_t(4 * b) * 64, key.data());
        dk.all(d.read<uint16_t>(kb, 128, size_t(b) * 128 * 2), key, size_t(b) * 128);
        blocks += (blocks.empty() ? "" : ", ") + std::to_string(b);
      }
    }
    report(name + " q4_qsa_prep q (positions 0..3, 2044..2051, 8996..8999)", dq);
    report(name + " q4_qsa_prep raw keys in the 8-slot tail", dt);
    report(name + " q4_qsa_prep compressed keys (blocks " + blocks + ")", dk);
  }
  // -- scores, then the select over them
  std::vector<float> dev600, dev65536;
  for (uint32_t n : {600u, 65536u}) {
    const uint32_t stride = 65536, p = 4 * n - 1;   // the grid spans max_len / 4 = 65536 blocks (max_len 262144)
    const std::vector<float> q = bf16_vals(512, -1.f, 1.f, 400 + n);
    const std::vector<uint16_t> keys = random_bf16(size_t(stride) * 128, -1.f, 1.f, 401 + n);
    Ctl ctl(d.ctx);
    ctl.at(p);
    l0::Mem qb = d.upload(q), kb = d.upload(keys), sb = d.filled(size_t(stride) * 4, qr::as_u32(-1.0f));
    launch(d, m, "q4_qsa_score", qk::kScoreWg, {stride / qk::kScoreWg, 1, 1}, {ctl.m, qb, kb, sb, stride});
    std::vector<float> got = d.read<float>(sb, stride), want(stride, -1.0f);
    for (uint32_t b = 0; b < n; ++b) want[b] = qr::qsa_score(q.data(), keys.data() + size_t(b) * 128);
    Diff ds;
    ds.all(got, want);
    report(name + " q4_qsa_score at " + std::to_string(n) + " blocks (grid over 65536, nothing past n)", ds);
    got.resize(n);
    (n == 600 ? dev600 : dev65536) = got;
    if (n == 600) {   // n 0 (p 2): nothing written
      ctl.at(2);
      l0::Mem s0 = d.filled(size_t(stride) * 4, qr::as_u32(-1.0f));
      launch(d, m, "q4_qsa_score", qk::kScoreWg, {stride / qk::kScoreWg, 1, 1}, {ctl.m, qb, kb, s0, stride});
      Diff d0;
      d0.all(d.read<float>(s0, stride), std::vector<float>(stride, -1.0f));
      report(name + " q4_qsa_score at p 2 (no complete block): nothing written", d0);
    }
  }
  {
    struct Case {
      std::string name;
      std::vector<float> s;
    };
    std::vector<Case> cases;
    cases.push_back({"n 512 (identity: p 2047..2050)", tied_scores(512, 0, 0, 500)});
    cases.push_back({"n 513 (p 2051..2054), 6 tied across the cut", tied_scores(513, 509, 515, 501)});
    cases.push_back({"n 513, 2 tied at 512 / 513", tied_scores(513, 511, 513, 502)});
    cases.push_back({"n 600, 9 tied across the cut", tied_scores(600, 506, 515, 503)});
    cases.push_back({"n 600, the device's own scores", dev600});
    {
      std::vector<float> z = tied_scores(600, 0, 0, 504);
      std::vector<uint32_t> idx(600);
      for (uint32_t b = 0; b < 600; ++b) idx[b] = b;
      std::shuffle(idx.begin(), idx.end(), std::mt19937(505));
      for (uint32_t i = 0; i < 100; ++i) z[idx[i]] = 0.0f;
      cases.push_back({"n 600, the cut inside 100 zeros", z});
    }
    {
      const std::vector<uint32_t> rows = qr::hex32(fx::kSelRows);
      for (uint32_t r = 0; r < 4; ++r) {
        std::vector<float> s(fx::kSelRowsN[r]);
        for (uint32_t b = 0; b < s.size(); ++b) s[b] = qr::as_f32(rows[size_t(r) * fx::kSelN + b]);
        cases.push_back({"the fixture's row " + std::to_string(r) + " (n " + std::to_string(s.size()) + ")", s});
      }
    }
    {
      std::vector<float> dec(513);
      for (uint32_t b = 0; b < dec.size(); ++b) dec[b] = 1000.0f - float(b);
      cases.push_back({"n 513 decreasing: p's own block (512) a candidate, losing", dec});
      dec[512] = 2000.0f;
      cases.push_back({"n 513: p's own block winning", dec});
    }
    for (const Case& c : cases) {
      bool ok = true;
      std::string why;
      for (uint32_t tail = 0; tail < 4 && ok; ++tail) {
        ok = select_case(d, m, c.s, 4 * uint32_t(c.s.size()) - 1 + tail, &why);
        if (!ok) why = " (tail " + std::to_string(tail) + ": " + why + ")";
      }
      report_bool(name + " q4_qsa_select " + c.name + ", tails 0..3: list, count, diag", ok, why);
    }
    for (int k = 0; k < 2; ++k) {   // n 65536 (max_len 262144): the device's scores; coarse scores, ties everywhere
      std::vector<float> s = dev65536;
      if (k == 1) {
        std::mt19937 rng(506);
        for (float& v : s) v = 0.25f * float(rng() % 4096);
      }
      std::string why;
      bool ok = true;
      for (uint32_t tail : {0u, 3u})
        if (ok && !select_case(d, m, s, 4 * 65536 - 1 + tail, &why)) {
          ok = false;
          why = " (tail " + std::to_string(tail) + ": " + why + ")";
        }
      report_bool(name + " q4_qsa_select n 65536 " + (k ? "coarse ties" : "the device's scores") + ", tails 0 and 3", ok, why);
    }
  }
}

// ---- 4. the route and the MoE --------------------------------------------------------------------------------------
void moe_tests(Dev& d) {
  {
    const std::string name = qk::route_variant(1);
    l0::Module m(d.ctx, kernels::path(name));
    const auto one = [&](const std::vector<float>& lg) {
      l0::Mem lb = d.upload(lg), rb = d.filled(qk::route::kWords * 4, 0xFFFFFFFFu);
      launch(d, m, "q4_route", qk::kRouteLanes, {1, 1, 1}, {lb, rb});
      return d.read<uint32_t>(rb, qk::route::kWords);
    };
    Diff dr;
    size_t tag = 0;
    const auto check = [&](const std::vector<float>& lg) {
      const std::vector<uint32_t> got = one(lg);
      dr.all(got, qr::route_row(qr::route(lg.data())), tag++ * qk::route::kWords);
      return got;
    };
    for (uint32_t s = 0; s < 8; ++s) {
      std::vector<float> lg = random_f32(qk::kRouterN, -3.f, 3.f, 600 + s);
      for (uint32_t e = qk::kExperts + 1; e < qk::kRouterN; ++e) lg[e] = 0.0f;
      check(lg);
    }
    const std::vector<uint32_t> L = qr::hex32(fx::kRouteLogits);
    bool fix = true;
    for (uint32_t r = 0; r < 4; ++r) {
      std::vector<float> lg(qk::kRouterN);
      for (uint32_t e = 0; e < qk::kRouterN; ++e) lg[e] = qr::as_f32(L[size_t(r) * qk::kRouterN + e]);
      const std::vector<uint32_t> got = check(lg);
      for (uint32_t k = 0; k < qk::kTopK; ++k) fix = fix && got[k] == fx::kRouteRuledIds[r * 10 + k];
    }
    bool tie = true;
    {
      std::vector<float> lg(qk::kRouterN, -4.0f);
      const uint32_t win[9] = {500, 3, 250, 128, 77, 400, 9, 311, 64};
      for (uint32_t j = 0; j < 9; ++j) lg[win[j]] = 2.0f + 0.125f * float(j);
      lg[300] = lg[7] = 1.0f;
      lg[qk::kExperts] = 0.5f;
      const std::vector<uint32_t> got = check(lg);
      tie = got[9] == 7;
      for (uint32_t j = 0; j < 9; ++j) tie = tie && got[j] == win[8 - j];
    }
    report(name + " route rows (8 random, the fixture's 4, a crafted tie)", dr);
    report_bool(name + " ids = the fixture's ruled ids (its tie (99, 431), the -1e30 row)", fix);
    report_bool(name + " a tie (300, 7) for the 10th: the lower id", tie);
  }
  const size_t gub = qr::gate_up_block_words(), dnb = qr::down_block_words();
  const std::vector<uint32_t> gu = random_blocks(gub * qk::kExperts, 700), dn = random_blocks(dnb * qk::kExperts, 701);
  const std::vector<uint32_t> sh_gu4 = random_blocks(gub, 702), sh_dn4 = random_blocks(dnb, 703);
  const std::vector<uint16_t> sh_gub = random_bf16(size_t(D) * 2 * qk::kInter, -0.03f, 0.03f, 704);
  const std::vector<uint16_t> sh_dnb = random_bf16(size_t(qk::kInter) * D, -0.05f, 0.05f, 705);
  const std::vector<uint16_t> x = random_bf16(D, -0.5f, 0.5f, 706);
  l0::Mem gb = d.upload(gu), db = d.upload(dn), xb = d.upload(x);
  const uint32_t ids[10] = {511, 0, 17, 128, 255, 256, 300, 383, 400, 480};
  const float w_all[10] = {rf(0.21875f), rf(0.15625f), rf(0.125f), rf(0.109375f), rf(0.09375f),
                           rf(0.0859375f), rf(0.078125f), rf(0.0625f), rf(0.046875f), rf(0.0390625f)};
  const float sg_all = rf(0.40625f);
  for (bool shb : {false, true}) {
    const std::string name = qk::moe_variant(1, shb);
    l0::Module m(d.ctx, kernels::path(name));
    l0::Mem sgu = shb ? d.upload(sh_gub) : d.upload(sh_gu4), sdn = shb ? d.upload(sh_dnb) : d.upload(sh_dn4);
    const std::vector<uint16_t> h_ref =
        qr::gate_up(ids, x.data(), gu.data(), shb ? nullptr : sh_gu4.data(), shb ? sh_gub.data() : nullptr, qk::kUpKs);
    std::vector<uint16_t> y_all;
    for (int c = -1; c <= 11; ++c) {   // -1 every slot; 0..9 routed slot c alone; 10 the shared slot alone; 11 a replay of -1
      const bool every = c < 0 || c == 11;
      float w[10];
      for (uint32_t k = 0; k < 10; ++k) w[k] = (every || c == int(k)) ? w_all[k] : 0.0f;
      const float sg = (every || c == 10) ? sg_all : 0.0f;
      std::vector<uint32_t> row(qk::route::kWords, 0);
      for (uint32_t k = 0; k < 10; ++k) {
        row[qk::route::kIds + k] = ids[k];
        row[qk::route::kWeights + k] = qr::as_u32(w[k]);
      }
      row[qk::route::kSharedGate] = qr::as_u32(sg);
      l0::Mem rb = d.upload(row), hb = d.zeros(size_t(qk::kSlots) * qk::kInter * 2), yb = d.zeros(size_t(D) * 2);
      launch(d, m, "q4_moe_gate_up", qk::moe_gate_up_wg(), {qk::moe_gate_up_groups(), 1, 1}, {rb, xb, gb, sgu, hb});
      launch(d, m, "q4_moe_down", qk::moe_down_wg(), {D / 16, 1, 1}, {rb, hb, db, sdn, yb});
      const std::vector<uint16_t> h = d.read<uint16_t>(hb, size_t(qk::kSlots) * qk::kInter), y = d.read<uint16_t>(yb, D);
      const std::vector<uint16_t> y_ref =
          qr::down(ids, w, sg, h_ref, dn.data(), shb ? nullptr : sh_dn4.data(), shb ? sh_dnb.data() : nullptr, qk::kDnKs);
      if (c == -1) {
        Diff dh;
        dh.all(h, h_ref);
        report(name + " gate_up h [11][640] (ids 511, 0, 17, ... over 512 blocks)", dh);
        y_all = y;
      }
      if (c == 11) {
        report_bool(name + " a replay of every slot", y == y_all);
        continue;
      }
      Diff dy;
      dy.all(y, y_ref);
      const std::string what = c < 0 ? "every slot" : c == 10 ? "the shared slot alone" : "routed slot " + std::to_string(c) + " alone (id " + std::to_string(ids[c]) + ")";
      report(name + " down + combine, " + what, dy);
    }
  }
}

// ---- 5. eager attention, 6. the gated head ------------------------------------------------------------------------
struct Caches {
  uint32_t rows;
  std::vector<uint16_t> k, v;
  Caches(uint32_t r, uint32_t seed) : rows(r), k(random_bf16(size_t(r) * KVN, -1.f, 1.f, seed)), v(random_bf16(size_t(r) * KVN, -1.f, 1.f, seed + 1)) {}
};
std::vector<uint32_t> list_row(const qr::Selection& sel) {
  std::vector<uint32_t> row(qk::kListRow, 0);
  std::copy(sel.list.begin(), sel.list.end(), row.begin());
  row[qk::kCountWord] = sel.count();
  return row;
}

void eager_tests(Dev& d) {
  const std::string name = qk::qsa_attn_variant(1, true);
  l0::Module m(d.ctx, kernels::path(name));
  const Caches c(9001, 800);
  const std::vector<float> q = random_f32(QN, -2.f, 2.f, 802), gate = random_f32(QN, -4.f, 4.f, 803);
  l0::Mem kb = d.upload(c.k), vb = d.upload(c.v), qb = d.upload(q), gb = d.upload(gate), ob = d.zeros(size_t(QN) * 2);
  const std::vector<float> scores = random_f32(2250, 0.f, 4.f, 804);
  Ctl ctl(d.ctx);
  for (uint32_t p : {1u, 2050u, 2051u, 2400u, 9000u}) {
    const qr::Selection sel = qr::qsa_select(scores.data(), p);
    l0::Mem lb = d.upload(list_row(sel));
    ctl.at(p);
    launch(d, m, "q4_qsa_attn_eager", qk::kEagerWg, {QH, 1, 1}, {ctl.m, qb, gb, kb, vb, lb, ob});
    std::vector<uint16_t> want(QN);
    for (uint32_t h = 0; h < QH; ++h) {
      const qr::EagerHead e = qr::qsa_attn_eager(q.data(), gate.data(), c.k.data(), c.v.data(), sel.list.data(), sel.count(), h);
      std::copy(e.out.begin(), e.out.end(), want.begin() + size_t(h) * HD);
    }
    Diff dd;
    dd.all(d.read<uint16_t>(ob, QN), want);
    report(name + " at p " + std::to_string(p) + " (" + std::to_string(sel.count()) + " listed)", dd);
  }
}

void gated_tests(Dev& d) {
  const std::string name = qk::gated_head_sig_variant(1);
  l0::Module m(d.ctx, kernels::path(name));
  const uint32_t ON = qk::kGdnHeads * qk::kGdnHd;
  const auto run = [&](const std::vector<float>& qkvz, const std::vector<float>& o, const std::vector<uint16_t>& gw) {
    l0::Mem zb = d.upload(qkvz), obuf = d.upload(o), wb = d.upload(gw), xb = d.zeros(size_t(ON) * 2);
    launch(d, m, "prep_gated_head", 128, {qk::kGdnHeads, 1, 1}, {zb, obuf, wb, xb});
    return d.read<uint16_t>(xb, ON);
  };
  {
    const std::vector<float> qkvz = random_f32(qk::kQkvzN, -6.f, 6.f, 900), o = random_f32(ON, -2.f, 2.f, 901);
    const std::vector<uint16_t> gw = random_bf16(128, 0.5f, 1.5f, 902);
    std::vector<uint16_t> want(ON);
    qr::gated_head_sig(qkvz.data(), o.data(), gw.data(), want.data());
    Diff dd;
    dd.all(run(qkvz, o, gw), want);
    report(name + " random", dd);
  }
  {
    const std::vector<uint16_t> xo = qr::fixture_bf16(true, 60, ON, 7, 0.25f), z = qr::fixture_bf16(false, 61, ON, 7, 0.125f, 6.0f),
                                gw = qr::fixture_bf16(false, 62, 128, 7, 0.125f, 1.5f), want = qr::hex16(fx::kGated);
    std::vector<float> o(ON), qkvz(qk::kQkvzN, 0.0f);
    for (uint32_t i = 0; i < ON; ++i) {
      o[i] = f32(xo[i]);
      qkvz[qr::kZOff + i] = f32(z[i]);
    }
    Diff dd;
    dd.all(run(qkvz, o, gw), want);
    report(name + " the fixture's case (torch's output)", dd);
  }
}

// ---- 7. flash -----------------------------------------------------------------------------------------------------
double cosine(const std::vector<double>& a, const std::vector<double>& b) {
  double dot = 0, na = 0, nb = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return na == 0 && nb == 0 ? 1.0 : dot / std::sqrt(na * nb);
}

struct Flash {
  l0::Module m, v2;
  l0::Mem part, part2, ob;
  explicit Flash(Dev& d)
      : m(d.ctx, kernels::path(qk::qsa_attn_variant(1, false))),
        v2(d.ctx, kernels::path(qk::attn_v2_q4_variant(1))),
        part(d.zeros(size_t(QH) * qk::kAttnTgt * qk::kAttnPart * 4)),
        part2(d.zeros(size_t(QH) * qk::kAttnTgt * qk::kAttnPart * 4)),
        ob(d.zeros(size_t(QN) * 2)) {}
  std::vector<uint16_t> sparse(Dev& d, const Ctl& ctl, const l0::Mem& q, const l0::Mem& gate, const l0::Mem& k, const l0::Mem& v,
                               const l0::Mem& list) {
    launch(d, m, "q4_qsa_attn", qk::kAttnWg, {KVH, qk::kAttnTgt, 1}, {ctl.m, q, k, v, list, part});
    launch(d, m, "q4_qsa_reduce", qk::kAttnWg, {QH, 1, 1}, {ctl.m, part, gate, list, ob});
    return d.read<uint16_t>(ob, QN);
  }
  std::vector<uint16_t> dense(Dev& d, const Ctl& ctl, const l0::Mem& q, const l0::Mem& gate, const l0::Mem& k, const l0::Mem& v) {
    launch(d, v2, "attn_decode_v2", qk::kAttnWg, {KVH, qk::kAttnTgt, 1}, {ctl.m, q, k, v, part2});
    launch(d, v2, "attn_reduce_v2", qk::kAttnWg, {QH, 1, 1}, {ctl.m, part2, gate, ob});
    return d.read<uint16_t>(ob, QN);
  }
};

void flash_tests(Dev& d) {
  Flash fl(d);
  const std::string name = qk::qsa_attn_variant(1, false);
  const Caches c(131073, 1000);   // max_len 131073: [max_len][2][256] bf16 K and V
  const std::vector<float> q = random_f32(QN, -2.f, 2.f, 1002), gate = random_f32(QN, -4.f, 4.f, 1003);
  const std::vector<float> sat(QN, 30.0f);   // sigmoid(30) = 1 in fp32 and in bf16
  l0::Mem kb = d.upload(c.k), vb = d.upload(c.v), qb = d.upload(q), gb = d.upload(gate), satb = d.upload(sat);
  const std::vector<float> scores = random_f32(32768, 0.f, 4.f, 1004);
  Ctl ctl(d.ctx);
  for (uint32_t p : {1u, 2050u, 2051u, 9000u, 131072u}) {
    const qr::Selection sel = qr::qsa_select(scores.data(), p);
    l0::Mem lb = d.upload(list_row(sel));
    ctl.at(p);
    const std::vector<uint16_t> o = fl.sparse(d, ctl, qb, gb, kb, vb, lb);
    double worst = 1.0;
    for (uint32_t h = 0; h < QH; ++h) {
      const std::vector<double> want = qr::qsa_attn_fp64(q.data(), c.k.data(), c.v.data(), sel.list.data(), sel.count(), h);
      std::vector<double> a(HD), b(HD);
      for (uint32_t dd = 0; dd < HD; ++dd) {
        a[dd] = f32(o[size_t(h) * HD + dd]);
        b[dd] = want[dd] * double(rf(qr::sigmoid_t(gate[size_t(h) * HD + dd])));
      }
      worst = std::min(worst, cosine(a, b));
    }
    const bool ok = worst >= 0.99999;
    std::printf("  %s%s at p %u (%u listed): worst cosine %.7f against fp64 x rne(sigmoid(gate))\n", ok ? "" : "FAIL ", name.c_str(), p,
                sel.count(), worst);
    if (!ok) ++failures;
  }
  // at depth < 2052 (identity lists): against attention v2's dense result, the gate saturated on both sides
  for (uint32_t p : {100u, 1000u, 2050u}) {
    const qr::Selection sel = qr::qsa_select(scores.data(), p);
    l0::Mem lb = d.upload(list_row(sel));
    ctl.at(p);
    const std::vector<uint16_t> a = fl.sparse(d, ctl, qb, satb, kb, vb, lb), b = fl.dense(d, ctl, qb, satb, kb, vb);
    double worst = 1.0;
    int wu = 0;
    for (uint32_t h = 0; h < QH; ++h) {
      std::vector<double> x(HD), y(HD);
      for (uint32_t dd = 0; dd < HD; ++dd) {
        x[dd] = f32(a[size_t(h) * HD + dd]);
        y[dd] = f32(b[size_t(h) * HD + dd]);
        wu = std::max(wu, ulps16(a[size_t(h) * HD + dd], b[size_t(h) * HD + dd]));
      }
      worst = std::min(worst, cosine(x, y));
    }
    const bool ok = worst >= 0.99999;
    std::printf("  %s%s at p %u (identity, %u listed) against %s: worst cosine %.7f, worst %d bf16 ulp\n", ok ? "" : "FAIL ",
                name.c_str(), p, sel.count(), qk::attn_v2_q4_variant(1).c_str(), worst, wu);
    if (!ok) ++failures;
  }
}

// ---- --bench-attn ------------------------------------------------------------------------------------------------
double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}
double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0.0 : v[v.size() / 2];
}

int bench_attn(Dev& d) {
  const uint32_t R = 100, N = 20, ROUNDS = 3;
  Flash fl(d);
  const Caches c(2049, 1100);
  const std::vector<float> q = random_f32(QN, -2.f, 2.f, 1102), gate = random_f32(QN, -4.f, 4.f, 1103);
  l0::Mem kb = d.upload(c.k), vb = d.upload(c.v), qb = d.upload(q), gb = d.upload(gate);
  Ctl ctl(d.ctx);
  std::printf("bench-attn: %u launch pairs a list, %u warm-up replays, %u rounds of %u replays interleaved A / B; the median round\n",
              R, 3u, ROUNDS, N);
  for (uint32_t p : {512u, 1024u, 2048u}) {
    std::vector<float> none(1, 0.0f);
    const qr::Selection sel = qr::qsa_select(none.data(), p);   // n <= 512: the identity list 0..p
    l0::Mem lb = d.upload(list_row(sel));
    ctl.at(p);
    l0::Kernel ka = bind(fl.m, "q4_qsa_attn", qk::kAttnWg, {ctl.m, qb, kb, vb, lb, fl.part});
    l0::Kernel kr = bind(fl.m, "q4_qsa_reduce", qk::kAttnWg, {ctl.m, fl.part, gb, lb, fl.ob});
    l0::Kernel va = bind(fl.v2, "attn_decode_v2", qk::kAttnWg, {ctl.m, qb, kb, vb, fl.part2});
    l0::Kernel vr = bind(fl.v2, "attn_reduce_v2", qk::kAttnWg, {ctl.m, fl.part2, gb, fl.ob});
    l0::CmdList la = l0::CmdList::regular(d.ctx), lv = l0::CmdList::regular(d.ctx);
    for (uint32_t r = 0; r < R; ++r) {
      la.launch(ka, KVH, qk::kAttnTgt, 1);
      la.launch(kr, QH, 1, 1);
      lv.launch(va, KVH, qk::kAttnTgt, 1);
      lv.launch(vr, QH, 1, 1);
    }
    la.close();
    lv.close();
    const auto replay = [&](l0::CmdList& l) {
      const auto t0 = std::chrono::steady_clock::now();
      d.q.execute(l, &d.f);
      d.f.wait();
      return seconds_since(t0);
    };
    for (int w = 0; w < 3; ++w) {
      replay(la);
      replay(lv);
    }
    std::vector<double> ta, tv;
    for (uint32_t round = 0; round < ROUNDS; ++round) {
      double sa = 0, sv = 0;
      for (uint32_t i = 0; i < N; ++i) {
        sa += replay(la);
        sv += replay(lv);
      }
      ta.push_back(sa / (N * R) * 1e6);
      tv.push_back(sv / (N * R) * 1e6);
    }
    std::printf("bench-attn: depth %u (%u keys): %s %.2f us, %s %.2f us per launch pair (rounds %.2f / %.2f / %.2f vs "
                "%.2f / %.2f / %.2f)\n",
                p, p + 1, qk::qsa_attn_variant(1, false).c_str(), median(ta), qk::attn_v2_q4_variant(1).c_str(), median(tv),
                ta[0], ta[1], ta[2], tv[0], tv[1], tv[2]);
  }
  return 0;
}

// ---- --ple-rate ----------------------------------------------------------------------------------------------------
struct PleTable {   // 16 + 16 host-USM ranges in the primary context: every device of it reads them
  std::vector<uint64_t> sizes, offs;
  std::array<uint64_t, 3> mult{};
  std::vector<std::unique_ptr<l0::Mem>> q, s;
  std::vector<uint64_t> ptrs, consts;
  size_t bytes = 0;
};
// One device's gather list: `n` launches of q4_ple_gather_M1_BF16, each with its own Control block, id ring, e and
// ids_out (n 1: the narrow mode; 16: the wide one).
struct PleRig {
  l0::Context& ctx;
  l0::Queue q;
  l0::Fence f;
  l0::CmdList imm;
  l0::Module mod;
  l0::Mem ptrs, consts;
  std::vector<std::unique_ptr<l0::Mem>> ctl, ring, e, ids;
  std::vector<l0::Kernel> ks;
  l0::CmdList list;
  std::vector<std::vector<uint32_t>> seq;   // per launch: the tokens it has been given, by position
  PleRig(l0::Context& c, const PleTable& t, uint32_t n)
      : ctx(c), q(c), f(q), imm(l0::CmdList::immediate(c)), mod(c, kernels::path(qk::ple_gather_variant(1, true))),
        ptrs(c, l0::MemKind::Device, 32 * 8), consts(c, l0::MemKind::Device, t.consts.size() * 8), list(l0::CmdList::regular(c)),
        seq(n) {
    imm.copy(ptrs.ptr(), t.ptrs.data(), 32 * 8);
    imm.copy(consts.ptr(), t.consts.data(), t.consts.size() * 8);
    ks.reserve(n);
    for (uint32_t j = 0; j < n; ++j) {
      ctl.push_back(std::make_unique<l0::Mem>(c, l0::MemKind::Shared, sizeof(runtime::Control)));
      std::memset(ctl[j]->ptr(), 0, sizeof(runtime::Control));
      ring.push_back(std::make_unique<l0::Mem>(c, l0::MemKind::Device, 16 * 4));
      e.push_back(std::make_unique<l0::Mem>(c, l0::MemKind::Device, size_t(D) * 2));
      ids.push_back(std::make_unique<l0::Mem>(c, l0::MemKind::Device, 16 * 8));
      ks.push_back(bind(mod, "q4_ple_gather", qk::kPleWg, {*ctl[j], *ring[j], ptrs, consts, *e[j], *ids[j]}));
      list.launch(ks[j], qk::kPleHeads, 1, 1);
    }
    list.close();
  }
  // Launch j's next token at position `pos`.
  void set(uint32_t j, uint32_t pos, uint32_t token) {
    runtime::Control* c = ctl[j]->as<runtime::Control>();
    c->pos = pos;
    c->n_active = 1;
    c->cur_token[0] = token;
    if (seq[j].size() <= pos) seq[j].resize(pos + 1);
    seq[j][pos] = token;
  }
  // Launch j's readback (ids and e) against the host twin over the host ranges: the mismatching values.
  size_t check(uint32_t j, uint32_t pos, const PleTable& t) {
    std::vector<uint64_t> got(16);
    std::vector<uint16_t> eg(D);
    imm.copy(got.data(), ids[j]->ptr(), 16 * 8);
    imm.copy(eg.data(), e[j]->ptr(), size_t(D) * 2);
    const loader::Q4PleHistory hh = qr::ple_history(pos, [&](uint32_t p) { return seq[j][p]; });
    const std::array<uint64_t, 16> want = loader::q4_ple_ids(hh.t0, hh.t1, hh.t2, t.mult, t.sizes, t.offs);
    size_t bad = 0;
    for (uint32_t h = 0; h < 16; ++h) {
      bad += got[h] != want[h];
      const uint64_t r = want[h] - t.offs[h];
      uint16_t row[qk::kPleDim];
      qr::ple_row(t.q[h]->as<int8_t>() + r * qk::kPleDim, f32(t.s[h]->as<uint16_t>()[r]), row);
      for (uint32_t i = 0; i < qk::kPleDim; ++i) bad += eg[h * qk::kPleDim + i] != row[i];
    }
    return bad;
  }
};

int ple_rate(double gb) {
  const model::Qwen4ExpDesc& md = model::qwen4exp();
  const uint32_t REPLAYS = 10000;
  const size_t row_b = qk::kPleDim + 2;   // int8 row + bf16 scale (decision 7's default form)
  PleTable t;
  {
    const std::vector<uint64_t> primes = loader::q4_ple_primes(md.ple_base, qk::kPleHeads, 0);
    const double real = double(loader::q4_ple_total_rows(primes)) * double(row_b);
    const double f = gb * 1e9 / real;
    for (uint64_t p : primes) t.sizes.push_back(std::max<uint64_t>(1, uint64_t(double(p) * f)));
    t.offs = loader::q4_ple_offsets(t.sizes);
    t.mult = loader::q4_ple_multipliers(md.vocab, md.ngram, 0, md.ple_seed);
    t.bytes = size_t(loader::q4_ple_total_rows(t.sizes)) * row_b;
    std::printf("ple-rate: %.1f GB requested (the real table %.2f GB): 16 heads of %llu..%llu rows, %.2f GB of host USM\n", gb,
                real / 1e9, (unsigned long long)*std::min_element(t.sizes.begin(), t.sizes.end()),
                (unsigned long long)*std::max_element(t.sizes.begin(), t.sizes.end()), double(t.bytes) / 1e9);
  }
  try {
    loader::check_q4_ple_host_fit(t.bytes, loader::q4_mem_available());
  } catch (const std::exception& e) {
    std::printf("SKIP ple-rate: %s\n", e.what());
    return 77;
  }
  l0::Context c0(0);
  const uint32_t ndev = l0::Context::gpu_count();
  const auto t_alloc = std::chrono::steady_clock::now();
  for (uint32_t h = 0; h < 16; ++h) {
    t.q.push_back(std::make_unique<l0::Mem>(c0, l0::MemKind::Host, size_t(t.sizes[h]) * qk::kPleDim, loader::kQ4PlePage));
    t.s.push_back(std::make_unique<l0::Mem>(c0, l0::MemKind::Host, size_t(t.sizes[h]) * 2, loader::kQ4PlePage));
    uint64_t x = 0x9E3779B97F4A7C15ull * (h + 1);
    uint64_t* qw = t.q[h]->as<uint64_t>();
    for (size_t i = 0; i < t.q[h]->size() / 8; ++i) qw[i] = loader::q4_splitmix64(x + i);
    uint16_t* sw = t.s[h]->as<uint16_t>();
    for (uint64_t r = 0; r < t.sizes[h]; ++r) sw[r] = uint16_t(0x3C00u | (r % 251));   // ~0.0078 .. 0.0156, finite
  }
  for (uint32_t r = 0; r < 32; ++r) t.ptrs.push_back(reinterpret_cast<uint64_t>((r < 16 ? t.q[r] : t.s[r - 16])->ptr()));
  t.consts.assign(t.mult.begin(), t.mult.end());
  t.consts.insert(t.consts.end(), t.sizes.begin(), t.sizes.end());
  t.consts.insert(t.consts.end(), t.offs.begin(), t.offs.end());
  std::printf("ple-rate: pinned and filled in %.1f s; %u GPU(s)\n", seconds_since(t_alloc), ndev);
  std::unique_ptr<l0::Context> c1;
  if (ndev > 1) c1 = std::make_unique<l0::Context>(c0, 1);

  int bad_runs = 0;
  std::mt19937 rng(1234);
  for (uint32_t n : {1u, 16u}) {
    const char* mode = n == 1 ? "narrow" : "wide";
    std::vector<std::unique_ptr<PleRig>> rigs;
    rigs.push_back(std::make_unique<PleRig>(c0, t, n));
    if (c1) rigs.push_back(std::make_unique<PleRig>(*c1, t, n));
    // configurations: {0}, {1}, {0, 1}
    std::vector<std::vector<uint32_t>> cfgs = {{0}};
    if (c1) {
      cfgs.push_back({1});
      cfgs.push_back({0, 1});
    }
    for (const std::vector<uint32_t>& cfg : cfgs) {
      std::vector<double> ts;
      ts.reserve(REPLAYS);
      size_t bad = 0, checked = 0;
      for (uint32_t i = 0; i < REPLAYS; ++i) {
        for (uint32_t r : cfg)
          for (uint32_t j = 0; j < n; ++j) rigs[r]->set(j, i, rng() % md.vocab);
        const auto t0 = std::chrono::steady_clock::now();
        for (uint32_t r : cfg) rigs[r]->q.execute(rigs[r]->list, &rigs[r]->f);
        for (uint32_t r : cfg) rigs[r]->f.wait();
        ts.push_back(seconds_since(t0));
        for (uint32_t r : cfg) {   // every replay: every launch (narrow) or launch i % 16 (wide)
          bad += rigs[r]->check(i % n, i, t);
          ++checked;
        }
      }
      const double med = median(ts);
      const double tokens = double(n) * double(cfg.size());   // per replay
      const double rows = tokens * qk::kPleHeads;
      std::string devs;
      for (uint32_t r : cfg) devs += (devs.empty() ? "" : "+") + std::to_string(r);
      std::printf("ple-rate: %s, device %s: median %.2f us a replay (%u replays, %.0f tokens = %.0f rows each): %.3f us per token, "
                  "implied bus rate %.3f GB/s (%.0f rows x %zu B); readback %s (%zu launches checked)\n",
                  mode, devs.c_str(), med * 1e6, REPLAYS, tokens, rows, med * 1e6 / tokens, rows * double(row_b) / med / 1e9, rows,
                  row_b, bad == 0 ? "every row equal" : ("FAIL: " + std::to_string(bad) + " values differ").c_str(), checked);
      if (bad) ++bad_runs;
    }
  }
  return bad_runs ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
  bool bench = false, rate = false;
  double gb = 16.0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--bench-attn") {
      bench = true;
    } else if (a == "--ple-rate") {
      rate = true;
      if (i + 1 < argc && argv[i + 1][0] != '-') gb = std::atof(argv[++i]);
    } else {
      std::fprintf(stderr, "usage: qwen4exp_kernels_test [--bench-attn | --ple-rate [GB]]\n");
      return 2;
    }
  }
  try {
    if (l0::Context::gpu_count() == 0) {
      std::puts("SKIP qwen4exp_kernels_test: no GPU");
      return 77;
    }
  } catch (const std::exception& e) {
    std::printf("SKIP qwen4exp_kernels_test: no Level Zero GPU (%s)\n", e.what());
    return 77;
  }
  if (rate) return ple_rate(gb > 0 ? gb : 16.0);
  Dev d;
  if (bench) return bench_attn(d);
  std::printf("qwen4exp_kernels_test on %s\n", d.ctx.name().c_str());
  hc_tests(d);
  ple_tests(d);
  qsa_tests(d);
  moe_tests(d);
  eager_tests(d);
  gated_tests(d);
  flash_tests(d);
  if (failures) {
    std::printf("qwen4exp_kernels_test FAILED (%d)\n", failures);
    return 1;
  }
  std::puts("qwen4exp_kernels_test OK");
  return 0;
}
