// flash_split_test - 3l (2): the split-K prefill attention for short chunks deep in the cache
// (pf_flash_attn_split / pf_flash_attn_kv8{,_raw}_split + pf_flash_merge) against today's
// kernels (pf_flash_attn, kv8_widen + pf_flash_attn_kv8 raw) and an fp64 host reference.
//
//   flash_split_test            the check: per shape (pos, C) and span count, both paths on the
//                               same random Q / K / V (bf16, and int8 + fp16 scales), every pf_o
//                               word of rows < C finite; on sampled rows x 24 heads the fp64
//                               reference's cosine (worst) and max abs error of each path. Bar:
//                               the split's worst cosine >= 0.99999 and its max error <= 2 x
//                               today's + 1e-6 (the merge re-associates the softmax; it may not
//                               move further from the exact attention than today's rounding does).
//   flash_split_test --time <pos,...> [--rows 1,16,...] [--spans 0,512,...]
//                               kernel times: one FA layer's attention as each path issues it,
//                               16 times in one list (Qwen3.8's FA layers), arms interleaved in a
//                               rotating order after a warm-up, 7 rounds, medians. A span of 0 is
//                               today's path; int8 arms: today's widen + raw, widen + raw split,
//                               in-place split.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/kv8.h"
#include "kernels/kernels.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "pf_harness.h"

namespace {

constexpr uint32_t kQH = 24, kKvH = 4, kHD = 256, kKT = 64;

uint32_t pad256(uint32_t x) { return (x + 255u) & ~255u; }
uint32_t c8_of(uint32_t C) { return (C + 7u) & ~7u; }

// n spans of whole 64-key tiles over [0, pos): span = ceil(pos / n) rounded up to 64; the span
// count actually launched is ceil(pos / span), so every span starts below pos.
uint32_t span_for(uint32_t pos, uint32_t n) { return ((pos + n - 1) / n + kKT - 1) / kKT * kKT; }
uint32_t spans_of(uint32_t pos, uint32_t span) { return (pos + span - 1) / span; }

std::vector<uint32_t> list_of(const std::string& s) {
  std::vector<uint32_t> v;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, ',')) v.push_back(uint32_t(std::stoul(t)));
  return v;
}

// A cheap deterministic fill (the 128k caches are 134 M values a plane).
struct Lcg {
  uint64_t x;
  explicit Lcg(uint64_t s) : x(s * 6364136223846793005ull + 1442695040888963407ull) {}
  uint32_t next() { x = x * 6364136223846793005ull + 1442695040888963407ull; return uint32_t(x >> 33); }
  float uni() { return float(next() & 0xFFFFFF) / float(0x1000000) * 2.0f - 1.0f; }   // [-1, 1)
};

struct Caches {
  uint32_t depth = 0;
  std::vector<uint16_t> q, kb, vb;     // bf16 Q [C][24][256], K / V [depth][4][256]
  std::vector<int8_t> k8, v8;          // int8 rows
  std::vector<uint16_t> ks, vs;        // fp16 scales [depth][4]
};

Caches make(uint32_t pos, uint32_t C, float qscale) {
  Caches c;
  c.depth = pos + C;
  Lcg r(pos * 31u + C);
  c.q.resize(size_t(C) * kQH * kHD);
  for (auto& e : c.q) e = common::f32_to_bf16(r.uni() * 1.7f * qscale);
  const size_t n = size_t(c.depth) * kKvH * kHD;
  c.kb.resize(n);
  c.vb.resize(n);
  c.k8.resize(n);
  c.v8.resize(n);
  for (size_t i = 0; i < n; ++i) {
    c.kb[i] = common::f32_to_bf16(r.uni() * 1.7f);
    c.vb[i] = common::f32_to_bf16(r.uni() * 1.7f);
    c.k8[i] = int8_t(int(r.next() % 255u) - 127);
    c.v8[i] = int8_t(int(r.next() % 255u) - 127);
  }
  c.ks.resize(size_t(c.depth) * kKvH);
  c.vs.resize(c.ks.size());
  for (size_t i = 0; i < c.ks.size(); ++i) {
    c.ks[i] = common::kv8::f32_to_f16_rne(float(1 + r.next() % 100u) / 4096.0f);
    c.vs[i] = common::kv8::f32_to_f16_rne(float(1 + r.next() % 100u) / 4096.0f);
  }
  return c;
}

float f16f(uint16_t h) { return common::kv8::f16f(h); }

// fp64 attention of row r, head h (scale 1 / 16, key <= pos + r), int8 when `i8`.
std::vector<double> reference(const Caches& c, uint32_t pos, uint32_t r, uint32_t h, bool i8) {
  const uint32_t j = h / (kQH / kKvH), n = pos + r + 1;
  std::vector<double> s(n), o(kHD, 0.0);
  double mx = -INFINITY;
  const uint16_t* q = &c.q[(size_t(r) * kQH + h) * kHD];
  for (uint32_t t = 0; t < n; ++t) {
    const size_t at = (size_t(t) * kKvH + j) * kHD;
    const double sk = i8 ? f16f(c.ks[size_t(t) * kKvH + j]) : 1.0;
    double d = 0;
    for (uint32_t x = 0; x < kHD; ++x)
      d += double(common::bf16_to_f32(q[x])) * (i8 ? double(c.k8[at + x]) * sk : double(common::bf16_to_f32(c.kb[at + x])));
    s[t] = d / 16.0;
    mx = std::max(mx, s[t]);
  }
  double l = 0;
  for (uint32_t t = 0; t < n; ++t) {
    const double p = std::exp(s[t] - mx);
    l += p;
    const size_t at = (size_t(t) * kKvH + j) * kHD;
    const double sv = i8 ? f16f(c.vs[size_t(t) * kKvH + j]) : 1.0;
    for (uint32_t x = 0; x < kHD; ++x)
      o[x] += p * (i8 ? double(c.v8[at + x]) * sv : double(common::bf16_to_f32(c.vb[at + x])));
  }
  for (auto& e : o) e /= l;
  return o;
}

struct Dev2 : pf_harness::Dev {
  l0::Module fm{ctx, kernels::path("pf_flash_attn")};
  l0::Module sm{ctx, kernels::path("pf_flash_attn_split")};
  l0::Module mm{ctx, kernels::path("pf_flash_merge")};
  l0::Module rm{ctx, kernels::path("pf_flash_attn_kv8_raw")};
  l0::Module rsm{ctx, kernels::path("pf_flash_attn_kv8_raw_split")};
  l0::Module ism{ctx, kernels::path("pf_flash_attn_kv8_split")};
  l0::Kernel flash{fm, "pf_flash_attn"}, split{sm, "pf_flash_attn"}, merge{mm, "pf_flash_merge"};
  l0::Kernel raw{rm, "pf_flash_attn_kv8"}, widen{rm, "kv8_widen"}, raw_split{rsm, "pf_flash_attn_kv8"};
  l0::Kernel in_split{ism, "pf_flash_attn_kv8"};
  Dev2() {
    for (l0::Kernel* k : {&flash, &split, &raw, &raw_split, &in_split}) k->group_size(96);
    merge.group_size(256);
    widen.group_size(256);
  }
};

// The device side of one shape: every buffer, and the launches of each arm appended to a list.
struct Shape {
  uint32_t pos, C, rows, c8;
  l0::Mem q, kb, vb, k8, v8, ks, vs, raw, o, part;
  Shape(Dev2& d, const Caches& c, uint32_t pos_, uint32_t C_, uint32_t max_spans)
      : pos(pos_), C(C_), rows(pad256(C_)), c8(c8_of(C_)),
        q(pf_harness::upload(d.ctx, d.imm, c.q)), kb(pf_harness::upload(d.ctx, d.imm, c.kb)),
        vb(pf_harness::upload(d.ctx, d.imm, c.vb)), k8(pf_harness::upload(d.ctx, d.imm, c.k8)),
        v8(pf_harness::upload(d.ctx, d.imm, c.v8)), ks(pf_harness::upload(d.ctx, d.imm, c.ks)),
        vs(pf_harness::upload(d.ctx, d.imm, c.vs)),
        raw(d.ctx, l0::MemKind::Device, c.kb.size() * 2 * sizeof(uint16_t)),
        o(d.ctx, l0::MemKind::Device, size_t(kQH) * rows * kHD * sizeof(float)),
        part(d.ctx, l0::MemKind::Device, size_t(max_spans) * kQH * c8 * (kHD + 2) * sizeof(float)) {}
  uint16_t* kraw() { return raw.as<uint16_t>(); }
  uint16_t* vraw() { return raw.as<uint16_t>() + size_t(pos + C) * kKvH * kHD; }
  float* ml(uint32_t n) { return part.as<float>() + size_t(n) * kQH * c8 * kHD; }
};

enum Form { kBf16, kRaw, kInPlace };
const char* form_name(Form f) { return f == kBf16 ? "bf16" : f == kRaw ? "int8 raw" : "int8 in-place"; }

// Appends one FA layer's attention: today's path at span 0, else the split at `span`.
void append(Dev2& d, l0::CmdList& list, Shape& s, Form f, uint32_t span) {
  const uint32_t gx = (s.C + 7u) / 8u, count = (s.pos + s.C) * kKvH * kHD;
  if (f == kRaw) {
    d.widen.arg_ptr(0, s.k8.ptr());
    d.widen.arg_ptr(1, s.v8.ptr());
    d.widen.arg_ptr(2, s.kraw());
    d.widen.arg_ptr(3, s.vraw());
    d.widen.arg(4, count);
    list.launch(d.widen, (count + 4095u) / 4096u, 1, 1);
  }
  const uint32_t n = span ? spans_of(s.pos, span) : 1;
  l0::Kernel& k = f == kBf16 ? (span ? d.split : d.flash) : f == kRaw ? (span ? d.raw_split : d.raw) : d.in_split;
  CHECK(f != kInPlace || span != 0);
  uint32_t a = 0;
  k.arg_ptr(a++, s.q.ptr());
  if (f == kBf16) {
    k.arg_ptr(a++, s.kb.ptr());
    k.arg_ptr(a++, s.vb.ptr());
  } else {
    k.arg_ptr(a++, f == kRaw ? static_cast<void*>(s.kraw()) : s.k8.ptr());
    k.arg_ptr(a++, s.ks.ptr());
    k.arg_ptr(a++, f == kRaw ? static_cast<void*>(s.vraw()) : s.v8.ptr());
    k.arg_ptr(a++, s.vs.ptr());
  }
  k.arg_ptr(a++, span ? s.part.ptr() : s.o.ptr());
  k.arg(a++, s.pos);
  k.arg(a++, s.C);
  k.arg(a++, s.rows);
  if (span) {
    k.arg_ptr(a++, s.ml(n));
    k.arg(a++, span);
  }
  list.launch(k, gx, kKvH, n);
  if (!span) return;
  d.merge.arg_ptr(0, s.part.ptr());
  d.merge.arg_ptr(1, s.ml(n));
  d.merge.arg_ptr(2, s.o.ptr());
  d.merge.arg(3, s.C);
  d.merge.arg(4, s.rows);
  d.merge.arg(5, n);
  list.launch(d.merge, s.C, kQH, 1);
}

std::vector<float> run_once(Dev2& d, Shape& s, Form f, uint32_t span) {
  d.imm.fill(s.o.ptr(), 0x7FC00000u, s.o.size());   // NaN: a row not written fails the finiteness check
  auto list = l0::CmdList::regular(d.ctx);
  append(d, list, s, f, span);
  list.close();
  d.q.execute(list, &d.fence);
  d.fence.wait();
  std::vector<float> out(s.o.size() / sizeof(float));
  pf_harness::download(d.imm, out, s.o);
  return out;
}

struct Err {
  double worst_cos = 1.0, max_abs = 0.0;
};

Err against(const std::vector<float>& got, const std::vector<std::vector<double>>& ref,
            const std::vector<std::pair<uint32_t, uint32_t>>& rh, uint32_t rows) {
  Err e;
  for (size_t i = 0; i < rh.size(); ++i) {
    const float* g = &got[(size_t(rh[i].second) * rows + rh[i].first) * kHD];
    double ab = 0, aa = 0, bb = 0;
    for (uint32_t x = 0; x < kHD; ++x) {
      ab += g[x] * ref[i][x];
      aa += double(g[x]) * g[x];
      bb += ref[i][x] * ref[i][x];
      e.max_abs = std::max(e.max_abs, std::fabs(g[x] - ref[i][x]));
    }
    e.worst_cos = std::min(e.worst_cos, ab / std::sqrt(aa * bb));
  }
  return e;
}

int check(Dev2& d) {
  struct Case { uint32_t pos, C, n; };
  const std::vector<Case> cases = {{64, 1, 1},     {65, 1, 2},      {1000, 16, 4},   {4095, 7, 16},
                                   {8191, 64, 8},  {32768, 1, 128}, {32768, 16, 64}, {32768, 128, 16},
                                   {32700, 33, 9}, {131072, 1, 256}, {131071, 64, 32}, {131072, 128, 14},
                                   {100000, 256, 7}};
  bool ok = true;
  for (const Case& cs : cases) {
    const Caches c = make(cs.pos, cs.C, 2.0f);
    const uint32_t span = span_for(cs.pos, cs.n), n = spans_of(cs.pos, span);
    Shape s(d, c, cs.pos, cs.C, n);
    // sampled rows x every head, the fp64 reference of both forms
    std::vector<uint32_t> rws = {0, cs.C - 1, cs.C / 2, std::min(cs.C - 1, 7u)};
    std::sort(rws.begin(), rws.end());
    rws.erase(std::unique(rws.begin(), rws.end()), rws.end());
    std::vector<std::pair<uint32_t, uint32_t>> rh;
    for (uint32_t r : rws)
      for (uint32_t h = 0; h < kQH; ++h) rh.push_back({r, h});
    for (Form f : {kBf16, kRaw, kInPlace}) {
      std::vector<std::vector<double>> ref;
      for (auto [r, h] : rh) ref.push_back(reference(c, cs.pos, r, h, f != kBf16));
      const auto base = run_once(d, s, f == kInPlace ? kRaw : f, 0);
      const auto got = run_once(d, s, f, span);
      size_t nonfinite = 0;
      double diff = 0, mag = 0;
      for (uint32_t h = 0; h < kQH; ++h)
        for (uint32_t r = 0; r < cs.C; ++r)
          for (uint32_t x = 0; x < kHD; ++x) {
            const size_t i = (size_t(h) * s.rows + r) * kHD + x;
            nonfinite += !std::isfinite(got[i]) || !std::isfinite(base[i]);
            diff = std::max(diff, double(std::fabs(got[i] - base[i])));
            mag = std::max(mag, double(std::fabs(base[i])));
          }
      const Err eb = against(base, ref, rh, s.rows), es = against(got, ref, rh, s.rows);
      const bool pass = nonfinite == 0 && es.worst_cos >= 0.99999 && es.max_abs <= 2.0 * eb.max_abs + 1e-6;
      std::printf("pos %6u C %3u %-13s spans %3u (span %6u): today cos %.8f err %.3g | split cos %.8f err %.3g"
                  " | split - today max %.3g of %.3g %s\n",
                  cs.pos, cs.C, form_name(f), n, span, eb.worst_cos, eb.max_abs, es.worst_cos, es.max_abs, diff, mag,
                  pass ? "ok" : "FAIL");
      std::fflush(stdout);
      ok = ok && pass;
    }
  }
  std::puts(ok ? "flash_split_test: PASS" : "flash_split_test: FAIL");
  return ok ? 0 : 1;
}

int time_arms(Dev2& d, const std::vector<uint32_t>& poss, const std::vector<uint32_t>& rowsv,
              const std::vector<uint32_t>& spans) {
  using Clock = std::chrono::steady_clock;
  for (uint32_t pos : poss)
    for (uint32_t C : rowsv) {
      const Caches c = make(pos, C, 2.0f);
      uint32_t max_n = 1;
      for (uint32_t sp : spans)
        if (sp) max_n = std::max(max_n, spans_of(pos, sp));
      Shape s(d, c, pos, C, max_n);
      struct Arm { Form f; uint32_t span; std::string name; l0::CmdList list; };
      std::vector<Arm> arms;
      for (Form f : {kBf16, kRaw, kInPlace})
        for (uint32_t sp : spans) {
          if (f == kInPlace && sp == 0) continue;
          if (sp && sp >= pos + kKT) continue;   // one span: the merge alone
          Arm a{f, sp, std::string(form_name(f)) + (sp ? " span " + std::to_string(sp) + " n " +
                                                             std::to_string(spans_of(pos, sp))
                                                       : " today"),
                l0::CmdList::regular(d.ctx)};
          for (uint32_t layer = 0; layer < 16; ++layer) append(d, a.list, s, f, sp);
          a.list.close();
          arms.push_back(std::move(a));
        }
      auto once = [&](Arm& a) {
        const auto t0 = Clock::now();
        d.q.execute(a.list, &d.fence);
        d.fence.wait();
        return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
      };
      for (int w = 0; w < 2; ++w)
        for (auto& a : arms) once(a);
      std::vector<std::vector<double>> ms(arms.size());
      for (uint32_t round = 0; round < 7; ++round)
        for (size_t i = 0; i < arms.size(); ++i) {
          const size_t a = (round + i) % arms.size();
          ms[a].push_back(once(arms[a]));
        }
      for (size_t a = 0; a < arms.size(); ++a) {
        std::sort(ms[a].begin(), ms[a].end());
        std::printf("time pos %6u C %4u %-32s 16 layers %9.3f ms [%.3f, %.3f]\n", pos, C, arms[a].name.c_str(),
                    ms[a][3], ms[a].front(), ms[a].back());
      }
      std::fflush(stdout);
    }
  std::puts("flash_split_test --time: done");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Dev2 d;
  if (argc > 1 && std::string(argv[1]) == "--time") {
    CHECK(argc > 2);
    const auto poss = list_of(argv[2]);
    std::vector<uint32_t> rowsv = {1, 16, 64, 128, 256, 512}, spans = {0, 512, 1024, 2048, 4096, 8192, 16384};
    for (int i = 3; i + 1 < argc; i += 2) {
      if (std::string(argv[i]) == "--rows") rowsv = list_of(argv[i + 1]);
      else if (std::string(argv[i]) == "--spans") spans = list_of(argv[i + 1]);
      else CHECK(false);
    }
    return time_arms(d, poss, rowsv, spans);
  }
  return check(d);
}
