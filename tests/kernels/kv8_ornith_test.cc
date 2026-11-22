// kv8_ornith_test - PERF-64: kv8.cl's binaries at Ornith's 16 q / 2 kv heads (`_Q16KV2`) BITWISE their Qwen3.8 twins
// on the same heads. Every kernel here computes a q head (or a kv head's rows) from that head's inputs alone, so the
// Ornith build run on Ornith-shaped buffers and the Qwen3.8 build run on the same values laid out at 24 / 4 give the
// same bits head for head; the Qwen3.8 twins are held to src/common/kv8.h and the fp64 attention by
// kv8_kernels_test / kv8_flash_bitwise_test / flash_split_test. No checkpoint.
//
//   W  the writers: attn_prep_M<M>_Q16KV2_KV8 (M = 1..4, QKV_S 2, n_active < M too), the MTP head's _S1 build and the
//      prefill build pf_attn_prep_kv8_Q16KV2 - q (rotated), the gate, the int8 K / V rows and their fp16 scales, and the
//      rows outside the written window keep their canary. Head h and kv head j map to Qwen3.8's h and j.
//   D  decode v2 over int8 (attn_decode_v2_kv8 + attn_reduce_v2_kv8) at M = 1..4: Ornith's q head 8 j + i reads kv head
//      j; on Qwen3.8's 24 / 4 it is head 12 j + i over kv heads 2 j and 2 j + 1, both holding Ornith's kv head j.
//   F  prefill: kv8_widen + the raw 8-row flash (HPW 8), the raw pair, the split read in place and the raw split (each
//      with pf_flash_merge), pf_o head for head; the gate pf_attn_gate_kv8 on the merged rows.
//   P  the paged prefill twins (the sequence slots, the pool): the writer, the raw flash (its kv8_widen), the pair, the
//      two splits - over a scattered page table, bitwise the unpaged Ornith binaries on the same positions.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "common/kv8.h"
#include "kernels/kernels.h"
#include "kernels/prefill/pf_kernels.h"
#include "kernels/shape_suffix.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "prefill/pf_harness.h"
#include "runtime/buffers.h"
#include "runtime/control.h"

namespace {

constexpr uint32_t kHD = 256, kTgt = runtime::DecodeScratch::kAttnV2Blocks, kPage = kernels::kKvPage;

struct Shape {
  uint32_t q, kv;
  uint32_t gqa() const { return q / kv; }
  uint32_t qkv_n() const { return (2 * q + 2 * kv) * kHD; }
  uint32_t k_off() const { return 2 * q * kHD; }
  uint32_t v_off() const { return k_off() + kv * kHD; }
};
constexpr Shape kOrn{16, 2}, kQwen{24, 4};

// runtime::prefill::flash_split_count: the spans of `span` keys a split chunk at `pos` walks.
uint32_t split_count(uint32_t pos, uint32_t span) { return (pos + span - 1) / span; }

// Ornith's q head h = 8 j + i on Qwen3.8's heads (kv heads 2 j, 2 j + 1 each hold Ornith's kv head j).
uint32_t qwen_head(uint32_t h) { return 12 * (h / 8) + h % 8; }

std::mt19937 g_rng(64);
std::vector<float> rnd_f32(size_t n, float sigma) {
  std::normal_distribution<float> nd(0.0f, sigma);
  std::vector<float> v(n);
  for (float& e : v) {
    e = nd(g_rng);
    if (e == 0.0f) e = 1e-3f;   // no -0.0f: the +0.0f slice padding stays exact
  }
  return v;
}
std::vector<uint16_t> rnd_bf16(size_t n, float sigma) {
  std::vector<uint16_t> v(n);
  const std::vector<float> f = rnd_f32(n, sigma);
  for (size_t i = 0; i < n; ++i) v[i] = common::f32_to_bf16(f[i]);
  return v;
}
// An int8 cache of L positions at `kv` heads: rows and fp16 scales.
struct Cache {
  std::vector<int8_t> k, v;
  std::vector<uint16_t> ks, vs;
};
Cache rnd_cache(uint32_t L, uint32_t kv) {
  Cache c;
  c.k.resize(size_t(L) * kv * kHD);
  c.v.resize(c.k.size());
  c.ks.resize(size_t(L) * kv);
  c.vs.resize(c.ks.size());
  std::uniform_int_distribution<int> b(-127, 127), s(1, 100);
  for (size_t i = 0; i < c.k.size(); ++i) {
    c.k[i] = int8_t(b(g_rng));
    c.v[i] = int8_t(b(g_rng));
  }
  for (size_t i = 0; i < c.ks.size(); ++i) {
    c.ks[i] = common::kv8::f32_to_f16_rne(float(s(g_rng)) / 4096.0f);
    c.vs[i] = common::kv8::f32_to_f16_rne(float(s(g_rng)) / 4096.0f);
  }
  return c;
}
// Ornith's cache on Qwen3.8's 4 kv heads: kv head j' is Ornith's j' / 2.
Cache to_qwen(const Cache& o, uint32_t L) {
  Cache c;
  c.k.resize(size_t(L) * 4 * kHD);
  c.v.resize(c.k.size());
  c.ks.resize(size_t(L) * 4);
  c.vs.resize(c.ks.size());
  for (uint32_t p = 0; p < L; ++p)
    for (uint32_t j = 0; j < 4; ++j) {
      const size_t src = size_t(p) * 2 + j / 2, dst = size_t(p) * 4 + j;
      std::memcpy(&c.k[dst * kHD], &o.k[src * kHD], kHD);
      std::memcpy(&c.v[dst * kHD], &o.v[src * kHD], kHD);
      c.ks[dst] = o.ks[src];
      c.vs[dst] = o.vs[src];
    }
  return c;
}
// Per-row q-head rows [rows][heads][256] of Ornith on Qwen3.8's heads by qwen_head (other heads zero).
template <class T>
std::vector<T> heads_to_qwen(const std::vector<T>& o, uint32_t rows) {
  std::vector<T> q(size_t(rows) * kQwen.q * kHD, T(0));
  for (uint32_t m = 0; m < rows; ++m)
    for (uint32_t h = 0; h < kOrn.q; ++h)
      std::memcpy(&q[(size_t(m) * kQwen.q + qwen_head(h)) * kHD], &o[(size_t(m) * kOrn.q + h) * kHD], kHD * sizeof(T));
  return q;
}

pf_harness::Dev* g_dev = nullptr;
template <class T>
l0::Mem up(const std::vector<T>& v) { return pf_harness::upload(g_dev->ctx, g_dev->imm, v); }
template <class T>
std::vector<T> down(const l0::Mem& m) {
  std::vector<T> v(m.size() / sizeof(T));
  pf_harness::download(g_dev->imm, v, m);
  return v;
}
l0::Mem dev_mem(size_t bytes, uint32_t fill = 0) {
  l0::Mem m(g_dev->ctx, l0::MemKind::Device, std::max<size_t>(bytes, 4));
  g_dev->imm.fill(m.ptr(), fill, m.size());
  return m;
}
template <class F>
void run(F append) {
  auto list = l0::CmdList::regular(g_dev->ctx);
  append(list);
  list.close();
  g_dev->q.execute(list, &g_dev->fence);
  g_dev->fence.wait();
}

size_t g_bad = 0;
void verdict(const char* part, const std::string& what, size_t bad) {
  std::printf("%s %-46s %zu mismatches -- %s\n", part, what.c_str(), bad, bad ? "FAIL" : "PASS");
  g_bad += bad;
}
template <class T>
size_t diff(const T* a, const T* b, size_t n) {
  size_t d = 0;
  for (size_t i = 0; i < n; ++i) d += std::memcmp(a + i, b + i, sizeof(T)) != 0;
  return d;
}

// --- W: the writers ----------------------------------------------------------------------------------------------

struct WriterOut {
  std::vector<uint8_t> q, gate;   // fp32 (decode) or bf16 (PF) q rows; fp32 gate
  std::vector<int8_t> k, v;
  std::vector<uint16_t> ks, vs;
};
// One writer launch at `sh`: partials [S][rows][qkv_n], the cache of L positions pre-filled with canaries.
WriterOut run_writer(const std::string& variant, Shape sh, uint32_t S, uint32_t rows, uint32_t n_act, uint32_t pos,
                     uint32_t L, bool pf, const std::vector<float>& partials, const std::vector<float>& fa,
                     const std::vector<float>& rope, const std::vector<uint32_t>* pages = nullptr) {
  l0::Module mod(g_dev->ctx, kernels::path(variant));
  l0::Kernel k(mod, "attn_prep_kv8");
  k.group_size(256);
  l0::Mem ctrl(g_dev->ctx, l0::MemKind::Shared, sizeof(runtime::Control));
  auto* c = ctrl.as<runtime::Control>();
  *c = runtime::Control{};
  c->pos = pos;
  c->n_active = n_act;
  (void)S;
  l0::Mem dp = up(partials), dfa = up(fa), drope = up(rope);
  l0::Mem dq = dev_mem(size_t(rows) * sh.q * kHD * 4), dg = dev_mem(size_t(rows) * sh.q * kHD * 4);
  l0::Mem dk = dev_mem(size_t(L) * sh.kv * kHD, 0x5A5A5A5Au), dv = dev_mem(size_t(L) * sh.kv * kHD, 0x5A5A5A5Au);
  l0::Mem dks = dev_mem(size_t(L) * sh.kv * 2, 0xBEEFBEEFu), dvs = dev_mem(size_t(L) * sh.kv * 2, 0xBEEFBEEFu);
  std::unique_ptr<l0::Mem> dpg;
  k.arg_ptr(0, ctrl.ptr());
  k.arg_ptr(1, dp.ptr());
  k.arg_ptr(2, dfa.ptr());
  k.arg_ptr(3, drope.ptr());
  k.arg_ptr(4, dq.ptr());
  k.arg_ptr(5, pf ? nullptr : dg.ptr());
  k.arg_ptr(6, dk.ptr());
  k.arg_ptr(7, dv.ptr());
  k.arg_ptr(8, dks.ptr());
  k.arg_ptr(9, dvs.ptr());
  if (pages) {
    dpg = std::make_unique<l0::Mem>(up(*pages));
    k.arg_ptr(10, dpg->ptr());
  }
  g_dev->run(k, sh.q + sh.kv, rows);
  WriterOut o;
  o.q = down<uint8_t>(dq);
  if (pf) o.q.resize(size_t(rows) * sh.q * kHD * 2);
  o.gate = down<uint8_t>(dg);
  o.k = down<int8_t>(dk);
  o.v = down<int8_t>(dv);
  o.ks = down<uint16_t>(dks);
  o.vs = down<uint16_t>(dvs);
  return o;
}

// Ornith's partials [S][rows][9216] on Qwen3.8's [S][rows][14336]: q || gate of head h, K and V of kv head j at theirs.
std::vector<float> partials_to_qwen(const std::vector<float>& o, uint32_t S, uint32_t rows) {
  std::vector<float> q(size_t(S) * rows * kQwen.qkv_n(), 0.0f);
  for (uint32_t r = 0; r < S * rows; ++r) {
    const float* src = &o[size_t(r) * kOrn.qkv_n()];
    float* dst = &q[size_t(r) * kQwen.qkv_n()];
    std::memcpy(dst, src, size_t(kOrn.k_off()) * 4);
    std::memcpy(dst + kQwen.k_off(), src + kOrn.k_off(), size_t(kOrn.kv) * kHD * 4);
    std::memcpy(dst + kQwen.v_off(), src + kOrn.v_off(), size_t(kOrn.kv) * kHD * 4);
  }
  return q;
}

// Ornith's writer against Qwen3.8's on the same heads: rows of q / gate (head h <-> h), K / V / scales (j <-> j).
size_t cmp_writer(const WriterOut& o, const WriterOut& w, uint32_t rows, uint32_t L, bool pf) {
  const size_t qb = pf ? 2 : 4;
  size_t bad = 0;
  for (uint32_t m = 0; m < rows; ++m) {
    const size_t on = size_t(m) * kOrn.q * kHD * qb, wn = size_t(m) * kQwen.q * kHD * qb, n = size_t(kOrn.q) * kHD * qb;
    bad += diff(&o.q[on], &w.q[wn], n);
    if (!pf) bad += diff(&o.gate[size_t(m) * kOrn.q * kHD * 4], &w.gate[size_t(m) * kQwen.q * kHD * 4],
                         size_t(kOrn.q) * kHD * 4);
  }
  for (uint32_t p = 0; p < L; ++p)
    for (uint32_t j = 0; j < kOrn.kv; ++j) {
      const size_t a = size_t(p) * kOrn.kv + j, b = size_t(p) * kQwen.kv + j;
      bad += diff(&o.k[a * kHD], &w.k[b * kHD], kHD) + diff(&o.v[a * kHD], &w.v[b * kHD], kHD);
      bad += (o.ks[a] != w.ks[b]) + (o.vs[a] != w.vs[b]);
    }
  return bad;
}

std::vector<float> rope_table(uint32_t L) {
  std::vector<float> rope(size_t(L) * 64);
  for (uint32_t p = 0; p < L; ++p)
    for (uint32_t i = 0; i < 32; ++i) {
      const double th = p * std::pow(1e7, -double(i) / 32.0);
      rope[size_t(p) * 64 + i] = float(std::cos(th));
      rope[size_t(p) * 64 + 32 + i] = float(std::sin(th));
    }
  return rope;
}

void part_w() {
  constexpr uint32_t L = 128;
  std::vector<float> fa = rnd_f32(512, 0.3f);
  for (float& e : fa) e += 1.0f;
  const std::vector<float> rope = rope_table(2 * kPage);
  for (uint32_t M = 1; M <= 4; ++M) {
    std::vector<float> part = rnd_f32(size_t(2) * M * kOrn.qkv_n(), 0.7f);
    for (uint32_t n_act : {M, M > 1 ? M - 1 : M}) {
      const uint32_t pos = 37 + M;
      const auto o = run_writer(kernels::attn_prep_kv8_variant(M, 16, 2), kOrn, 2, M, n_act, pos, L, false, part, fa,
                                rope);
      const auto w = run_writer(kernels::attn_prep_kv8_variant(M), kQwen, 2, M, n_act, pos, L, false,
                                partials_to_qwen(part, 2, M), fa, rope);
      // Every position of the cache is compared, so the canaries outside the window are Qwen3.8's (kv8_kernels_test).
      verdict("W", kernels::attn_prep_kv8_variant(M, 16, 2) + " n_active " + std::to_string(n_act),
              cmp_writer(o, w, n_act, L, false));
    }
    const std::string s1 = kernels::attn_prep_s1_kv8_variant(M, 16, 2);
    if (!std::ifstream(kernels::path(s1)).good()) {
      std::printf("W %s: not built (B70_MTP off), skipped\n", s1.c_str());
      continue;
    }
    std::vector<float> one = rnd_f32(size_t(M) * kOrn.qkv_n(), 0.7f);
    const auto o = run_writer(s1, kOrn, 1, M, M, 5, L, false, one, fa, rope);
    const auto w = run_writer(kernels::attn_prep_s1_kv8_variant(M), kQwen, 1, M, M, 5, L, false,
                              partials_to_qwen(one, 1, M), fa, rope);
    verdict("W", s1, cmp_writer(o, w, M, L, false));
  }
  constexpr uint32_t C = 37, pos = 200, PL = 256;
  const std::vector<float> part = rnd_f32(size_t(C) * kOrn.qkv_n(), 0.7f);
  const auto o = run_writer(kernels::pf_attn_prep_kv8_variant(16, 2), kOrn, 1, C, C, pos, PL, true, part, fa, rope);
  const auto w = run_writer(kernels::pf_attn_prep_kv8_variant(), kQwen, 1, C, C, pos, PL, true,
                            partials_to_qwen(part, 1, C), fa, rope);
  verdict("W", kernels::pf_attn_prep_kv8_variant(16, 2), cmp_writer(o, w, C, PL, true));
  // P: the paged prefill writer over a reversed two-page table, rows straddling the page edge.
  {
    constexpr uint32_t PC = 40, ppos = kPage - 20, PL2 = 2 * kPage;
    const std::vector<uint32_t> pages = {1, 0};
    const std::vector<float> pp = rnd_f32(size_t(PC) * kOrn.qkv_n(), 0.7f);
    const auto a = run_writer(kernels::pf_attn_prep_kv8_variant(16, 2), kOrn, 1, PC, PC, ppos, PL2, true, pp, fa, rope);
    const auto b = run_writer(kernels::kv_paged(kernels::pf_attn_prep_kv8_variant(16, 2)), kOrn, 1, PC, PC, ppos, PL2,
                              true, pp, fa, rope, &pages);
    size_t bad = diff(a.q.data(), b.q.data(), a.q.size());
    for (uint32_t p = 0; p < PL2; ++p) {
      const uint32_t r = pages[p / kPage] * kPage + p % kPage;
      bad += diff(&a.k[size_t(p) * 2 * kHD], &b.k[size_t(r) * 2 * kHD], 2 * kHD) +
             diff(&a.v[size_t(p) * 2 * kHD], &b.v[size_t(r) * 2 * kHD], 2 * kHD) +
             diff(&a.ks[size_t(p) * 2], &b.ks[size_t(r) * 2], 2) + diff(&a.vs[size_t(p) * 2], &b.vs[size_t(r) * 2], 2);
    }
    verdict("P", kernels::kv_paged(kernels::pf_attn_prep_kv8_variant(16, 2)), bad);
  }
}

// --- D: decode v2 over int8 --------------------------------------------------------------------------------------

std::vector<uint16_t> run_decode(const std::string& variant, Shape sh, const Cache& c, uint32_t M, uint32_t pos,
                                 const std::vector<float>& q, const std::vector<float>& g) {
  l0::Module mod(g_dev->ctx, kernels::path(variant));
  l0::Kernel dec(mod, "attn_decode_v2_kv8"), red(mod, "attn_reduce_v2_kv8");
  dec.group_size(256);
  red.group_size(256);
  l0::Mem ctrl(g_dev->ctx, l0::MemKind::Shared, sizeof(runtime::Control));
  auto* cc = ctrl.as<runtime::Control>();
  *cc = runtime::Control{};
  cc->pos = pos;
  cc->n_active = M;
  l0::Mem dk = up(c.k), dv = up(c.v), dks = up(c.ks), dvs = up(c.vs), dq = up(q), dg = up(g);
  l0::Mem part = dev_mem(size_t(sh.q) * kTgt * M * 258 * 4, 0x7FC00000u);
  l0::Mem out = dev_mem(size_t(M) * sh.q * kHD * 2);
  dec.arg_ptr(0, ctrl.ptr());
  dec.arg_ptr(1, dq.ptr());
  dec.arg_ptr(2, dk.ptr());
  dec.arg_ptr(3, dks.ptr());
  dec.arg_ptr(4, dv.ptr());
  dec.arg_ptr(5, dvs.ptr());
  dec.arg_ptr(6, part.ptr());
  red.arg_ptr(0, ctrl.ptr());
  red.arg_ptr(1, part.ptr());
  red.arg_ptr(2, dg.ptr());
  red.arg_ptr(3, out.ptr());
  g_dev->run2(dec, sh.kv, kTgt, red, sh.q, M);
  return down<uint16_t>(out);
}

void part_d() {
  constexpr uint32_t L = 4096;
  const Cache oc = rnd_cache(L, 2), qc = to_qwen(oc, L);
  for (uint32_t M = 1; M <= 4; ++M)
    for (uint32_t pos : {0u, 63u, 2047u, 4092u}) {
      if (pos + M > L) continue;
      std::vector<float> q = rnd_f32(size_t(M) * kOrn.q * kHD, 1.0f), g = rnd_f32(size_t(M) * kOrn.q * kHD, 1.5f);
      for (size_t i = 0; i < q.size(); ++i) q[i] *= (i / kHD) % 3 == 0 ? 3.0f : 1.0f;
      for (float& e : g) e = common::bf16_to_f32(common::f32_to_bf16(e));
      const auto a = run_decode(kernels::attn_v2_kv8_variant(M, kTgt, 16, 2), kOrn, oc, M, pos, q, g);
      const auto b = run_decode(kernels::attn_v2_kv8_variant(M, kTgt), kQwen, qc, M, pos, heads_to_qwen(q, M),
                                heads_to_qwen(g, M));
      size_t bad = 0;
      for (uint32_t m = 0; m < M; ++m)
        for (uint32_t h = 0; h < kOrn.q; ++h)
          bad += diff(&a[(size_t(m) * kOrn.q + h) * kHD], &b[(size_t(m) * kQwen.q + qwen_head(h)) * kHD], kHD);
      verdict("D", kernels::attn_v2_kv8_variant(M, kTgt, 16, 2) + " pos " + std::to_string(pos), bad);
    }
}

// --- F / P: the prefill flash forms ------------------------------------------------------------------------------

// One chunk's operands at `sh`: bf16 rotated q [C][q][256], the int8 cache (and its widened twin, built by kv8_widen).
struct Chunk {
  Shape sh;
  uint32_t pos, C, rows, c8;
  l0::Mem q, k, v, ks, vs, kr, vr, o, part;
  std::unique_ptr<l0::Mem> pages;
  Chunk(Shape s, uint32_t p, uint32_t c, const std::vector<uint16_t>& qv, const Cache& cache, uint32_t n_split)
      : sh(s), pos(p), C(c), rows((c + 255u) & ~255u), c8((c + 7u) & ~7u), q(up(qv)), k(up(cache.k)), v(up(cache.v)),
        ks(up(cache.ks)), vs(up(cache.vs)), kr(dev_mem(cache.k.size() * 2)), vr(dev_mem(cache.v.size() * 2)),
        o(dev_mem(size_t(s.q) * rows * kHD * 4)),
        part(dev_mem(size_t(std::max(n_split, 1u)) * s.q * c8 * (kHD + 2) * 4)) {}
  float* ml(uint32_t n) { return part.as<float>() + size_t(n) * sh.q * c8 * kHD; }
};

enum class Form { Raw, Pair, SplitInPlace, SplitRaw };
const char* form_name(Form f) {
  return f == Form::Raw ? "raw" : f == Form::Pair ? "raw_pair" : f == Form::SplitInPlace ? "split" : "raw_split";
}
std::string form_variant(Form f, Shape sh) {
  const unsigned q = sh.q, kv = sh.kv;
  return f == Form::Raw ? kernels::pf_flash_attn_kv8_raw_variant(q, kv)
         : f == Form::Pair ? kernels::pf_flash_attn_kv8_raw_pair_variant(q, kv)
         : f == Form::SplitInPlace ? kernels::pf_flash_attn_kv8_split_variant(q, kv)
                                   : kernels::pf_flash_attn_kv8_raw_split_variant(q, kv);
}

// The form over one chunk into ch.o (merged for the splits); `paged` binds the _PG twin (ch.pages set).
void run_form(Chunk& ch, Form f, uint32_t span, bool paged) {
  const std::string v = paged ? kernels::kv_paged(form_variant(f, ch.sh)) : form_variant(f, ch.sh);
  const std::string widen_v =
      paged ? kernels::kv_paged(kernels::pf_flash_attn_kv8_raw_variant(ch.sh.q, ch.sh.kv))
            : kernels::pf_flash_attn_kv8_raw_variant(ch.sh.q, ch.sh.kv);
  l0::Module fm(g_dev->ctx, kernels::path(v)), wm(g_dev->ctx, kernels::path(widen_v));
  l0::Module mm(g_dev->ctx, kernels::path(kernels::pf_flash_merge_variant(ch.sh.q, ch.sh.kv)));
  l0::Kernel fk(fm, "pf_flash_attn_kv8"), wk(wm, "kv8_widen"), mk(mm, "pf_flash_merge");
  const bool pair = f == Form::Pair, split = f == Form::SplitInPlace || f == Form::SplitRaw;
  const bool raw = f != Form::SplitInPlace;
  fk.group_size(pair ? 16 * 2 * 2 : 16 * ch.sh.gqa());
  wk.group_size(256);
  mk.group_size(256);
  const uint32_t count = (ch.pos + ch.C) * ch.sh.kv * kHD, n = split ? split_count(ch.pos, span) : 0;
  uint32_t a = 0;
  wk.arg_ptr(a++, ch.k.ptr());
  wk.arg_ptr(a++, ch.v.ptr());
  wk.arg_ptr(a++, ch.kr.ptr());
  wk.arg_ptr(a++, ch.vr.ptr());
  wk.arg(a++, count);
  if (paged) wk.arg_ptr(a++, ch.pages->ptr());
  a = 0;
  fk.arg_ptr(a++, ch.q.ptr());
  fk.arg_ptr(a++, raw ? ch.kr.ptr() : ch.k.ptr());
  fk.arg_ptr(a++, ch.ks.ptr());
  fk.arg_ptr(a++, raw ? ch.vr.ptr() : ch.v.ptr());
  fk.arg_ptr(a++, ch.vs.ptr());
  fk.arg_ptr(a++, split ? ch.part.ptr() : ch.o.ptr());
  fk.arg(a++, ch.pos);
  fk.arg(a++, ch.C);
  fk.arg(a++, ch.rows);
  if (split) {
    fk.arg_ptr(a++, ch.ml(n));
    fk.arg(a++, span);
  }
  if (paged) fk.arg_ptr(a++, ch.pages->ptr());
  if (split) {
    mk.arg_ptr(0, ch.part.ptr());
    mk.arg_ptr(1, ch.ml(n));
    mk.arg_ptr(2, ch.o.ptr());
    mk.arg(3, ch.C);
    mk.arg(4, ch.rows);
    mk.arg(5, n);
  }
  run([&](l0::CmdList& list) {
    if (raw) list.launch(wk, (count + 4095u) / 4096u, 1, 1);
    if (pair)
      list.launch(fk, (ch.C + 15u) / 16u, ch.sh.kv, ch.sh.gqa() / 2u);
    else
      list.launch(fk, (ch.C + 7u) / 8u, ch.sh.kv, split ? n : 1u);
    if (split) list.launch(mk, ch.C, ch.sh.q, 1);
  });
}

size_t cmp_o(Chunk& a, Chunk& b, bool qwen_b) {
  const std::vector<float> oa = down<float>(a.o), ob = down<float>(b.o);
  size_t bad = 0;
  for (uint32_t h = 0; h < kOrn.q; ++h) {
    const uint32_t hb = qwen_b ? qwen_head(h) : h;
    bad += diff(&oa[size_t(h) * a.rows * kHD], &ob[size_t(hb) * b.rows * kHD], size_t(a.C) * kHD);
  }
  return bad;
}

void part_f() {
  struct Case { uint32_t pos, C, span; };   // span: the splits' (0: raw / pair only)
  for (const Case cs : {Case{0, 64, 0}, Case{777, 300, 256}, Case{4096, 1, 512}, Case{2000, 2048, 0},
                        Case{9000, 16, 1024}, Case{5000, 64, 512}}) {
    const uint32_t L = ((cs.pos + cs.C + 255u) & ~255u);
    const Cache oc = rnd_cache(L, 2), qc = to_qwen(oc, L);
    const std::vector<uint16_t> q = rnd_bf16(size_t(cs.C) * kOrn.q * kHD, 1.0f);
    const uint32_t n = cs.span ? split_count(cs.pos, cs.span) : 0;
    std::vector<Form> forms = {Form::Raw, Form::Pair};
    if (cs.span) forms.insert(forms.end(), {Form::SplitInPlace, Form::SplitRaw});
    for (Form f : forms) {
      Chunk a(kOrn, cs.pos, cs.C, q, oc, n), b(kQwen, cs.pos, cs.C, heads_to_qwen(q, cs.C), qc, n);
      run_form(a, f, cs.span, false);
      run_form(b, f, cs.span, false);
      char what[96];
      std::snprintf(what, sizeof what, "%s_Q16KV2 pos %u C %u span %u", form_name(f), cs.pos, cs.C, cs.span);
      verdict("F", what, cmp_o(a, b, true));
      if (f == Form::Raw) {   // the gate on the merged rows, head h <-> h on Qwen3.8's partials layout
        const std::vector<float> part = rnd_f32(size_t(cs.C) * kOrn.qkv_n(), 1.5f);
        std::vector<float> ow = down<float>(a.o), oq(size_t(kQwen.q) * a.rows * kHD, 0.0f);
        std::memcpy(oq.data(), ow.data(), ow.size() * 4);
        auto gate = [&](const std::string& v, Shape sh, const std::vector<float>& o, const std::vector<float>& p) {
          l0::Module gm(g_dev->ctx, kernels::path(v));
          l0::Kernel gk(gm, "pf_attn_gate_kv8");
          gk.group_size(256);
          l0::Mem dO = up(o), dp = up(p), out = dev_mem(size_t(cs.C) * sh.q * kHD * 2);
          gk.arg_ptr(0, dO.ptr());
          gk.arg_ptr(1, dp.ptr());
          gk.arg_ptr(2, out.ptr());
          gk.arg(3, a.rows * kHD);
          g_dev->run(gk, sh.q, cs.C);
          return down<uint16_t>(out);
        };
        const auto ga = gate(kernels::pf_attn_gate_kv8_variant(16, 2), kOrn, ow, part);
        const auto gb = gate(kernels::pf_attn_gate_kv8_variant(), kQwen, oq, partials_to_qwen(part, 1, cs.C));
        size_t bad = 0;
        for (uint32_t m = 0; m < cs.C; ++m)
          bad += diff(&ga[size_t(m) * kOrn.q * kHD], &gb[size_t(m) * kQwen.q * kHD], size_t(kOrn.q) * kHD);
        verdict("F", "pf_attn_gate_kv8_Q16KV2 C " + std::to_string(cs.C), bad);
      }
      if (cs.pos + cs.C <= kPage) continue;
      // P: the paged twin over a scattered table (the cache's pages in reverse after a spare one), against `a`.
      const uint32_t np = (cs.pos + cs.C + kPage - 1) / kPage;
      std::vector<uint32_t> pages(np);
      for (uint32_t i = 0; i < np; ++i) pages[i] = np - i;
      Cache pc;
      pc.k.assign(size_t(np + 1) * kPage * 2 * kHD, 0);
      pc.v = pc.k;
      pc.ks.assign(size_t(np + 1) * kPage * 2, 0);
      pc.vs = pc.ks;
      for (uint32_t p = 0; p < L; ++p) {
        const size_t r = size_t(pages[p / kPage]) * kPage + p % kPage;
        std::memcpy(&pc.k[r * 2 * kHD], &oc.k[size_t(p) * 2 * kHD], 2 * kHD);
        std::memcpy(&pc.v[r * 2 * kHD], &oc.v[size_t(p) * 2 * kHD], 2 * kHD);
        std::memcpy(&pc.ks[r * 2], &oc.ks[size_t(p) * 2], 4);
        std::memcpy(&pc.vs[r * 2], &oc.vs[size_t(p) * 2], 4);
      }
      Chunk p(kOrn, cs.pos, cs.C, q, pc, n);
      p.pages = std::make_unique<l0::Mem>(up(pages));
      run_form(p, f, cs.span, true);
      std::snprintf(what, sizeof what, "%s_Q16KV2_PG pos %u C %u span %u", form_name(f), cs.pos, cs.C, cs.span);
      verdict("P", what, cmp_o(a, p, false));
    }
  }
}

}  // namespace

int main() {
  pf_harness::Dev d;
  g_dev = &d;
  part_w();
  part_d();
  part_f();
  CHECK_EQ(g_bad, size_t(0));
  std::printf("kv8_ornith_test: PASS\n");
  return 0;
}
