// Spec 8 M2 and plan 8b's Review Focus 1, 3, 4, 5: the verify list at M = k + 1 with
// per-row GDN slots, and commit.
//
// S0 = a prompt prefilled to pos n with its pending id; the reference is T plain M = 1
// steps from S0 (verify(0) + commit(0, argmax)): ids g[0..T], each step's logits and
// the GDN state after the first kMaxDraft + 1 steps.
//
//   M2a  for k = 1..3: restore S0, feed the true greedy drafts g[1..k], verify(k):
//        row r's logits vs the reference step r (cosine >= 0.99999), row r's GDN slot vs
//        the reference state after step r (cosine >= 0.99999, max abs recorded), and
//        row r's argmax == g[r+1].
//   M2b  (Review Focus 1) for k = 1..3 and every j in 0..k: restore S0, drafts g[1..j]
//        then a WRONG id at j+1 (and junk after), verify(k), commit(j, argmax row j),
//        then 64 plain greedy ids == g[j+1 .. j+64] (the run that never saw the
//        rejected drafts). Run at n % 16 == 14, so pos % 16 wraps inside the verify
//        rows (the conv ring), and at n % 16 == 5.
//   RF4  spec 7b's C1 case A with MTP on (state and KV carry the head), plus the same
//        restore driving speculative iterations: bitwise equal to the straight run.
//   RF5  (M5) a fixed pattern of k and forced j over 12 iterations, twice from S0 and
//        once from reset + prefill: every verify/draft logit row and the final state
//        bitwise identical.
//   RF3  max_len 16384 at pos 16380: max_verify_k() falls 3, 2, 1, 0; the iterations run
//        to pos 16384 without a throw and produce the plain run's ids; then verify and
//        draft refuse.
//
// usage: mtp_verify_test <snapshot>   (cwd = the source tree)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "golden_common.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/loader.h"
#include "model/qwen35.h"
#include "runtime/buffers.h"
#include "runtime/control.h"
#include "runtime/engine.h"

namespace {
using model::Qwen35;
constexpr uint32_t kMaxLen = 16384;
constexpr size_t V = Qwen35::kVocab, VU = Qwen35::kVocabUsed;
constexpr size_t kGdnBytes = size_t(48) * 48 * 128 * 128 * 4;

double cosine(const float* a, const float* b, size_t n, double* max_abs = nullptr) {
  double ab = 0, aa = 0, bb = 0, ma = 0;
  for (size_t i = 0; i < n; ++i) {
    ab += double(a[i]) * b[i];
    aa += double(a[i]) * a[i];
    bb += double(b[i]) * b[i];
    ma = std::fmax(ma, std::fabs(double(a[i]) - b[i]));
  }
  if (max_abs) *max_abs = ma;
  return ab / std::sqrt(aa * bb);
}
uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
  const auto* b = static_cast<const uint8_t*>(p);
  for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
  return h;
}
std::vector<uint32_t> head(const std::vector<uint32_t>& v, size_t n) {
  return std::vector<uint32_t>(v.begin(), v.begin() + n);
}

struct Ctx {
  l0::Context& ctx;
  runtime::Engine& e;
  runtime::Control* ctl;
  l0::CmdList imm;
};

// A session snapshot: state + KV [0, pos) + the pending id.
struct Snap {
  l0::Mem st, kv;
  uint32_t pos, pending;
  Snap(Ctx& c) : st(c.ctx, l0::MemKind::Host, c.e.state_bytes()),
                 kv(c.ctx, l0::MemKind::Host, c.e.kv_bytes(c.e.pos())), pos(c.e.pos()),
                 pending(c.ctl->cur_token[0]) {
    c.e.save_state(st.ptr());
    c.e.save_kv(0, pos, kv.ptr());
  }
  void restore(Ctx& c) const {
    c.e.load_state(st.ptr(), pos);
    c.e.load_kv(0, pos, kv.ptr());
    c.ctl->cur_token[0] = pending;
    c.ctl->n_active = 1;
  }
};

std::vector<float> logits_rows(Ctx& c, const float* dev, uint32_t rows) {
  std::vector<float> v(size_t(rows) * V);
  c.imm.copy(v.data(), dev, v.size() * 4);
  return v;
}
std::vector<float> gdn_slot(Ctx& c, uint32_t s) {
  std::vector<float> v(kGdnBytes / 4);
  const void* src = s == 0 ? c.e.buffers().gdn_state.ptr()
                           : c.e.mtp_buffers()->gdn_spec.as<uint8_t>() + size_t(s - 1) * kGdnBytes;
  c.imm.copy(v.data(), src, kGdnBytes);
  return v;
}

struct Ref {
  std::vector<uint32_t> g;                     // g[0] = pending at S0, g[i+1] = argmax of step i
  std::vector<std::vector<float>> logits;      // step r's logits, r <= kMaxDraft
  std::vector<std::vector<float>> state;       // GDN state after step r, r <= kMaxDraft
};
Ref reference(Ctx& c, const Snap& s0, uint32_t T) {
  s0.restore(c);
  Ref r;
  r.g.push_back(c.ctl->cur_token[0]);
  for (uint32_t i = 0; i < T; ++i) {
    c.e.verify(0);
    if (i <= runtime::Engine::kMaxDraft) {
      r.logits.push_back(logits_rows(c, c.e.verify_logits_device(), 1));
    }
    c.e.commit(0, c.e.verify_ids()[0]);
    if (i <= runtime::Engine::kMaxDraft) r.state.push_back(gdn_slot(c, c.ctl->gdn_live));
    r.g.push_back(c.ctl->cur_token[0]);
  }
  return r;
}

bool m2(Ctx& c, const Snap& s0, const Ref& ref) {
  bool ok = true;
  // M2a
  for (uint32_t k = 1; k <= runtime::Engine::kMaxDraft; ++k) {
    s0.restore(c);
    for (uint32_t i = 1; i <= k; ++i) c.ctl->cur_token[i] = ref.g[i];
    c.e.verify(k);
    const std::vector<float> lg = logits_rows(c, c.e.verify_logits_device(), k + 1);
    for (uint32_t r = 0; r <= k; ++r) {
      const double cl = cosine(lg.data() + size_t(r) * V, ref.logits[r].data(), VU);
      const bool lbits = std::memcmp(lg.data() + size_t(r) * V, ref.logits[r].data(), V * 4) == 0;
      const std::vector<float> sl = gdn_slot(c, (c.ctl->gdn_live + r) % runtime::MtpBuffers::kSlots);
      double ma = 0;
      const double cs = cosine(sl.data(), ref.state[r].data(), sl.size(), &ma);
      const bool sbits = std::memcmp(sl.data(), ref.state[r].data(), kGdnBytes) == 0;
      const bool id_ok = c.e.verify_ids()[r] == ref.g[r + 1];
      std::printf("  M2a k=%u row %u: logits cos %.9f%s  GDN slot cos %.9f max abs %.3e%s  id %s\n",
                  k, r, cl, lbits ? " (bitwise)" : "", cs, ma, sbits ? " (bitwise)" : "",
                  id_ok ? "ok" : "DIFFERS");
      ok &= cl >= 0.99999 && cs >= 0.99999 && id_ok;
    }
    c.e.commit(k, c.e.verify_ids()[k]);
  }
  // M2b
  for (uint32_t k = 1; k <= runtime::Engine::kMaxDraft; ++k)
    for (uint32_t j = 0; j <= k; ++j) {
      s0.restore(c);
      for (uint32_t i = 1; i <= k; ++i) {
        uint32_t d = ref.g[i];
        if (i == j + 1) d = (ref.g[i] + 7919) % uint32_t(VU);           // the first rejection
        if (i > j + 1) d = (ref.g[i] + 104729 * i) % uint32_t(VU);      // junk after it
        c.ctl->cur_token[i] = d;
      }
      c.e.verify(k);
      bool rows_ok = true;
      for (uint32_t r = 0; r <= j; ++r) rows_ok &= c.e.verify_ids()[r] == ref.g[r + 1];
      c.e.commit(j, c.e.verify_ids()[j]);
      const std::vector<uint32_t> ids = c.e.generate(64);
      size_t diff = 0;
      for (size_t i = 0; i < ids.size(); ++i) diff += ids[i] != ref.g[j + 1 + i];
      std::printf("  M2b k=%u j=%u: accepted rows %s, 64 further ids: %zu differ\n", k, j,
                  rows_ok ? "ok" : "DIFFER", diff);
      ok &= rows_ok && diff == 0;
    }
  return ok;
}

// One greedy speculative iteration at depth k (0 = plain step): returns ids emitted.
std::vector<uint32_t> iterate(Ctx& c, uint32_t k) {
  std::vector<uint32_t> out{c.ctl->cur_token[0]};   // the pending id, emitted by this step
  if (k == 0) {
    c.e.verify(0);
    c.e.commit(0, c.e.verify_ids()[0]);
    return out;
  }
  c.e.draft(k);
  c.e.verify(k);
  uint32_t j = 0;
  while (j < k && c.e.draft_ids()[j] == c.e.verify_ids()[j]) ++j;
  for (uint32_t i = 0; i < j; ++i) out.push_back(c.e.draft_ids()[i]);
  c.e.commit(j, c.e.verify_ids()[j]);
  return out;
}
}  // namespace

int main(int argc, char** argv) {
  const std::string snap = argc > 1 ? argv[1] : "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ";
  const auto ids = golden::read_ids("tests/golden/prompts/long32k.ids");
  CHECK(ids.size() > kMaxLen);
  l0::Context ctx(0);
  runtime::Engine e(ctx, loader::load(ctx, snap, kMaxLen, /*mtp=*/true), kMaxLen);
  Ctx c{ctx, e, e.buffers().control.as<runtime::Control>(), l0::CmdList::immediate(ctx)};
  bool ok = true;

  // Sizes with the head (Review Focus 4): +hh0 in the state, a 17th KV layer.
  CHECK_EQ(e.state_bytes(), size_t(166723584) + 10240);
  CHECK_EQ(e.kv_bytes(2048), size_t(134217728) / 16 * 17);

  // --- M2 at two prompt lengths ----------------------------------------------------
  for (uint32_t n : {1998u, 2053u}) {
    e.reset();
    e.prefill(head(ids, n));
    std::printf("M2 at n = %u (n %% 16 = %u)\n", n, n % 16);
    const Snap s0(c);
    const Ref ref = reference(c, s0, 72);
    ok &= m2(c, s0, ref);
  }
  std::printf("M2: %s\n", ok ? "PASS" : "FAIL");

  // --- Review Focus 4: spec 7b C1 case A with MTP on --------------------------------
  {
    const uint32_t p = 4395;
    const auto other = [&] {
      auto o = golden::read_ids("tests/golden/prompts/code.ids");
      std::vector<uint32_t> v;
      while (v.size() < 3000) v.insert(v.end(), o.begin(), o.end());
      v.resize(3000);
      return v;
    }();
    const std::vector<uint32_t> next = {ids[p]};
    e.reset();
    e.prefill(head(ids, p));
    const Snap s(c);
    e.prefill(next);
    const auto x = e.generate(64);
    std::vector<uint32_t> xs;
    for (int it = 0; it < 8; ++it) {
      const auto o = iterate(c, 3);
      xs.insert(xs.end(), o.begin(), o.end());
    }
    e.reset();
    e.prefill(other);
    s.restore(c);
    l0::Mem kv2(ctx, l0::MemKind::Host, e.kv_bytes(p));
    e.save_kv(0, p, kv2.ptr());
    const bool kv_same = std::memcmp(s.kv.ptr(), kv2.ptr(), e.kv_bytes(p)) == 0;
    e.prefill(next);
    const auto y = e.generate(64);
    std::vector<uint32_t> ys;
    for (int it = 0; it < 8; ++it) {
      const auto o = iterate(c, 3);
      ys.insert(ys.end(), o.begin(), o.end());
    }
    std::printf("RF4: restore at %u over a 3000-id state: KV [0,%u) x17 exact %s, 64 plain ids %s,"
                " 8 speculative iterations (%zu ids) %s\n",
                p, p, kv_same ? "yes" : "NO", x == y ? "equal" : "DIFFER", xs.size(),
                xs == ys ? "equal" : "DIFFER");
    ok &= kv_same && x == y && xs == ys;
  }

  // --- Review Focus 5 (M5): replay determinism over a mixed pattern -----------------
  {
    const uint32_t n = 3001;
    const uint32_t ks[12] = {3, 1, 2, 3, 3, 0, 2, 1, 3, 2, 3, 1};
    const uint32_t js[12] = {3, 0, 1, 2, 0, 0, 2, 1, 1, 0, 3, 1};
    auto run = [&]() {
      uint64_t h = 1469598103934665603ull;
      for (int it = 0; it < 12; ++it) {
        const uint32_t k = ks[it];
        if (k) {
          c.e.draft(k);
          const auto q = logits_rows(c, c.e.mtp_logits_device(), k);
          h = fnv(q.data(), q.size() * 4, h);
        }
        c.e.verify(k);
        const auto p = logits_rows(c, c.e.verify_logits_device(), k + 1);
        h = fnv(p.data(), p.size() * 4, h);
        c.e.commit(js[it], c.e.verify_ids()[js[it]]);
      }
      l0::Mem st(ctx, l0::MemKind::Host, c.e.state_bytes());
      c.e.save_state(st.ptr());
      h = fnv(st.ptr(), c.e.state_bytes(), h);
      h = fnv(&c.ctl->pos, 4, h);
      return h;
    };
    e.reset();
    e.prefill(head(ids, n));
    const Snap s0(c);
    const uint64_t a = run();
    s0.restore(c);
    const uint64_t b = run();
    e.reset();
    e.prefill(head(ids, n));
    const uint64_t d = run();
    std::printf("RF5: three runs of a 12-iteration pattern: %016llx %016llx %016llx -> %s\n",
                (unsigned long long)a, (unsigned long long)b, (unsigned long long)d,
                a == b && b == d ? "bitwise" : "DIFFER");
    ok &= a == b && b == d;
  }

  // --- Review Focus 3: the max_len edge ---------------------------------------------
  {
    const uint32_t n = 16380;
    e.reset();
    e.prefill(head(ids, n));
    const Snap s0(c);
    std::vector<uint32_t> plain;
    while (e.pos() < kMaxLen) {
      const auto o = iterate(c, 0);
      plain.insert(plain.end(), o.begin(), o.end());
    }
    s0.restore(c);
    std::vector<uint32_t> spec, limits;
    while (e.pos() < kMaxLen) {
      const uint32_t k = e.max_verify_k();
      limits.push_back(k);
      const auto o = iterate(c, k);
      spec.insert(spec.end(), o.begin(), o.end());
    }
    bool threw_v = false, threw_d = false;
    try {
      e.verify(0);
    } catch (const std::runtime_error&) {
      threw_v = true;
    }
    try {
      e.draft(1);
    } catch (const std::runtime_error&) {
      threw_d = true;
    }
    std::string ls;
    for (uint32_t k : limits) ls += std::to_string(k) + " ";
    std::printf("RF3: from pos %u: max_verify_k per iteration %s; %zu ids, plain %zu ids, %s; at"
                " pos %u verify %s, draft %s\n",
                n, ls.c_str(), spec.size(), plain.size(), spec == plain ? "equal" : "DIFFER",
                e.pos(), threw_v ? "refused" : "RAN", threw_d ? "refused" : "RAN");
    ok &= limits.front() == 3 && spec == plain && threw_v && threw_d && e.pos() == kMaxLen;
  }

  CHECK(ok);
  std::puts("mtp_verify_test OK");
  return 0;
}
