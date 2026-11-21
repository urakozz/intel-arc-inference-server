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
//        then 64 plain greedy ids == the run that never saw the rejected drafts: the same
//        verify(k) with the TRUE drafts g[1..k], commit(j), 64 plain ids - bitwise (spec 28
//        N3 + N4: rows 0..j do not depend on rows past j). Since spec 28b the verify rows run
//        gemm_sm and are no longer bitwise M = 1, so the 64 ids against the plain run's
//        g[j+1 .. j+64] are graded by spec 28 N6: identical, or
//        from the first difference the f32 judge rejects no more of the ids than of the plain run's
//        (tests/server/spec_judge.h), the golden tie rule without a judge - before 28b the two
//        references were the same run. M2a's logits rows likewise: cosine >= 0.99999 against the
//        plain row, or no further from the f32 judge's row than the plain row (x kCosSlack). Run at n % 16 == 14, so pos % 16 wraps inside the
//        verify rows (the conv ring), and at n % 16 == 5.
//   RF4  spec 7b's C1 case A with MTP on (state and KV carry the head), plus the same
//        restore driving speculative iterations: bitwise equal to the straight run.
//   RF5  (M5) a fixed pattern of k and forced j over 12 iterations, twice from S0 and
//        once from reset + prefill: every verify/draft logit row and the final state
//        bitwise identical.
//   RF3  max_len 16384 at pos 16380: max_verify_k() falls 3, 2, 1, 0; the iterations run
//        to pos 16384 without a throw and produce the plain run's ids; then verify and
//        draft refuse.
//
//   W32 / W64 (spec 32, `... 16384 rule deep`): the deep lists against the 16-row list from the same state - (a) every
//        row bitwise the 16-row list's row (rows past 15: after 16-row verifies committing 15 of the same inputs);
//        (b) for j in 0, 7, 15, 16, 31, 32, 47, 62 a wrong draft at j + 1 and junk after leave rows 0..j bitwise and the
//        next 64 plain ids bitwise the clean run's; (c) greedy lookup iterations with commits in between emit the
//        same ids through the deep list as through the 16-row one (the captured lists replayed across commits);
//        (d) after a deep verify committing 5 W / 8 rows, a snapshot's state, conv-ring window and hidden row bitwise
//        the 16-row lists' at the same position (a verify row is not bitwise a plain step's) and, after its load and
//        after a 64-id prefill, the next 64 plain ids bitwise that session's.
//   W16  (open-work 28, `mtp_verify_test <snapshot> 16384 rule wide`: the int8 head, whose engine builds the 16-row
//        list) at the M2 positions: (a) for j in 0, 1, 4, 7, 11, 14 the 16-row verify with drafts g[1..j] true, a WRONG
//        id at j + 1 and junk after - its rows 0..j bitwise the all-true verify's (no later row leaks into an
//        earlier one) - then commit(j) and 64 plain ids bitwise the all-true run's; (b) every row of the all-true
//        verify against the plain step's row by the f32 judge (spec_judge::row: cosine >= 0.99999 against plain, or a
//        distance to f32 at most kCosSlack (2) x the plain row's).
//
// usage: mtp_verify_test <snapshot> [max_len = 16384] [vattn = rule] [wide | deep]   (cwd = the source tree)
//
// Spec 28c: `vattn` v2 | dpas | rule (test-only) - the verify lists' attention forced (Engine::recapture_verify) or
// the capture's measured rule. Where DPAS is bound the verify rows are no longer bitwise their M = 1 steps: M2a's
// cosine bars hold, its "(bitwise)" marks go.
//
// Spec 8 §12 (MTP at every context length; box queue row 6): with max_len above 16384
// (mtp_verify_{32k,128k}_test) the same cases run at that length, plus M2 deep in the
// cache (n = max_len - 1026, n % 16 = 14) and RF4 at max_len / 2 + 4395 (spec 7 C1 with
// MTP at ~64k on the 131072 run); RF3 sits at max_len - 4. The prompt is long32k.ids
// repeated as far as the run needs. At 16384 every case and line is as before.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "../server/greedy_tie.h"
#include "../server/spec_judge.h"
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
uint32_t kMaxLen = 16384;   // argv[2]
constexpr size_t V = Qwen35::kVocab, VU = Qwen35::kVocabUsed;
// One GDN state slot: gdn_layers x 48 x 128 x 128 fp32 (48 layers on Qwen3.8), set from the loaded model's descriptor in main().
size_t kGdnBytes = 0;

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

// The committed GDN state (Engine::gdn_state_now: the live slot, or with the deferred build the pending rows flushed).
std::vector<float> committed_state(Ctx& c) {
  std::vector<float> v(kGdnBytes / 4);
  c.imm.copy(v.data(), c.e.gdn_state_now(), kGdnBytes);
  return v;
}

struct Ref {
  std::vector<uint32_t> g;                     // g[0] = pending at S0, g[i+1] = argmax of step i
  std::vector<greedy_tie::RefStep> steps;      // step i's decision (top-2, margin) - it decided g[i+1]
  std::vector<std::vector<float>> logits;      // step r's logits, r <= kMaxDraft
  std::vector<std::vector<float>> state;       // GDN state after step r, r <= kMaxDraft
};
Ref reference(Ctx& c, const Snap& s0, uint32_t T) {
  s0.restore(c);
  Ref r;
  r.g.push_back(c.ctl->cur_token[0]);
  for (uint32_t i = 0; i < T; ++i) {
    c.e.verify(0);
    const std::vector<float> row = logits_rows(c, c.e.verify_logits_device(), 1);
    r.steps.push_back(greedy_tie::ref_step(row.data(), uint32_t(VU)));
    if (i <= runtime::Engine::kMaxDraft) r.logits.push_back(row);
    c.e.commit(0, c.e.verify_ids()[0]);
    if (i <= runtime::Engine::kMaxDraft) r.state.push_back(committed_state(c));
    r.g.push_back(c.ctl->cur_token[0]);
  }
  return r;
}

// `prompt`: the ids prefilled to S0 (the f32 judge's context, with the forced g[0..]); `tag` names the judge.
bool m2(Ctx& c, const Snap& s0, const Ref& ref, const std::vector<uint32_t>& prompt, const std::string& tag) {
  bool ok = true;
  const golden::ForcedJudge fj;
  // N4's judge: the context prompt + g[0 .. kMaxDraft + 1]; judge row r + 1 decides g[r + 1], verify row r's id.
  const spec_judge::RowJudge rj = spec_judge::open_rows(
      fj, tag + "_m2a", prompt, std::vector<uint32_t>(ref.g.begin(), ref.g.begin() + runtime::Engine::kMaxDraft + 2),
      uint32_t(V));
  // M2a
  for (uint32_t k = 1; k <= runtime::Engine::kMaxDraft; ++k) {
    // Row r's state: its slot after verify(k); with the deferred build (no per-row state) the committed state after
    // verify(k) + commit(r).
    std::vector<std::vector<float>> rs;
    if (c.e.gdn_deferred())
      for (uint32_t r = 0; r <= k; ++r) {
        s0.restore(c);
        for (uint32_t i = 1; i <= k; ++i) c.ctl->cur_token[i] = ref.g[i];
        c.e.verify(k);
        c.e.commit(r, c.e.verify_ids()[r]);
        rs.push_back(committed_state(c));
      }
    s0.restore(c);
    for (uint32_t i = 1; i <= k; ++i) c.ctl->cur_token[i] = ref.g[i];
    c.e.verify(k);
    const std::vector<float> lg = logits_rows(c, c.e.verify_logits_device(), k + 1);
    for (uint32_t r = 0; r <= k; ++r) {
      double lma = 0;
      const double cl = cosine(lg.data() + size_t(r) * V, ref.logits[r].data(), VU, &lma);
      const bool lbits = std::memcmp(lg.data() + size_t(r) * V, ref.logits[r].data(), V * 4) == 0;
      const std::vector<float> sl =
          c.e.gdn_deferred() ? rs[r] : gdn_slot(c, (c.ctl->gdn_live + r) % runtime::MtpBuffers::kSlots);
      double ma = 0;
      const double cs = cosine(sl.data(), ref.state[r].data(), sl.size(), &ma);
      const bool sbits = std::memcmp(sl.data(), ref.state[r].data(), kGdnBytes) == 0;
      const bool id_ok = c.e.verify_ids()[r] == ref.g[r + 1];
      int le = 0;
      std::frexp(std::fabs(ref.logits[r][ref.g[r + 1]]), &le);
      const spec_judge::RowVerdict rv = spec_judge::row(lg.data() + size_t(r) * V, ref.logits[r].data(), rj, r + 1,
                                                        cl, uint32_t(VU));
      std::printf("  M2a k=%u row %u: logits cos %.9f max abs %.3e (%.2f bf16 ulp of the top)%s  GDN slot cos %.9f max "
                  "abs %.3e%s  id %s%s\n",
                  k, r, cl, lma, lma / std::ldexp(1.0, le - 8), lbits ? " (bitwise)" : "", cs, ma,
                  sbits ? " (bitwise)" : "", id_ok ? "ok" : "DIFFERS", spec_judge::row_line(rv).c_str());
      // spec 28 N4: the logits row by the f32 judge where one exists
      ok &= rv.pass && cs >= 0.99999 && id_ok;
    }
    c.e.commit(k, c.e.verify_ids()[k]);
  }
  // M2b
  for (uint32_t k = 1; k <= runtime::Engine::kMaxDraft; ++k)
    for (uint32_t j = 0; j <= k; ++j) {
      // the run that never saw a rejected draft: every draft true, the same verify(k), commit(j)
      s0.restore(c);
      for (uint32_t i = 1; i <= k; ++i) c.ctl->cur_token[i] = ref.g[i];
      c.e.verify(k);
      c.e.commit(j, c.e.verify_ids()[j]);
      const std::vector<uint32_t> clean = c.e.generate(64);
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
      for (size_t i = 0; i < ids.size(); ++i) diff += ids[i] != clean[i];
      // against the plain M = 1 run (spec 28 N6): identical, else the f32 judge, else the golden tie rule
      const std::vector<uint32_t> pg(ref.g.begin() + j + 1, ref.g.begin() + j + 1 + ids.size());
      const std::vector<greedy_tie::RefStep> ps(ref.steps.begin() + j, ref.steps.begin() + j + ids.size());
      std::vector<uint32_t> ctx = prompt;
      ctx.insert(ctx.end(), ref.g.begin(), ref.g.begin() + j + 1);
      const spec_judge::StreamVerdict sv = spec_judge::streams(fj, tag + "_m2b", ctx, pg, ps, ids, uint32_t(V),
                                                               uint32_t(VU));
      std::printf("  M2b k=%u j=%u: accepted rows %s, 64 further ids: %zu differ from the clean run; the plain run: %s\n",
                  k, j, rows_ok ? "ok" : "DIFFER", diff, sv.line().c_str());
      ok &= rows_ok && diff == 0 && sv.pass;
    }
  return ok;
}

// W16 (header): the 16-row list against 16 plain steps from S0.
bool w16(Ctx& c, const Snap& s0, const std::vector<uint32_t>& prompt, const std::string& tag) {
  constexpr uint32_t K = runtime::Engine::kWideK;
  bool ok = true;
  s0.restore(c);
  std::vector<uint32_t> g{c.ctl->cur_token[0]};
  std::vector<std::vector<float>> plain;
  for (uint32_t i = 0; i <= K + 1; ++i) {
    c.e.verify(0);
    if (i <= K) plain.push_back(logits_rows(c, c.e.verify_logits_device(), 1));
    c.e.commit(0, c.e.verify_ids()[0]);
    g.push_back(c.ctl->cur_token[0]);
  }
  const auto wide = [&](uint32_t j) {   // drafts g[1..j] true, then a wrong id and junk (j = K: every draft true)
    s0.restore(c);
    for (uint32_t i = 1; i <= K; ++i)
      c.ctl->wide_cur[i] = i <= j ? g[i] : i == j + 1 ? (g[i] + 7919) % uint32_t(VU) : (g[i] + 104729 * i) % uint32_t(VU);
    c.e.verify(K);
  };
  wide(K);
  const std::vector<float> all = logits_rows(c, c.e.verify_logits_device(), K + 1);
  // (b)
  const golden::ForcedJudge fj;
  const spec_judge::RowJudge rj =
      spec_judge::open_rows(fj, tag + "_w16", prompt, std::vector<uint32_t>(g.begin(), g.begin() + K + 2), uint32_t(V));
  for (uint32_t r = 0; r <= K; ++r) {
    const double cl = cosine(all.data() + size_t(r) * V, plain[r].data(), VU);
    const spec_judge::RowVerdict rv = spec_judge::row(all.data() + size_t(r) * V, plain[r].data(), rj, r + 1, cl,
                                                      uint32_t(VU));
    const bool id_ok = c.e.verify_ids()[r] == g[r + 1];
    std::printf("  W16 row %2u: logits cos against plain %.9f, id %s%s\n", r, cl, id_ok ? "ok" : "DIFFERS",
                spec_judge::row_line(rv).c_str());
    ok &= rv.pass;
  }
  // (a)
  for (uint32_t j : {0u, 1u, 4u, 7u, 11u, 14u}) {
    wide(K);
    c.e.commit(j, c.e.verify_ids()[j]);
    const std::vector<uint32_t> clean = c.e.generate(64);
    wide(j);
    const std::vector<float> lg = logits_rows(c, c.e.verify_logits_device(), j + 1);
    const bool rows_ok = std::memcmp(lg.data(), all.data(), size_t(j + 1) * V * 4) == 0;
    c.e.commit(j, c.e.verify_ids()[j]);
    const std::vector<uint32_t> ids = c.e.generate(64);
    std::printf("  W16 j=%2u: rows 0..%u %s the all-true verify's, 64 further ids %s\n", j, j,
                rows_ok ? "bitwise" : "DIFFER from", ids == clean ? "bitwise the clean run's" : "DIFFER");
    ok &= rows_ok && ids == clean;
  }
  return ok;
}

// PERF-56 probe for (d): every buffer a plain step may read that a restore does not write, by name - the decode
// scratch, the head's buffers and the whole KV (main and head, rows past pos too) - as hashes.
using BufHashes = std::vector<std::pair<std::string, uint64_t>>;
BufHashes buf_hashes(Ctx& c) {
  BufHashes out;
  std::vector<uint8_t> h;
  const auto add = [&](const char* name, const l0::Mem* m) {
    if (!m || m->size() == 0) return;
    h.resize(m->size());
    c.imm.copy(h.data(), m->ptr(), h.size());
    out.emplace_back(name, fnv(h.data(), h.size()));
  };
  runtime::DecodeBuffers& b = c.e.buffers();
  add("control", &b.control);
  add("kv_k", &b.kv_k), add("kv_v", &b.kv_v);
  add("resid", &b.resid), add("x", &b.x), add("partials", &b.partials), add("ab_out", &b.ab_out);
  add("norm_sumsq", &b.norm_sumsq), add("gdn_o", &b.gdn_o), add("attn_q", &b.attn_q), add("attn_gate", &b.attn_gate);
  add("attn_part", &b.attn_part), add("attn_out", &b.attn_out), add("logits", &b.logits);
  add("argmax_part", &b.argmax_part), add("moe", b.moe);
  if (runtime::MtpBuffers* m = c.e.mtp_buffers()) {
    add("mtp.hctl", &m->hctl), add("mtp.gdn_spec", &m->gdn_spec), add("mtp.kv_k", &m->kv_k), add("mtp.kv_v", &m->kv_v);
    add("mtp.hh", &m->hh), add("mtp.dh", &m->dh), add("mtp.logits", &m->logits);
  }
  return out;
}
// The scratch a restore leaves as it was, zeroed: the decode scratch, the head's hidden rows past row 0 (row 0 is the
// snapshot's), its speculative GDN slots, draft hidden and logits.
void zero_scratch(Ctx& c) {
  runtime::DecodeBuffers& b = c.e.buffers();
  for (l0::Mem* m : {&b.resid, &b.x, &b.partials, &b.ab_out, &b.norm_sumsq, &b.gdn_o, &b.attn_q, &b.attn_gate,
                     &b.attn_part, &b.attn_out, &b.logits, &b.argmax_part, b.moe})
    if (m && m->size()) c.imm.fill(m->ptr(), 0u, m->size());
  if (runtime::MtpBuffers* m = c.e.mtp_buffers()) {
    const size_t row = size_t(c.e.model().desc->hidden) * 2;
    c.imm.fill(static_cast<uint8_t*>(m->hh.ptr()) + row, 0u, m->hh.size() - row);
    for (l0::Mem* x : {&m->gdn_spec, &m->dh, &m->logits}) c.imm.fill(x->ptr(), 0u, x->size());
  }
}
// The whole KV zeroed (main and head; the restore after it writes rows [0, pos) back).
void zero_kv(Ctx& c) {
  runtime::DecodeBuffers& b = c.e.buffers();
  for (l0::Mem* m : {&b.kv_k, &b.kv_v}) c.imm.fill(m->ptr(), 0u, m->size());
  if (runtime::MtpBuffers* m = c.e.mtp_buffers())
    for (l0::Mem* x : {&m->kv_k, &m->kv_v}) c.imm.fill(x->ptr(), 0u, x->size());
}

// W32 / W64 (header): the deep list of W rows against the 16-row list from the same state.
bool w_deep(Ctx& c, const Snap& s0, uint32_t W) {
  constexpr uint32_t K16 = runtime::Engine::kWideK;
  const uint32_t K = W - 1;
  bool ok = true;
  s0.restore(c);
  std::vector<uint32_t> g{c.ctl->cur_token[0]};   // the plain run: g[0] pending, g[i + 1] step i's id
  for (uint32_t i = 0; i < 4 * W; ++i) {
    c.e.verify(0);
    c.e.commit(0, c.e.verify_ids()[0]);
    g.push_back(c.ctl->cur_token[0]);
  }
  const auto deep = [&](uint32_t j) {   // drafts g[1..j] true, then a wrong id and junk (j = K: every draft true)
    s0.restore(c);
    for (uint32_t i = 1; i <= K; ++i)
      c.ctl->deep_cur[i] = i <= j ? g[i] : i == j + 1 ? (g[i] + 7919) % uint32_t(VU) : (g[i] + 104729 * i) % uint32_t(VU);
    c.e.verify(K);
  };
  deep(K);
  const std::vector<float> all = logits_rows(c, c.e.verify_logits_device(), W);
  // (a) rows 16 q .. 16 q + 15 against the 16-row list after q 16-row verifies of the same inputs, each committing 15
  // rows with the next chunk's first input as the pending id (whatever the rows chose: the inputs are what match).
  size_t diff_rows = 0;
  s0.restore(c);
  for (uint32_t q = 0; q < W / 16; ++q) {
    for (uint32_t i = 1; i <= K16; ++i) c.ctl->wide_cur[i] = g[16 * q + i];
    c.e.verify(K16);
    const std::vector<float> lg = logits_rows(c, c.e.verify_logits_device(), 16);
    for (uint32_t r = 0; r < 16; ++r)
      diff_rows += std::memcmp(lg.data() + size_t(r) * V, all.data() + size_t(16 * q + r) * V, V * 4) != 0;
    c.e.commit(K16, g[16 * q + 16]);
  }
  std::printf("  W%u (a): %zu of %u rows differ from the 16-row list's (bitwise)\n", W, diff_rows, W);
  ok &= diff_rows == 0;
  // (b) a wrong draft at j + 1 and junk after: rows 0..j bitwise the all-true verify's, the next 64 plain ids too
  for (uint32_t j : {0u, 7u, 15u, 16u, 31u, 32u, 47u, 62u}) {
    if (j >= K) continue;
    deep(K);
    c.e.commit(j, c.e.verify_ids()[j]);
    const std::vector<uint32_t> clean = c.e.generate(64);
    deep(j);
    const std::vector<float> lg = logits_rows(c, c.e.verify_logits_device(), j + 1);
    const bool rows_ok = std::memcmp(lg.data(), all.data(), size_t(j + 1) * V * 4) == 0;
    c.e.commit(j, c.e.verify_ids()[j]);
    const std::vector<uint32_t> ids = c.e.generate(64);
    std::printf("  W%u j=%2u: rows 0..%u %s the all-true verify's, 64 further ids %s\n", W, j, j,
                rows_ok ? "bitwise" : "DIFFER from", ids == clean ? "bitwise the clean run's" : "DIFFER");
    ok &= rows_ok && ids == clean;
  }
  // (c) the lists replayed across commits: greedy lookup iterations from S0 with the plain run's ids as drafts (one in
  // three corrupted at a moving row), at W rows and at 16 rows - the same ids emitted, then the same 64 plain ids.
  const auto run = [&](uint32_t k) {
    s0.restore(c);
    std::vector<uint32_t> out;
    for (uint32_t it = 0; out.size() < 2 * W; ++it) {
      const size_t t0 = out.size();   // g[t0] is the pending id when the run so far matched the plain run
      uint32_t* cur = k == K16 ? c.ctl->wide_cur : c.ctl->deep_cur;
      for (uint32_t i = 1; i <= k; ++i) cur[i] = t0 + i < g.size() ? g[t0 + i] : 0;
      if (it % 3 == 2) cur[1 + (it * 13) % k] = (cur[1 + (it * 13) % k] + 7919) % uint32_t(VU);
      out.push_back(c.ctl->cur_token[0]);
      c.e.verify(k);
      uint32_t j = 0;
      while (j < k && cur[j + 1] == c.e.verify_ids()[j]) ++j;
      for (uint32_t i = 0; i < j; ++i) out.push_back(cur[i + 1]);
      c.e.commit(j, c.e.verify_ids()[j]);
    }
    return out;
  };
  const std::vector<uint32_t> a = run(K), b = run(K16);
  const size_t n = std::min(a.size(), b.size());
  const bool same = std::equal(a.begin(), a.begin() + n, b.begin());
  std::printf("  W%u (c): %zu ids emitted through %u-row verifies and commits %s the 16-row lists'\n", W, n, W,
              same ? "equal" : "DIFFER from");
  ok &= same;
  // (d) the state paths after a deep verify: an all-true verify of W rows committing j = 5 W / 8 (20 at 32 rows, 40 at
  // 64: past the first chain link, so the state comes from the replay and the ring rows from the conv scratch), then
  // (i) a snapshot saved - bitwise the one the 16-row lists leave at the same position - loaded after 8 more steps,
  // and (ii) 64 ids prefilled straight after the commit; each time the next 64 plain ids bitwise that session's.
  {
    const uint32_t j = 5 * W / 8;
    const std::vector<uint32_t> fill(g.begin() + 1, g.begin() + 65);   // any 64 ids, the same in both arms
    const auto to_j = [&](bool deep_arm) {   // pos = S0 + j + 1, the pending id g[j + 1]
      if (deep_arm) {
        deep(K);
        // g[j + 1], as the 16-row arm commits: row j's own argmax is the 16-row list's (a), and on Ornith that may
        // differ from the plain step's at a near-tie (W16 rows are judged, not bitwise, against plain steps there)
        c.e.commit(j, g[j + 1]);
        return;
      }
      s0.restore(c);   // the 16-row lists over the same inputs, as (a): 15 rows a verify committed, the last j % 16
      for (uint32_t q = 0; 16 * q <= j; ++q) {
        for (uint32_t i = 1; i <= K16; ++i) c.ctl->wide_cur[i] = g[16 * q + i];
        c.e.verify(K16);
        const uint32_t jj = 16 * (q + 1) <= j ? K16 : j - 16 * q;
        c.e.commit(jj, g[16 * q + jj + 1]);
      }
    };
    // What a snapshot carries that the session reads again: the GDN state, the conv ring's window (the 3 positions
    // before pos; a deep commit copies only those of its rows 16.. into the ring, so older slots may hold other
    // rows than plain steps leave - never read) and the head's hidden row.
    const model::ModelDesc& md = *c.e.model().desc;   // PERF-56: Ornith's widths too
    const size_t ring_row = size_t(md.gdn_conv_dim()) * 2, layers = md.gdn_layers;
    // PERF-56: the KV rows [0, pos) too, and each half's ids on its own line's verdict
    struct Probe {
      BufHashes bufs;                           // after the restore, before the 64 ids
      std::vector<uint32_t> zs, zk;             // the 64 ids again: scratch zeroed / KV past pos zeroed, after a restore
    };
    const auto paths = [&](bool deep_arm, std::vector<uint8_t>& state, std::vector<uint8_t>& kv,
                           std::vector<uint32_t>& after, Probe& pr) {
      to_j(deep_arm);
      const Snap s(c);
      const uint8_t* h = s.st.as<uint8_t>();
      kv.assign(s.kv.as<uint8_t>(), s.kv.as<uint8_t>() + c.e.kv_bytes(s.pos));
      state.assign(h, h + kGdnBytes);
      for (size_t l = 0; l < layers; ++l)
        for (uint32_t t = 1; t <= 3; ++t) {
          const uint8_t* r = h + kGdnBytes + (l * 32 + (s.pos - t) % 32) * ring_row;
          state.insert(state.end(), r, r + ring_row);
        }
      state.insert(state.end(), h + kGdnBytes + layers * 32 * ring_row, h + c.e.state_bytes());
      c.e.generate(8);
      s.restore(c);
      pr.bufs = buf_hashes(c);
      std::vector<uint32_t> ids = c.e.generate(64);
      s.restore(c);
      zero_scratch(c);
      pr.zs = c.e.generate(64);
      zero_kv(c);
      s.restore(c);
      pr.zk = c.e.generate(64);
      to_j(deep_arm);   // and straight on into a prefill (the deferred rows flushed by it, not by a snapshot)
      std::vector<uint32_t> f{c.ctl->cur_token[0]};
      f.insert(f.end(), fill.begin(), fill.end() - 1);
      c.e.prefill(f);
      after = c.e.generate(64);
      return ids;
    };
    std::vector<uint8_t> st_deep, st_plain, kv_deep, kv_plain;
    std::vector<uint32_t> d_after, p_after;
    Probe d_pr, p_pr;
    const std::vector<uint32_t> d_ids = paths(true, st_deep, kv_deep, d_after, d_pr),
                                p_ids = paths(false, st_plain, kv_plain, p_after, p_pr);
    const bool st_ok = st_deep == st_plain, kv_ok = kv_deep == kv_plain, load_ok = d_ids == p_ids,
               pf_ok = d_after == p_after;
    std::printf("  W%u (d) j=%u: snapshot (state, ring window, hidden row) %s the 16-row lists', KV rows [0, pos) %s; "
                "64 ids after load %s, 64 ids after a 64-id prefill %s\n",
                W, j, st_ok ? "bitwise" : "DIFFERS from", kv_ok ? "bitwise" : "DIFFER", load_ok ? "bitwise" : "DIFFER",
                pf_ok ? "bitwise" : "DIFFER");
    ok &= st_ok && kv_ok && load_ok && pf_ok;
    if (!load_ok || !pf_ok) {   // PERF-56 probe: what the two sessions still differ in, and which zeroing aligns them
      std::string differ;
      for (size_t i = 0; i < d_pr.bufs.size(); ++i)
        if (d_pr.bufs[i].second != p_pr.bufs[i].second) differ += " " + d_pr.bufs[i].first;
      size_t first = 0;
      while (first < d_ids.size() && d_ids[first] == p_ids[first]) ++first;
      std::printf("  W%u (d) probe: after load the buffers differ in%s; the first id after load differs at step %zu "
                  "(%u / %u); after load with the scratch zeroed the 64 ids %s, with the KV past pos zeroed %s\n",
                  W, differ.empty() ? " none" : differ.c_str(), first, first < d_ids.size() ? d_ids[first] : 0u,
                  first < p_ids.size() ? p_ids[first] : 0u, d_pr.zs == p_pr.zs ? "bitwise" : "DIFFER",
                  d_pr.zk == p_pr.zk ? "bitwise" : "DIFFER");
    }
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
// Spec 28c diagnostic: `trace <prompt ids> <n> <k>` - the prompt prefilled, n plain steps recorded (logits per step),
// then greedy speculative iterations at depth k from the same state: every verify row's logits against the plain
// step of the same position; prints each row below cosine 0.9999 (the first 20) and the worst.
int trace(Ctx& c, const std::vector<uint32_t>& prompt, uint32_t n, uint32_t k) {
  c.e.reset();
  c.e.prefill(prompt);
  const Snap s0(c);
  std::vector<std::vector<float>> ref;
  std::vector<uint32_t> g{c.ctl->cur_token[0]};
  for (uint32_t i = 0; i < n; ++i) {
    c.e.verify(0);
    ref.push_back(logits_rows(c, c.e.verify_logits_device(), 1));
    c.e.commit(0, c.e.verify_ids()[0]);
    g.push_back(c.ctl->cur_token[0]);
  }
  s0.restore(c);
  uint32_t t = 0, shown = 0;
  double worst = 1;
  while (t + k + 1 < n) {
    const uint32_t pos = c.e.pos();
    c.e.draft(k);
    c.e.verify(k);
    const std::vector<float> lg = logits_rows(c, c.e.verify_logits_device(), k + 1);
    uint32_t j = 0;
    while (j < k && c.e.draft_ids()[j] == c.e.verify_ids()[j]) ++j;
    for (uint32_t r = 0; r <= j; ++r) {
      const double cl = cosine(lg.data() + size_t(r) * V, ref[t + r].data(), VU);
      worst = std::min(worst, cl);
      if (cl < 0.9999 && shown++ < 20)
        std::printf("  trace: pos %u row %u of M %u (step %u): logits cos %.6f, id %u vs plain %u\n", pos + r, r,
                    k + 1, t + r, cl, c.e.verify_ids()[r], g[t + r + 1]);
      if (c.e.verify_ids()[r] != g[t + r + 1]) {
        const uint32_t a = g[t + r + 1], b = c.e.verify_ids()[r];
        const float* vr = lg.data() + size_t(r) * V;
        std::printf("trace: first id difference at step %u (pos %u, row %u of M %u); worst cos %.6f; plain row: "
                    "plain id %u %.5f, verify id %u %.5f; verify row: %.5f / %.5f; this row's cos %.6f\n",
                    t + r, pos + r, r, k + 1, worst, a, ref[t + r][a], b, ref[t + r][b], vr[a], vr[b],
                    cosine(vr, ref[t + r].data(), VU));
        return 1;
      }
    }
    c.e.commit(j, c.e.verify_ids()[j]);
    t += j + 1;
  }
  std::printf("trace: %u steps equal; worst row cos %.6f\n", t, worst);
  return 0;
}
}  // namespace

int main(int argc, char** argv) {
  const std::string snap = argc > 1 ? argv[1] : "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ";
  golden::load_end_of_turn(snap);
  if (argc > 2) kMaxLen = uint32_t(std::stoul(argv[2]));
  CHECK(kMaxLen >= 16384 && kMaxLen % 256 == 0);
  const auto ids = [] {
    std::vector<uint32_t> v = golden::read_ids("tests/golden/prompts/long32k.ids");
    const std::vector<uint32_t> file = v;
    while (v.size() <= kMaxLen) v.insert(v.end(), file.begin(), file.end());
    return v;
  }();
  CHECK(ids.size() > kMaxLen);
  const bool deep = kMaxLen > 16384;
  if (deep) std::printf("max_len %u: M2 also at n = %u, RF4 also at %u\n", kMaxLen, kMaxLen - 1026,
                        kMaxLen / 2 + 4395);
  l0::Context ctx(0);
  // open-work 28: W16 alone, the int8 head; spec 32: W32 / W64 alone (deep), the same
  const bool wide_mode = argc > 4 && (std::string(argv[4]) == "wide" || std::string(argv[4]) == "deep");
  runtime::Engine e(ctx,
                    loader::load(ctx, snap, kMaxLen, /*mtp=*/true,
                                 wide_mode ? loader::LmHeadForm::Int8 : loader::LmHeadForm::Checkpoint),
                    kMaxLen);
  if (argc > 3) {   // spec 28c
    const std::string v = argv[3];
    CHECK(v == "v2" || v == "dpas" || v == "rule");
    if (v != "rule")
      e.recapture_verify(runtime::attn_forced(v == "v2" ? runtime::VerifyAttnForce::V2 : runtime::VerifyAttnForce::Dpas));
    std::printf("verify attention: %s (spec 28c)\n", v.c_str());
  }
  const model::ModelDesc& md = *e.model().desc;
  if (argc > 7 && std::string(argv[4]) == "trace") {   // spec 28c diagnostic
    kGdnBytes = size_t(md.gdn_layers) * md.gdn_v_heads * 128 * 128 * 4;
    Ctx c{ctx, e, e.buffers().control.as<runtime::Control>(), l0::CmdList::immediate(ctx)};
    return trace(c, golden::read_ids(argv[5]), uint32_t(std::stoul(argv[6])), uint32_t(std::stoul(argv[7])));
  }
  const size_t gdn_layers = md.gdn_layers, fa_layers = md.fa_layers;
  kGdnBytes = gdn_layers * md.gdn_v_heads * 128 * 128 * 4;
  Ctx c{ctx, e, e.buffers().control.as<runtime::Control>(), l0::CmdList::immediate(ctx)};
  bool ok = true;

  // Sizes with the head (Review Focus 4): +hh0 in the state, a 17th KV layer.
  // Qwen3.8: 166723584 + 10240 and 134217728 / 16 x 17 (the gdn_state + conv ring of
  // 48 GDN layers; 2048 positions of 16 FA layers' K and V). PERF-56: at the loaded model's widths (Ornith's wide and
  // deep lists run here too).
  CHECK_EQ(e.state_bytes(), kGdnBytes + gdn_layers * 32 * md.gdn_conv_dim() * 2 + md.hidden * 2);   // the ring 32 deep
  // Spec 12b: kv heads x 256 x 2 B per layer and position at bf16, kv heads x (256 + 2) B at int8.
  CHECK_EQ(e.kv_bytes(2048), size_t(2048) * 2 * (fa_layers + 1) * md.fa_kv_heads *
                                 (e.kv_cache() == runtime::KvCache::Int8 ? (256 + 2) : 256 * 2));

  if (argc > 4 && std::string(argv[4]) == "deep") {   // W32 / W64 (header)
    CHECK(e.deep_built(runtime::Engine::kDeepK32) && e.deep_built(runtime::Engine::kDeepK64));
    for (uint32_t n : {1998u, 2053u})
      for (uint32_t W : {32u, 64u}) {
        e.reset();
        e.prefill(head(ids, n));
        std::printf("W%u at n = %u\n", W, n);
        const Snap s0(c);
        ok &= w_deep(c, s0, W);
      }
    std::printf("W32 / W64: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
  }
  if (wide_mode) {   // W16 (header)
    CHECK(e.wide_built());
    for (uint32_t n : {1998u, 2053u}) {
      e.reset();
      e.prefill(head(ids, n));
      std::printf("W16 at n = %u\n", n);
      const Snap s0(c);
      ok &= w16(c, s0, head(ids, n), "mtp_verify_" + md.name + "_wide_int8_n" + std::to_string(n));
    }
    std::printf("W16: %s\n", ok ? "PASS" : "FAIL");
    return spec_judge::exit_code(ok, "mtp_verify_wide_test");
  }

  // --- M2 at two prompt lengths ----------------------------------------------------
  std::vector<uint32_t> m2_at = {1998u, 2053u};
  if (deep) m2_at.push_back(kMaxLen - 1026);
  for (uint32_t n : m2_at) {
    e.reset();
    e.prefill(head(ids, n));
    std::printf("M2 at n = %u (n %% 16 = %u)\n", n, n % 16);
    const Snap s0(c);
    const Ref ref = reference(c, s0, 72);
    ok &= m2(c, s0, ref, head(ids, n),
             "mtp_verify_" + e.model().desc->name + "_" + runtime::kv_cache_name(e.kv_cache()) + "_n" + std::to_string(n));
  }
  std::printf("M2: %s\n", ok ? "PASS" : "FAIL");

  // --- Review Focus 4: spec 7b C1 case A with MTP on --------------------------------
  std::vector<uint32_t> rf4_at = {4395u};
  if (deep) rf4_at.push_back(kMaxLen / 2 + 4395);
  for (const uint32_t p : rf4_at) {
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
    const uint32_t n = kMaxLen - 4;
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
  return spec_judge::exit_code(ok, "mtp_verify_test");
}
