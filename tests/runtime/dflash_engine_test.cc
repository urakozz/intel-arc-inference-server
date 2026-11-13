// Spec 19c on the card: DFlash in the decode loop - runtime::Engine with the real drafter (runtime::dflash::
// DFlashDrafter) beside the gate checkpoint (int8 head, bf16 KV).
//
// S0 = a prompt prefilled to pos n (the drafter's ring as prefill left it) with its pending id; the reference is T
// plain steps from S0 (verify(0) + commit(0, argmax) - each projects its one row): ids g[0..T], and for the first 8
// steps the logits, the GDN state after the step, the step's tap row and, at the end, the ring rows of positions
// n .. n + 7.
//
//   L    the lists: verify at M = 1..8 = decode_launches (774 on Qwen3.8; the taps are copies), the draft list 87,
//        the projection 8; the snapshot is the plain engine's state plus the drafter's whole ring (spec 19d,
//        decision 5: 41,943,040 B).
//   SR   (spec 19d) save_state / load_state carry the ring: after a reset() (a zero ring) the restore brings the
//        ring back bitwise, every slot of every layer's K and V.
//   (Spec 28b, the operator's ruling 2026-11-03: the verify rows run gemm_sm - logits rows by the f32 judge where one
//   exists, taps and committed ring rows bitwise the M = 8 verify's (N3), their distance to the plain steps recorded;
//   the ids against the plain run by spec 28 N6 - tests/server/spec_judge.h.)
//   T1   (Task 1, Review Focus 1-2) for k = 1..7: restore S0, the true greedy drafts g[1..k], verify(k): every row's
//        TAPS against the reference step's tap row (cosine >= 0.99999; bitwise expected - the M-row kernels sum each
//        row as M = 1 does - and recorded); commit(k): the ring rows of positions n .. n + k against the reference
//        ring's (worst row cosine >= 0.9999, bitwise recorded) - the commit projected exactly the verify's rows
//        0..k. In P (below) the rejected rows' positions keep S0's ring rows bitwise: rows past j are never
//        projected, and the newest id has no tap.
//   D3   (spec 19 §6, the M2 extension to M = 5..8) the same verifies: row r's logits and the GDN state slot
//        (live + r) % 8 against the reference step r (cosine >= 0.99999; bitwise is recorded), row r's argmax
//        == g[r + 1].
//   P    (partial acceptance; the GDN state at the last accepted token) for k = 7 and k = 4, every j in 0..k:
//        restore S0, drafts g[1..j], a wrong id at j + 1, junk after; verify(k); commit(j, argmax row j): the live
//        GDN slot equals the reference state after step j (cosine >= 0.99999, bitwise recorded) - the slot index
//        is the whole commit, no copy and no replay of the accepted rows (vLLM's per-token replay state, for
//        contrast, re-applies the accepted rows' rank-1 updates); the ring as T1 / Review Focus 1 say; then 64 plain
//        greedy ids == the run that never saw the rejected drafts - the same verify(k) with every draft true,
//        commit(j), 64 plain ids - bitwise (spec 28 N3 + N4: rows 0..j do not depend on rows past j; the conv
//        ring's stale positions past n + j are overwritten before any row reads them). Since spec 28b the verify
//        rows run gemm_sm, so against the plain run's g[j + 1 .. j + 64] they must be identical up to a first
//        difference the golden tie rule classes as a near-tie (tests/server/greedy_tie.h). Run at n % 16 == 14 so
//        pos % 16 wraps inside the verify rows.
//   D2   (greedy is lossless, on the card) from S0, the loop the server runs - d.draft(K) -> verify(K) -> the
//        longest agreeing prefix -> commit - at K = 7, 3 and 1: the 160 ids equal g; acceptance printed.
//   RF3  near max_len: max_verify_k() falls 7 .. 0, the loop runs to max_len - 1 and emits the plain run's ids;
//        then verify(1) refuses.
//   PF   (Task 3, the prefill tail) a 3000-id prompt: one prefill() call fills the ring with the taps of its last
//        2048 positions (the window) - the same ring, BITWISE, from the prompt prefilled in two calls split at 1024
//        (a multiple of 64: prefill_split_test's bitwise case) and in chunks of 1000 (chunk boundaries move no
//        per-row value... except the GDN scan's 64-chunks: gated at row cosine, median >= 0.9998); and against the
//        decode path (16 ids prefilled, the rest ingested - verify(0) + commit(0) per id, each projecting its row):
//        the window's rows at median cosine >= 0.999 (prefill's GEMMs vs decode's GEMVs), worst printed; and no row of
//        the window is left zero (every position the ring reads back was projected).
//
// usage: dflash_engine_test <target snapshot> <drafter> [max_len = 16384]   (cwd = the source tree)
// SKIP (77) when either checkpoint is missing.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "check.h"
#include "../server/greedy_tie.h"
#include "../server/spec_judge.h"
#include "golden_common.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"
#include "loader/dflash_loader.h"
#include "loader/loader.h"
#include "loader/snapshot.h"
#include "model/qwen35.h"
#include "runtime/capture.h"
#include "runtime/control.h"
#include "runtime/dflash/dflash_drafter.h"
#include "runtime/dflash/dflash_sizes.h"
#include "runtime/engine.h"
#include "runtime/verify_attn.h"

namespace {
using model::Qwen35;
uint32_t kMaxLen = 16384;   // argv[3]
constexpr size_t V = Qwen35::kVocab, VU = Qwen35::kVocabUsed;
constexpr uint32_t kK = 7;   // DFlash 2's block of 8
size_t kGdnBytes = 0, kTapRow = 0;

double cosine(const float* a, const float* b, size_t n) {
  double ab = 0, aa = 0, bb = 0;
  for (size_t i = 0; i < n; ++i) {
    ab += double(a[i]) * b[i];
    aa += double(a[i]) * a[i];
    bb += double(b[i]) * b[i];
  }
  return aa == 0 && bb == 0 ? 1.0 : ab / std::sqrt(aa * bb);
}
std::vector<uint32_t> head(const std::vector<uint32_t>& v, size_t n) {
  return std::vector<uint32_t>(v.begin(), v.begin() + n);
}

struct Ctx {
  l0::Context& ctx;
  runtime::Engine& e;
  runtime::dflash::DFlashDrafter& d;
  runtime::Control* ctl;
  l0::CmdList imm;
  template <class T>
  std::vector<T> read(const void* src, size_t bytes) {
    std::vector<T> v(bytes / sizeof(T));
    imm.copy(v.data(), src, bytes);
    return v;
  }
};

// The ring rows of positions [p, p + n): per position, every layer's K then V rows (bf16).
std::vector<uint16_t> ring_rows(Ctx& c, uint32_t p, uint32_t n) {
  runtime::dflash::DFlashBuffers& b = c.d.buffers();
  const model::DFlashDesc& dd = c.d.desc();
  const size_t kvn = dd.kv_n(), slots = dd.window[0];
  std::vector<uint16_t> out;
  const std::vector<uint16_t> k = c.read<uint16_t>(b.ring_k.ptr(), b.ring_k.size()),
                              v = c.read<uint16_t>(b.ring_v.ptr(), b.ring_v.size());
  for (uint32_t i = 0; i < n; ++i)
    for (uint32_t l = 0; l < dd.layers; ++l)
      for (const auto* r : {&k, &v}) {
        const size_t base = (size_t(l) * slots + (p + i) % slots) * kvn;
        out.insert(out.end(), r->begin() + base, r->begin() + base + kvn);
      }
  return out;
}
size_t ring_row_elems(const Ctx& c) { return size_t(c.d.desc().layers) * 2 * c.d.desc().kv_n(); }

// bf16 rows as floats, and the worst per-position cosine of `n` positions of two ring_rows() results.
std::vector<float> widen(const uint16_t* h, size_t n) {
  std::vector<float> f(n);
  for (size_t i = 0; i < n; ++i) {
    const uint32_t b = uint32_t(h[i]) << 16;
    std::memcpy(&f[i], &b, 4);
  }
  return f;
}
double worst_row_cos(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, size_t row, uint32_t n) {
  double w = 1.0;
  for (uint32_t i = 0; i < n; ++i) {
    const std::vector<float> fa = widen(a.data() + i * row, row), fb = widen(b.data() + i * row, row);
    w = std::min(w, cosine(fa.data(), fb.data(), row));
  }
  return w;
}

// A session snapshot: state (since spec 19d with the drafter's whole ring) + KV [0, pos) + the pending id.
struct Snap {
  l0::Mem st, kv;
  uint32_t pos, pending;
  explicit Snap(Ctx& c)
      : st(c.ctx, l0::MemKind::Host, c.e.state_bytes()),
        kv(c.ctx, l0::MemKind::Host, std::max(size_t(1), c.e.kv_bytes(c.e.pos()))),
        pos(c.e.pos()),
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

std::vector<float> logits_rows(Ctx& c, uint32_t rows) {
  return c.read<float>(c.e.verify_logits_device(), size_t(rows) * V * 4);
}
std::vector<float> gdn_slot(Ctx& c, uint32_t s) {
  const void* src = s == 0 ? c.e.buffers().gdn_state.ptr()
                           : c.e.drafter_slots()->as<uint8_t>() + size_t(s - 1) * kGdnBytes;
  return c.read<float>(src, kGdnBytes);
}
std::vector<uint16_t> tap_row(Ctx& c, uint32_t r) {
  return c.read<uint16_t>(c.d.taps().as<uint8_t>() + size_t(r) * kTapRow * 2, kTapRow * 2);
}

struct Ref {
  std::vector<uint32_t> g;                     // g[0] = pending at S0, g[i + 1] = argmax of step i
  std::vector<std::vector<float>> logits;      // step r's logits, r <= kK
  std::vector<std::vector<float>> state;       // GDN state after step r, r <= kK
  std::vector<std::vector<uint16_t>> taps;     // step r's tap row, r <= kK
  std::vector<uint16_t> ring;                  // positions n .. n + kK after the reference steps
  std::vector<greedy_tie::RefStep> steps;      // step i's decision (it decided g[i + 1]) - spec 28 N6's tie rule
};
Ref reference(Ctx& c, const Snap& s0, uint32_t T) {
  s0.restore(c);
  Ref r;
  r.g.push_back(c.ctl->cur_token[0]);
  for (uint32_t i = 0; i < T; ++i) {
    c.e.verify(0);
    const std::vector<float> row = logits_rows(c, 1);
    r.steps.push_back(greedy_tie::ref_step(row.data(), uint32_t(VU)));
    if (i <= kK) {
      r.logits.push_back(row);
      r.taps.push_back(tap_row(c, 0));
    }
    c.e.commit(0, c.e.verify_ids()[0]);
    if (i <= kK) r.state.push_back(gdn_slot(c, c.ctl->gdn_live));
    r.g.push_back(c.ctl->cur_token[0]);
    if (i == kK) r.ring = ring_rows(c, s0.pos, kK + 1);
  }
  return r;
}

// The M = 8 verify (k = kK, true drafts) of S0: each row's taps and, after commit(kK), the ring rows n .. n + kK - the
// reference of spec 28 N3 in the engine (a row's taps and its projection do not depend on M >= 2).
struct M8 {
  std::vector<std::vector<uint16_t>> taps;
  std::vector<uint16_t> ring;
};
// Spec 19 v2: the real queues and buffers, including flushes that must precede a tap overwrite.
void batched_case(Ctx& c, const Snap& s0) {
  struct Frame {
    runtime::dflash::DFlashDraftOut draft;
    std::vector<uint32_t> ids;
    std::vector<float> logits;
    std::vector<uint16_t> ring;
    uint32_t accepted, next;
  };
  std::vector<Frame> frames;
  s0.restore(c);
  const auto initial_ring = ring_rows(c, 0, c.d.window());
  for (unsigned i = 0; i < 2; ++i) {
    Frame f;
    f.draft = c.d.draft(c.e, kK, 0.0f, 0);
    c.e.verify(kK);
    f.ids.assign(c.e.verify_ids(), c.e.verify_ids() + kK + 1);
    f.logits = logits_rows(c, kK + 1);
    f.accepted = 0;
    while (f.accepted < kK && f.draft.ids[f.accepted] == f.ids[f.accepted]) ++f.accepted;
    f.next = f.ids[f.accepted];
    c.e.commit(f.accepted, f.next);
    f.ring = ring_rows(c, 0, c.d.window());
    frames.push_back(std::move(f));
  }
  const Snap committed(c);
  auto batched = [&] {
    s0.restore(c);
    for (size_t i = 0; i < frames.size(); ++i) {
      const Frame& f = frames[i];
      const uint64_t before = c.e.queue_submissions();
      const auto& draft = c.d.draft_with([&](l0::CmdList& list) { c.e.verify(kK, &list); }, kK, 0.0f, 0);
      CHECK(c.e.queue_submissions() == before + 1);
      CHECK(std::memcmp(&draft, &f.draft, sizeof draft) == 0);
      CHECK(f.ids == std::vector<uint32_t>(c.e.verify_ids(), c.e.verify_ids() + kK + 1));
      const auto logits = logits_rows(c, kK + 1);
      CHECK(std::memcmp(logits.data(), f.logits.data(), logits.size() * sizeof(float)) == 0);
      c.e.commit(f.accepted, f.next, true);
      CHECK(c.e.queue_submissions() == before + 1);   // projection rides on the next iteration
      CHECK(ring_rows(c, 0, c.d.window()) == (i ? frames[i - 1].ring : initial_ring));
    }
  };
  batched();
  const uint64_t before = c.e.queue_submissions();
  const Snap saved(c);   // a snapshot materializes the deferred projection
  CHECK(c.e.queue_submissions() == before + 1);
  CHECK(std::memcmp(saved.st.ptr(), committed.st.ptr(), c.e.state_bytes()) == 0);
  c.e.flush_projection();
  CHECK(c.e.queue_submissions() == before + 1);

  l0::Mem state(c.ctx, l0::MemKind::Host, c.e.state_bytes());
  l0::Mem kv(c.ctx, l0::MemKind::Host, c.e.kv_bytes(committed.pos));
  for (unsigned mode = 0; mode < 7; ++mode) {
    auto boundary = [&] {
      switch (mode) {
        case 0: c.e.save_state(state.ptr()); break;
        case 1: c.e.save_kv(0, c.e.pos(), kv.ptr()); break;
        case 2: (void)c.e.generate(1); break;
        case 3: c.e.ingest({760}); break;
        case 4: c.e.prefill({760}); break;
        case 5: c.e.reset(); break;
        case 6: s0.restore(c); break;
      }
    };
    committed.restore(c);
    boundary();
    const auto expected_ring = ring_rows(c, 0, c.d.window());
    const Snap expected(c);
    batched();
    boundary();
    CHECK(ring_rows(c, 0, c.d.window()) == expected_ring);
    const Snap actual(c);
    CHECK(actual.pos == expected.pos && actual.pending == expected.pending);
    CHECK(std::memcmp(actual.st.ptr(), expected.st.ptr(), c.e.state_bytes()) == 0);
  }
  s0.restore(c);
  std::printf("BATCH: one submit per iteration; draft/verify, snapshots, plain/ingest/prefill, reset/restore bitwise\n");
}

M8 m8_reference(Ctx& c, const Snap& s0, const Ref& ref) {
  M8 m;
  s0.restore(c);
  for (uint32_t i = 1; i <= kK; ++i) c.ctl->cur_token[i] = ref.g[i];
  c.e.verify(kK);
  for (uint32_t r = 0; r <= kK; ++r) m.taps.push_back(tap_row(c, r));
  c.e.commit(kK, c.e.verify_ids()[kK]);
  m.ring = ring_rows(c, s0.pos, kK + 1);
  return m;
}

// Do the M = k + 1 verify and the M = 8 one bind the same attention kernel at S0's depth (runtime/verify_attn.h:
// spec 28c's table puts M = 2..5 on v2 below 4096 keys and M = 6..8 on attn_verify_dpas from the first key)? Where
// they do, a row's taps and its projection are bitwise the M = 8 verify's (N3, unchanged). Where they do not (the
// coordinator's ruling 2026-11-04: not a weakened bitwise check), the rows are judged like every other two-kernel
// pair in spec 28 (spec_judge.h's N4 row rule, golden::kCosSlack): the logits and ids by the f32 judge as before,
// and the taps / ring rows - no f32 reference exists for them - by their cosine distance to the plain M = 1 rows,
// at most kCosSlack x the M = 8 verify's distance to the same rows.
bool same_attn_as_m8(Ctx& c, const Snap& s0, uint32_t k) {
  const model::ModelDesc& d = *c.e.model().desc;
  return runtime::verify_attn_kind(k + 1, s0.pos + 1, false, d.fa_q_heads, d.fa_kv_heads) ==
         runtime::verify_attn_kind(kK + 1, s0.pos + 1, false, d.fa_q_heads, d.fa_kv_heads);
}
// The two-kernel judgement: `cos` (this path vs plain) against `cos8` (the M = 8 verify vs plain).
bool near_m8(double cos, double cos8) { return golden::cos_dist_ratio(cos, cos8) <= golden::kCosSlack; }

// T1 + D3: verify(k) with the true drafts, then commit(k). Since spec 28b (the operator's ruling 2026-11-03): the
// logits rows by the f32 judge where one exists (spec_judge.h; else cosine >= 0.99999), the taps and the committed
// ring rows BITWISE those of the M = 8 verify (N3); their cosines against the plain M = 1 steps are recorded.
bool t1_d3(Ctx& c, const Snap& s0, const Ref& ref, const std::vector<uint32_t>& prompt, const std::string& tag) {
  bool ok = true;
  const M8 m8 = m8_reference(c, s0, ref);
  const spec_judge::RowJudge rj = spec_judge::open_rows(
      golden::ForcedJudge{}, tag + "_d3", prompt, std::vector<uint32_t>(ref.g.begin(), ref.g.begin() + kK + 2),
      uint32_t(V));
  for (uint32_t k = 1; k <= kK; ++k) {
    const bool same = same_attn_as_m8(c, s0, k);
    s0.restore(c);
    for (uint32_t i = 1; i <= k; ++i) c.ctl->cur_token[i] = ref.g[i];
    c.e.verify(k);
    const std::vector<float> lg = logits_rows(c, k + 1);
    for (uint32_t r = 0; r <= k; ++r) {
      const double cl = cosine(lg.data() + size_t(r) * V, ref.logits[r].data(), VU);
      const bool lbits = std::memcmp(lg.data() + size_t(r) * V, ref.logits[r].data(), V * 4) == 0;
      const std::vector<float> sl = gdn_slot(c, (c.ctl->gdn_live + r) % runtime::TapVerifyDims::kSlots);
      const double cs = cosine(sl.data(), ref.state[r].data(), sl.size());
      const bool sbits = std::memcmp(sl.data(), ref.state[r].data(), kGdnBytes) == 0;
      const std::vector<uint16_t> tr = tap_row(c, r);
      const bool tbits = tr == ref.taps[r];
      const double ct = worst_row_cos(tr, ref.taps[r], tr.size(), 1);
      const bool id_ok = c.e.verify_ids()[r] == ref.g[r + 1];
      const bool t8bits = tr == m8.taps[r];
      const double ct8 = worst_row_cos(m8.taps[r], ref.taps[r], tr.size(), 1);
      const bool t8 = same ? t8bits : near_m8(ct, ct8);
      const spec_judge::RowVerdict rv = spec_judge::row(lg.data() + size_t(r) * V, ref.logits[r].data(), rj, r + 1,
                                                        cl, uint32_t(VU));
      char m8s[160];
      if (same) std::snprintf(m8s, sizeof m8s, "%s", t8bits ? "bitwise" : "DIFFER");
      else
        std::snprintf(m8s, sizeof m8s, "another attention kernel - distance to plain x%.2f of its %.9f (slack %.1f) %s",
                      golden::cos_dist_ratio(ct, ct8), ct8, golden::kCosSlack, t8 ? "ok" : "FAIL");
      std::printf("  D3 k=%u row %u: logits cos %.9f%s  GDN slot cos %.9f%s  taps cos %.9f%s (the M = 8 row's: %s)  id "
                  "%s%s\n", k, r, cl, lbits ? " (bitwise)" : "", cs, sbits ? " (bitwise)" : "", ct,
                  tbits ? " (bitwise)" : "", m8s, id_ok ? "ok" : "DIFFERS", spec_judge::row_line(rv).c_str());
      ok &= rv.pass && cs >= 0.99999 && t8 && id_ok;
    }
    c.e.commit(k, c.e.verify_ids()[k]);
    const std::vector<uint16_t> ring = ring_rows(c, s0.pos, k + 1);
    const bool rbits =
        std::equal(ring.begin(), ring.end(), ref.ring.begin(), ref.ring.begin() + ring.size());
    const double rc = worst_row_cos(ring, ref.ring, ring_row_elems(c), k + 1);
    const bool r8bits = std::equal(ring.begin(), ring.end(), m8.ring.begin());
    const double rc8 = worst_row_cos(m8.ring, ref.ring, ring_row_elems(c), k + 1);
    const bool r8 = same ? r8bits : near_m8(rc, rc8);
    char m8s[160];
    if (same) std::snprintf(m8s, sizeof m8s, "%s", r8bits ? "bitwise" : "DIFFER");
    else
      std::snprintf(m8s, sizeof m8s, "another attention kernel - distance to plain x%.2f of its %.9f (slack %.1f) %s",
                    golden::cos_dist_ratio(rc, rc8), rc8, golden::kCosSlack, r8 ? "ok" : "FAIL");
    std::printf("  T1 k=%u: ring rows of positions %u..%u vs the plain reference's: worst cos %.9f%s (recorded); the "
                "M = 8 verify's: %s\n", k, s0.pos, s0.pos + k, rc, rbits ? " (bitwise)" : "", m8s);
    ok &= r8;
  }
  return ok;
}

// P: partial acceptance at depth k. The committed ring rows BITWISE the M = 8 verify's (N3; their cosine against the
// plain steps recorded); the 64 further ids against the clean run (bitwise) and the plain run (spec 28 N6).
bool partial(Ctx& c, const Snap& s0, const Ref& ref, uint32_t k, const std::vector<uint32_t>& prompt,
             const std::string& tag) {
  bool ok = true;
  const M8 m8 = m8_reference(c, s0, ref);
  const golden::ForcedJudge fj;
  const size_t row = ring_row_elems(c);
  const std::vector<uint16_t> ring0 = [&] {
    s0.restore(c);
    return ring_rows(c, s0.pos, kK + 1);
  }();
  for (uint32_t j = 0; j <= k; ++j) {
    // the run that never saw a rejected draft: every draft true, the same verify(k), commit(j), 64 ids
    s0.restore(c);
    for (uint32_t i = 1; i <= k; ++i) c.ctl->cur_token[i] = ref.g[i];
    c.e.verify(k);
    c.e.commit(j, c.e.verify_ids()[j]);
    const std::vector<uint32_t> clean = c.e.generate(64);
    s0.restore(c);
    for (uint32_t i = 1; i <= k; ++i) {
      uint32_t dft = ref.g[i];
      if (i == j + 1) dft = (ref.g[i] + 7919) % uint32_t(VU);         // the first rejection
      if (i > j + 1) dft = (ref.g[i] + 104729 * i) % uint32_t(VU);    // junk after it
      c.ctl->cur_token[i] = dft;
    }
    c.e.verify(k);
    bool rows_ok = true;
    for (uint32_t r = 0; r <= j; ++r) rows_ok &= c.e.verify_ids()[r] == ref.g[r + 1];
    c.e.commit(j, c.e.verify_ids()[j]);
    const std::vector<float> live = gdn_slot(c, c.ctl->gdn_live);
    const double cs = cosine(live.data(), ref.state[j].data(), live.size());
    const bool sbits = std::memcmp(live.data(), ref.state[j].data(), kGdnBytes) == 0;
    const std::vector<uint16_t> ring = ring_rows(c, s0.pos, kK + 1);
    const bool same = same_attn_as_m8(c, s0, k);
    const bool committed_bits = std::equal(ring.begin(), ring.begin() + (j + 1) * row, m8.ring.begin());
    const double committed_cos = worst_row_cos(ring, ref.ring, row, j + 1);
    const double m8_cos = worst_row_cos(m8.ring, ref.ring, row, j + 1);
    const bool committed = same ? committed_bits : near_m8(committed_cos, m8_cos);
    char cm[200];
    if (same) std::snprintf(cm, sizeof cm, "%s", committed ? "projected (the M = 8 rows, bitwise)" : "WRONG");
    else
      std::snprintf(cm, sizeof cm, "projected (another attention kernel than M = 8: distance to plain x%.2f of the "
                    "M = 8 rows' %.6f, slack %.1f) %s", golden::cos_dist_ratio(committed_cos, m8_cos), m8_cos,
                    golden::kCosSlack, committed ? "ok" : "WRONG");
    const bool untouched = std::equal(ring.begin() + (j + 1) * row, ring.end(), ring0.begin() + (j + 1) * row);
    const std::vector<uint32_t> ids = c.e.generate(64);
    size_t diff = 0;
    for (size_t i = 0; i < ids.size(); ++i) diff += ids[i] != clean[i];
    const std::vector<uint32_t> pg(ref.g.begin() + j + 1, ref.g.begin() + j + 1 + ids.size());
    const std::vector<greedy_tie::RefStep> ps(ref.steps.begin() + j, ref.steps.begin() + j + ids.size());
    std::vector<uint32_t> ctx = prompt;
    ctx.insert(ctx.end(), ref.g.begin(), ref.g.begin() + j + 1);
    const spec_judge::StreamVerdict sv = spec_judge::streams(fj, tag + "_p", ctx, pg, ps, ids, uint32_t(V), uint32_t(VU));
    std::printf("  P k=%u j=%u: live slot %u, GDN state vs the state after step %u cos %.9f%s, accepted rows %s, "
                "ring rows %u..%u %s (cos vs plain %.6f), %u..%u %s, 64 further ids: %zu differ from the clean run; "
                "the plain run: %s\n",
                k, j, c.ctl->gdn_live, j, cs, sbits ? " (bitwise)" : "", rows_ok ? "ok" : "DIFFER", s0.pos,
                s0.pos + j, cm, committed_cos, s0.pos + j + 1,
                s0.pos + kK, untouched ? "untouched" : "WRITTEN", diff, sv.line().c_str());
    ok &= cs >= 0.99999 && rows_ok && committed && untouched && diff == 0 && sv.pass;
  }
  return ok;
}

// One greedy DFlash iteration at depth k (0 = a plain step): returns the ids it emits.
std::vector<uint32_t> iterate(Ctx& c, uint32_t k, uint64_t* drafted, uint64_t* accepted) {
  std::vector<uint32_t> out{c.ctl->cur_token[0]};
  k = std::min(k, c.e.max_verify_k());
  if (k == 0) {
    c.e.verify(0);
    c.e.commit(0, c.e.verify_ids()[0]);
    return out;
  }
  const runtime::dflash::DFlashDraftOut& o = c.d.draft(c.e, k, 0.0f, 0);
  std::vector<uint32_t> drafts(o.ids, o.ids + k);
  c.e.verify(k);
  uint32_t j = 0;
  while (j < k && drafts[j] == c.e.verify_ids()[j]) ++j;
  for (uint32_t i = 0; i < j; ++i) out.push_back(drafts[i]);
  c.e.commit(j, c.e.verify_ids()[j]);
  *drafted += k;
  *accepted += j;
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::string snap, dsnap;
  try {
    snap = loader::resolve_snapshot(argc > 1 ? argv[1] : "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ");
    dsnap = loader::resolve_snapshot(argc > 2 ? argv[2] : "z-lab/Qwen3.8-27B-DFlash2");
  } catch (const std::exception& ex) {
    std::printf("SKIP: %s\n", ex.what());
    return 77;
  }
  if (argc > 3) kMaxLen = uint32_t(std::stoul(argv[3]));
  CHECK(kMaxLen >= 16384 && kMaxLen % 256 == 0);
  const auto ids = [] {
    std::vector<uint32_t> v = golden::read_ids("tests/golden/prompts/long32k.ids");
    const std::vector<uint32_t> file = v;
    while (v.size() <= kMaxLen) v.insert(v.end(), file.begin(), file.end());
    return v;
  }();
  l0::Context ctx(0);
  loader::LoadedModel model = loader::load(ctx, snap, kMaxLen, /*mtp=*/false, loader::LmHeadForm::Int8);
  loader::DFlashWeights w = loader::load_drafter(ctx, dsnap, model, loader::DraftFormat::Int8);
  const uint32_t head_rows = model.linears.at({loader::kTopLevel, model::LinearId::LmHead}).shape.N;
  runtime::dflash::DFlashDrafter d(ctx, std::move(w), head_rows, kMaxLen, kK);
  runtime::Engine e(ctx, std::move(model), kMaxLen, /*debug_resid=*/false, runtime::default_kv_cache(), &d);
  d.attach(e, /*draft_vocab=*/false);
  const model::ModelDesc& md = *e.model().desc;
  kGdnBytes = size_t(md.gdn_layers) * md.gdn_v_heads * 128 * 128 * 4;
  kTapRow = d.tap_layers().size() * size_t(md.hidden);
  Ctx c{ctx, e, d, e.buffers().control.as<runtime::Control>(), l0::CmdList::immediate(ctx)};
  bool ok = true;

  // --- L ------------------------------------------------------------------------------
  {
    bool l_ok = e.has_drafter() && e.max_draft() == kK && e.gdn_slots() == 8;
    for (uint32_t M = 1; M <= kK + 1; ++M) l_ok &= e.verify_step(M).kernel_count == runtime::decode_launches(md);
    l_ok &= d.draft_step()->kernel_count == runtime::dflash::draft_launches(d.desc());
    l_ok &= d.project_step().kernel_count == runtime::dflash::project_launches(d.desc());
    l_ok &= e.state_bytes() == kGdnBytes + size_t(md.gdn_layers) * 16 * 10240 * 2 + 41943040;   // + the ring (19d)
    l_ok &= d.ring_bytes() == 41943040 && d.key() != 0;
    std::printf("L: verify M = 1..8 %zu launches each, draft %zu, projection %zu, state %zu B: %s\n",
                e.verify_step(1).kernel_count, d.draft_step()->kernel_count, d.project_step().kernel_count,
                e.state_bytes(), l_ok ? "ok" : "WRONG");
    std::printf("%s\n", e.memory_line().c_str());
    ok &= l_ok;
  }

  // --- T1, D3, P at two prompt lengths (n % 16 = 14 wraps the conv ring inside the rows) --------------------
  for (uint32_t n : {1998u, 2062u}) {
    e.reset();
    e.prefill(head(ids, n));
    std::printf("n = %u (n %% 16 = %u)\n", n, n % 16);
    const Snap s0(c);
    batched_case(c, s0);
    {   // SR (spec 19d): the snapshot carries the ring - a reset zeroes it, the restore brings it back bitwise
      const std::vector<uint16_t> before = ring_rows(c, 0, d.window());
      e.reset();
      s0.restore(c);
      const bool sr = ring_rows(c, 0, d.window()) == before;
      std::printf("SR at n = %u: the ring after reset() + restore %s\n", n, sr ? "equals the saved one (bitwise)"
                                                                               : "DIFFERS");
      ok &= sr;
    }
    const Ref ref = reference(c, s0, 200);
    const std::vector<uint32_t> prompt = head(ids, n);
    const std::string tag = "dflash_engine_int8_n" + std::to_string(n);
    const bool a = t1_d3(c, s0, ref, prompt, tag);
    const bool p7 = partial(c, s0, ref, 7, prompt, tag), p4 = partial(c, s0, ref, 4, prompt, tag);
    std::printf("T1 + D3 at n = %u: %s; P: %s\n", n, a ? "PASS" : "FAIL", p7 && p4 ? "PASS" : "FAIL");
    ok &= a && p7 && p4;

    // --- D2 on the card: the server's greedy loop at K = 7, 3, 1 ---------------------------------------
    for (uint32_t k : {kK, 3u, 1u}) {
      s0.restore(c);
      std::vector<uint32_t> got;
      uint64_t drafted = 0, accepted = 0, iters = 0;
      while (got.size() < 160) {
        const std::vector<uint32_t> o = iterate(c, k, &drafted, &accepted);
        got.insert(got.end(), o.begin(), o.end());
        ++iters;
      }
      got.resize(160);
      size_t diff = 0;
      for (size_t i = 0; i < got.size(); ++i) diff += got[i] != ref.g[i];
      // spec 28 N6: steps2[i] decided ref.g[i] (ref.steps[i - 1]; g[0] is the prefill's)
      std::vector<greedy_tie::RefStep> steps2{{UINT32_MAX, UINT32_MAX, 1e30}};
      steps2.insert(steps2.end(), ref.steps.begin(), ref.steps.begin() + 159);
      const spec_judge::StreamVerdict sv = spec_judge::streams(
          golden::ForcedJudge{}, tag + "_d2", prompt, std::vector<uint32_t>(ref.g.begin(), ref.g.begin() + 160), steps2,
          got, uint32_t(V), uint32_t(VU));
      std::printf("  D2 K=%u: 160 ids, %zu differ from plain (%s); %llu iterations, acceptance %.3f (%.2f ids an "
                  "iteration)\n",
                  k, diff, sv.line().c_str(), (unsigned long long)iters,
                  drafted ? double(accepted) / double(drafted) : 0.0, double(160) / double(iters));
      ok &= sv.pass;
    }
  }

  // --- RF3: the last positions of max_len ------------------------------------------------------------------
  {
    const uint32_t n = kMaxLen - 12;
    e.reset();
    e.prefill(head(ids, n));
    const Snap s0(c);
    std::vector<uint32_t> plain = e.generate(kMaxLen - 1 - n);
    s0.restore(c);
    std::vector<uint32_t> got, ks;
    uint64_t drafted = 0, accepted = 0;
    while (c.e.pos() < kMaxLen - 1) {
      ks.push_back(c.e.max_verify_k());
      const std::vector<uint32_t> o = iterate(c, kK, &drafted, &accepted);
      got.insert(got.end(), o.begin(), o.end());
    }
    got.resize(std::min(got.size(), plain.size()));
    bool refused = false;
    try {
      c.e.verify(1);
    } catch (const std::runtime_error&) {
      refused = true;
    }
    const bool rf3 = std::equal(got.begin(), got.end(), plain.begin()) && refused && ks.front() == kK &&
                     c.e.max_verify_k() == 0;
    std::printf("RF3: from pos %u, max_verify_k %u .. %u, %zu ids equal the plain run's: %s, verify(1) at pos %u %s\n",
                n, ks.front(), c.e.max_verify_k(), got.size(), rf3 ? "yes" : "NO", c.e.pos(),
                refused ? "refused" : "ran");
    ok &= rf3;
  }

  // --- PF: the prefill tail ----------------------------------------------------------------------------------
  {
    const uint32_t n = 3000, W = d.window();
    const size_t row = ring_row_elems(c);
    const auto window_rows = [&] { return ring_rows(c, n - W, W); };
    e.reset();
    e.prefill(head(ids, n));
    const std::vector<uint16_t> one = window_rows();
    e.reset();
    e.prefill(head(ids, 1024));
    e.prefill(std::vector<uint32_t>(ids.begin() + 1024, ids.begin() + n));
    const std::vector<uint16_t> split = window_rows();
    e.reset();
    e.prefill(head(ids, n), 1000);
    const std::vector<uint16_t> chunked = window_rows();
    e.reset();
    e.prefill(head(ids, 16));
    e.ingest(std::vector<uint32_t>(ids.begin() + 16, ids.begin() + n));
    const std::vector<uint16_t> decode = window_rows();
    const auto stats = [&](const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, double* med) {
      std::vector<double> cs(W);
      for (uint32_t i = 0; i < W; ++i) {
        const std::vector<float> fa = widen(a.data() + i * row, row), fb = widen(b.data() + i * row, row);
        cs[i] = cosine(fa.data(), fb.data(), row);
      }
      std::sort(cs.begin(), cs.end());
      *med = cs[W / 2];
      return cs.front();
    };
    double m_chunk = 0, m_dec = 0;
    const bool split_bits = split == one;
    const double w_chunk = stats(chunked, one, &m_chunk), w_dec = stats(decode, one, &m_dec);
    bool nonzero = true;   // every window row was projected (a zero row is a position prefill missed)
    for (uint32_t i = 0; i < W; ++i) {
      bool any = false;
      for (size_t x = 0; x < row && !any; ++x) any = one[i * row + x] != 0;
      nonzero &= any;
    }
    // Informational (spec 19f): the window's worst rows by position. On long32k they are the positions the model
    // itself amplifies a rounding-sized perturbation at (prefill_split_test's header: 987, 1050, 1212, 1565, 2091),
    // and they move in every pair of paths (one / chunked / decode) alike.
    for (const auto* o : {&chunked, &decode}) {
      std::vector<std::pair<double, uint32_t>> cs(W);
      for (uint32_t i = 0; i < W; ++i) {
        const std::vector<float> fa = widen(o->data() + i * row, row), fb = widen(one.data() + i * row, row);
        cs[i] = {cosine(fa.data(), fb.data(), row), n - W + i};
      }
      std::sort(cs.begin(), cs.end());
      std::printf("PF %s vs one, the 6 worst rows:", o == &chunked ? "chunks of 1000" : "the decode path");
      for (int k = 0; k < 6; ++k) std::printf(" %u (%.4f)", cs[k].second, cs[k].first);
      std::printf("\n");
    }
    // The chunked bar is 0.9998 (operator, 2026-10-27): measured on the card (plan 19f, card 0) the chunks of 1000
    // gave median cosine 0.9998879 against the earlier 0.9999, the worst rows at long32k positions 2091, 1564 / 1565,
    // 987, 1051, 1212 - the amplifier positions above, and prefill_split_test's own split bar at l0 is median 0.9998.
    // The split (bitwise), decode (0.999) and every-row-projected bars are unchanged.
    const bool pf = split_bits && m_chunk >= 0.9998 && m_dec >= 0.999 && nonzero;
    std::printf("PF: the window's %u ring rows after prefill(%u): split at 1024 %s; chunks of 1000 median cos %.7f "
                "(worst %.6f); the decode path median cos %.6f (worst %.6f); every row projected %s: %s\n",
                W, n, split_bits ? "bitwise" : "DIFFERS", m_chunk, w_chunk, m_dec, w_dec, nonzero ? "yes" : "NO",
                pf ? "PASS" : "FAIL");
    ok &= pf;
  }

  std::printf("dflash_engine_test: %s\n", ok ? "PASS" : "FAIL");
  return spec_judge::exit_code(ok, "dflash_engine_test");
}
